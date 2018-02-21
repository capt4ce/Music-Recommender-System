import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz

class CF_kNN():
    #==============================================================
    # User - Item Collaborative Filtering based Recommender System
    # using k-Nearest-Neighbor
    #==============================================================

    def __init__(self):
        self.training_data = None
        self.training_data_pivot = None
        self.training_data_sparse_matrix = None
        self.model_knn = None
        self.user_id_col = None
        self.item_id_col = None
        self.pivot_id_col = None

    def create(self, training_data, user_id_col, item_id_col, pivot_id_col):
        # self.training_data = training_data.groupby([item_id_col]).agg({pivot_id_col: 'count'}).reset_index()
        self.training_data = training_data
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.pivot_id_col = pivot_id_col

        # grouping the duplicate user - item couple record
        grouped_train_data = self.training_data.groupby([item_id_col,user_id_col]).agg({pivot_id_col: 'sum'}).reset_index()

        # pivoting the grouped training data
        # making item - user list
        self.training_data_pivot = grouped_train_data.pivot(index=item_id_col, columns=user_id_col, values=pivot_id_col).fillna(0)

        # converting the list into Compressed Sparse Row (CSR) matrix
        self.training_data_sparse_matrix = csr_matrix(self.training_data_pivot.values)

        # specifiying to use cosine similarity as distance measure and brute algorithm
        self.model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')

        # training the model
        print('training...')
        self.model_knn.fit(self.training_data_sparse_matrix)

    # get unique items (songs) corresponding to a given user
    def getUserItems(self, user):
        user_data = self.training_data[self.training_data[self.user_id_col] == user]
        user_items = list(user_data[self.item_id_col].unique())
        return user_items


    def recommend(self, user, no_of_recommendations, item=None):
        # choosing an item for recommendation reference either from:
        # random item based on the song from the pivot set
        # or from the item supplied
        if item==None:
            query_index = np.random.choice(self.training_data_pivot.shape[0])
        else:
            ratio_tuples = []
            for i in self.training_data_pivot.index.values:
                ratio = fuzz.ratio(i.lower(), item.lower())
                if ratio >= 75:
                    current_query_index = self.training_data_pivot.index.tolist().index(i)
                    ratio_tuples.append((i, ratio, current_query_index))
            print('fuzzy ratio query results:\n' + str(ratio_tuples) + '\n')
            try:
                query_index = max(ratio_tuples, key = lambda x: x[1])[2] # get the index of the best artist match in the data
            except:
                print 'The item suplied didn\'t match any item in the data. Try again'
                return None

        # computing the distance and listing items that located closest to the query_index item
        distances, indices = self.model_knn.kneighbors(self.training_data_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = no_of_recommendations+1)

        # initializing recommendation dataframe
        df = pd.DataFrame(columns=['user_id', 'song', 'distance', 'rank'])

        # iterating the recommendation and moving them into the dataframe
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print 'Recommendations for {0}:\n'.format(self.training_data_pivot.index[query_index])
            else:
                # print '{0}: {1}, with distance of {2}:'.format(i, self.training_data_pivot.index[indices.flatten()[i]], distances.flatten()[i])
                df.loc[len(df)] = [user, self.training_data_pivot.index[indices.flatten()[i]], distances.flatten()[i], i]
        
        return df
