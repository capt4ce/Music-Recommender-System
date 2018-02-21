import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class SVD():
    def __init__(self):
        self.training_data = None
        self.training_data_pivot = None
        self.corr = None
        self.item_df = None
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
        # making user - item list
        # so that we can get the exact list of the items from the pivot
        # while for SVD we actually need item - user pivot list
        self.training_data_pivot = grouped_train_data.pivot(index=user_id_col, columns=item_id_col, values=pivot_id_col).fillna(0)

        self.item_df = pd.DataFrame(self.training_data_pivot.columns)
        self.item_df.columns=['song']

        # applying dimensionality reduction / compression
        # setting 12 latent variables
        # and 17 random states
        SVD = TruncatedSVD(n_components=12, random_state=17)

        # fitting pivot data to SVD model
        SVD_matrix = SVD.fit_transform(self.training_data_pivot.values.T)

        # constructing Pearson product-moment correlation coefficients matrix based on the SVD matrix
        self.corr = np.corrcoef(SVD_matrix)

    def recommend(self, no_of_recommendations, item=None):
        # matching the items with the supplied item
        # if no item supplied, choose a random item from the item_df
        if item==None:
            # query_index = (self.item_df.index[self.item_df[self.item_id_col] == 'Somebody To Love - Justin Bieber'])
            query_index = [np.random.choice(len(self.item_df))]
        else:
            query_index = (self.item_df.index[self.item_df[self.item_id_col] == item])
        print(query_index)

        if len(query_index)==0:
            print('No recommendation can be made for this item')
            return None

        corr_query_index = pd.DataFrame(self.corr[query_index[0]])
        corr_query_index.columns = ['corr']

        # merging item data with their corresponding coffey_hands
        df = pd.merge(self.item_df,corr_query_index, left_index=True, right_index=True).sort_values(['corr'], ascending=False)

        return df.head(no_of_recommendations)