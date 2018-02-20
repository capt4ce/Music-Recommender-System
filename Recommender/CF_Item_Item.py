import pandas as pd
import numpy as np

class CF_Item_Item():
    def __init__(self):
        self.training_data = None
        self.user_id_col = None
        self.item_id_col = None
        # self.no_of_recommendations = None
        # self.recommendations = None

    def create(self, training_data, user_id_col, item_id_col):
        self.training_data = training_data
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col

    # get unique users for a given item (song)
    def getItemUsers(self, item):
        item_data = self.training_data[self.training_data[self.item_id_col] == item]
        item_users = set(item_data[self.user_id_col].unique())
        return item_users

    # get unique items (songs) corresponding to a given user
    def getUserItems(self, user):
        user_data = self.training_data[self.training_data[self.user_id_col] == user]
        user_items = list(user_data[self.item_id_col].unique())
        return user_items
    
    def getAllItems(self):
        all_items = list(self.training_data[self.item_id_col].unique())
        return all_items

    def constructCoocuranceMatrix(self, user_items, all_items):
        # get users for all imtes in user_items
        user_items_users = []
        for i in range(0,len(user_items)):
            user_items_users.append(self.getItemUsers(user_items[i]))

        # initializing cooccurence_matrix
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_items), len(all_items))), float)

        # calculating similarity between user_items and unique all_items in the training data
        # filling is column wise / along the rows
        for i in range(0,len(all_items)):
            # calculate unique listeners (users) of song (item) i
            items_i_data = self.training_data[self.training_data[self.item_id_col] == all_items[i]]
            users_i = set(items_i_data[self.user_id_col].unique())

            for j in range(0,len(user_items)):
                # get unique listeners (users) of song (item) j
                users_j = user_items_users[j]
                    
                # calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)

                # calculate cooccurence_matrix[j,i] as Jaccard Index
                # it uses IOU (Intersection Over Union)
                if len(users_intersection) != 0:
                    # calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        return cooccurence_matrix


    def generateTopRecommendation(self, user, no_of_recommendations, cooccurence_matrix, user_items, all_items):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        #=============================================================================
        # Calculate the weighted average score in the coocurance matrix of user items
        #=============================================================================
        
        # weighted average score = sum of each item (of all_item) occurance corresponding to each user_item / number of user_items
        # it is to get the weight for each item in all_items for doing fair scoring 
        item_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        # converting the weight scores into a list to be enumerated
        # it takes the first element (index 0) because the resulting array from the previous operation is in a multidimntional array
        item_scores = np.array(item_scores)[0].tolist()

        # relating the score with the song index
        # and sort it based on the score in descending order
        sorted_score_idx = sorted(((score,i) for i, score in enumerate(item_scores)), reverse=True)

        # creating empty recommendation dataframe
        df = pd.DataFrame(columns=['user_id', 'song', 'score', 'rank'])

        # filling the dataframe with the top recommendations
        rank = 1
        for i in range(0,len(sorted_score_idx)):
            # copy the data to the recommendation dataframe if:
            #    the score of the corresponding item is not null
            #    the corresponding item presents in the all_item based on its index is not present in the current user_items list to recommend something out of his listening list
            #    the rank is not more than no_of_recomendation requested
            if ~np.isnan(sorted_score_idx[i][0]) and all_items[sorted_score_idx[i][1]] not in user_items and rank<= no_of_recommendations:
                df.loc[len(df)] = [user, all_items[sorted_score_idx[i][1]], sorted_score_idx[i][0], rank]
                rank = rank+1

        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def recommend(self, user, no_of_recommendations):
        user_items = self.getUserItems(user)
        print("No. of unique items (songs) for the user: %d" % len(user_items))

        all_items = self.getAllItems()
        print("No. of unique items (songs) in the training set: %d" % len(all_items))

        cooccurence_matrix = self.constructCoocuranceMatrix(user_items, all_items)

        return self.generateTopRecommendation(user, no_of_recommendations,cooccurence_matrix, user_items, all_items)

