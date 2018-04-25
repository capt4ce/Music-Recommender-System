import pandas
import numpy as np
import math

class ContentRecommender:
    def __init__(self,label_map_path = '../../dataset/deep_learning/label_map(boolean_mood).csv', main_song_label_path = '../../dataset/main_song_labels.csv', main_labels_path = '../../dataset/main_labels.csv', user_rating_path = '../../dataset/main_user_rating.csv'):
        self.label_map_path = label_map_path
        self.main_song_label_path = main_song_label_path
        self.main_labels_path = main_labels_path
        self.user_rating_path = user_rating_path
        self.song_df = pandas.read_csv(self.main_song_label_path, sep='\t')
        self.TF = None
        self.user_profile = None
        self.rating_df = None
        self.similarityMatrix = None

    def constructLabelVector(self):
        labels_df = pandas.read_csv(self.main_labels_path, sep='\t')
        labels_df['label_count'] = 1

        # finding TF
        TF = labels_df.groupby(['track_id', 'label'], as_index=False).count().rename(columns={'label_count': 'label_count_TF'})
        label_distinct = labels_df[['label', 'track_id']].drop_duplicates()

        # finding DF
        DF = label_distinct.groupby('label', as_index=False).count().rename(columns={'track_id': 'label_count_DF'})
        a=math.log10(len(np.unique(labels_df['track_id'])))
        DF['IDF']=a-np.log10(DF['label_count_DF'])
        TF = pandas.merge(TF,DF,on = 'label', how = 'left', sort = False)
        TF['TF-IDF']=TF['label_count_TF']*TF['IDF']

        # calculating vector
        Vect_len = TF[['track_id', 'TF-IDF']]
        Vect_len['TF-IDF-Sq'] = Vect_len['TF-IDF']**2
        Vect_len = Vect_len.groupby(['track_id'], as_index=False).sum().rename(columns={'TF-IDF-Sq':'TF-IDF-Sq-Sum'})[['track_id', 'TF-IDF-Sq-Sum']]
        Vect_len['vect_len'] = np.sqrt(Vect_len[['TF-IDF-Sq-Sum']].sum(axis=1))

        # calculating the weight of the song
        TF = pandas.merge(TF, Vect_len, on='track_id', how='left', sort=False)
        TF['label_wt'] = TF['TF-IDF']/TF['vect_len']
        self.TF = TF
        return TF

    def constructUserProfile(self):
        TF = self.TF
        rating_df = pandas.read_csv(self.user_rating_path, sep='\t')
        rating_df = rating_df[rating_df['rating']!=0]
        user_distinct = rating_df['username'].drop_duplicates()
        user_profile = pandas.DataFrame()
        i=1

        for user in user_distinct:            
            user_data = rating_df[rating_df['username']==user]
            user_data = pandas.merge(user_data,TF, on='track_id', how='inner')
            user_data_processed = user_data.groupby('label', as_index=False).sum().rename(columns={'label_wt': 'label_pref'})[['label', 'label_pref']]
            user_data_processed['user'] = user
            user_profile = user_profile.append(user_data_processed, ignore_index=True)
            i+=1

        self.rating_df = rating_df
        self.user_profile = user_profile
        return user_profile

    def constructCosineSimilarity(self):
        rating_df = self.rating_df
        user_profile = self.user_profile
        TF= self.TF
        distinct_users=np.unique(rating_df['username'])
        label_merge_all=pandas.DataFrame()

        i=1
        for user in distinct_users:
            # analizing user data one by one
            print(str(i)+' of '+str(len(distinct_users))+' users')
            user_profile_all= user_profile[user_profile['user']==user]
            distinct_songs = np.unique(TF['track_id'])
            j=1
            for song in distinct_songs:

                if j%300==0:
                    print('song: ', j , 'out of: ', len(distinct_songs) , 'with user: ', i , 'out of: ', len(distinct_users))

                # analizing song one by one
                TF_song= TF[TF['track_id']==song]
                label_merge = pandas.merge(TF_song,user_profile_all,on = 'label', how = 'left', sort = False)
                label_merge['label_pref']=label_merge['label_pref'].fillna(0)
                
                # listing label_value= weight of the label * label profile of the user
                label_merge['label_value']=label_merge['label_wt']*label_merge['label_pref']

                # getting the label weight of the current user-song pair
                label_wt_val=np.sqrt(np.sum(np.square(label_merge['label_wt']), axis=0))
                
                # getting the label value of the current user-song pair
                label_pref_val=np.sqrt(np.sum(np.square(user_profile_all['label_pref']), axis=0))

                # summing the label_value (rating) of user-song pair
                label_merge_final = label_merge.groupby(['user','track_id']).agg({'label_value': 'sum'}).rename(columns = {'label_value': 'score'}).reset_index()

                # score = score / (label weight * label value)
                label_merge_final['score']=label_merge_final['score']/(label_wt_val*label_pref_val)

                label_merge_all = label_merge_all.append(label_merge_final, ignore_index=True)
                j=j+1
            i=i+1
            label_merge_all=label_merge_all.sort_values(by=['user','score']).reset_index(drop=True)
            
        self.similarityMatrix = label_merge_all
        return label_merge_all

    def prepareRecommendation(self):
        self.constructLabelVector()
        self.constructUserProfile()
        self.constructCosineSimilarity()

    def recommend(self,username):
        recommendations = self.similarityMatrix[self.similarityMatrix['user']==username].sort_values(by='score', ascending=False)
        recommendation_detail = pandas.merge(recommendations, self.song_df, on='track_id', how='inner')
        return recommendation_detail

if __name__ == '__main__':
    recommender = ContentRecommender()
    recommender.prepareRecommendation()
    print(recommender.recommend('ali').head(10))
