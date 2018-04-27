import pandas
import numpy as np
import math
from shutil import copyfile
import random
import string
import operator
from Recommender.DeepContent.LabellingModel import SongLabellingModel
# from LabellingModel import SongLabellingModel


class ContentRecommender:
    def __init__(self,label_map_path = '../../dataset/deep_learning/label_map(boolean_mood).csv', main_song_label_path = '../../dataset/main_song_labels.csv', main_labels_path = '../../dataset/main_labels.csv', user_rating_path = '../../dataset/main_user_rating.csv', model_path='bestcheckpoint-0.04- 0.34.hdf5', song_preview_dir='../../dataset/song_preview', label_map_boolean_path='../../dataset/deep_learning/label_map(boolean_mood).csv', matrix_path = 'similarityDataframe.csv', user_profile_path = 'userProfile.csv', label_vector_path = 'labelVector.csv'):
        self.label_map_path = label_map_path
        self.main_song_label_path = main_song_label_path
        self.main_labels_path = main_labels_path
        self.user_rating_path = user_rating_path
        self.song_df = pandas.read_csv(self.main_song_label_path, sep='\t')
        self.labels_df = pandas.read_csv(self.main_labels_path, sep='\t')

        self.matrix_path = matrix_path
        self.user_profile_path = user_profile_path
        self.label_vector_path = label_vector_path
        self.TF = None
        self.user_profile = None
        self.rating_df = pandas.read_csv(self.user_rating_path, sep='\t')
        self.similarityMatrix = None

        self.model_path = model_path
        self.song_preview_dir = song_preview_dir
        self.label_map_boolean_path = label_map_boolean_path
        self.labels = ['Alternative & Punk',  'Rock',  'Traditional',  'Urban',  'Pop',  'Other',
                        'Western Hip-Hop/Rap', 'Metal', 'Western Pop', 'Electronica', 'Punk', 'Indie Rock', 'Alternative', '70s Rock', 'Adult Alternative Rock', 'Electric Blues',
                        'Country', 'Jazz', 'Contemporary R&B/Soul', 'Alternative Rock', 'Acoustic Blues', '60s Rock', 'Classic Country', 'Emo & Hardcore', 'Mainstream Rock', 'Classic R&B/Soul',
                        'Synth Pop', 'General Mainstream Rock', 'Pop Punk', 'Adult Alternative Pop', 'Black Metal', 'Old School Hip-Hop/Rap', 'Latin Pop', 'General Latin Pop', 'Brit Rock', 'New Wave Pop',
                        'Classic Hard Rock', 'Adult Contemporary', 'East Coast Rap', 'European Pop', 'Latin Rock', 'Hard Rock', 'Religious',
                        'happiness', 'sadness', 'anger', 'neutral',
                        'slow', 'medium', 'fast']

    def constructLabelVector(self):
        labels_df = self.labels_df
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
        TF.to_csv(self.label_vector_path, sep='\t', encoding='utf-8', index=False)
        return TF

    def constructUserProfile(self):
        TF = self.TF
        rating_df = self.rating_df
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
        user_profile.to_csv(self.user_profile_path, sep='\t', encoding='utf-8', index=False)
        return user_profile

    def constructCosineSimilarity(self, user=None):
        rating_df = self.rating_df
        user_profile = self.user_profile
        TF= self.TF
        if user == None:
            distinct_users=np.unique(rating_df['username'])
            label_merge_all=pandas.DataFrame()
        else:
            distinct_users=[user]
            label_merge_all = self.similarityMatrix
            label_merge_all = label_merge_all[label_merge_all['user']!=user]
        print(distinct_users)

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
            
        label_merge_all.to_csv(self.matrix_path, sep='\t', encoding='utf-8', index=False)
        self.similarityMatrix = label_merge_all
        return label_merge_all

    def prepareRecommendation(self, reconstruct=True):
        if reconstruct:
            self.constructLabelVector()
            self.constructUserProfile()
            self.constructCosineSimilarity()
        else:
            self.TF = pandas.read_csv(self.label_vector_path, sep='\t')
            self.user_profile = pandas.read_csv(self.user_profile_path, sep='\t')
            self.rating_df = pandas.read_csv(self.user_rating_path, sep='\t')
            self.similarityMatrix = pandas.read_csv(self.matrix_path, sep='\t')


    def recommend(self, username, no_of_recommendation=10):
        print(self.rating_df[self.rating_df['username']==username])
        recommendations = self.similarityMatrix[self.similarityMatrix['user']==username].sort_values(by='score', ascending=False)
        if len(recommendations)==0:
            recommendations = self.similarityMatrix.sort_values(by='score', ascending=False)
        recommendation_detail = pandas.merge(recommendations, self.song_df, on='track_id', how='inner')
        recommendation_detail = pandas.merge(recommendation_detail, self.rating_df[self.rating_df['username']==username], on='track_id', how='left')[['user','track_id','score', 'title', 'preview_file', 'mp3_file','rating']].fillna(0)
        print(len(recommendation_detail))
        recommendation_detail = recommendation_detail.drop_duplicates('track_id').reset_index(drop=True)
        print(len(recommendation_detail))
        return recommendation_detail[:no_of_recommendation]
    
    def titleSearch(self, title, username = None, no_of_result=10):
        result = self.song_df[self.song_df['title'].str.contains(title, False)][:no_of_result].reset_index(drop=True)
        return pandas.merge(result, self.rating_df[self.rating_df['username']==username], on='track_id', how='left').fillna(0)
    
    def labelSearch(self, query, username,no_of_result=10):
        queries = query.split('_')
        recommendations = self.similarityMatrix[self.similarityMatrix['user']==username].sort_values(by='score', ascending=False)
        if len(recommendations)==0:
            recommendations = self.similarityMatrix.sort_values(by='score', ascending=False)
        
        # print(recommendations.head(10))
        analize_df = pandas.merge(recommendations,self.labels_df, on='track_id', how='inner')        
        print(self.labels_df.head(10))
        for q in queries:
            query_satisfied = analize_df[analize_df['label']==q]
            analize_df = analize_df[analize_df['track_id'].isin(query_satisfied['track_id'])]
        result = analize_df[recommendations.columns].drop_duplicates('track_id').sort_values(by='score', ascending=False)
        if len(result)>0:
            print(self.labels_df[self.labels_df['track_id'] == result.track_id.iloc[0]])
            result = pandas.merge(result[:no_of_result], self.song_df, on='track_id', how='inner')
            return pandas.merge(result, self.rating_df, on='track_id', how='left').fillna(0)
        else:
            return []

    def rateSong(self, username, track_id, rating):
        print(self.rating_df.head(10))
        self.rating_df.loc[len(self.rating_df)]=[username, track_id, rating]
        print(self.rating_df.head(10))
        self.constructUserProfile()
        self.constructCosineSimilarity(username)
        print(self.rating_df.head(10))
        self.rating_df.to_csv(self.user_rating_path, sep='\t', encoding='utf-8', index=False)
        return self

    def analyzeNewSong(self, filepath):
        genre_list = self.labels[:43]
        mood_list = self.labels[43:47]
        tempo_list = self.labels[47:]

        selected_labels=[]

        lModel = SongLabellingModel(self.model_path, self.song_preview_dir, self.label_map_boolean_path)
        model = lModel.getModel()
        result = lModel.predict(filepath)
        normalized = (result-np.min(result))/(np.max(result)-np.min(result))
        print(normalized)
        
        genre = normalized[0][:43]
        mood = normalized[0][43:47]
        tempo = normalized[0][47:]

        selected_genre = [(i,j) for (i,j) in zip(genre,genre_list) if i >= np.average(genre)]
        selected_mood = [(i,j) for (i,j) in zip(mood,mood_list) if i >= np.average(mood)]
        selected_tempo = [(i,j) for (i,j) in zip(tempo,tempo_list) if i >= np.max(tempo)]

        selected_labels = [i[1] for i in selected_genre]+[i[1] for i in selected_mood]+[i[1] for i in selected_tempo]

        # return a list of labels
        return selected_labels
    def addNewSong(self, song_title, song_path):
        labels = []
        # labeling the song
        song_output_dir = 'dataset/song_mp3/'
        song_labels = self.analyzeNewSong(song_path)
        print(song_labels)

        # adding new song to song_df and labels_df
        track_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(18))
        while len(self.song_df[self.song_df['track_id']==track_id])>0:
            track_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(18))
        
        self.song_df.loc[len(self.song_df)]=[track_id, track_id, song_title, track_id+'.mp3', track_id+'.mp3']

        for label in song_labels:
            self.labels_df.loc[len(self.labels_df)]=[track_id,label]

        print(self.song_df[self.song_df['track_id']==track_id])
        print(self.labels_df[self.labels_df['track_id']==track_id])
        
        # reconstruct labels vector, user_profile, and similarity matrix
        self.prepareRecommendation()

        # saving the song data and labels
        self.song_df.to_csv(self.main_song_label_path, sep='\t', encoding='utf-8', index=False)
        self.labels_df.to_csv(self.main_labels_path, sep='\t', encoding='utf-8', index=False)
        copyfile(song_path, song_output_dir+track_id+'.mp3')
        return True


if __name__ == '__main__':
    recommender = ContentRecommender()
    recommender.prepareRecommendation()
    print(recommender.recommend('ali'),5)
