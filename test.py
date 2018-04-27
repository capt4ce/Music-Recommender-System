import pandas
import numpy as np
from Recommender.DeepContent.Recommender import ContentRecommender
Recommender = ContentRecommender('dataset/deep_learning/label_map(boolean_mood).csv', 'dataset/main_song_labels.csv', 'dataset/main_labels.csv', 'dataset/main_user_rating.csv','Recommender/DeepContent/bestcheckpoint-1501.06- 0.34.hdf5','dataset/song_preview/','dataset/deep_learning/label_map(boolean_mood).csv')
Recommender.prepareRecommendation(False)

# print(Recommender.labelSearch('sadness_slow', 'ali'))
# print(Recommender.recommend('ujwal'))
print(Recommender.addNewSong('aaa','/home/capt4ce/projects/_ref_major_project/music-auto_tagging-keras/data/bensound-cute.mp3'))