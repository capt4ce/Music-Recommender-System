import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys  
from . import LabellingModel

reload(sys)   
sys.setdefaultencoding('utf8')

class DeepContentRecommender():

    def __init__(self, song_df, tag_df, user_df):
        self.song_df = song_df          # unique song with meta data
        self.user_df = user_df          # unique user with listening song_ids
        self.tag_df = tag_df            # raw list of song-tag pair for easy tag-specific or song-related tag information to be extracted
        self.cosine_sim = None

    def _buildModel(self):

        # input network

        # k-level Convolutional Neural Network with ReLUs intead of sigmoid as activation function

        pass
    
    def train(self):
        # train data configuration

        # model construction

        # training

        # saving model

        # running similarity process
        self._prepare_recommendation()
        pass
    
    def _prepare_recommendation(self):
        # using Count Vectorizer and place them in a space and cosine similarity score in taken when title is supplied
        # loading model

        # recommending set of songs, with a song file as a query

        # use the CountVectorizer() instead of TF-IDF, because you do not want to down-weight the presence of tags relative to the songs
        tags = self.tag_df.groupby('song_id').agg({'tags': lambda x: ' '.join(x)}).reset_index()
        self.song_df = pd.merge(self.song_df, tags, on='song_id', how='inner')
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.song_df['tags'].values.astype('U'))
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)         

    def recommendFromQuery(self,song_title, no_of_recommendation):
        self.song_df = self.song_df.reset_index(drop=True)
        indices = pd.Series(self.song_df.index, index=self.song_df['title'])
        idx = indices[song_title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:no_of_recommendation+1]
        song_indices = [i[0] for i in sim_scores]
        return self.song_df.iloc[song_indices].reset_index(drop=True)

    def recommendFromFile(self, filepath):
        lModel = LabellingModel()
        lModel.predict(filepath)