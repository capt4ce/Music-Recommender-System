import pandas
import time
import numpy as np
import pdb
from tqdm import tqdm
import librosa

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint

import keras
from sklearn.model_selection import train_test_split

import os.path
import math

class SongLabellingModel():

    def __init__(self, model_path=None, song_preview_dir=None, label_map_boolean_path=None):
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'use5-bestcheckpoint-0.08- 0.35.hdf5'#'use2-bestcheckpoint-0.02- 0.25.hdf5'#'use-bestcheckpoint-0.08- 0.26.hdf5'#'DLModel.h5'

        if song_preview_dir:
            self.song_preview_dir = song_preview_dir
        else:
            self.song_preview_dir = '../../dataset/song_preview/'

        if label_map_boolean_path:
            self.label_map_boolean_path = label_map_boolean_path
        else:
            self.label_map_boolean_path = '../../dataset/deep_learning/label_map(boolean_mood).csv'

        self.model = None

    def buildModelCNN(self):
        '''Instantiate the MusicTaggerCNN architecture.
        Note that when using TensorFlow,
        for best performance you should set
        `image_dim_ordering="tf"` in your Keras config
        at ~/.keras/keras.json.

        For preparing mel-spectrogram input, see
        `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
        You will need to install [Librosa](http://librosa.github.io/librosa/)
        to use it.

        # Arguments
            weights: one of `None` (random initialization)
                or "msd" (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            include_top: whether to include the 1 fully-connected
                layer (output layer) at the top of the network.
                If False, the network outputs 256-dim features.


        # Returns
            A Keras model instance.
        '''
        K.set_image_dim_ordering('th')

        # Determine proper input shape
        input_shape = (1, 96, 1366)

        melgram_input = Input(shape=input_shape)

        channel_axis = 1
        freq_axis = 2
        time_axis = 3

        # Input block
        x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)

        # Conv block 1
        x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)

        # Conv block 2
        x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

        # Conv block 3
        x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

        # Conv block 4
        x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)

        # Conv block 5
        x = Convolution2D(64, 3, 3, border_mode='same', name='conv5')(x)
        x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

        # Output
        x = Flatten()(x)
        x = Dense(50, activation='sigmoid', name='output')(x)
        # if include_top:
        #     x = Dense(50, activation='sigmoid', name='output')(x)

        # Create model
        model = Model(melgram_input, x)

        return model    

    # for converting audio file to melgram data
    def _compute_melgram(self, audio_path):
        ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
        96 == #mel-bins and 1366 == #time frame

        parameters
        ----------
        audio_path: path for the audio file.
                    Any format supported by audioread will work.
        More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

        '''

        # mel-spectrogram parameters
        SR = 12000
        N_FFT = 512
        N_MELS = 96
        HOP_LEN = 256
        DURA = 29.12  # to make it 1366 frame..

        src, sr = librosa.load(audio_path, sr=SR)  # whole signal
        n_sample = src.shape[0]
        n_sample_fit = int(DURA*SR)

        if n_sample < n_sample_fit:  # if too short
            src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
        elif n_sample > n_sample_fit:  # if too long
            src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
        # logam = librosa.logamplitude
        logam = librosa.power_to_db
        melgram = librosa.feature.melspectrogram
        ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                            n_fft=N_FFT, n_mels=N_MELS)**2,
                    ref=1.0)
        ret = ret[np.newaxis, np.newaxis, :]
        return ret

    def _dataset_to_array(self, df):
        features = np.zeros((0, 1, 96, 1366))
        
        # df = df.reindex(np.random.permutation(df.index)).head(1000)
        label_matrix = df.copy()
        label_matrix = label_matrix.drop(['track_id', 'song_id', 'title', 'preview_file'], axis=1).as_matrix()
        for idx,row in tqdm(df.iterrows(), "converting song audio file to features"):
            melgram = self._compute_melgram(self.song_preview_dir+row['preview_file'])
            features = np.concatenate((features, melgram), axis=0)
            
        if os.path.isfile('song_features-1500.npy'):
            os.remove('song_features-1500.npy')
        if os.path.isfile('song_labels-1500.npy'):
            os.remove('song_labels-1500.npy')
        np.save('song_features-1500.npy', features)
        np.save('song_labels-1500.npy', label_matrix)
        return features, label_matrix

    def _train_in_chunk(self, model, X_train, X_test, Y_train, Y_test, iteration):
        channel = 1
        epochs = 10#50
        batch_size = 10
        verbose = 1

        # normal_checkpoint_name = 'checkpoint-'+str(iteration)+'.{epoch:02d}-{val_acc: .2f}.hdf5'
        best_checkpoint_name = 'bestcheckpoint-'+str(iteration)+'.{epoch:02d}-{val_acc: .2f}.hdf5'

        # normal_checkpoint = ModelCheckpoint(normal_checkpoint, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        best_checkpoint = ModelCheckpoint(best_checkpoint_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callback_list = [best_checkpoint]

        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, Y_test), callbacks=callback_list)
        # model.save(self.model_path)
        path_separated = self.model_path.split('.')
        modelName = path_separated[0]+str(iteration)+'.'+path_separated[1]

        if os.path.isfile(self.model_path):
            os.rename(self.model_path,modelName)
        model.save_weights(self.model_path)
        os.remove('training_status.txt')
        f = open('training_status.txt', 'w')
        f.write('%d' % iteration)
        f.close()

    def train(self, model=None, start=0):
        print(model)
        if model==None:
            model = self.buildModelCNN()
        
        print(model)
        
        if os.path.isfile(self.model_path):
            model.load_weights(self.model_path)
        
        print(model)
        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

        if os.path.isfile('randomized.csv'):
            df = pandas.read_csv('randomized.csv', sep='\t')            
        else:
            df = pandas.read_csv(self.label_map_boolean_path, sep='\t')
            df = df.reindex(np.random.permutation(df.index))
            df.to_csv('randomized.csv', sep='\t', encoding='utf-8', index=False)
        df = df[start:]

        print(len(df))

        chunk_content = 1500 #1000
        split_ratio=0.8
        chunk_train = chunk_content#int(chunk_content*split_ratio)
        random_state=7

        # if the numpy file is already there, so no repeated feature sampling again
        pass_melgram = False

        for i in range(math.ceil(len(df)/chunk_content)):
            # if (i*chunk_train)+chunk_content>=len(df):
            #         part_df = df[i*chunk_train:]
            # else:
            #     part_df = df[i*chunk_train:(i*chunk_train)+chunk_content]

            # if pass_melgram:
            #     pass_melgram = False
            #     X = np.load('song_features-1000.npy')
            #     Y = np.load('song_labels-1000.npy')
            # else:
            #     print('Getting features from '+str(i*chunk_train)+'/'+str((i*chunk_train)+len(part_df))+' dataset...')
            #     X, Y = self._dataset_to_array(part_df)
            # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
            # del X
            # del Y
            # print('Training from '+str(i*chunk_content)+'/'+str((i*chunk_content)+len(part_df))+' dataset...')
            # self._train_in_chunk(model, X_train, X_test, Y_train, Y_test, (i*chunk_content)+i)
            # del X_train
            # del X_test
            # del Y_train
            # del Y_test
            if (i*chunk_train)+chunk_content>=len(df):
                part_df = df[i*chunk_train:]
            else:
                part_df = df[i*chunk_train:(i*chunk_train)+chunk_content]

            if pass_melgram:
                pass_melgram = False
                X = np.load('song_features-1500.npy')
                Y = np.load('song_labels-1500.npy')
            else:
                print('Getting features from '+str(i*chunk_train)+'/'+str((i*chunk_train)+len(part_df))+' dataset...')
                X, Y = self._dataset_to_array(part_df)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
            print('X_train shape: '+ str(X_train.shape))
            print('X_test shape: '+ str(X_test.shape))
            print('Y_train shape: '+ str(Y_train.shape))
            print('Y_test shape: '+ str(Y_test.shape))
            del X
            del Y
            print('Training from '+str(i*chunk_content)+'/'+str((i*chunk_content)+len(part_df))+' dataset...')
            self._train_in_chunk(model, X_train, X_test, Y_train, Y_test, (i*chunk_content)+i)
            del X_train
            del X_test
            del Y_train
            del Y_test
        del df
        self.model = model
        return model

    def getModel(self, train=False, start=0):
        self.model = self.buildModelCNN()
        print(os.path.isfile(self.model_path))
        print(start)
        print(train)
        if train==False:
            print ('a')
            self.model.load_weights(self.model_path)
            return self.model
            # return self.model.load_weights(self.model_path)
        else:
            print ('b')
            # return 'b'
            self.train(self.model, start)
            return self.model
    
    def predict(self, filepath):
        model = self.model
        feature = self._compute_melgram(filepath)
        result = model.predict(feature)
        
        return result
        # # adding result to the DF
        # new_entry = None

        # df = pandas.read_csv(self.label_map_boolean_path, sep='\t')
        # new_df = df.append(new_entry)
        # return new_df




if __name__ == '__main__':
    labellingModel = SongLabellingModel()
    model = labellingModel.getModel(True)
