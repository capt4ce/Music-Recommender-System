import os
import pandas as pd
import json
import urllib
from urllib.request import urlopen
from urllib.parse import urlencode
import math
import numbers
import sys


import time
from functools import wraps


import pygn.pygn as pygn

clientID = '897996368-7A5061BC9D13DC0AF3FC9ECB9E141964'
userID = '43728440884220450-CDE3B31ACCC9AFA11AD32B43712D706E'


def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry

@retry(urllib.error.URLError, tries=100, delay=60, backoff=2)
def retry_request(track, artist):
    return pygn.search(clientID=clientID, userID=userID, artist=artist, track=track)


def access_info(track, artist):
    try:
        return pygn.search(clientID=clientID, userID=userID, artist=artist, track=track)
    except:
        return retry_request(track, artist)
        raise


df = pd.read_csv('../dataset/selected_song_based_on_tags (copy).csv', sep='\t')

# initializing preview_url column to Null if the column doesnt exit initially
if 'moods' not in df.columns:
    df['moods']=None
    df['genres']=None
    df['tempos']=None
    df['gracenote']=None
    print('column doesn\'t exist yet')

if 'genres' not in df.columns:
    df['genres']=None

len_selected = len(df)
unprocessed_num = df['moods'].isnull().sum()
i = len_selected - unprocessed_num
processed_df = df
# processed_df = df

print('total song: '+str(len_selected))
print('unprocessed song: '+str(unprocessed_num))

for idx, row in processed_df.iterrows():
    if (idx < i):
        continue

    i=i+1
    sys.stdout.write('\r')
    sys.stdout.write('['+str(i)+'/'+str(len_selected)+']')
    sys.stdout.flush()
    if (row['moods'] == None or (isinstance(row['moods'],numbers.Real) and math.isnan(row['moods']))):
        title_data = row['title'].split(' - ')
        # print(title_data[0] + ' ' + title_data[-1])
        
        response = access_info(title_data[0], title_data[-1])

        # print(response)

        if (response is None):
            response
            processed_df.moods.iloc[[idx]]='not found'
            processed_df.genres.iloc[[idx]]='not found'
            processed_df.tempos.iloc[[idx]]='not found'
            processed_df.gracenote.iloc[[idx]]='not found'
        
        else:
            if ('mood' in response):
                moods_arr=[]
                for key,mood in response['mood'].items():
                    # print(mood)
                    # exit()
                    moods_arr.append(mood['TEXT'])
                moods = str(moods_arr)
            else:
                moods = 'not found'
            
            if ('genre' in response):
                genres_arr=[]
                for key,genre in response['genre'].items():
                    genres_arr.append(genre['TEXT'])
                genres = str(genres_arr)
            else:
                genres = 'not found'
                
            if ('tempo' in response):
                tempo_arr=[]
                for key,tempo in response['tempo'].items():
                    tempo_arr.append(tempo['TEXT'])
                tempos = str(tempo_arr)
            else:
                tempos = 'not found'

            processed_df.moods.iloc[[idx]]=moods
            processed_df.genres.iloc[[idx]]=genres
            processed_df.tempos.iloc[[idx]]=tempos
            processed_df.gracenote.iloc[[idx]]=response

    if (i%10==0 or i==unprocessed_num):
        processed_df.to_csv('../dataset/selected_song_based_on_tags (copy).csv', sep='\t', encoding='utf-8', index=False)
        print('['+str(i)+'/'+str(len_selected)+'] : ' + title_data[0])
    
    if (i==unprocessed_num):
        break

processed_df.to_csv('../dataset/selected_song_based_on_tags (copy).csv', sep='\t', encoding='utf-8', index=False)
print('Finished!')