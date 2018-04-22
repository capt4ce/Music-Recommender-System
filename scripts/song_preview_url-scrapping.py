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
def retry_request(url):
    return json.load(urlopen(url))

# import threading
# from functools import wraps

# def delay(delay=0.):
#     """
#     Decorator delaying the execution of a function for a while.
#     """
#     def wrap(f):
#         @wraps(f)
#         def delayed(*args, **kwargs):
#             timer = threading.Timer(delay, f, args=args, kwargs=kwargs)
#             timer.start()
#         return delayed
#     return wrap

# @delay(3.0)
# def retry_request(url):
#     return json.load(urlopen(url))

def access_info(url):
    try:
        return json.load(urlopen(url))
    except:
        return retry_request(url)
        raise


df = pd.read_csv('../dataset/selected_song_based_on_tags.csv', sep='\t')

# initializing preview_url column to Null if the column doesnt exit initially
if 'preview_url' not in df.columns:
    df['preview_url']=None
    df['preview_info']=None
    print('column doesn\'t exist yet')

if 'genres' not in df.columns:
    df['genres']=None

len_selected = len(df)
unprocessed_num = df['preview_url'].isnull().sum()
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
    sys.stdout.write(str(i))
    sys.stdout.flush()
    if (row['preview_url'] == None or (isinstance(row['preview_url'],numbers.Real) and math.isnan(row['preview_url']))):
        info_url = ('https://itunes.apple.com/search?'+urlencode([('term',row['title'])]))
        # print(info_url)

        # try:
        #     response = json.load(urlopen(info_url))
        # except urllib.error.HTTPError:
        #     print(info_url)
        #     result = urlopen(info_url)
        #     print(result)
        #     print(json.load(result))
        #     break

        response = access_info(info_url)

        # try: 
        if (len(response['results']) == 0):
            preview_info = 'not found'
            preview_url = 'not found'
            genres= 'not found'
        else:
            preview_info = response['results'][0]['trackName']+' - '+response['results'][0]['artistName']

            if ('previewUrl' in response['results'][0]):
                preview_url = response['results'][0]['previewUrl']
            else:
                preview_url = 'not found'

            if ('genres' in response['results'][0]):
                genres = str(response['results'][0]['genres'])
            else:
                genres = 'not found'

        # except:
        #     print(response)
        #     print(row['title'])
        #     break

        processed_df.preview_url.iloc[[idx]]=preview_url
        processed_df.preview_info.iloc[[idx]]=preview_info
        processed_df.genres.iloc[[idx]]=genres

    if (i%10==0 or i==unprocessed_num):
        processed_df.to_csv('../dataset/selected_song_based_on_tags.csv', sep='\t', encoding='utf-8', index=False)
        print('['+str(i)+'/'+str(len_selected)+'] : ' + preview_url)
    
    if (i==unprocessed_num):
        break

processed_df.to_csv('../dataset/selected_song_based_on_tags.csv', sep='\t', encoding='utf-8', index=False)
print('Finished!')


# print(processed_df.head())


# i=0
# len_selected = len(df)
# processed_df = df.copy()

# def get_url(x):
#     global i, processed_df
#     i=i+1
#     x['preview_url']='bbb'
#     if (i%20==0):
#         processed_df.to_csv('../dataset/selected_song_based_on_tags.csv', sep='\t', encoding='utf-8', index=False)
#         print(str(i)+'/'+str(len_selected)+' songs has been scrapped')
#     return x

# processed_df = df.apply(get_url, axis=1)


