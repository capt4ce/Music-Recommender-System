import os
import pandas as pd
import json

def _extractSongTags(file_path):
    tags_df = pd.DataFrame(columns=['track_id', 'tags'])
    data = json.load(open(file_path))
    tags = [tag[0].lower().replace(' ','_') for tag in data['tags']]
    tags_df['tags'] = tags
    tags_df['track_id'] = data['track_id']
    return tags_df

def _fileProgress(ith_file, total_files, filename):
    print('[{0}/{1}] {2}'.format(ith_file, total_files, filename))

def extractBulk(dirPath):
    song_tags_df = pd.DataFrame(columns=['track_id', 'tags'])

    print('Directory path: '+ dirPath)

    # walking through a directory to get do bulk time-frequency representation of the audio
    for root, subdirs, files in os.walk(dirPath):
        i = 0
        for filename in files:
            i = i + 1
            if filename.endswith('.json'):
                _fileProgress(i, len(files), filename)

                file_path = os.path.join(root, filename)
                # song_id, tags = _extractSongTags(file_path)
                # # print(song_id, tags)
                # song_tags_df.loc[len(song_tags_df)] = [song_id, tags]

                song_tags_df = song_tags_df.append(_extractSongTags(file_path))
                # return song_tags_df
    return song_tags_df

if __name__ == '__main__':
    song_tags_df = extractBulk('../dataset/raw/lastfm_subset')

    # normalizing similar tags
    similar_tags = [{'favorite':['favorites','favourites','favourite','favorite_songs']}]
    for i in similar_tags:
        for key, j in i.items():
            for k in j:
                song_tags_df.replace(to_replace=k, value=key, inplace=True)
    print(song_tags_df)

    
    song_tags_df.to_csv('../dataset/LAST_FM_tags.csv', sep='\t', encoding='utf-8', index=False)