import os
import pandas as pd
import hdf5_getters

def _extractSongData(file_path, filename):
    # song_id, title, release, artist_name, year
    h5 = hdf5_getters.open_h5_file_read(file_path)
    track_id = filename[:-3]
    song_id = hdf5_getters.get_song_id(h5).decode('UTF-8')
    dig7_id = hdf5_getters.get_track_7digitalid(h5)
    title = hdf5_getters.get_title(h5).decode('UTF-8')
    release = hdf5_getters.get_release(h5).decode('UTF-8')
    artist_name = hdf5_getters.get_artist_name(h5).decode('UTF-8')
    year = hdf5_getters.get_year(h5)
    h5.close()
    # print(song_id, track_id, dig7_id, title, release, artist_name, year)
    return track_id, song_id, dig7_id, title, release, artist_name, year

def _fileProgress(ith_file, total_files, filename):
    print('[{0}/{1}] {2}'.format(ith_file, total_files, filename))

def extractBulk(dirPath):
    song_data_df = pd.DataFrame(columns=['track_id', 'song_id', 'dig7_id', 'title', 'release', 'artist_name', 'year'])

    print('Directory path: '+ dirPath)

    # walking through a directory to get do bulk time-frequency representation of the audio
    for root, subdirs, files in os.walk(dirPath):
        i = 0
        for filename in files:
            i = i + 1
            if filename.endswith('.h5'):
                _fileProgress(i, len(files), filename)

                file_path = os.path.join(root, filename)
                # song_id, tags = _extractSongTags(file_path)
                # # print(song_id, tags)
                song_data_df.loc[len(song_data_df)] = _extractSongData(file_path, filename)

                # song_tags_df = song_tags_df.append(_extractSongData(file_path))
                # return song_data_df
    return song_data_df

if __name__ == '__main__':
    # print(list(filter(lambda x: x[:3] == 'get', hdf5_getters.__dict__.keys())))
    song_data_df = extractBulk('../dataset/MillionSongSubset')
    print(song_data_df)
    song_data_df.to_csv('../dataset/MSD_songs.csv', sep='\t', encoding='utf-8', index=False)