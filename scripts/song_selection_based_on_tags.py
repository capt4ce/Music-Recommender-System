import pandas
import numpy as np
import csv

song_df2 = pandas.read_csv('../dataset/MSD_songs.csv', sep='\t')
song_df2['title'] = song_df2['title'].map(str) + ' - ' + song_df2['artist_name']
song_df2 = song_df2.drop(['dig7_id', 'release', 'artist_name', 'year'], axis=1)
tag_df2 = pandas.read_csv('../dataset/LAST_FM_tags.csv', sep='\t')
tag_df2 = pandas.merge(tag_df2,song_df2, on='track_id', how='inner')

popular_tag_df = tag_df2.groupby('tags').agg({'tags': 'count'}).sort_values('tags',ascending=False)
popular_tag_df = popular_tag_df[:50]
print('Number of tags to be selected: '+str(len(popular_tag_df)))

popular_processed_tag_df = pandas.merge(tag_df2, pandas.DataFrame({'tags':popular_tag_df.index}), on='tags', how='inner')

selected_songs_df = popular_processed_tag_df.groupby('track_id', as_index=False).last()
print('Number of songs having the selected tags: '+str(len(selected_songs_df)))
selected_songs_df.head()

selected_songs_df.to_csv('../dataset/selected_song_based_on_tags.csv', sep='\t', encoding='utf-8', index=False)