import pandas as pd
import os
import sys
import scipy
from scipy import sparse

from pandas.api.types import CategoricalDtype





def fetch_data():
    df = pd.read_csv("Data/ANALYTICS_PUBLIC_AWEME_REC_NINJA_AGG_2022_2.csv")
    df = df[~((df["SONG_NAME"] == "OdeToJoy_Beethoven_Essentials") & (df["NUM_PLAYED"]==1))]
    profiles = df["PROFILE_ID"].unique()
    songs = df["SONG_NAME"].unique()
    shape = (len(profiles), len(songs))
    profile_cat = CategoricalDtype(categories=sorted(profiles), ordered=True)
    song_cat = CategoricalDtype(categories=sorted(songs), ordered=True)
    user_index = df["PROFILE_ID"].astype(profile_cat).cat.codes
    movie_index = df["SONG_NAME"].astype(song_cat).cat.codes
    return profiles, songs,sparse.coo_matrix((df["NUM_PLAYED"], (user_index, movie_index)), shape=shape).tocsr()



