import pandas as pd
import numpy as np
import tmdbsimple as tmdb
import requests
import tqdm
import multiprocessing
import fastparquet as fp
import sys
import src.helper.helper as hlp
import src.helper.secret as sec

# Initialization
MOVIE_URI = "https://www.themoviedb.org/movie/"
FILE_DIR = "/datadisk/dev/AIDA_movie_genre/src/"
DATA_DIR_RAW = FILE_DIR + "../data/raw/"
DATA_DIR_INTERIM = FILE_DIR + "../data/interim/"

VERSION_ID_OUT = "v2"
FILE_PATH_DF_IN = DATA_DIR_INTERIM + "df_cleaned_v1.gzip"
FILE_PATH_DF_OUT = DATA_DIR_INTERIM + f"df_cleaned_{VERSION_ID_OUT}.gzip"

#=============================================================================
def load_movie_details(df):
    for idx, row in df.iterrows():
        p_rd = get_movie_data(row["id"])
        df.loc[idx, "release_date"] = p_rd
    return df


def get_movie_data(mid: int):
    try:
        m = tmdb.Movies(mid).info()
    except Exception as e:
        print(f"Error loading movie {mid}, Exception: {sys.exc_info()}")
        return None

    # Get poster url
    if m['release_date'] != None:
        try:
            p_rd = pd.Timestamp(tmdb.Movies(mid).info()['release_date'])
        except Exception as e:
            print(f"No valid Timestamp for movie {mid}, release date set to None")
            p_rd = None
    else:
        print(f"Release date not set for movie {mid}")
        p_rd = None

    return p_rd


def parallelize_dataframe(df, func):
    num_cores = multiprocessing.cpu_count()-1  #leave one free to not freeze machine
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)    
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
#=============================================================================

# Open TDB session
tmdb.API_KEY = sec.TMDB_API_KEY
tmdb.REQUESTS_SESSION = requests.Session()

# Load dataframe to change
df = pd.read_parquet(FILE_PATH_DF_IN)

# Add release date
l_df = []
split_size = 2275
start_count = 100
end_count = (len(df) // split_size) + 1

for i in tqdm.tqdm(range(start_count, end_count)):
    s = i * split_size
    e = ((i + 1) * split_size) - 1
    if e > len(df):
        e = len(df) - 1

    df_tmp = df[s:e+1].copy()
    df_tmp = parallelize_dataframe(df_tmp, load_movie_details)
    #df_tmp = load_movie_details(df_tmp)
    l_df.append(df_tmp)
    df_tmp.to_parquet(DATA_DIR_RAW + f'df_{s}_{e}_{VERSION_ID_OUT}.gzip', compression='gzip')

# Put it together
df = pd.concat(l_df)
df.isna().sum()
df.head()

