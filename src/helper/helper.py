'''
That module contains all helper functions of the project
'''
import pandas as pd
import multiprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K


def recall_m(y_true, y_pred):
    """Calculate recall"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """Calculate precision"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """Calcuates F1 metric"""
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def parallelize_dataframe(df: pd.DataFrame, func: object, num_cpu: int = 0) -> pd.DataFrame:
    '''
    Process a dataframe using multiprocessing
    Params:
        df: Pandas Dataframe to process
        func: Function containing the df processing code
              Hint: Use iterrows within that function
              Example:
                def function_2_call(df: pd.DataFrame) -> pd.DataFrame:
                    for idx, row in df.iterrows():
                        ...
                        ...
                        ...
                return df
        num_cpu: Number of CPUs to use. If set to -1, all CPUs used.

    Returns:
        Pandas Dataframe or list object
    '''
    num_cores = multiprocessing.cpu_count()
    if num_cpu == 0:
      use_cores = num_cores - 1
    elif num_cpu >= num_cores:
      use_cores = num_cores - 1
    else:
      use_cores = num_cpu

    num_partitions = use_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(use_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def plot_correlations(df: pd.DataFrame, num_cols: list = None, show_diag: bool = True, figsize=(8, 5)):
    """Plot the correlation matrix using Pearson."""
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=np.bool))

    if isinstance(num_cols, type(None)):
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.set_title("Correlation of columns")
    if show_diag:
        sns.heatmap(
            df[num_cols].corr(), cmap=cmap, vmin=-1, vmax=1, annot=True, ax=ax)
    else:
        sns.heatmap(
            df[num_cols].corr(), cmap=cmap, vmin=-1, vmax=1, annot=True, mask=mask, ax=ax)

    plt.show()


def get_dist_of_simple_genre_combis(df: pd.DataFrame, genre_cols: list, verbose: bool = False) -> list:
    """
    Get number of genres per simple genre combinations
    """
    # Calc distibution of one or multi genres per movie
    l_rowsum = []
    l_count_gen = []
    for idx, row in df.iterrows():
        l_rowsum.append(row[genre_cols].sum())

    gen_dist = np.array(l_rowsum)
    for i in range(0, len(genre_cols)):
        count = len(np.where(gen_dist==i)[0])
        l_count_gen.append(count)
        if verbose:
            print(f"Number of movies holding {i} genres: {count}")
    return l_count_gen