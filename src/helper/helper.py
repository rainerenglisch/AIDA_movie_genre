'''
That module contains all helper functions of the project
'''
import pandas as pd
import multiprocessing
import numpy as np

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