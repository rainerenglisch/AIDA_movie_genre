'''
That module contains all helper functions of the project
'''
import pandas as pd

def parallelize_dataframe(df: pd.DataFrame, func: object) -> pd.DataFrame:
    '''
    Process a dataframe using multiprocessing
    Params:
        df: Pandas Dataframe to process
        func: Function containing the df processing code
              Hint: Use iterrows within that function
              The function needs that interface:
              
    Returns:
        Pandas Dataframe containing the results
    '''
    num_cores = multiprocessing.cpu_count()-1  #leave one free to not freeze machine
    num_partitions = num_cores #number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df