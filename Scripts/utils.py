from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
import sys
sys.path.append("..")
import os
pd.set_option("max_columns", 10000)

import json

from tqdm import tqdm

import copy

from Scripts.saving import submit

def reduce_mem_usage(df, skip_cols_pattern='SK_ID_'):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        if skip_cols_pattern in col:
            print(f"don't optimize index {col}")

        else:
            col_type = df[col].dtype

            if col_type != object:

                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

            else:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

from sklearn.preprocessing import LabelEncoder
def encode_columns(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].values)

unique_roles = set()
def onehot_lists(series_lists):

    def str_to_list(s):
        global unique_roles
        s = s[1:-1].split("', '")
        s[0] = s[0][1:]
        s[-1] = s[-1][:-1]
        s = set(s)
        unique_roles = unique_roles | s
        return s

    series_lists = series_lists.apply(str_to_list)

    new_data = {role:list() for role in unique_roles}

    for role in unique_roles:
        for l in series_lists:
            new_data[role].append( role in l )

    new_data = pd.DataFrame(new_data)

    return new_data

def make_diff_shifts(row, n=1):
    """
    row - np-array
    """
    if n == 0:
        return row

    return row[n:]- row[:-n]

def linreg_trend(Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    X = range(len(Y))

    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx

    trend_a = (Sxy * N - Sy * Sx)/det
    trend_b = (Sxx * Sy - Sx * Sxy)/det
    return trend_a
