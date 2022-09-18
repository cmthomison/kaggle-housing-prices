"""
Functions used throughout the process to load/manipulate/push data.
"""

import pandas as pd
import os


def load_train(path='~/Documents/projects/kaggle-housing-prices/data/train.csv'):
    
    data = pd.read_csv(path)

    return data

def col_types(df:pd.DataFrame):

    # Numeric
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    num_cols = df.select_dtypes(include=numerics).columns.tolist()

    # String
    strings = ['object', 'string']

    string_cols = df.select_dtypes(include=strings).columns.tolist()

    return num_cols, string_cols