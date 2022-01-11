import pandas as pd
import numpy as np
from dython.nominal import associations, numerical_encoding

def find_cateorical_columns(data):
    categorical_columns = []
    for col in data.columns:
        levels = len(list(data[col].value_counts().index))
        if(levels < 10):
            categorical_columns.append(col)
    return tuple(categorical_columns)

def change_dtype(data):
    for col in data.columns:
        levels = len(list(data[col].value_counts().index))
        if(levels < 10):
            data[col] = data[col].astype('category')
    return data

def match_dtypes(real,synthetic):
    for col in real.columns:
            synthetic[col]=synthetic[col].astype(real[col].dtypes.name)
    return synthetic

def PCD(real,synthetic):
    real = change_dtype(real)
    synthetic = match_dtypes(real,synthetic)
    return np.round(np.linalg.norm((associations(real,nan_strategy='drop_samples',compute_only=True)['corr'] - associations(synthetic,nan_strategy='drop_samples',compute_only=True)['corr']),ord='fro'), 4)
