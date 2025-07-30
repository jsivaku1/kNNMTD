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
    """
    Calculates the Pairwise Correlation Difference (PCD).
    This version includes a check to prevent errors if the association calculation fails.
    """
    real = change_dtype(real)
    synthetic = match_dtypes(real,synthetic)
    
    # Calculate associations for real and synthetic data
    real_corr_matrix = associations(real, nan_strategy='drop_samples', compute_only=True)['corr']
    synth_corr_matrix = associations(synthetic, nan_strategy='drop_samples', compute_only=True)['corr']
    
    # --- ROBUSTNESS CHECK ---
    # Check if either of the correlation matrices are None, which can happen with certain datasets
    if real_corr_matrix is None or synth_corr_matrix is None:
        # You can either return a specific value like NaN or raise a more informative error
        # Returning NaN is often safer for automated pipelines.
        print("Warning: Correlation matrix could not be computed for real or synthetic data. PCD cannot be calculated.")
        return np.nan

    # Calculate the Frobenius norm of the difference
    pcd_score = np.linalg.norm((real_corr_matrix - synth_corr_matrix), ord='fro')
    
    return np.round(pcd_score, 4)
