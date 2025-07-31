import pandas as pd
import numpy as np
from dython.nominal import associations

def analyze_columns(df):
    """
    Analyzes dataframe columns and categorizes them as suitable for 
    classification or regression based on dtype and unique value counts.
    """
    categorical_cols = []
    numerical_cols = []
    
    for col in df.columns:
        # Rule for categorical: object/category dtype, or integer with few unique values (e.g., <= 15)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            categorical_cols.append({'name': col, 'type': str(df[col].dtype)})
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 15:
            categorical_cols.append({'name': col, 'type': str(df[col].dtype)})
        # Rule for numerical: float, or integer with many unique values
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append({'name': col, 'type': str(df[col].dtype)})
            
    return {'categorical': categorical_cols, 'numerical': numerical_cols}


def PCD(real,synthetic):
    """
    Calculates the Pairwise Correlation Difference (PCD).
    This version includes a check to prevent errors if the association calculation fails.
    """
    if real.empty or synthetic.empty:
        return np.nan

    # dython's 'associations' can handle mixed types, no need for manual dtype changes here.
    real_corr_matrix = associations(real, nan_strategy='drop_samples', compute_only=True)['corr']
    synth_corr_matrix = associations(synthetic, nan_strategy='drop_samples', compute_only=True)['corr']
    
    if real_corr_matrix is None or synth_corr_matrix is None:
        print("Warning: Correlation matrix could not be computed. PCD cannot be calculated.")
        return np.nan

    pcd_score = np.linalg.norm((real_corr_matrix - synth_corr_matrix), ord='fro')
    
    return np.round(pcd_score, 4)
