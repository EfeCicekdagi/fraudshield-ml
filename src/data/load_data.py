import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw dataset in a memory-efficient manner.
    
    Args:
        filepath (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame with optimized data types.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The dataset file was not found at '{filepath}'. "
                                f"Please ensure the file is present in the specified location.")

    # Optimized dtypes to save memory on large datasets
    dtypes = {
        'step': 'int32',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'object',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8'
    }

    try:
        df = pd.read_csv(filepath, dtype=dtypes)
        return df
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")
