import pandas as pd
import os
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

def load_raw_data(filepath: str | os.PathLike) -> pd.DataFrame:
    """
    Loads the raw dataset in a memory-efficient manner.
    
    Args:
        filepath (str | os.PathLike): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame with optimized data types.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    filepath_str = str(filepath)
    if not os.path.exists(filepath_str):
        logger.error(f"File not found: {filepath_str}")
        raise FileNotFoundError(f"Error: The dataset file was not found at '{filepath_str}'. "
                                f"Please ensure the file is present in the specified location.")

    logger.info(f"Loading raw data from {filepath_str}")
    
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
        df = pd.read_csv(filepath_str, dtype=dtypes)
        logger.info(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise RuntimeError(f"An error occurred while loading the data: {e}")
