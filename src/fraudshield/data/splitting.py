import pandas as pd
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

def temporal_train_val_test_split(df: pd.DataFrame, temporal_column: str = 'step', train_size: float = 0.70, val_size: float = 0.15):
    """
    Splits the dataframe into Train, Validation, and Test sets based on the temporal column.
    Ensures that records with the same temporal value are not scattered across different splits.
    
    Args:
        df (pd.DataFrame): Dataframe containing the temporal column.
        temporal_column (str): Name of the column representing time (e.g., 'step').
        train_size (float): Approximate proportion of data for the training set.
        val_size (float): Approximate proportion of data for the validation set.
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if temporal_column not in df.columns:
        logger.error(f"Missing temporal column: {temporal_column}")
        raise ValueError(f"The dataframe must contain a '{temporal_column}' column for temporal splitting.")
        
    logger.info(f"Starting temporal split using column '{temporal_column}' (Train: {train_size}, Val: {val_size}).")
    
    # Ensure data is sorted chronologically
    df_sorted = df.sort_values(by=temporal_column).reset_index(drop=True)
    
    n_total = len(df_sorted)
    target_train_idx = int(n_total * train_size)
    target_val_idx = int(n_total * (train_size + val_size))
    
    def find_split_index(target_idx):
        if target_idx >= n_total:
            return n_total
        
        step_value = df_sorted.loc[target_idx, temporal_column]
        last_idx_of_step = df_sorted[df_sorted[temporal_column] == step_value].index.max()
        return last_idx_of_step + 1

    train_end_idx = find_split_index(target_train_idx)
    val_end_idx = find_split_index(target_val_idx)
    
    if val_end_idx <= train_end_idx:
        val_end_idx = train_end_idx
        
    train_df = df_sorted.iloc[:train_end_idx].copy()
    val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
    test_df = df_sorted.iloc[val_end_idx:].copy()
    
    logger.info(f"Split completed. Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows.")
    return train_df, val_df, test_df
