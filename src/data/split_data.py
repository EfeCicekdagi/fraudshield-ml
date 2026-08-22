import pandas as pd

def temporal_train_val_test_split(df: pd.DataFrame, train_size: float = 0.70, val_size: float = 0.15):
    """
    Splits the dataframe into Train, Validation, and Test sets based on the 'step' column.
    Ensures that records with the same 'step' value are not scattered across different splits,
    preventing data leakage and maintaining chronological integrity.
    
    Args:
        df (pd.DataFrame): Dataframe containing a 'step' column.
        train_size (float): Approximate proportion of data for the training set.
        val_size (float): Approximate proportion of data for the validation set.
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if 'step' not in df.columns:
        raise ValueError("The dataframe must contain a 'step' column for temporal splitting.")
        
    # Ensure data is sorted chronologically
    df_sorted = df.sort_values(by='step').reset_index(drop=True)
    
    n_total = len(df_sorted)
    target_train_idx = int(n_total * train_size)
    target_val_idx = int(n_total * (train_size + val_size))
    
    def find_split_index(target_idx):
        if target_idx >= n_total:
            return n_total
        
        step_value = df_sorted.loc[target_idx, 'step']
        # Find the last index where the step is equal to step_value
        # This ensures all records with `step_value` go to the left of the split (if we include up to this index)
        # Actually, it's safer to find the first index of the *next* step to make a clean cut.
        # Let's find the maximum index where step == step_value
        last_idx_of_step = df_sorted[df_sorted['step'] == step_value].index.max()
        
        # The slice will be df_sorted.iloc[:last_idx_of_step + 1]
        return last_idx_of_step + 1

    train_end_idx = find_split_index(target_train_idx)
    val_end_idx = find_split_index(target_val_idx)
    
    # Handle edge case where val_end_idx might be <= train_end_idx due to huge step blocks
    if val_end_idx <= train_end_idx:
        val_end_idx = train_end_idx
        
    train_df = df_sorted.iloc[:train_end_idx].copy()
    val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
    test_df = df_sorted.iloc[val_end_idx:].copy()
    
    return train_df, val_df, test_df
