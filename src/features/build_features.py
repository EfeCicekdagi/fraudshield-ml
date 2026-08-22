import pandas as pd
import numpy as np

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time, transaction, and balance features for the dataset.
    Safely drops raw identifying columns while preserving target columns for baseline tracking.
    
    Args:
        df (pd.DataFrame): Raw dataframe containing required columns.
        
    Returns:
        pd.DataFrame: A new DataFrame with engineered features.
    """
    # Create a copy to avoid SettingWithCopyWarning or mutating original df directly
    df = df.copy()

    # --- 1. Time Features ---
    # step is 1 hour of time.
    df['hour_of_day'] = ((df['step'] - 1) % 24).astype('int8')
    df['day'] = ((df['step'] - 1) // 24 + 1).astype('int16')
    df['week'] = ((df['day'] - 1) // 7 + 1).astype('int8')
    
    # --- 2. Transaction Features ---
    # is_risky_type: True if TRANSFER or CASH_OUT
    df['is_risky_type'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype('int8')
    df['amount_log1p'] = np.log1p(df['amount']).astype('float32')
    df['amount_is_zero'] = (df['amount'] == 0).astype('int8')

    # --- 3. Source Account Balance Features ---
    df['orig_balance_delta'] = (df['oldbalanceOrg'] - df['newbalanceOrig']).astype('float32')
    # error_balance_orig = expected new balance - actual new balance
    # expected = oldbalance - amount (for most outgoing transactions)
    df['error_balance_orig'] = (df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']).astype('float32')
    df['abs_error_balance_orig'] = df['error_balance_orig'].abs()
    
    df['orig_oldbalance_is_zero'] = (df['oldbalanceOrg'] == 0).astype('int8')
    df['orig_newbalance_is_zero'] = (df['newbalanceOrig'] == 0).astype('int8')
    
    # amount to oldbalance ratio, avoiding zero division
    df['amount_to_oldbalance_orig_ratio'] = np.where(
        df['oldbalanceOrg'] == 0, 
        0.0, 
        df['amount'] / df['oldbalanceOrg']
    ).astype('float32')
    
    # --- 4. Destination Account Balance Features ---
    df['dest_balance_delta'] = (df['newbalanceDest'] - df['oldbalanceDest']).astype('float32')
    # expected new dest balance = oldbalance + amount
    df['error_balance_dest'] = (df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']).astype('float32')
    df['abs_error_balance_dest'] = df['error_balance_dest'].abs()
    
    df['dest_oldbalance_is_zero'] = (df['oldbalanceDest'] == 0).astype('int8')
    df['dest_newbalance_is_zero'] = (df['newbalanceDest'] == 0).astype('int8')
    
    df['amount_to_oldbalance_dest_ratio'] = np.where(
        df['oldbalanceDest'] == 0, 
        0.0, 
        df['amount'] / df['oldbalanceDest']
    ).astype('float32')
    
    # --- 5. Account Type Features ---
    # 'C' for Customer, 'M' for Merchant
    # Convert nameOrig and nameDest to string before taking the first character
    df['orig_account_type'] = df['nameOrig'].astype(str).str[0].astype('category')
    df['dest_account_type'] = df['nameDest'].astype(str).str[0].astype('category')
    
    # Drop raw identifiers to avoid leakage / high cardinality noise
    # We keep 'isFraud' and 'isFlaggedFraud' for baseline comparison, 
    # but they must NOT be used as features during training.
    columns_to_drop = ['nameOrig', 'nameDest']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    # Ensure no uncontrolled NaNs or Infs from our math
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill remaining NaNs introduced by inf replacements with 0
    df.fillna(0, inplace=True)
    
    return df
