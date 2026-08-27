import pandas as pd
import numpy as np
from fraudshield.logging_config import setup_logger

logger = setup_logger(__name__)

def validate_raw_data(df: pd.DataFrame, required_columns: list[str]) -> dict:
    """
    Performs data validation and basic exploratory checks on the raw DataFrame.
    
    Args:
        df (pd.DataFrame): The raw data loaded.
        required_columns (list[str]): List of columns expected in the DataFrame.
        
    Returns:
        dict: A dictionary containing validation metrics and findings.
    """
    logger.info("Starting raw data validation...")
    results = {}
    
    # 1. Required Columns Check
    missing_cols = [col for col in required_columns if col not in df.columns]
    results['missing_required_columns'] = missing_cols
    results['has_all_required_columns'] = len(missing_cols) == 0
    
    # Early exit if required columns are missing to prevent key errors
    if not results['has_all_required_columns']:
        logger.warning(f"Validation failed. Missing required columns: {missing_cols}")
        return results

    # 2. Basic Shape and Types
    results['num_rows'] = len(df)
    results['num_cols'] = len(df.columns)
    results['data_types'] = df.dtypes.to_dict()
    
    # 3. Missing values
    results['missing_values'] = df.isnull().sum().to_dict()
    
    # 4. Completely duplicated rows
    results['duplicate_rows'] = int(df.duplicated().sum())
    
    # 5. Negative values check
    results['negative_amounts'] = int((df['amount'] < 0).sum())
    results['negative_oldbalanceOrg'] = int((df['oldbalanceOrg'] < 0).sum())
    results['negative_newbalanceOrig'] = int((df['newbalanceOrig'] < 0).sum())
    results['negative_oldbalanceDest'] = int((df['oldbalanceDest'] < 0).sum())
    results['negative_newbalanceDest'] = int((df['newbalanceDest'] < 0).sum())
    
    # 6. Flag constraints check (must be 0 or 1)
    results['isFraud_valid_values'] = bool(df['isFraud'].isin([0, 1]).all())
    results['isFlaggedFraud_valid_values'] = bool(df['isFlaggedFraud'].isin([0, 1]).all())
    
    # 7. Transaction types
    results['transaction_types'] = df['type'].value_counts().to_dict()
    
    # 8. Transaction types with fraud
    fraud_df = df[df['isFraud'] == 1]
    results['fraud_transaction_types'] = fraud_df['type'].value_counts().to_dict()
    
    # 9. Fraud Rate
    total_fraud = len(fraud_df)
    results['total_fraud'] = total_fraud
    results['fraud_rate'] = float(total_fraud / len(df)) if len(df) > 0 else 0.0
    
    # 10. isFraud vs isFlaggedFraud
    crosstab = pd.crosstab(df['isFraud'], df['isFlaggedFraud'])
    results['fraud_vs_flagged'] = crosstab.to_dict()
    
    logger.info(f"Validation completed successfully. Found {total_fraud} fraud cases ({results['fraud_rate']:.4%}).")
    return results
