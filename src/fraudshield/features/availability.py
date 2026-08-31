import os
import pandas as pd

FEATURE_AVAILABILITY = {
    # Time
    'step': 'PRE_TRANSACTION_AVAILABLE',
    'hour_of_day': 'PRE_TRANSACTION_AVAILABLE',
    'day': 'PRE_TRANSACTION_AVAILABLE',
    
    # Core Transaction
    'type': 'PRE_TRANSACTION_AVAILABLE',
    'amount': 'PRE_TRANSACTION_AVAILABLE',
    'amount_log1p': 'PRE_TRANSACTION_AVAILABLE',
    'is_risky_type': 'PRE_TRANSACTION_AVAILABLE',
    
    # Origin Account
    'oldbalanceOrg': 'PRE_TRANSACTION_AVAILABLE',
    'orig_oldbalance_is_zero': 'PRE_TRANSACTION_AVAILABLE',
    'amount_to_oldbalance_orig_ratio': 'PRE_TRANSACTION_AVAILABLE',
    'orig_account_type': 'PRE_TRANSACTION_AVAILABLE',
    
    # Destination Account (assuming we can query internal db before approving)
    'oldbalanceDest': 'PRE_TRANSACTION_AVAILABLE',
    'dest_oldbalance_is_zero': 'PRE_TRANSACTION_AVAILABLE',
    'amount_to_oldbalance_dest_ratio': 'PRE_TRANSACTION_AVAILABLE',
    'dest_account_type': 'PRE_TRANSACTION_AVAILABLE',
    
    # Post-transaction only (not available before processing)
    'newbalanceOrig': 'POST_TRANSACTION_ONLY',
    'newbalanceDest': 'POST_TRANSACTION_ONLY',
    'orig_newbalance_is_zero': 'POST_TRANSACTION_ONLY',
    'dest_newbalance_is_zero': 'POST_TRANSACTION_ONLY',
    'orig_balance_delta': 'POST_TRANSACTION_ONLY',
    'dest_balance_delta': 'POST_TRANSACTION_ONLY',
    
    # Simulation Artifacts
    'error_balance_orig': 'SYNTHETIC_ARTIFACT',
    'abs_error_balance_orig': 'SYNTHETIC_ARTIFACT',
    'error_balance_dest': 'SYNTHETIC_ARTIFACT',
    'abs_error_balance_dest': 'SYNTHETIC_ARTIFACT',
    
    # Forbidden/Target
    'isFraud': 'TARGET',
    'isFlaggedFraud': 'TARGET',
    'nameOrig': 'FORBIDDEN_IDENTIFIER',
    'nameDest': 'FORBIDDEN_IDENTIFIER'
}

def check_feature_safety(feature_list):
    """
    Raises ValueError if any feature is not PRE_TRANSACTION_AVAILABLE.
    """
    for f in feature_list:
        if f not in FEATURE_AVAILABILITY:
            raise ValueError(f"Feature {f} not documented in availability map!")
            
        status = FEATURE_AVAILABILITY[f]
        if status != 'PRE_TRANSACTION_AVAILABLE':
            raise ValueError(f"Feature {f} is unsafe! Status: {status}")
            
    return True

def generate_availability_report(output_path):
    lines = [
        "# Feature Availability Report",
        "",
        "| Feature | Availability Status |",
        "|---------|---------------------|"
    ]
    
    for f, status in FEATURE_AVAILABILITY.items():
        lines.append(f"| {f} | {status} |")
        
    lines.extend([
        "",
        "## Destination Balance Availability Notes",
        "**Scenario A (Internal Destination Balance Available):** If the destination account is within the same institution, `oldbalanceDest` can be queried in real-time before transaction approval.",
        "**Scenario B (Destination Balance Unavailable):** If the destination is external, `oldbalanceDest` might not be known. In a true real-world system, this feature might need to be dropped or imputed."
    ])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
