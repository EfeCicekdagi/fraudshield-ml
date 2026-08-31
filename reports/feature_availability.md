# Feature Availability Report

| Feature | Availability Status |
|---------|---------------------|
| step | PRE_TRANSACTION_AVAILABLE |
| hour_of_day | PRE_TRANSACTION_AVAILABLE |
| day | PRE_TRANSACTION_AVAILABLE |
| type | PRE_TRANSACTION_AVAILABLE |
| amount | PRE_TRANSACTION_AVAILABLE |
| amount_log1p | PRE_TRANSACTION_AVAILABLE |
| is_risky_type | PRE_TRANSACTION_AVAILABLE |
| oldbalanceOrg | PRE_TRANSACTION_AVAILABLE |
| orig_oldbalance_is_zero | PRE_TRANSACTION_AVAILABLE |
| amount_to_oldbalance_orig_ratio | PRE_TRANSACTION_AVAILABLE |
| orig_account_type | PRE_TRANSACTION_AVAILABLE |
| oldbalanceDest | PRE_TRANSACTION_AVAILABLE |
| dest_oldbalance_is_zero | PRE_TRANSACTION_AVAILABLE |
| amount_to_oldbalance_dest_ratio | PRE_TRANSACTION_AVAILABLE |
| dest_account_type | PRE_TRANSACTION_AVAILABLE |
| newbalanceOrig | POST_TRANSACTION_ONLY |
| newbalanceDest | POST_TRANSACTION_ONLY |
| orig_newbalance_is_zero | POST_TRANSACTION_ONLY |
| dest_newbalance_is_zero | POST_TRANSACTION_ONLY |
| orig_balance_delta | POST_TRANSACTION_ONLY |
| dest_balance_delta | POST_TRANSACTION_ONLY |
| error_balance_orig | SYNTHETIC_ARTIFACT |
| abs_error_balance_orig | SYNTHETIC_ARTIFACT |
| error_balance_dest | SYNTHETIC_ARTIFACT |
| abs_error_balance_dest | SYNTHETIC_ARTIFACT |
| isFraud | TARGET |
| isFlaggedFraud | TARGET |
| nameOrig | FORBIDDEN_IDENTIFIER |
| nameDest | FORBIDDEN_IDENTIFIER |

## Destination Balance Availability Notes
**Scenario A (Internal Destination Balance Available):** If the destination account is within the same institution, `oldbalanceDest` can be queried in real-time before transaction approval.
**Scenario B (Destination Balance Unavailable):** If the destination is external, `oldbalanceDest` might not be known. In a true real-world system, this feature might need to be dropped or imputed.