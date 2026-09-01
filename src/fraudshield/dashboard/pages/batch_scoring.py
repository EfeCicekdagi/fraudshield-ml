import streamlit as st
import pandas as pd
import io
import time
from fraudshield.dashboard.api_client import get_api_client

st.title("Batch Scoring")

client = get_api_client()

st.markdown("""
Upload a CSV file containing transactions. The file must contain the following columns:
`step`, `type`, `amount`, `oldbalanceOrg`, `orig_account_type`, `dest_account_type`
""")

# Provide template
template_csv = "step,type,amount,oldbalanceOrg,orig_account_type,dest_account_type\n1,PAYMENT,100,1000,C,C"
st.download_button("Download Template", data=template_csv, file_name="template.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def sanitize_csv_export(df: pd.DataFrame) -> pd.DataFrame:
    """Escapes values starting with =, +, -, @ to prevent spreadsheet formula injection."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: f"'{x}" if isinstance(x, str) and x.startswith(('=', '+', '-', '@')) else x)
    return df

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} rows.")
        st.dataframe(df.head())
        
        required_cols = {'step', 'type', 'amount', 'oldbalanceOrg', 'orig_account_type', 'dest_account_type'}
        if not required_cols.issubset(df.columns):
            st.error(f"Missing required columns. Found: {list(df.columns)}")
            st.stop()
            
        if st.button("Start Batch Processing"):
            # Chunking exactly to avoid MAX_BATCH_SIZE limits on API
            MAX_BATCH_SIZE = 1000
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            records = df.to_dict("records")
            total = len(records)
            
            processed = 0
            for i in range(0, total, MAX_BATCH_SIZE):
                chunk = records[i:i + MAX_BATCH_SIZE]
                try:
                    res_chunk = client.predict_batch(chunk)
                    results.extend(res_chunk)
                except Exception as e:
                    st.error(f"Error processing chunk {i//MAX_BATCH_SIZE + 1}: {str(e)}")
                    break
                
                processed += len(chunk)
                progress = min(processed / total, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processed {processed} / {total}")
                
            if len(results) == total:
                st.success("Batch processing complete!")
                res_df = pd.DataFrame(results)
                
                alerts = res_df[res_df['risk_level'].isin(['HIGH', 'CRITICAL'])]
                st.metric("Alerts Generated", len(alerts))
                
                safe_df = sanitize_csv_export(res_df)
                csv = safe_df.to_csv(index=False)
                st.download_button("Download Results", data=csv, file_name="results.csv", mime="text/csv")
                
    except Exception as e:
        st.error("Could not parse CSV file. Ensure it is a valid format.")
