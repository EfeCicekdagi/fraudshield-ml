import streamlit as st
from fraudshield.dashboard.api_client import get_api_client

st.title("FraudShield ML Overview")

client = get_api_client()

# Check Readiness
if not client.get_health_ready():
    st.error("API is currently unavailable or Model is not loaded. Please check backend logs.")
    st.stop()

st.success("API is Online & Ready.")

try:
    summary = client.get_dashboard_summary()
except Exception as e:
    st.error(f"Failed to fetch summary: {str(e)}")
    st.stop()

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Open Cases", summary.get("total_open_cases", 0))
col2.metric("Critical/High Cases", summary.get("critical_high_cases", 0))
col3.metric("Confirmed Fraud", summary.get("confirmed_fraud_count", 0))
col4.metric("Median Open Probability", f"{summary.get('median_probability', 0):.4f}")

# Distribution
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Status Distribution")
    dist = summary.get("status_distribution", {})
    if dist:
        import pandas as pd
        df_status = pd.DataFrame(list(dist.items()), columns=["Status", "Count"])
        st.bar_chart(df_status.set_index("Status"))
    else:
        st.info("No data available.")

with col_right:
    st.subheader("Risk Distribution (Open Cases)")
    risk = summary.get("risk_distribution", {})
    if risk:
        import pandas as pd
        df_risk = pd.DataFrame(list(risk.items()), columns=["Risk Level", "Count"])
        st.bar_chart(df_risk.set_index("Risk Level"))
    else:
        st.info("No data available.")

# Recent Cases preview
st.subheader("Recent Open Cases")
try:
    cases_resp = client.get_cases(limit=5, status="NEW")
    items = cases_resp.get("items", [])
    if items:
        import pandas as pd
        df = pd.DataFrame(items)
        st.dataframe(df[["case_id", "inference_id", "calibrated_probability", "risk_level", "created_at"]])
    else:
        st.info("No NEW cases found.")
except Exception as e:
    st.warning("Could not fetch recent cases.")
