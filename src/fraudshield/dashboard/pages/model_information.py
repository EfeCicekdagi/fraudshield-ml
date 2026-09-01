import streamlit as st
from fraudshield.dashboard.api_client import get_api_client

st.title("Model Information")

client = get_api_client()

try:
    info = client.get_model_info()
    
    st.header(f"Model: {info.get('model_type', 'Unknown')} (v{info.get('model_version', 'Unknown')})")
    
    st.subheader("Configuration")
    st.write(f"**Calibration Method:** {info.get('calibration_method')}")
    st.write(f"**Decision Threshold:** {info.get('decision_threshold')}")
    st.write(f"**Explanation Method:** {info.get('explanation_method')}")
    st.write(f"**Bundle Format:** {info.get('bundle_format')}")
    
    st.subheader("Risk Policy")
    st.json(info.get("risk_policy", {}))
    
    st.subheader("Limitations")
    for lim in info.get("known_limitations", []):
        st.warning(lim)
        
    st.info("The calibrated score is an empirically calibrated estimate, not a true independent fraud probability.")
    
except Exception as e:
    st.error(f"Could not load model info: {e}")
