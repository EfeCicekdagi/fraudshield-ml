import streamlit as st
from fraudshield.dashboard.api_client import get_api_client

st.title("Live Scoring")

client = get_api_client()

with st.form("live_score_form"):
    st.subheader("Transaction Details")
    col1, col2 = st.columns(2)
    
    with col1:
        step = st.number_input("Step (Hours)", min_value=1, value=1, step=1)
        type_val = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0, value=100.0, step=10.0)
        oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=1000.0, step=10.0)
        
    with col2:
        orig_account_type = st.selectbox("Origin Account Type", ["C", "M"], index=0)
        dest_account_type = st.selectbox("Destination Account Type", ["C", "M"], index=0)
        
    submitted = st.form_submit_button("Score Transaction")

if submitted:
    payload = {
        "step": step,
        "type": type_val,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "orig_account_type": orig_account_type,
        "dest_account_type": dest_account_type
    }
    
    try:
        with st.spinner("Scoring..."):
            res = client.predict_single(payload, explain=False)
            st.session_state["last_prediction"] = res
            st.session_state["last_payload"] = payload
    except Exception as e:
        st.error(f"Error during scoring: {str(e)}")

if "last_prediction" in st.session_state:
    res = st.session_state["last_prediction"]
    st.divider()
    st.subheader("Result")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", res.get("risk_level", "UNKNOWN"))
    col2.metric("Probability", f"{res.get('calibrated_probability', 0):.4f}")
    col3.metric("Fraud Score (Logit)", f"{res.get('fraud_score', 0):.4f}")
    
    st.write(f"**Inference ID:** `{res.get('inference_id')}`")
    st.write(f"**Reason Codes:** {', '.join(res.get('reason_codes', []))}")
    
    if st.button("Generate Explanation (IG)"):
        try:
            with st.spinner("Calculating Integrated Gradients..."):
                exp_res = client.predict_single(st.session_state["last_payload"], explain=True)
                st.session_state["last_prediction"] = exp_res
                st.rerun()
        except Exception as e:
            st.error(f"Explanation failed: {str(e)}")
            
    if res.get("explanation", {}).get("enabled"):
        st.subheader("Feature Contributions")
        import pandas as pd
        contribs = res.get("top_contributors", [])
        if contribs:
            df = pd.DataFrame(contribs)
            st.bar_chart(df.set_index("feature")["attribution"])

    st.divider()
    st.subheader("Case Management")
    if st.button("Create Case from this Transaction"):
        try:
            case_payload = {
                "inference_id": res["inference_id"],
                "calibrated_probability": res["calibrated_probability"],
                "risk_level": res["risk_level"],
                "reason_codes": res["reason_codes"],
                "model_version": res["model_version"],
                "priority": "HIGH" if res["risk_level"] in ["CRITICAL", "HIGH"] else "LOW"
            }
            case_res = client.create_case(case_payload)
            st.success(f"Case `{case_res['case_id']}` created successfully!")
        except Exception as e:
            st.error(f"Failed to create case: {str(e)}")
