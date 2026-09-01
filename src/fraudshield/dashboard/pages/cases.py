import streamlit as st
import pandas as pd
from fraudshield.dashboard.api_client import get_api_client

st.title("Case Queue")

client = get_api_client()

col1, col2 = st.columns(2)
with col1:
    filter_status = st.selectbox("Status", ["", "NEW", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"])
with col2:
    filter_risk = st.selectbox("Risk Level", ["", "CRITICAL", "HIGH", "MEDIUM", "LOW"])

try:
    cases_resp = client.get_cases(limit=100, status=filter_status or None, risk_level=filter_risk or None)
    items = cases_resp.get("items", [])
    total = cases_resp.get("total", 0)
except Exception as e:
    st.error(f"Failed to fetch cases: {e}")
    items = []
    total = 0

st.write(f"Total Cases: {total} (showing up to 100)")

if items:
    df = pd.DataFrame(items)
    # Display basic info
    st.dataframe(df[["case_id", "status", "priority", "risk_level", "calibrated_probability", "created_at"]])
    
    st.divider()
    st.subheader("Case Details & Actions")
    selected_id = st.selectbox("Select Case to View/Update", df["case_id"].tolist())
    
    if selected_id:
        try:
            case = client.get_case(selected_id)
            c1, c2 = st.columns(2)
            c1.write(f"**Status:** {case['status']}")
            c1.write(f"**Priority:** {case['priority']}")
            c1.write(f"**Risk Level:** {case['risk_level']}")
            
            c2.write(f"**Prob:** {case['calibrated_probability']:.4f}")
            c2.write(f"**Inference ID:** {case['inference_id']}")
            c2.write(f"**Note:** {case['analyst_note']}")
            
            with st.expander("Update Case"):
                with st.form("update_form"):
                    new_status = st.selectbox("New Status", ["NEW", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"], index=["NEW", "UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"].index(case['status']))
                    new_note = st.text_area("Analyst Note")
                    actor = st.text_input("Actor (Your Name)", value="analyst_1")
                    
                    submitted = st.form_submit_button("Update")
                    if submitted:
                        payload = {
                            "version": case["version"],
                            "status": new_status,
                            "analyst_note": new_note or None,
                            "actor": actor
                        }
                        try:
                            client.update_case(selected_id, payload)
                            st.success("Case updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                            
            with st.expander("Event History"):
                history = client.get_case_history(selected_id)
                for h in history:
                    st.write(f"**{h['timestamp']}** | {h['actor']} changed status to {h['new_status']}")
                    if h.get('note'):
                        st.caption(f"Note: {h['note']}")
        except Exception as e:
            st.error(f"Error loading case: {e}")
else:
    st.info("No cases found matching the criteria.")
