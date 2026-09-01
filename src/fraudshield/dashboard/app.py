import streamlit as st

st.set_page_config(
    page_title="FraudShield ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define pages
overview = st.Page("pages/overview.py", title="Overview", icon="📊", default=True)
live_scoring = st.Page("pages/live_scoring.py", title="Live Scoring", icon="⚡")
batch_scoring = st.Page("pages/batch_scoring.py", title="Batch Scoring", icon="📁")
cases = st.Page("pages/cases.py", title="Case Queue", icon="📋")
model_info = st.Page("pages/model_information.py", title="Model Info", icon="🧠")

# Navigation
pg = st.navigation({
    "Dashboard": [overview, model_info],
    "Scoring": [live_scoring, batch_scoring],
    "Case Management": [cases]
})

# Run the selected page
pg.run()
