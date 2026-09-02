from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pages import model_performance, monitoring, overview, transaction_scoring


st.set_page_config(page_title="Fraud Risk Platform", layout="wide")

st.title("Real-Time Transaction Fraud Risk & Monitoring Platform")
st.caption("Portfolio/research system — not a production banking fraud system.")

page = st.sidebar.radio(
    "Page",
    ["Overview", "Transaction Scoring", "Model Performance", "Monitoring"],
)

{
    "Overview": overview.render,
    "Transaction Scoring": transaction_scoring.render,
    "Model Performance": model_performance.render,
    "Monitoring": monitoring.render,
}[page]()
