# Streamlit dashboard for the AML Transaction Monitoring System
# Run with: streamlit run dashboard/app.py

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# API base URL - make sure the FastAPI server is running
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AML Monitor",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AML Transaction Monitoring")
st.caption("Real-time fraud detection powered by XGBoost + SHAP")

# Check if API is online
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.success(f"API Online · Model: {health['model']} · Version: {health['version']}")
except:
    st.error("API is offline. Run: uvicorn api.main:app --reload")
    st.stop()

st.divider()

# Layout - two columns
left, right = st.columns([1, 1])

# ── LEFT: Score a transaction ──────────────────────────────────
with left:
    st.subheader("Score a Transaction")

    with st.form("score_form"):
        step = st.number_input("Step (time)", min_value=1, value=1)
        tx_type = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=0.0, value=181.0)

        st.markdown("**Origin Account**")
        col1, col2 = st.columns(2)
        old_orig = col1.number_input("Balance Before", min_value=0.0, value=181.0)
        new_orig = col2.number_input("Balance After", min_value=0.0, value=0.0)

        st.markdown("**Destination Account**")
        col3, col4 = st.columns(2)
        old_dest = col3.number_input("Balance Before ", min_value=0.0, value=0.0)
        new_dest = col4.number_input("Balance After ", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Score Transaction", use_container_width=True)

    if submitted:
        payload = {
            "step": step,
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest
        }

        response = requests.post(f"{API_URL}/score", json=payload).json()

        if response["flagged"]:
            st.error(f"🚨 FRAUD DETECTED — Probability: {response['probability']:.4f}")
        else:
            st.success(f"✅ LEGIT — Probability: {response['probability']:.4f}")

        st.json(response)

# ── RIGHT: Fraud alerts feed ───────────────────────────────────
with right:
    st.subheader("Recent Fraud Alerts")

    if st.button("Refresh Alerts", use_container_width=True):
        st.rerun()

    try:
        alerts = requests.get(f"{API_URL}/alerts").json()

        if not alerts:
            st.info("No fraud alerts yet. Score a transaction on the left.")
        else:
            df = pd.DataFrame(alerts)
            df["probability"] = df["probability"].round(4)
            df.columns = ["ID", "TX ID", "Probability", "Step", "Type", "Amount"]

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Simple bar chart of amounts flagged
            st.markdown("**Flagged amounts by type**")
            chart_data = df.groupby("Type")["Amount"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(chart_data["Type"], chart_data["Amount"], color="#ef4444")
            ax.set_xlabel("Transaction Type")
            ax.set_ylabel("Total Amount")
            ax.set_title("Total Flagged Amount by Type")
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not load alerts: {e}")

st.divider()

# ── BOTTOM: Model stats ────────────────────────────────────────
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "XGBoost")
col2.metric("AUC-ROC", "0.9998")
col3.metric("Recall", "99.88%")
col4.metric("Precision", "86.69%")