import requests
import streamlit as st

from src.config import settings


def render() -> None:
    st.header("Transaction Scoring")
    api_url = st.text_input("API URL", settings.api_base_url)
    amount = st.number_input("Transaction amount", min_value=0.0, value=125.50)
    product = st.selectbox("ProductCD", ["W", "C", "R", "H", "S"])
    card1 = st.number_input("card1", min_value=0, value=12345)
    card4 = st.selectbox("card4", ["visa", "mastercard", "discover", "american express"])
    device = st.selectbox("DeviceType", ["desktop", "mobile"])
    if st.button("Score transaction"):
        payload = {
            "TransactionAmt": amount,
            "ProductCD": product,
            "card1": card1,
            "card4": card4,
            "DeviceType": device,
        }
        try:
            response = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.metric("Fraud Probability", f"{result['fraud_probability']:.3f}")
            st.write("Prediction:", result.get("prediction"))
            st.write("Risk Level:", result["risk_level"])
            st.write("Decision:", result["decision"])
            st.write("Threshold:", result["threshold"])
            st.write("Reason Codes:", result["reason_codes"])
        except requests.RequestException as exc:
            st.error(
                "API call failed. Make sure Terminal 1 is running `uvicorn api.main:app --reload`. "
                f"Details: {exc}"
            )
