import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================
# Streamlit Page Config
# ============================
st.set_page_config(
    page_title="Fraud Detection System with SHAP",
    layout="wide"
)

st.title("💳 Fraud Detection System with SHAP Explainability")

# ============================
# Load Model & Scaler
# ============================
model = joblib.load("../outputs/fraud_model.pkl")
scaler = joblib.load("../outputs/scaler.pkl")

FEATURES = [
    "transaction_amount",
    "transaction_time",
    "account_age_days",
    "num_prev_transactions",
    "is_international"
]

# ============================
# Sidebar Inputs
# ============================
st.sidebar.header("🧾 Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount", min_value=0.0, value=500.0, step=50.0
)

hour = st.sidebar.slider(
    "Transaction Time (Hour)", 0, 23, 12
)

account_age = st.sidebar.number_input(
    "Account Age (Days)", min_value=0, value=365
)

prev_txns = st.sidebar.number_input(
    "Previous Transactions", min_value=0, value=20
)

international = st.sidebar.selectbox(
    "International Transaction", [0, 1]
)

# ============================
# Prediction Button
# ============================
if st.sidebar.button("🚨 Check Fraud Risk"):

    # ----------------------------
    # Create input DataFrame
    # ----------------------------
    input_df = pd.DataFrame([{
        "transaction_amount": amount,
        "transaction_time": hour,
        "account_age_days": account_age,
        "num_prev_transactions": prev_txns,
        "is_international": international
    }])

    # ----------------------------
    # Scale input (IMPORTANT)
    # ----------------------------
    input_scaled = scaler.transform(input_df)

    # ----------------------------
    # Prediction
    # ----------------------------
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("🔮 Prediction Result")

    if pred == 1:
        st.error(f"❌ Fraud Detected — Risk: {prob:.2f}%")
    else:
        st.success(f"✅ Transaction Safe — Fraud Risk: {prob:.2f}%")

    # ----------------------------
    # Risk Level
    # ----------------------------
    st.subheader("📊 Risk Level")

    if prob < 30:
        st.info("🟢 Low Risk")
    elif prob < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # ============================
    # SHAP Explainability
    # ============================
    st.subheader("🧠 SHAP Explanation")

    explainer = shap.TreeExplainer(model)

    # SHAP values for THIS input
    shap_values = explainer.shap_values(input_scaled)

    # ----------------------------
    # ✅ SAFE handling for binary classification
    # ----------------------------
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]   # class 1 = fraud
    else:
        shap_vals = shap_values[0]

    shap_vals = np.array(shap_vals).flatten()

    # Ensure same length
    shap_vals = shap_vals[:len(FEATURES)]

    # ----------------------------
    # SHAP DataFrame
    # ----------------------------
    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Value": shap_vals
    }).sort_values(by="SHAP Value", key=abs)

    # ----------------------------
    # SHAP Bar Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["red" if v > 0 else "green" for v in shap_df["SHAP Value"]]

    ax.barh(
        shap_df["Feature"],
        shap_df["SHAP Value"],
        color=colors
    )

    ax.set_title("Feature Impact on Fraud Prediction")
    ax.set_xlabel("SHAP Value")

    st.pyplot(fig)
