import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

st.title("💳 Fraud Detection System with SHAP Explainability")

# -------------------------------
# Load model & scaler
# -------------------------------
model = joblib.load("../outputs/fraud_model.pkl")
scaler = joblib.load("../outputs/scaler.pkl")

# -------------------------------
# Load reference data (for SHAP background)
# -------------------------------
data = pd.read_csv("../data/transactions.csv")
X_background = data.drop("fraud", axis=1)
X_background_scaled = scaler.transform(X_background)

# Create SHAP explainer (IMPORTANT: no check_additivity here)
explainer = shap.Explainer(model, X_background_scaled)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🧾 Transaction Input")

amount = st.sidebar.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=500.0,
    step=10.0
)

time = st.sidebar.slider(
    "Transaction Time (Hour)",
    0, 23, 12
)

account_age = st.sidebar.number_input(
    "Account Age (Days)",
    min_value=0,
    value=365
)

prev_txns = st.sidebar.number_input(
    "Previous Transactions",
    min_value=0,
    value=20
)

international = st.sidebar.selectbox(
    "International Transaction",
    [0, 1]
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("🚨 Check Fraud Risk"):

    # Create input dataframe
    input_df = pd.DataFrame([{
        "transaction_amount": amount,
        "transaction_time": time,
        "account_age_days": account_age,
        "num_prev_transactions": prev_txns,
        "is_international": international
    }])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    # -------------------------------
    # Display Prediction
    # -------------------------------
    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — Risk: {prob*100:.2f}%")
    else:
        st.success(f"✅ Transaction Safe — Fraud Risk: {prob*100:.2f}%")

    # -------------------------------
    # Risk Band
    # -------------------------------
    st.subheader("📊 Risk Level")

    if prob < 0.3:
        st.info("🟢 Low Risk")
    elif prob < 0.6:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # -------------------------------
# SHAP Explainability (FIXED)
# -------------------------------
st.subheader("🧠 SHAP Explainability")

# Compute SHAP values FIRST
shap_values = explainer(
    input_scaled,
    check_additivity=False
)

# Extract SHAP values for FRAUD class (class 1)
shap_vals = shap_values.values[0, :, 1]

shap_df = pd.DataFrame({
    "Feature": input_df.columns,
    "SHAP Value": shap_vals
}).sort_values(by="SHAP Value", key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(
    shap_df["Feature"],
    shap_df["SHAP Value"],
    color=["red" if v > 0 else "green" for v in shap_df["SHAP Value"]]
)
ax.invert_yaxis()
ax.set_title("Feature Impact on Fraud Prediction")

st.pyplot(fig)
