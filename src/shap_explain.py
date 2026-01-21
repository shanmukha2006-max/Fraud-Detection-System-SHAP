import pandas as pd
import shap
import joblib

# Load model & scaler
model = joblib.load("../outputs/fraud_model.pkl")
scaler = joblib.load("../outputs/scaler.pkl")

# Load data
df = pd.read_csv("../data/transactions.csv")
X = df.drop("fraud", axis=1)

# Scale
X_scaled = scaler.transform(X)

# Create explainer (NO check_additivity here)
explainer = shap.Explainer(model, X_scaled)

# ⬇️ Disable additivity check HERE (THIS IS THE KEY)
shap_values = explainer(
    X_scaled,
    check_additivity=False
)

print("✅ SHAP values computed successfully")

# Plot fraud class explanations (class 1)
shap.summary_plot(
    shap_values.values[:, :, 1],
    X,
    show=True
)
