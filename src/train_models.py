import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report

df = pd.read_csv("../data/fraud_data.csv")

X = df.drop("fraud", axis=1)
y = df["fraud"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cost-sensitive learning
model = RandomForestClassifier(
    n_estimators=200,
    class_weight={0:1, 1:8},
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.3
)

model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

# Anomaly Detection
anomaly_model = IsolationForest(contamination=0.03, random_state=42)
anomaly_model.fit(X_scaled)

joblib.dump(model, "../outputs/fraud_model.pkl")
joblib.dump(scaler, "../outputs/scaler.pkl")
joblib.dump(anomaly_model, "../outputs/anomaly_model.pkl")

print("✅ Models saved")
