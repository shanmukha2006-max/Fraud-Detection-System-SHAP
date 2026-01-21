# 🛡️ Fraud Detection System with SHAP Explainability

A **production-ready Machine Learning project** that detects fraudulent financial transactions and explains predictions using **SHAP (Explainable AI)**.  
Built with **Scikit-learn, Streamlit, and SHAP**, this system combines **high accuracy**, **imbalanced-data handling**, and **model transparency**.

---

## 🚀 Project Highlights

- ✅ Realistic imbalanced fraud dataset
- ✅ High-performance fraud classification model
- ✅ Saved model & scaler (production ready)
- ✅ Interactive Streamlit dashboard
- ✅ SHAP explainability for every prediction
- ✅ Clear risk scoring (Low / Medium / High)

---

## 🧠 Problem Statement

Fraudulent transactions are **rare but extremely costly**.  
Traditional ML models act as black boxes, making it difficult to trust predictions.

This project solves:
- Accurate fraud detection
- Explainability (why a transaction is risky)
- Interactive visualization for users

---

## 🧩 Features Used

| Feature | Description |
|------|------------|
| transaction_amount | Amount of the transaction |
| transaction_time | Hour of transaction (0–23) |
| account_age_days | Age of account in days |
| num_prev_transactions | Number of previous transactions |
| is_international | Domestic (0) / International (1) |

---

## 🏗️ Project Structure

Fraud-Detection-Project/
│
├── data/
│ └── fraud_data.csv
│
├── src/
│ ├── generate_data.py
│ ├── fraud_detection.py
│ ├── fraud_detection_improved.py
│ ├── train_and_save_model.py
│ ├── shap_explain.py
│ └── app.py
│
├── outputs/
│ ├── fraud_model.pkl
│ └── scaler.pkl
│
├── notebooks/
├── reports/
├── requirements.txt
└── README.md


---

## ⚙️ Tech Stack

- Python
- Scikit-learn
- Pandas / NumPy
- Streamlit
- SHAP
- Matplotlib / Seaborn
- Joblib

---

## 📊 Model Performance

- Handles severely imbalanced data
- High recall for fraud cases
- Probability-based risk scoring
- Production-stable pipeline

**Sample Classification Report**
precision recall f1-score
Fraud (1) 1.00 0.86 0.92

---

## 🔍 SHAP Explainability

The system explains **each prediction visually**:

- 🔴 Features increasing fraud risk
- 🟢 Features reducing fraud risk
- Contribution strength of each feature

This ensures:
- Transparency
- Trust
- Audit readiness

---

## 🖥️ Streamlit Dashboard

### Dashboard Features
- User-friendly input sliders
- Real-time fraud probability
- Risk categorization:
  - 🟢 Low Risk
  - 🟡 Medium Risk
  - 🔴 High Risk
- SHAP feature impact bar chart

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository
```bash
git clone https://github.com/shanmukha2006-max/Fraud-Detection-System-SHAP.git
cd Fraud-Detection-System-SHAP


2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Generate Dataset
python src/generate_data.py

5️⃣ Train & Save Model
python src/train_and_save_model.py

6️⃣ Run Streamlit App
streamlit run src/app.py


Open in browser:

http://localhost:8501

🎯 Use Cases

Banking & fintech fraud detection

Risk assessment systems

Explainable AI demonstrations

Academic projects

ML portfolio showcase

📈 Future Enhancements

Real-time transaction ingestion

API deployment (FastAPI)

Cloud deployment (AWS / Azure)

Advanced anomaly detection models

👨‍💻 Author

CH SHANMUKHA VENKATA LAKSHMAN
Machine Learning | Data Science | Explainable AI

⭐ Final Note

This project demonstrates:

Strong ML fundamentals

Real-world problem solving

Explainable AI (highly valued skill)

Production-ready ML pipeline

⭐ If you like this project, give it a star!
