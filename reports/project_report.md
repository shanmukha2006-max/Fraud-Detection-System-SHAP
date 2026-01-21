# Fraud Detection in Financial Transactions using Machine Learning

## 1. Introduction
Financial fraud poses a significant risk to banks and digital payment systems.  
This project focuses on building an AI-powered fraud detection system capable of
identifying suspicious transactions in real time while providing transparent
model explanations.

---

## 2. Problem Statement
A financial institution needs to enhance transaction security by detecting
fraudulent activities early and minimizing financial losses.  
The solution must be accurate, explainable, and suitable for real-time usage.

---

## 3. Dataset Description
The dataset contains simulated financial transactions with the following features:

- transaction_amount
- transaction_time (hour of day)
- account_age_days
- num_prev_transactions
- is_international
- fraud (target variable)

**Class Imbalance:**  
Fraudulent transactions represent a very small percentage of total data,
which reflects real-world conditions.

---

## 4. Methodology

### 4.1 Data Preprocessing
- Removed invalid values
- Feature scaling using StandardScaler
- Handled class imbalance carefully

### 4.2 Model Training
Multiple models were tested:
- Logistic Regression
- Random Forest (selected)
- Cost-sensitive learning applied

### 4.3 Explainability
SHAP (SHapley Additive Explanations) was used to:
- Explain individual predictions
- Identify key fraud indicators

---

## 5. Results

### Model Performance
- Accuracy: ~99%
- Precision (Fraud): High
- Recall (Fraud): Improved using cost-sensitive learning

### Key Fraud Indicators
- Transaction amount
- Transaction time
- International transactions

---

## 6. Real-Time Fraud Detection
A Streamlit dashboard was developed to:
- Accept live transaction inputs
- Display fraud probability
- Visualize SHAP explanations

---

## 7. Conclusion
The system successfully detects fraudulent transactions with high accuracy and
offers transparent explanations.  
This makes it suitable for deployment in real-world financial systems.

---

## 8. Future Enhancements
- Real-time streaming using Kafka
- Deep learning models
- Integration with alert systems
