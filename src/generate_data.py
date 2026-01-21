import pandas as pd
import numpy as np

np.random.seed(42)

n = 8000

data = {
    "transaction_amount": np.random.exponential(200, n),
    "transaction_time": np.random.randint(0, 24, n),
    "account_age_days": np.random.randint(10, 4000, n),
    "num_prev_transactions": np.random.randint(1, 200, n),
    "is_international": np.random.choice([0, 1], n, p=[0.85, 0.15])
}

df = pd.DataFrame(data)

# Fraud logic (realistic)
df["fraud"] = (
    (df.transaction_amount > 1500).astype(int) |
    ((df.is_international == 1) & (df.transaction_time < 5)).astype(int)
)

df.to_csv("../data/fraud_data.csv", index=False)

print("✅ Fraud dataset created")
print(df["fraud"].value_counts())
