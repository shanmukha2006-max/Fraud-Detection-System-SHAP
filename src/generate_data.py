import pandas as pd
import numpy as np

np.random.seed(42)
rows = 6000

data = {
    "transaction_amount": np.random.exponential(scale=200, size=rows),
    "transaction_time": np.random.randint(0, 24, size=rows),
    "account_age_days": np.random.randint(10, 4000, size=rows),
    "num_prev_transactions": np.random.randint(0, 300, size=rows),
    "is_international": np.random.choice([0, 1], size=rows, p=[0.85, 0.15]),
}

df = pd.DataFrame(data)

# Fraud logic (realistic)
df["fraud"] = (
    (df["transaction_amount"] > 1500) |
    ((df["is_international"] == 1) & (df["transaction_amount"] > 700)) |
    ((df["account_age_days"] < 30) & (df["transaction_amount"] > 500))
).astype(int)

df.to_csv("../data/transactions.csv", index=False)

print("✅ Realistic fraud dataset created")
print(df["fraud"].value_counts())
