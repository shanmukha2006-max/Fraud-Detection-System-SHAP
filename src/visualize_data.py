import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("../data/transactions.csv")

print(df.head())
print("\nFraud distribution:")
print(df["fraud"].value_counts())

# Plot fraud distribution
plt.figure()
sns.countplot(x="fraud", data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# Transaction amount vs fraud
plt.figure()
sns.boxplot(x="fraud", y="transaction_amount", data=df)
plt.title("Transaction Amount by Fraud")
plt.show()

# International vs fraud
plt.figure()
sns.countplot(x="is_international", hue="fraud", data=df)
plt.title("International Transactions & Fraud")
plt.show()
