import time
import random
import pandas as pd

def stream_transaction():
    return pd.DataFrame([{
        "transaction_amount": random.uniform(10, 5000),
        "transaction_time": random.randint(0, 23),
        "account_age_days": random.randint(10, 4000),
        "num_prev_transactions": random.randint(1, 200),
        "is_international": random.choice([0,1])
    }])
