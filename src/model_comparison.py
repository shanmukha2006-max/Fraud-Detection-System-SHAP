import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compare_models(df):
    X = df.drop("fraud", axis=1)
    y = df["fraud"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:,1]
        results[name] = roc_auc_score(y_test, preds)

    return results
