import shap
import numpy as np

def explain_prediction(model, X_row, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)[1][0]

    explanation = []
    for name, val in zip(feature_names, shap_values):
        if abs(val) > 0.1:
            impact = "increased" if val > 0 else "decreased"
            explanation.append(f"{name} {impact} fraud risk")

    return explanation
