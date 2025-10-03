import shap
import numpy as np

def shap_values(model, X):
    clf = model.named_steps['clf']
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(model.named_steps['scaler'].transform(X))
    return sv
