import joblib, os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

def build_model():
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    return pipe

def fit(model, X, y):
    return model.fit(X, y)

def evaluate(model, X, y):
    proba = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, model.predict(X))
    return {'roc_auc': float(auc), 'f1': float(f1)}

def save(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load(path):
    return joblib.load(path)
