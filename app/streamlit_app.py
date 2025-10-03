import pandas as pd
import streamlit as st
from predictor.model import load
from predictor.featurize import smiles_to_ecfp

st.set_page_config(page_title='AdverseEventPredictor', page_icon='💊', layout='wide')
st.title('AdverseEventPredictor 💊')
st.caption('Demo only — not for clinical use.')

uploaded_model = st.file_uploader('Upload trained model (.joblib)', type='joblib')
uploaded_csv = st.file_uploader('Upload CSV with SMILES', type='csv')

if uploaded_model and uploaded_csv:
    import joblib, numpy as np
    model = joblib.load(uploaded_model)
    df = pd.read_csv(uploaded_csv)
    X = np.array([smiles_to_ecfp(s) for s in df['smiles']])
    proba = model.predict_proba(X)[:,1]
    df['predicted_risk'] = proba
    st.write(df)
    st.bar_chart(df['predicted_risk'])
else:
    st.info('Upload a model and drug CSV to begin.')
