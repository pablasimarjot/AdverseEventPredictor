# AdverseEventPredictor 💊🧬
AI to predict adverse drug reactions (ADRs) from molecular structure + target info.

**Why this repo matters**
- Drug safety is a critical bottleneck in pharma/biotech.
- Shows recruiters and collaborators you can build AI pipelines with **real-world impact**.
- Combines cheminformatics (RDKit), ML (scikit-learn), and explainability (SHAP).
- Includes **Streamlit app**, **Dockerfile**, and **CI/CD**.

⚠️ **Disclaimer**: Demo project using public/synthetic data. Not for clinical use.

---

## Quickstart
```bash
pip install -r requirements.txt

# Download and preprocess SIDER dataset (drug–side effect associations)
python scripts/download_sider.py

# Train a model
python scripts/train.py --outdir artifacts/

# Predict side effects for new compounds
python scripts/predict.py --model artifacts/model.joblib --input data/example_drugs.csv

# Launch app
streamlit run app/streamlit_app.py
```

---

## Repo structure
```
AdverseEventPredictor/
├─ app/                   # Streamlit app
├─ predictor/             # Core library
├─ scripts/               # Data + train + predict scripts
├─ tests/                 # Unit tests
├─ artifacts/             # (created) trained models
├─ requirements.txt
├─ Dockerfile
└─ README.md
```
## Potential Insights
```
<img width="1200" height="800" alt="adverse_risk_chart" src="https://github.com/user-attachments/assets/2959e6c5-9458-4068-b741-7e144618e9d0" />
<img width="1200" height="800" alt="adverse_shap_importance" src="https://github.com/user-attachments/assets/abdabf98-3746-4b8a-b148-b5f373474287" />

