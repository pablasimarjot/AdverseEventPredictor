import argparse, pandas as pd
from predictor.featurize import smiles_to_ecfp
from predictor.model import load

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    args = p.parse_args()

    model = load(args.model)
    df = pd.read_csv(args.input)
    X = [smiles_to_ecfp(s) for s in df['smiles']]
    import numpy as np
    X = np.array(X)
    proba = model.predict_proba(X)[:,1]
    df['predicted_risk'] = proba
    df.to_csv('predictions.csv', index=False)
    print('Wrote predictions.csv')

if __name__ == '__main__':
    main()
