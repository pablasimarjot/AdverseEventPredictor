import argparse, os, json, pandas as pd
from predictor.featurize import smiles_to_ecfp
from predictor.model import build_model, fit, evaluate, save

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='data/example_sider.csv')
    p.add_argument('--outdir', type=str, default='artifacts')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    X = [smiles_to_ecfp(s) for s in df['smiles']]
    import numpy as np
    X = np.array(X)
    y = df['side_effect_label']

    model = build_model()
    model = fit(model, X, y)
    metrics = evaluate(model, X, y)

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, 'model.joblib')
    save(model, outpath)
    with open(os.path.join(args.outdir, 'metrics.json'),'w') as f:
        json.dump(metrics, f, indent=2)
    print('Saved model to', outpath)
    print('Metrics:', metrics)

if __name__ == '__main__':
    main()
