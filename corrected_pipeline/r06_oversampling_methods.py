"""Supplementary Table 12 for the corrected re-run: XGBoost with no oversampling, random oversampling, SMOTE and ADASYN
(same per-class targets, same training/test isolates)."""
import os
import json, sys, numpy as np, pandas as pd, xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r02_train import splits
D = os.path.dirname(os.path.abspath(__file__))
rows = []
for drug in ["AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH", "KAN", "LEV", "LZD", "MXF", "RIF"]:
    S = splits(drug); X, y, R = S["X"], S["y"], S["R"]; tr, te = S["train_orig"], S["test"]; K = S["K"]
    rc = int(np.argmax(np.load(f"{D}/data/{drug}.npz")["mic_levels"] > float(np.load(f"{D}/data/{drug}.npz")["ecoff"]) * 1.05))
    kmin = int(np.bincount(y[tr]).min())
    for name, s in (("none", None), ("random", RandomOverSampler(sampling_strategy=S["target"], random_state=42)),
                    ("smote", SMOTE(sampling_strategy=S["target"], random_state=42, k_neighbors=max(1, min(5, kmin - 1)))),
                    ("adasyn", ADASYN(sampling_strategy=S["target"], random_state=42, n_neighbors=max(1, min(5, kmin - 1))))):
        try:
            Xs, ys = (X[tr], y[tr]) if s is None else s.fit_resample(X[tr], y[tr])
            m = xgb.XGBClassifier(random_state=42, n_jobs=16).fit(Xs, ys)
            P = np.zeros((len(te), K)); P[:, m.classes_] = m.predict_proba(X[te]); p = P.argmax(1); yt = y[te]
            r = dict(drug=drug, sampler=name, n_train=len(ys), dda=float(np.mean(np.abs(p - yt) <= 1)),
                     auc_ovr=roc_auc_score(yt, P, multi_class="ovr", labels=list(range(K))), bal_acc=balanced_accuracy_score(yt, p),
                     macro_f1=f1_score(yt, p, average="macro", zero_division=0), R_recall=recall_score(R[te], (p >= rc).astype(int), zero_division=0))
        except Exception as e:
            r = dict(drug=drug, sampler=name, error=str(e)[:120])
        rows.append(r); print(json.dumps(r, default=float), flush=True)
pd.DataFrame(rows).to_csv(f"{D}/tables/s12_oversampling_methods.csv", index=False)
