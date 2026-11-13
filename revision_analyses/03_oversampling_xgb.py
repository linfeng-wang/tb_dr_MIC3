"""Reviewer 1, Methods 1: random oversampling vs SMOTE vs ADASYN (XGBoost, same
per-class target counts, same held-out test set)."""
import json, sys
import numpy as np, pandas as pd, xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score
from mic_common import *
ECOFF = dict(AMI=1, BDQ=0.25, CFZ=0.25, DLM=0.12, EMB=4, ETH=4, INH=0.1, KAN=4, LEV=1, LZD=1, MXF=1, RIF=0.5)
rows = []
for drug in DRUGS:
    d = load_drug(drug, oversample=True)
    Xr, yr = d["raw"]["X_train"], d["raw"]["train_target"][d["mx"]]
    Xte, yte = d["X_test"], d["y_test"]
    rc = int(np.argmax(d["mic_values"] > ECOFF[drug] * 1.05)); R = (yte >= rc).astype(int)
    tc = d["target_counts"]; kmin = int(yr.value_counts().min())
    samplers = {"none": None,
                "random": RandomOverSampler(sampling_strategy=tc, random_state=42),
                "smote": SMOTE(sampling_strategy=tc, random_state=42, k_neighbors=min(5, kmin - 1)),
                "adasyn": ADASYN(sampling_strategy=tc, random_state=42, n_neighbors=min(5, kmin - 1))}
    for name, s in samplers.items():
        try:
            Xs, ys = (Xr, yr) if s is None else s.fit_resample(Xr, yr)
            m = xgb.XGBClassifier(random_state=42).fit(Xs, ys)
            P = m.predict_proba(Xte); p = P.argmax(1)
            r = dict(drug=drug, sampler=name, n_train=len(ys), dda=dda(yte, p), auc_ovr=roc_auc_score(yte, P, multi_class="ovr"),
                     bal_acc=balanced_accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro", zero_division=0),
                     R_recall=recall_score(R, (p >= rc).astype(int), zero_division=0))
        except Exception as e:
            r = dict(drug=drug, sampler=name, error=str(e)[:150])
        rows.append(r); print(json.dumps(r), flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/03_oversampling_xgb.csv", index=False)
