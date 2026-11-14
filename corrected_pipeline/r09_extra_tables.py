"""Supplementary Table 13 (CNN sensitivity analyses) and Supplementary Table 14 (selected CNN hyperparameters) of the
corrected re-run. Metrics are the mean (SD) over the three training seeds, on the held-out test set."""
import os
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
D = os.path.dirname(os.path.abspath(__file__)); T = f"{D}/tables"
DRUGS = ["AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH", "KAN", "LEV", "LZD", "MXF", "RIF"]
SEEDS = (42, 43, 44)


def metrics(y, R, rc, P):
    p = P.argmax(1)
    return dict(dda=float(np.mean(np.abs(p - y) <= 1)), auc=float(roc_auc_score(y, P, multi_class="ovr", labels=list(range(P.shape[1])))),
                bal_acc=balanced_accuracy_score(y, p), R_recall=recall_score(R, (p >= rc).astype(int), zero_division=0))


rows, hp = [], []
for drug in DRUGS:
    z = np.load(f"{D}/data/{drug}.npz", allow_pickle=True); rc = int(np.argmax(z["mic_levels"] > float(z["ecoff"]) * 1.05))
    x = np.load(f"{D}/results/xgb_{drug}.npz", allow_pickle=True); y, R = x["y"], x["R"]
    for kind, model, main_lab, mode, alt_lab in (
            ("cnn_std", "Standard CNN", "Features in genomic-position order (reported model)", "perm", "Features in randomly permuted order"),
            ("cnn_bin", "Binary-supported CNN", "Phenotypic indicator in training (reported model)", "tbptrain", "TB-Profiler call in training")):
        f = f"{D}/results/{kind}_{drug}.npz"
        if not os.path.exists(f): continue
        m = np.load(f, allow_pickle=True); assert np.array_equal(m["ids"], x["ids"])
        sets = [(main_lab, [m[f"{kind}_s{s}"] for s in SEEDS])]
        g = f"{D}/results/sens_{mode}_{drug}.npz"
        if os.path.exists(g):
            a = np.load(g, allow_pickle=True); assert np.array_equal(a["ids"], x["ids"])
            sets.append((alt_lab, [a[f"s{s}"] for s in SEEDS]))
        for lab, Ps in sets:
            M = pd.DataFrame([metrics(y, R, rc, P) for P in Ps])
            rows.append(dict(drug=drug, model=model, analysis=lab, **{k: M[k].mean() for k in M}, **{k + "_sd": M[k].std(ddof=1) for k in M},
                             dda_min=M.dda.min(), dda_max=M.dda.max()))
        b = json.load(open(f"{D}/results/{kind}_{drug}.json"))["best_params"]
        hp.append(dict(drug=drug, model=model, **b))
pd.DataFrame(rows).to_csv(f"{T}/s13_sensitivity.csv", index=False)
pd.DataFrame(hp).to_csv(f"{T}/s14_hyperparameters.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).round(3).to_string(index=False))
