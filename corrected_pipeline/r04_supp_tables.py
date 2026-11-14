"""Supplementary Table 2 (oversampling counts) and Supplementary Table 6 (features) for the corrected re-run."""
import os
import json, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
DRUGS = ["AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH", "KAN", "LEV", "LZD", "MXF", "RIF"]
rows, feats = [], []
rep = json.load(open(f"{D}/data/build_report.json"))
for drug in DRUGS:
    info = json.load(open(f"{D}/results/xgb_{drug}.json")); z = np.load(f"{D}/data/{drug}.npz", allow_pickle=True)
    ref = max(info["counts"], key=lambda k: info["counts"][k]); refn = info["counts"][ref]
    for c in sorted(info["counts"], key=int):
        b, a = info["counts"][c], info["target"][c]
        rows.append(dict(drug=drug, mic=z["mic_levels"][int(c)], ref=(c == ref), before=b, after=a, before_pct=100 * b / refn, after_pct=100 * a / refn))
    feats.append(dict(drug=drug, genes=", ".join(rep[drug]["genes"]), n_features=rep[drug]["n_features"],
                      n_train=info["n_train_orig"], n_train_os=info["n_train_os"], n_val=info["n_val"], n_test=info["n_test"]))
pd.DataFrame(rows).to_csv(f"{D}/tables/s2_oversampling.csv", index=False)
pd.DataFrame(feats).to_csv(f"{D}/tables/s6_features.csv", index=False)
print(pd.DataFrame(feats).to_string(index=False))
r = pd.DataFrame(rows); print("after % of reference: min", round(r.after_pct.min(), 1), "max", round(r.after_pct.max(), 1))
