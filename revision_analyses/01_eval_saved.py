"""Re-evaluate the saved manuscript models on the held-out test set in a fixed
isolate order and store per-isolate predictions and class probabilities."""
import pickle, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from mic_common import *

rows = []
for drug in DRUGS:
    d = load_drug(drug, oversample=False)
    X, y, tbp = d["X_test"], d["y_test"], d["tbp_test"]
    res = {"y": y, "tbp": tbp, "phen": d["phen_test"], "cutoff": d["cutoff"], "mic_values": d["mic_values"]}
    for name, binary in [("cnn", False), ("cnn_bin", True)]:
        m, a = load_saved_cnn(drug, binary, X.shape[1])
        p = predict(m, X, tbp if binary else np.zeros(len(X)))
        res[name] = p
    for name, suff, binary in [("xgb", "", False), ("xgb_bin", "_bin", True)]:
        mdl = pickle.load(open(f"{IM}/saved_models/xgb_MIC_model_{drug}{suff}.pkl", "rb"))
        Xx = np.column_stack((X, tbp)) if binary else X
        res[name] = mdl.predict_proba(Xx)
    np.savez(f"{OUT}/pred_{drug}.npz", **res)
    r = {"drug": drug, "n_test": len(y)}
    for k in ["cnn_bin", "cnn", "xgb", "xgb_bin"]:
        pred = res[k].argmax(1)
        r[f"dda_{k}"] = round(dda(y, pred), 3)
        try:
            r[f"auc_{k}"] = round(roc_auc_score(y, res[k], multi_class="ovr"), 3)
        except Exception as e:
            r[f"auc_{k}"] = np.nan
    rows.append(r); print(r, flush=True)
import pandas as pd
pd.DataFrame(rows).to_csv(f"{OUT}/01_reproduced_table2.csv", index=False)
