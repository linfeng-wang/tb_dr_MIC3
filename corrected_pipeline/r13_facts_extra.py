"""Further numbers quoted in the text: Doubling Dilution Accuracy of a constant prediction of the most frequent MIC
category, the most frequent genotype and the category assigned to it in each CNN training run, predictions for
TB-Profiler-positive isolates relative to the ECOFF, and summary comparisons for the oversampling methods."""
import os
import numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__))
DRUGS = ["AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH", "KAN", "LEV", "LZD", "MXF", "RIF"]
SEEDS = (42, 43, 44)
pd.set_option("display.width", 250)
rows = []
for d in DRUGS:
    z = np.load(f"{D}/data/{d}.npz", allow_pickle=True); lv = [float(x) for x in z["mic_levels"]]; ecoff = float(z["ecoff"])
    r = {m: np.load(f"{D}/results/{m}_{d}.npz", allow_pickle=True) for m in ("xgb", "cnn_std", "cnn_bin")}
    te, y, tbp = r["xgb"]["test_idx"], r["xgb"]["y"], r["xgb"]["tbp"]
    X = z["X"][te]; K = len(lv); n_s = sum(l <= ecoff * 1.05 for l in lv)      # classes 0..n_s-1 are at or below the ECOFF
    cnt = np.bincount(y, minlength=K); mode = cnt.argmax()
    # most frequent genotype in the test set
    keys = np.array([hash(x.tobytes()) for x in X]); u, c = np.unique(keys, return_counts=True); top = keys == u[c.argmax()]
    row = dict(drug=d, ecoff=ecoff, ecoff_class=lv[n_s - 1], n=len(y), modal_mic=lv[mode], modal_dda=np.mean(np.abs(y - mode) <= 1),
               top_genotype_pct=100 * top.mean(), top_genotype_n_variants=int(X[top][0].sum()),
               wildtype_pct=100 * (X.sum(1) == 0).mean())
    for m in ("cnn_std", "cnn_bin"):
        row[f"{m}_top_pred"] = [lv[np.bincount(r[m][f"{m}_s{s}"][top].argmax(1), minlength=K).argmax()] for s in SEEDS]
        row[f"{m}_largest_pred_pct"] = [round(100 * np.bincount(r[m][f"{m}_s{s}"].argmax(1), minlength=K).max() / len(y)) for s in SEEDS]
    p = r["cnn_bin"]["cnn_bin_s42"].argmax(1); px = r["xgb"]["xgb_bin"].argmax(1); ps = r["cnn_std"]["cnn_std_s42"].argmax(1)
    row["tbp_pos"] = int(tbp.sum())
    row["tbp_pos_obs_above"] = int((y[tbp == 1] >= n_s).sum())
    row["tbp_pos_pred_above_cnn_bin"] = [int((r["cnn_bin"][f"cnn_bin_s{s}"].argmax(1)[tbp == 1] >= n_s).sum()) for s in SEEDS]
    row["tbp_neg_pred_above_cnn_bin"] = [int((r["cnn_bin"][f"cnn_bin_s{s}"].argmax(1)[tbp == 0] >= n_s).sum()) for s in SEEDS]
    row["tbp_pos_pred_above_xgb_bin"] = int((px[tbp == 1] >= n_s).sum()); row["tbp_neg_pred_above_xgb_bin"] = int((px[tbp == 0] >= n_s).sum())
    row["tbp_pos_pred_above_cnn_std"] = int((ps[tbp == 1] >= n_s).sum())
    row["n_obs_at_ecoff_tbp_pos"] = int(((y == n_s - 1) & (tbp == 1)).sum()); row["n_obs_at_ecoff"] = int((y == n_s - 1).sum())
    rows.append(row)
f = pd.DataFrame(rows)
print("== constant prediction of the most frequent category, most frequent genotype"); print(f.iloc[:, :13].round(3).to_string(index=False))
print("\n== TB-Profiler-positive / negative test isolates predicted above the ECOFF"); print(f.iloc[:, [0, 1, 2] + list(range(13, f.shape[1]))].to_string(index=False))
t2 = pd.read_csv(f"{D}/tables/table2.csv").merge(f[["drug", "modal_dda"]])
for m in ("xgb", "xgb_bin", "cnn_std", "cnn_bin"):
    dlt = t2[f"{m}_dda"] - t2.modal_dda
    print(f"{m} DDA minus constant prediction: min {dlt.min():+.3f} max {dlt.max():+.3f} | higher in", list(t2.drug[dlt > 0.005]), "| lower in", list(t2.drug[dlt < -0.005]))
print("constant-prediction DDA: mean %.2f range %.2f-%.2f" % (f.modal_dda.mean(), f.modal_dda.min(), f.modal_dda.max()))
print("\n== Table 2 comparisons")
print("XGB >= standard CNN (2 dp):", list(t2.drug[t2.xgb_dda.round(2) >= t2.cnn_std_dda.round(2)]))
print("bin CNN > std CNN (2 dp):", list(t2.drug[t2.cnn_bin_dda.round(2) > t2.cnn_std_dda.round(2)]), "| equal:", list(t2.drug[t2.cnn_bin_dda.round(2) == t2.cnn_std_dda.round(2)]))
print("bin CNN - XGB:", {d: round(v, 3) for d, v in zip(t2.drug, t2.cnn_bin_dda - t2.xgb_dda)})
print("XGB bin - XGB:", {d: round(v, 3) for d, v in zip(t2.drug, t2.xgb_bin_dda - t2.xgb_dda)})
print("bin SD <= 0.03:", list(t2.drug[t2.cnn_bin_dda_sd <= 0.03]), "| std SD <= 0.03:", list(t2.drug[t2.cnn_std_dda_sd <= 0.03]))
print("AUC range across the four models per drug:", {d: (round(min(v), 2), round(max(v), 2)) for d, v in zip(t2.drug, t2[["cnn_bin_auc", "cnn_std_auc", "xgb_auc", "xgb_bin_auc"]].values)})
print("max |AUC std CNN - XGB|:", (t2.cnn_std_auc - t2.xgb_auc).abs().round(3).tolist())
print("AUC bin CNN - std CNN:", {d: round(v, 3) for d, v in zip(t2.drug, t2.cnn_bin_auc - t2.cnn_std_auc)})
s12 = pd.read_csv(f"{D}/tables/s12_oversampling_methods.csv"); w = s12.pivot(index="drug", columns="sampler")
print("\n== oversampling (XGBoost)")
for k in ("dda", "auc_ovr", "bal_acc", "macro_f1", "R_recall"):
    a = w[k]["random"] - w[k]["none"]
    print(f"{k}: random > none in {int((a > 0).sum())}, < in {int((a < 0).sum())}, = in {int((a == 0).sum())} drugs | " + ", ".join(f"{d} {v:+.3f}" for d, v in a.items()))
print("R_recall none == 0:", list(w["R_recall"]["none"][w["R_recall"]["none"] == 0].index))
s10 = pd.read_csv(f"{D}/tables/s10_extra_metrics.csv"); print("\n== S10 ranges by model"); print(s10.groupby("model")[["bal_acc", "macro_f1", "R_recall", "R_precision"]].agg(["min", "max"]).round(2).to_string())
s11 = pd.read_csv(f"{D}/tables/s11_tbprofiler.csv"); print(s11.round(2).to_string(index=False))
