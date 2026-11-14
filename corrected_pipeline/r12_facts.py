"""Print the numbers quoted in the manuscript text from the tables of the corrected re-run (run after r03, r04, r06, r09)."""
import os
import json, numpy as np, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)); T = f"{D}/tables"
DRUGS = ["AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH", "KAN", "LEV", "LZD", "MXF", "RIF"]
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
t2 = pd.read_csv(f"{T}/table2.csv")
print("== Table 2 (CNN mean of 3 seeds, SD)")
print(t2[["drug", "cnn_bin_dda", "cnn_bin_dda_sd", "cnn_std_dda", "cnn_std_dda_sd", "xgb_dda", "xgb_bin_dda", "cnn_bin_auc", "cnn_bin_auc_sd",
          "cnn_std_auc", "cnn_std_auc_sd", "xgb_auc", "xgb_bin_auc"]].round(3).to_string(index=False))
for k in ("xgb_dda", "cnn_std_dda", "cnn_bin_dda", "xgb_bin_dda", "xgb_auc", "cnn_std_auc", "cnn_bin_auc"):
    v = t2[k]; print(f"{k}: mean {v.mean():.2f} SD {v.std(ddof=1):.2f} range {v.min():.2f}-{v.max():.2f}")
print("per-seed DDA:"); print(t2[["drug", "cnn_bin_dda_seeds", "cnn_std_dda_seeds"]].to_string(index=False))
print("\n== McNemar (seed 42; sig_up/sig_down = number of seeds with Holm P<0.05)")
print(pd.read_csv(f"{T}/s3_mcnemar_seed42.csv").round(4).to_string(index=False))
print("\n== class-wise AUC")
cw = pd.read_csv(f"{T}/s4_classwise_auc.csv"); print(cw.round(2).to_string(index=False))
print("\n== extra metrics (S10)")
s10 = pd.read_csv(f"{T}/s10_extra_metrics.csv"); print(s10.round(2).to_string(index=False))
print("\n== TB-Profiler (S11)"); print(pd.read_csv(f"{T}/s11_tbprofiler.csv").round(2).to_string(index=False))
print("\n== oversampling methods (S12)")
s12 = pd.read_csv(f"{T}/s12_oversampling_methods.csv"); print(s12.round(3).to_string(index=False))
p = s12.pivot(index="drug", columns="sampler")
for m in ("dda", "auc_ovr", "bal_acc", "R_recall"):
    for a, b in (("smote", "random"), ("adasyn", "random"), ("random", "none")):
        d = p[m][a] - p[m][b]; print(f"{m}: {a} - {b}: min {d.min():.3f} max {d.max():.3f} median {d.median():.3f}")
print("\n== sensitivity (S13)"); s13 = pd.read_csv(f"{T}/s13_sensitivity.csv"); print(s13.round(3).to_string(index=False))
for model, lab in (("Standard CNN", "perm - position"), ("Binary-supported CNN", "TB-Profiler train - phenotype train")):
    x = s13[s13.model == model]; a = x[x.analysis.str.contains("reported")].set_index("drug"); b = x[~x.analysis.str.contains("reported")].set_index("drug")
    for m in ("dda", "auc", "R_recall"):
        d = (b[m] - a[m]).dropna(); print(f"{lab} {m}: " + ", ".join(f"{k} {v:+.3f}" for k, v in d.items()))
print("\n== predicted class distributions (seed 42) and DDA in resistant / susceptible isolates")
for drug in DRUGS:
    x = np.load(f"{D}/results/xgb_{drug}.npz", allow_pickle=True); y = x["y"]; R = x["R"]; tbp = x["tbp"]
    z = np.load(f"{D}/data/{drug}.npz", allow_pickle=True); lv = z["mic_levels"]; K = len(lv)
    print(drug, "levels", list(lv), "| observed", np.bincount(y, minlength=K).tolist(), "| R", int(R.sum()), "tbp+", int(tbp.sum()))
    for k in ("cnn_bin", "cnn_std"):
        c = np.load(f"{D}/results/{k}_{drug}.npz", allow_pickle=True)
        for s in (42, 43, 44):
            p_ = c[f"{k}_s{s}"].argmax(1); ok = np.abs(p_ - y) <= 1
            print(f"   {k} s{s}: pred {np.bincount(p_, minlength=K).tolist()} DDA_R {ok[R == 1].mean():.2f} DDA_S {ok[R == 0].mean():.2f}"
                  + (f" | tbp+: pred {np.bincount(p_[tbp == 1], minlength=K).tolist()} obs {np.bincount(y[tbp == 1], minlength=K).tolist()}" if s == 42 else ""))
    for k in ("xgb", "xgb_bin"):
        p_ = x[k].argmax(1); ok = np.abs(p_ - y) <= 1
        print(f"   {k}: pred {np.bincount(p_, minlength=K).tolist()} DDA_R {ok[R == 1].mean():.2f} DDA_S {ok[R == 0].mean():.2f}")
print("\n== RIF attribution")
a = pd.read_csv(f"{T}/s5_attribution_RIF.csv")
print(a.sort_values("IGabs", ascending=False).head(20).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(a.sort_values("SHAP_abs", ascending=False).head(21).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
print(a[a.feature.isin(["rpoC c.-936T>G", "rpoB c.-61C>T", "rpoC c.-339T>C"])].to_string(index=False))
print(pd.read_csv(f"{T}/s5_carrier_lineages_RIF.csv").to_string(index=False))
