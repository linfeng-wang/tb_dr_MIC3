"""Compute all model-based tables of the corrected re-run from the saved test-set predictions.
CNN metrics are the mean (SD) over three training seeds; per-isolate analyses use seed 42 unless stated."""
import os
import json, glob
import numpy as np, pandas as pd
from scipy.stats import binom
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score
D = os.path.dirname(os.path.abspath(__file__)); T = f"{D}/tables"
DRUGS = ["AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH", "KAN", "LEV", "LZD", "MXF", "RIF"]
SEEDS = (42, 43, 44)
dda = lambda y, p: float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1))


def auc(y, P):
    return float(roc_auc_score(y, P, multi_class="ovr", labels=list(range(P.shape[1]))))


def load(drug):
    x = np.load(f"{D}/results/xgb_{drug}.npz", allow_pickle=True)
    out = dict(y=x["y"], tbp=x["tbp"], R=x["R"], ids=x["ids"], xgb=x["xgb"], xgb_bin=x["xgb_bin"])
    for k in ("cnn_std", "cnn_bin"):
        z = np.load(f"{D}/results/{k}_{drug}.npz", allow_pickle=True)
        assert np.array_equal(z["ids"], x["ids"])
        for s in SEEDS: out[f"{k}_s{s}"] = z[f"{k}_s{s}"]
    z = np.load(f"{D}/data/{drug}.npz", allow_pickle=True); out["levels"] = z["mic_levels"]
    out["rc"] = int(np.argmax(z["mic_levels"] > float(z["ecoff"]) * 1.05))
    return out


def extra(y, p, R, rc):
    Rp = (p >= rc).astype(int)
    return dict(dda=dda(y, p), bal_acc=balanced_accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro", zero_division=0),
                R_recall=recall_score(R, Rp, zero_division=0), R_precision=precision_score(R, Rp, zero_division=0), R_f1=f1_score(R, Rp, zero_division=0),
                dda_R=dda(y[R == 1], p[R == 1]) if R.any() else np.nan, dda_S=dda(y[R == 0], p[R == 0]))


t2, s3, s4, s10, s11, cw_rows = [], [], [], [], [], []
for drug in DRUGS:
    o = load(drug); y, R, tbp, rc = o["y"], o["R"], o["tbp"], o["rc"]
    row = dict(drug=drug, n_test=len(y))
    for k in ("cnn_bin", "cnn_std"):
        dd = [dda(y, o[f"{k}_s{s}"].argmax(1)) for s in SEEDS]; aa = [auc(y, o[f"{k}_s{s}"]) for s in SEEDS]
        row.update({f"{k}_dda": np.mean(dd), f"{k}_dda_sd": np.std(dd, ddof=1), f"{k}_auc": np.mean(aa), f"{k}_auc_sd": np.std(aa, ddof=1),
                    f"{k}_dda_seeds": dd, f"{k}_auc_seeds": aa})
    for k in ("xgb", "xgb_bin"):
        row[f"{k}_dda"] = dda(y, o[k].argmax(1)); row[f"{k}_auc"] = auc(y, o[k])
    t2.append(row)
    # McNemar (binary-supported vs standard CNN), per seed
    for s in SEEDS:
        okb = np.abs(o[f"cnn_bin_s{s}"].argmax(1) - y) <= 1; oks = np.abs(o[f"cnn_std_s{s}"].argmax(1) - y) <= 1
        b = int((oks & ~okb).sum()); c = int((~oks & okb).sum())
        p = min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)) if b + c else 1.0
        s3.append(dict(drug=drug, seed=s, n=len(y), dda_std=oks.mean(), dda_bin=okb.mean(), both=int((oks & okb).sum()), bin_only=c, std_only=b,
                       neither=int((~oks & ~okb).sum()), p=p))
    # class-wise AUC (mean over seeds)
    for c in range(len(o["levels"])):
        r = dict(drug=drug, mic=o["levels"][c], n_test=int((y == c).sum()))
        for k in ("cnn_bin", "cnn_std"):
            r[k] = np.mean([roc_auc_score((y == c).astype(int), o[f"{k}_s{s}"][:, c]) for s in SEEDS]) if 0 < (y == c).sum() < len(y) else np.nan
        for k in ("xgb", "xgb_bin"):
            r[k] = roc_auc_score((y == c).astype(int), o[k][:, c]) if 0 < (y == c).sum() < len(y) else np.nan
        cw_rows.append(r)
    # extra metrics
    for k, lab in (("xgb", "XGBoost"), ("xgb_bin", "Binary-supported XGBoost"), ("cnn_std", "CNN"), ("cnn_bin", "Binary-supported CNN")):
        if k.startswith("cnn"):
            e = pd.DataFrame([extra(y, o[f"{k}_s{s}"].argmax(1), R, rc) for s in SEEDS]).mean().to_dict()
        else:
            e = extra(y, o[k].argmax(1), R, rc)
        s10.append(dict(drug=drug, model=lab, n=len(y), n_R=int(R.sum()), **e))
    # TB-Profiler concordance
    conc = tbp == R
    r = dict(drug=drug, n=len(y), n_R=int(R.sum()), tbp_pos=int(tbp.sum()), sens=((tbp == 1) & (R == 1)).sum() / max(R.sum(), 1),
             spec=((tbp == 0) & (R == 0)).sum() / (R == 0).sum(), n_disc=int((~conc).sum()))
    for k in ("cnn_bin", "cnn_std", "xgb_bin"):
        ok = [np.abs((o[f"{k}_s{s}"] if k.startswith("cnn") else o[k]).argmax(1) - y) <= 1 for s in (SEEDS if k.startswith("cnn") else (0,))]
        r[f"{k}_conc"] = np.mean([v[conc].mean() for v in ok]); r[f"{k}_disc"] = np.mean([v[~conc].mean() for v in ok]) if (~conc).any() else np.nan
    s11.append(r)
t2 = pd.DataFrame(t2); s3 = pd.DataFrame(s3); cw = pd.DataFrame(cw_rows); s10 = pd.DataFrame(s10); s11 = pd.DataFrame(s11)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for r_, i in enumerate(o):
        run = max(run, (len(p) - r_) * p[i]); adj[i] = min(1, run)
    return adj


# Holm adjustment over the 12 drugs, separately for each training seed
s3["p_holm"] = np.nan
for s in SEEDS:
    s3.loc[s3.seed == s, "p_holm"] = holm(s3.loc[s3.seed == s, "p"].values)
m = s3[s3.seed == 42].copy()
agree = s3.assign(sig_up=lambda d: (d.p_holm < 0.05) & (d.bin_only > d.std_only), sig_down=lambda d: (d.p_holm < 0.05) & (d.bin_only < d.std_only)) \
    .groupby("drug")[["sig_up", "sig_down"]].sum()
m = m.merge(agree, left_on="drug", right_index=True)
for n, df in (("table2", t2), ("s3_mcnemar_seed42", m), ("s3_mcnemar_all_seeds", s3), ("s4_classwise_auc", cw), ("s10_extra_metrics", s10), ("s11_tbprofiler", s11)):
    df.to_csv(f"{T}/{n}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(t2[[c for c in t2.columns if not c.endswith("seeds")]].round(3).to_string(index=False))
print(m.round(4).to_string(index=False))
