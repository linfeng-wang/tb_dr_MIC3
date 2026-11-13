"""Additional test-set metrics, TB-Profiler concordance, paired McNemar test
and prediction distributions, all from the per-isolate predictions of 01."""
import numpy as np, pandas as pd
from scipy.stats import binom
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from mic_common import DRUGS, OUT, dda
ECOFF = dict(AMI=1, BDQ=0.25, CFZ=0.25, DLM=0.12, EMB=4, ETH=4, INH=0.1, KAN=4, LEV=1, LZD=1, MXF=1, RIF=0.5)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
met, tb, mc, thr = [], [], [], []
for drug in DRUGS:
    z = np.load(f"{OUT}/pred_{drug}.npz")
    y, tbp, phen_code = z["y"], z["tbp"], z["phen"]
    mv = z["mic_values"]; rc = int(np.argmax(mv > ECOFF[drug] * 1.05))      # first class above the ECOFF
    thr.append(dict(drug=drug, ecoff=ECOFF[drug], first_R_mic=mv[rc], ecoff_class=rc, code_cutoff_class=float(z["cutoff"]),
                    code_cutoff_mic=mv[int(z["cutoff"])], n_R_ecoff=int((y >= rc).sum()), n_R_code=int(phen_code.sum())))
    R = (y >= rc).astype(int)
    for k, lab in [("xgb", "XGBoost"), ("cnn", "CNN"), ("cnn_bin", "Binary-supported CNN")]:
        p = z[k].argmax(1); Rp = (p >= rc).astype(int)
        met.append(dict(drug=drug, model=lab, n=len(y), n_R=int(R.sum()), dda=dda(y, p),
                        bal_acc=balanced_accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro", zero_division=0),
                        R_recall=recall_score(R, Rp, zero_division=0), R_precision=precision_score(R, Rp, zero_division=0),
                        R_f1=f1_score(R, Rp, zero_division=0), S_recall=recall_score(1 - R, 1 - Rp, zero_division=0),
                        dda_R=dda(y[R == 1], p[R == 1]) if R.sum() else np.nan, dda_S=dda(y[R == 0], p[R == 0])))
    # TB-Profiler concordance on the test set and bin-CNN DDA by concordance
    pb, ps = z["cnn_bin"].argmax(1), z["cnn"].argmax(1)
    okb, oks = np.abs(pb - y) <= 1, np.abs(ps - y) <= 1
    conc = tbp == R
    tb.append(dict(drug=drug, n=len(y), n_R=int(R.sum()), tbp_pos=int(tbp.sum()),
                   sens=((tbp == 1) & (R == 1)).sum() / max(R.sum(), 1), spec=((tbp == 0) & (R == 0)).sum() / (R == 0).sum(),
                   n_disc=int((~conc).sum()), FN=int(((tbp == 0) & (R == 1)).sum()), FP=int(((tbp == 1) & (R == 0)).sum()),
                   dda_bin_conc=okb[conc].mean(), dda_bin_disc=okb[~conc].mean() if (~conc).any() else np.nan,
                   dda_bin_FN=okb[(tbp == 0) & (R == 1)].mean() if ((tbp == 0) & (R == 1)).any() else np.nan,
                   dda_bin_FP=okb[(tbp == 1) & (R == 0)].mean() if ((tbp == 1) & (R == 0)).any() else np.nan,
                   dda_std_conc=oks[conc].mean(), dda_std_disc=oks[~conc].mean() if (~conc).any() else np.nan))
    b = int((oks & ~okb).sum()); c = int((~oks & okb).sum())           # std only correct, bin only correct
    pval = min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)) if b + c else 1.0
    mc.append(dict(drug=drug, n=len(y), both=int((oks & okb).sum()), std_only=b, bin_only=c, neither=int((~oks & ~okb).sum()),
                   dda_std=oks.mean(), dda_bin=okb.mean(), p=pval))
    if drug in ("LZD", "RIF", "BDQ", "CFZ", "INH", "LEV"):
        K = len(mv)
        print(f"\n{drug}: classes {list(np.round(mv,3))}; observed {np.bincount(y, minlength=K)}")
        print("  std CNN predicted :", np.bincount(ps, minlength=K))
        print("  bin CNN pred, TBP=0:", np.bincount(pb[tbp == 0], minlength=K), " TBP=1:", np.bincount(pb[tbp == 1], minlength=K))
        print("  observed, TBP=0    :", np.bincount(y[tbp == 0], minlength=K), " TBP=1:", np.bincount(y[tbp == 1], minlength=K))
        print("  xgb predicted      :", np.bincount(z['xgb'].argmax(1), minlength=K))
met, tb, mc, thr = map(pd.DataFrame, (met, tb, mc, thr))
# Holm adjustment
o = np.argsort(mc.p.values); adj = np.empty(len(mc)); run = 0
for r, i in enumerate(o):
    run = max(run, (len(mc) - r) * mc.p.values[i]); adj[i] = min(1, run)
mc["p_holm"] = adj
for n, df in [("02_thresholds", thr), ("02_extra_metrics", met), ("02_tbprofiler", tb), ("02_mcnemar", mc)]:
    df.to_csv(f"{OUT}/{n}.csv", index=False)
    print("\n==", n); print(df.round(3).to_string(index=False))
