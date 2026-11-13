import os
"""Sensitivity check: XGBoost MIC prediction for BDQ, CFZ and INH with the gene
sets used by the manuscript models vs the gene sets named in the Methods.
Matrices are rebuilt from variants_full.csv with the same filters; MIC classes
and the per-class oversampling targets follow the notebooks."""
import numpy as np, pandas as pd, re, xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
v = pd.read_csv(f"{REPO}/cache/variants_full.csv", usecols=["sample_id", "gene", "change", "freq", "type"])
v = v[(v.type != "synonymous_variant") & (v.type != "non_coding_transcript_exon_variant") & (v.freq >= 0.8)]
cr = pd.read_csv(f"{REPO}/CRyPTIC_reuse_table_20231208.csv")
ids_wgs = set(pd.read_csv(f"{REPO}/cache/variants_full.csv", usecols=["sample_id"]).sample_id)
def mic_num(x):
    if pd.isna(x): return np.nan
    return float(re.search(r"[\d.]+", str(x)).group()) * (2 if ">" in str(x) else 1)
CFG = dict(
    BDQ=dict(used=["atpE", "pepQ"], methods=["mmpR5", "atpE", "pepQ"], lo=0.015625, hi=2.0, ecoff=0.25),
    CFZ=dict(used=["pepQ"], methods=["mmpR5", "pepQ"], lo=0.0625, hi=4.0, ecoff=0.25),
    INH=dict(used=["katG", "inhA", "kasA"], methods=["katG", "inhA", "kasA", "fabG1", "ahpC"], lo=0.025, hi=3.2, ecoff=0.1))
rows = []
for d, c in CFG.items():
    df = cr[cr.ENA_RUN.isin(ids_wgs) & cr[f"{d}_MIC"].notna()][["ENA_RUN", f"{d}_MIC"]].copy()
    m = df[f"{d}_MIC"].map(mic_num).clip(c["lo"], c["hi"])
    df["y"] = np.round(np.log2(m / c["lo"])).astype(int)
    rc = int(np.floor(np.log2(c["ecoff"] / c["lo"]) + 1e-6)) + 1
    tr_ids, te_ids = train_test_split(df.ENA_RUN.values, test_size=0.1, stratify=df.y.values, random_state=42)
    ymap = df.set_index("ENA_RUN").y
    for label, genes in [("as used in the manuscript models", c["used"]), ("as named in the Methods", c["methods"])]:
        vv = v[v.gene.isin(genes) & v.sample_id.isin(set(df.ENA_RUN))]
        M = pd.crosstab(vv.sample_id, vv.gene + " " + vv.change).clip(upper=1).reindex(df.ENA_RUN.values, fill_value=0)
        Xtr, Xte = M.loc[tr_ids].values, M.loc[te_ids].values; ytr, yte = ymap.loc[tr_ids].values, ymap.loc[te_ids].values
        cnt = pd.Series(ytr).value_counts(); top = cnt.max()
        tc = {k: int(max(n, min(top, n + 2000))) for k, n in cnt.items()}       # fixed number of duplicates per minority class
        Xs, ys = RandomOverSampler(sampling_strategy=tc, random_state=42).fit_resample(Xtr, ytr)
        K = int(df.y.max()) + 1
        mdl = xgb.XGBClassifier(random_state=42).fit(Xs, ys)
        P = np.zeros((len(yte), K)); P[:, mdl.classes_] = mdl.predict_proba(Xte); p = P.argmax(1); R = (yte >= rc).astype(int)
        cw = {}
        for k in np.unique(yte):
            cw[int(k)] = round(float(roc_auc_score((yte == k).astype(int), P[:, k])), 2)
        r = dict(drug=d, genes=", ".join(genes), feature_set=label, n_features=M.shape[1], n=len(df), n_test=len(yte), n_R_test=int(R.sum()),
                 dda=round(float(np.mean(np.abs(p - yte) <= 1)), 3), auc_ovr=round(float(roc_auc_score(yte, P[:, np.unique(yte)] / P[:, np.unique(yte)].sum(1, keepdims=True), multi_class="ovr")), 3) if len(np.unique(yte)) == K else round(float(np.mean(list(cw.values()))), 3),
                 bal_acc=round(float(balanced_accuracy_score(yte, p)), 3), R_recall=round(float(recall_score(R, (p >= rc).astype(int), zero_division=0)), 3),
                 R_precision=round(float(((p >= rc) & (R == 1)).sum() / max((p >= rc).sum(), 1)), 3), classwise_auc=cw)
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/07_geneset_sensitivity.csv", index=False)
