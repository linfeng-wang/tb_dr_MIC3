"""Rebuild the per-drug inputs for the corrected re-run.

Changes from the original data_gen_new.ipynb:
  * gene sets exactly as listed in the Methods (incl. mmpR5/Rv0678, fabG1, ahpC; AMI = rrs + eis);
  * rRNA variants (rrs, rrl) are kept; only synonymous variants are removed;
  * features ordered by genomic position (deterministic; the original used Python set order);
  * only variants present at allele frequency >= 0.8 in at least one isolate are kept (no all-zero columns);
  * isolate IDs are stored with every row.
MIC classes and TB-Profiler calls are rebuilt from the CRyPTIC tables and checked against the original arrays.
"""
import os
import numpy as np, pandas as pd, re, json, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); IM = f"{REPO}/individual_models"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GENES = dict(INH=["inhA", "katG", "kasA", "ahpC", "fabG1"], ETH=["ethA", "inhA"], EMB=["embA", "embB", "embC"],
             RIF=["rpoA", "rpoB", "rpoC"], LEV=["gyrA", "gyrB"], MXF=["gyrA", "gyrB"], KAN=["rrs", "eis"], AMI=["rrs", "eis"],
             BDQ=["mmpR5", "atpE", "pepQ"], CFZ=["mmpR5", "pepQ"], DLM=["fgd1", "fbiA", "fbiB", "fbiC", "fbiD", "ddn"], LZD=["rrl", "rplC"])
TBP = dict(INH="isoniazid", ETH="ethionamide", EMB="ethambutol", RIF="rifampicin", LEV="levofloxacin", MXF="moxifloxacin",
           KAN="kanamycin", AMI="amikacin", BDQ="bedaquiline", CFZ="clofazimine", DLM="delamanid", LZD="linezolid")
ECOFF = dict(AMI=1, BDQ=0.25, CFZ=0.25, DLM=0.12, EMB=4, ETH=4, INH=0.1, KAN=4, LEV=1, LZD=1, MXF=1, RIF=0.5)

v = pd.read_csv(f"{REPO}/cache/variants_full.csv", usecols=["sample_id", "genome_pos", "gene", "change", "freq", "type"])
v = v[v.type != "synonymous_variant"]
cr = pd.read_csv(f"{REPO}/CRyPTIC_reuse_table_20231208.csv")
tb = pd.read_csv(f"{REPO}/CRyPTIC_reuse_table_20231208_cleaned+tbprofiler.csv", usecols=["ENA_SAMPLE"] + list(TBP.values()))
df = cr.merge(tb, on="ENA_SAMPLE", how="inner")
wgs = set(v.sample_id)
report = {}
for drug, genes in GENES.items():
    old_y = np.load(f"{IM}/generated_data18122024/all_sample_drs_cryptic_{drug}.npy")
    old_t = np.load(f"{IM}/generated_data18122024/all_sample_drs_cryptic_{drug}-tbp.npy")
    old_y = pd.Series(old_y).replace({0.03125: 0.0625} if drug == "LZD" else {}).replace({0.015: 0.015625} if drug == "BDQ" else {}) \
        .replace({0.125: 0.25} if drug == "ETH" else {}).replace({6.4: 3.2} if drug == "INH" else {}).values
    levels = np.sort(np.unique(old_y))                                   # MIC classes used in the paper
    d = df[df.ENA_RUN.isin(wgs) & df[f"{drug}_MIC"].notna()].copy()
    s = d[f"{drug}_MIC"].astype(str)
    val = s.str.extract(r"([\d.]+)")[0].astype(float)
    val = np.where(s.str.startswith(">"), val * 2, val)
    li = np.log2(levels)
    idx = np.clip(np.round((np.log2(val.astype(float)) - li[0]) / 1.0), 0, len(levels) - 1).astype(int)
    # nearest class on the log2 scale (handles 0.12 -> 0.125, 0.06 -> 0.0625, <= and > boundaries)
    idx = np.array([int(np.argmin(np.abs(li - np.log2(x)))) if np.log2(x) <= li[-1] else len(li) - 1 for x in val])
    idx = np.where(np.log2(val) < li[0], 0, idx)
    d["y"] = idx; d["tbp"] = d[TBP[drug]].astype(int)
    d = d.sort_values("ENA_RUN").reset_index(drop=True)
    # features
    vv = v[v.gene.isin(genes) & v.sample_id.isin(set(d.ENA_RUN)) & (v.freq >= 0.8)].copy()
    vv["feat"] = vv.gene + " " + vv.change
    order = vv.drop_duplicates("feat").sort_values(["genome_pos", "feat"]).feat.values
    M = pd.crosstab(vv.sample_id, vv.feat).clip(upper=1).reindex(index=d.ENA_RUN.values, columns=order, fill_value=0)
    X = M.values.astype(np.int8)
    R = (levels[d.y.values] > ECOFF[drug] * 1.05).astype(int)
    np.savez_compressed(f"{OUT}/{drug}.npz", X=X, y=d.y.values, tbp=d.tbp.values, ids=d.ENA_RUN.values.astype(str),
                        features=order.astype(str), mic_levels=levels, ecoff=ECOFF[drug], R=R)
    same_counts = np.array_equal(np.bincount(d.y.values, minlength=len(levels)),
                                 np.bincount(np.searchsorted(levels, old_y), minlength=len(levels)))
    ct_new = pd.crosstab(d.y.values, d.tbp.values).values; ct_old = pd.crosstab(np.searchsorted(levels, old_y), old_t.astype(int)).values
    report[drug] = dict(n=len(d), n_old=len(old_y), class_counts_equal=bool(same_counts), class_x_tbp_equal=bool(np.array_equal(ct_new, ct_old)),
                        n_features=X.shape[1], genes={g: int((vv.drop_duplicates("feat").gene == g).sum()) for g in genes},
                        isolates_without_variant=int((X.sum(1) == 0).sum()), n_R=int(R.sum()))
    print(drug, report[drug], flush=True)
json.dump(report, open(f"{OUT}/build_report.json", "w"), indent=1)
