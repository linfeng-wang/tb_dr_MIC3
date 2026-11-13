import os
"""Identify which genes contribute the columns of the feature matrices used by
the saved models (generated_data18122024; AMI from generated_data30072025), by
matching per-column carrier counts with per-variant carrier counts."""
import numpy as np, pandas as pd, collections
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IM = f"{REPO}/individual_models"
v = pd.read_csv(f"{REPO}/cache/variants_full.csv", usecols=["sample_id", "genome_pos", "gene", "change", "freq", "type"])
cr = pd.read_csv(f"{REPO}/CRyPTIC_reuse_table_20231208.csv")
v = v[(v.gene != "PPE35") & (v.type != "synonymous_variant") & (v.type != "non_coding_transcript_exon_variant")]
METHODS = dict(INH="inhA katG kasA ahpC oxyR' fabG1", ETH="ethA inhA", EMB="embA embB embC", RIF="rpoA rpoB rpoC",
               LEV="gyrA gyrB", MXF="gyrA gyrB", KAN="rrs eis", AMI="rrs eis", BDQ="Rv0678 atpE pepQ mmpR5", CFZ="Rv0678 pepQ mmpR5",
               DLM="fgd1 fbiA fbiB fbiC fbiD ddn", LZD="rrl rplC")
for d in METHODS:
    folder = "generated_data30072025" if d == "AMI" else "generated_data18122024"
    X = np.load(f"{IM}/{folder}/all_sample_snps_cryptic_{d}.npy")
    cols = collections.Counter(int(c) for c in X.sum(0) if c > 0)
    ids = set(cr.loc[cr[f"{d}_MIC"].notna(), "ENA_RUN"])
    vv = v[v.sample_id.isin(ids)]
    n_all = vv.groupby("gene").apply(lambda g: (g.gene + g.change).nunique())          # any allele frequency
    vv8 = vv[vv.freq >= 0.8]
    cnt = vv8.groupby(["gene", "change"]).sample_id.nunique()
    res = []
    for g, s in cnt.groupby(level=0):
        c = collections.Counter(int(x) for x in s.values)
        match = sum(min(c[k], cols.get(k, 0)) for k in c)
        big = [int(x) for x in s.values if x >= 5]
        bigm = sum(1 for x in big if cols.get(x, 0) > 0)
        if len(big) >= 1: res.append((g, len(s), int(n_all.get(g, 0)), match, len(big), bigm))
    res = pd.DataFrame(res, columns=["gene", "n_var_af0.8", "n_var_any_af", "counts_matched", "n_var_ge5", "ge5_matched"])
    res["frac_ge5"] = (res.ge5_matched / res.n_var_ge5).round(2)
    hit = res[(res.frac_ge5 >= 0.95) & (res.n_var_ge5 >= 2)].sort_values("n_var_af0.8", ascending=False)
    print(f"\n{d}: matrix {X.shape}, non-zero columns {sum(cols.values())}; Methods genes: {METHODS[d]}")
    print("  genes whose variant carrier counts are all present in the matrix:")
    print("   ", "; ".join(f"{r.gene} (n={r._2}, any AF={r.n_var_any_af})" for r in hit.head(12).itertuples()))
    print("   sum n_var(AF>=0.8) of these:", int(hit["n_var_af0.8"].sum()), "| sum any AF:", int(hit["n_var_any_af"].sum()))
    mg = res[res.gene.isin(METHODS[d].split())]
    print("   Methods genes status:", "; ".join(f"{r.gene}: {r.ge5_matched}/{r.n_var_ge5} common variants matched" for r in mg.itertuples()))
