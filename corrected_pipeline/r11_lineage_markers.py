"""Lineage distribution and rifampicin phenotype of the carriers of the top-ranked rifampicin features of the corrected
binary-supported CNN (Supplementary Table 5), to separate resistance, compensatory and lineage-marker variants."""
import os
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.dirname(os.path.abspath(__file__))
z = np.load(f"{D}/data/RIF.npz", allow_pickle=True); ids = z["ids"]; X = z["X"]; feats = list(z["features"]); R = z["R"]; y = z["y"]
v = pd.read_csv(f"{REPO}/cache/variants_full.csv", usecols=["sample_id", "sublin"]).drop_duplicates("sample_id").set_index("sample_id")
lin = v["sublin"].astype(str).reindex(ids)
main = lin.str.extract(r"^(lineage\d+|La\d*|[A-Za-z]+)")[0].fillna("unknown"); main = main.where(~lin.str.contains(";"), "mixed").values
a = pd.read_csv(f"{D}/tables/s5_attribution_RIF.csv"); a = a[a.feature != "TB-Profiler resistance call"]
top = list(a.sort_values("IGabs", ascending=False).feature.head(15))
top += [f for f in a.sort_values("SHAP_abs", ascending=False).feature.head(20) if f not in top] + [f for f in ("rpoB c.-61C>T",) if f not in top]
s450 = X[:, feats.index("rpoB p.Ser450Leu")] == 1
tot = pd.Series(main).value_counts()
rows = []
for f in top:
    c = X[:, feats.index(f)] == 1; ml = pd.Series(main[c]).value_counts()
    best = ml.index[0]
    rows.append(dict(feature=f, carriers=int(c.sum()), pct_R=round(100 * R[c].mean(), 1), pct_with_S450L=round(100 * s450[c].mean(), 1),
                     top_lineage=best, pct_carriers_in_top=round(100 * ml.iloc[0] / c.sum(), 1), pct_of_top_lineage_carrying=round(100 * ml.iloc[0] / tot[best], 1),
                     by_lineage="; ".join(f"{k}:{v}" for k, v in ml.items())))
t = pd.DataFrame(rows); t.to_csv(f"{D}/tables/s5_carrier_lineages_RIF.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print("lineage totals:", tot.to_dict(), "| %R by lineage:", {k: round(100 * R[main == k].mean(), 1) for k in tot.index})
print(t.to_string(index=False))
