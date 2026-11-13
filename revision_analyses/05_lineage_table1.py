"""(a) Lineage distribution of the rifampicin promoter/3' rpoB haplotype
(rpoC c.-936T>G, rpoB c.-61C>T, rpoC c.-339T>C); (b) lineage composition of the
dataset; (c) check of resistant counts in Table 1 against the ECOFF definition."""
import numpy as np, pandas as pd, re, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
vf = f"{REPO}/variants_full.csv" if os.path.exists(f"{REPO}/variants_full.csv") else f"{REPO}/cache/variants_full.csv"
v = pd.read_csv(vf)
cr = pd.read_csv(f"{REPO}/CRyPTIC_reuse_table_20231208.csv")
v = v[v.sample_id.isin(set(cr.ENA_RUN))]
lin = v.drop_duplicates("sample_id").set_index("sample_id")["sublin"].astype(str)
main = lin.str.extract(r"^(lineage\d+|La\d*|[A-Za-z]+)")[0].fillna("unknown")
mixed = lin.str.contains(";")
main = main.where(~mixed, "mixed")
print("isolates with variants and lineage call:", len(lin))
print("main lineage composition:\n", main.value_counts().to_string())
rif = v[v.gene.isin(["rpoA", "rpoB", "rpoC"]) & (v.freq >= 0.8)].copy()
rif["snp"] = rif.gene + " " + rif.change
targets = ["rpoC c.-936T>G", "rpoB c.-61C>T", "rpoC c.-339T>C"]
car = {t: set(rif.loc[rif.snp == t, "sample_id"]) for t in targets}
for t in targets:
    s = pd.Series(list(car[t]))
    print(f"\n{t}: n carriers = {len(s)}")
    print("  main lineage:", main.reindex(s).value_counts().to_dict())
    print("  top sublineages:", lin.reindex(s).value_counts().head(6).to_dict())
allthree = car[targets[0]] & car[targets[1]] & car[targets[2]]
two = car[targets[0]] & car[targets[1]]
print("\ncarriers of c.-936 and c.-61:", len(two), "| all three:", len(allthree))
for nm, grp in [("c.-936 & c.-61", two), ("all three", allthree)]:
    s = pd.Series(list(grp)); ml = main.reindex(s).value_counts()
    print(nm, "main lineage:", ml.to_dict(), "| sublineage:", lin.reindex(s).value_counts().head(5).to_dict())
# share of each main lineage / sublineage that carries the haplotype
ml_all = main.value_counts(); ml_car = main.reindex(pd.Series(list(two))).value_counts()
print("\nproportion of each main lineage carrying c.-936 & c.-61:\n", (ml_car / ml_all).dropna().round(3).to_string())
sl_all = lin.value_counts(); sl_car = lin.reindex(pd.Series(list(two))).value_counts()
x = pd.DataFrame({"carriers": sl_car, "total": sl_all}).dropna().sort_values("carriers", ascending=False).head(8); x["prop"] = (x.carriers / x.total).round(3)
print("\nsublineages:\n", x.to_string())
# (c) Table 1 check
ECOFF = dict(AMI=1, BDQ=0.25, CFZ=0.25, DLM=0.12, EMB=4, ETH=4, INH=0.1, KAN=4, LEV=1, LZD=1, MXF=1, RIF=0.5)
def num(x):
    if pd.isna(x): return np.nan, None
    s = str(x); m = re.search(r"[\d.]+", s); return float(m.group()), ("<=" if "<=" in s else (">" if ">" in s else ""))
rows = []
for d, e in ECOFF.items():
    col = cr[f"{d}_MIC"]; vals = col.map(lambda x: num(x)[0]); sign = col.map(lambda x: num(x)[1])
    n = int(vals.notna().sum())
    above = (vals > e * 1.05) | ((sign == ">") & (vals >= e * 0.95))       # MIC > ECOFF
    at_or_above = above | ((vals >= e * 0.9) & (sign != "<="))               # MIC >= ECOFF
    bp = cr[f"{d}_BINARY_PHENOTYPE"].value_counts().to_dict()
    rows.append(dict(drug=d, n_MIC=n, R_gt_ecoff=int(above.sum()), pct_gt=round(100 * above.sum() / n, 1),
                     R_ge_ecoff=int(at_or_above.sum()), pct_ge=round(100 * at_or_above.sum() / n, 1),
                     binary_R=bp.get("R", 0), binary_S=bp.get("S", 0), pct_binary_R=round(100 * bp.get("R", 0) / max(bp.get("R", 0) + bp.get("S", 0), 1), 1)))
t = pd.DataFrame(rows); t.to_csv(f"{OUT}/05_table1_check.csv", index=False); print("\n", t.to_string(index=False))
