"""Supplementary Figure 4 from the SHAP values of r05_attribution.py, plotted as in the RIF notebook (cell 63):
top 20 features by absolute SHAP value, purple horizontal bars, labels '<gene>-<change>'."""
import os
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
drug = sys.argv[1] if len(sys.argv) > 1 else "RIF"
df = pd.read_csv(f"{D}/tables/s5_attribution_{drug}.csv")
df = df[df.feature != "TB-Profiler resistance call"]
shap_vec = df.SHAP_mean.values
feat_all = [f.replace(" ", "-", 1) for f in df.feature]
sorted_idx = np.argsort(-np.abs(shap_vec))
top_k = 20
idx_to_plot = sorted_idx[:top_k]
vals_to_plot = shap_vec[idx_to_plot]
feat_names = [feat_all[i] for i in idx_to_plot]
plt.figure(figsize=(10, 8))
order = np.argsort(vals_to_plot)
plt.barh([feat_names[i] for i in order], vals_to_plot[order], color="purple")
plt.axvline(0, color="black")
# no title: in the supplement the title of the original image is cropped and given in the figure heading instead
plt.xlabel("SHAP value")
plt.tight_layout()
plt.savefig(f"{D}/figs/supp_fig4_shap_{drug}.png", dpi=150, bbox_inches="tight")
print("\n".join(f"{n} {v:.4g}" for n, v in zip(feat_names, vals_to_plot)))
