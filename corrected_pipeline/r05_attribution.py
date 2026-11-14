"""RIF feature attribution for the binary-supported CNN (seed 42) of the corrected re-run.
  * integrated gradients (IG) on the predicted-class logit, zero baseline, 20 steps (IGdir signed, IGabs magnitude);
  * ablation: mean drop in the predicted-class logit when a variant is set to absent;
  * SHAP: expected gradients (shap.GradientExplainer) on the predicted-class probability, 200 training isolates as
    background, all test isolates; the TB-Profiler indicator is included as an extra input feature.
Feature names come from the feature list stored with the data, so labels are aligned with the input columns."""
import os
import sys, json, numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "revision_analyses"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mic_common import Model
from r02_train import splits
D = os.path.dirname(os.path.abspath(__file__)); dev = "cuda"
drug = sys.argv[1] if len(sys.argv) > 1 else "RIF"
S = splits(drug); info = json.load(open(f"{D}/results/cnn_bin_{drug}.json")); hp = info["best_params"]
names = np.load(f"{D}/data/{drug}.npz", allow_pickle=True)["features"]
model = Model(S["X"].shape[1], num_classes=S["K"], num_filters=hp["num_filters"], num_conv_layers=hp["num_conv_layers"],
              num_dense_neurons=hp["num_dense_neurons"], num_dense_layers=hp["num_dense_layers"],
              conv_dropout_rate=hp["conv_dropout_rate"], dense_dropout_rate=hp["dense_dropout_rate"]).to(dev)
model.load_state_dict(torch.load(f"{D}/models/cnn_bin_{drug}_s42.pt", map_location=dev)); model.eval()
te = S["test"]; X = S["X"][te]; aux = S["tbp"][te].astype(np.float32)
xt = torch.tensor(X).unsqueeze(1).to(dev); at = torch.tensor(aux).to(dev)
with torch.no_grad():
    base = model(xt, at); tgt = base.argmax(1)
N, Fn = X.shape; steps = 20
igd = np.zeros((N, Fn)); abl = np.zeros((N, Fn))
alphas = torch.linspace(0, 1, steps, device=dev).view(-1, 1, 1)
for b in range(N):
    xs = (alphas * xt[b:b + 1]).detach().requires_grad_(True)
    model(xs, at[b:b + 1].repeat(steps))[:, tgt[b]].sum().backward()
    igd[b] = (xt[b, 0] * xs.grad.mean(0)[0]).detach().cpu().numpy()
with torch.no_grad():
    bt = base.gather(1, tgt.view(-1, 1)).squeeze(1)
    for i in np.where(X.sum(0) > 0)[0]:
        xa = xt.clone(); xa[:, 0, i] = 0
        abl[:, i] = (bt - model(xa, at).gather(1, tgt.view(-1, 1)).squeeze(1)).cpu().numpy()


class Wrap(nn.Module):                      # SNP vector + indicator as one input row
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z): return torch.softmax(self.m(z[:, :, :-1], z[:, 0, -1]), 1)


import shap
rng = np.random.RandomState(42); bg_idx = rng.choice(S["train_orig"], 200, replace=False)
bg = torch.tensor(np.column_stack([S["X"][bg_idx], S["R"][bg_idx]]).astype(np.float32)).unsqueeze(1).to(dev)
ev = torch.tensor(np.column_stack([X, aux]).astype(np.float32)).unsqueeze(1).to(dev)
expl = shap.GradientExplainer(Wrap(model), bg)
sv = np.zeros((N, Fn + 1))
for i in range(0, N, 50):
    vals, _ = expl.shap_values(ev[i:i + 50], ranked_outputs=1, nsamples=200)
    sv[i:i + 50] = np.asarray(vals[0] if isinstance(vals, list) else vals).reshape(min(50, N - i), -1)[:, :Fn + 1]
lab = list(names) + ["TB-Profiler resistance call"]
df = pd.DataFrame({"feature": names, "carriers_test": X.sum(0).astype(int), "ablation": abl.mean(0), "IGdir": igd.mean(0),
                   "IGabs": np.abs(igd).mean(0), "SHAP_mean": sv[:, :Fn].mean(0), "SHAP_abs": np.abs(sv[:, :Fn]).mean(0)})
df.loc[len(df)] = ["TB-Profiler resistance call", int(aux.sum()), np.nan, np.nan, np.nan, sv[:, Fn].mean(), np.abs(sv[:, Fn]).mean()]
df.to_csv(f"{D}/tables/s5_attribution_{drug}.csv", index=False)
print(df.sort_values("IGabs", ascending=False).head(16).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print(df.sort_values("SHAP_abs", ascending=False).head(12)[["feature", "carriers_test", "SHAP_mean", "SHAP_abs"]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
top = df[df.feature != "TB-Profiler resistance call"].sort_values("SHAP_abs", ascending=False).head(20).iloc[::-1]
plt.figure(figsize=(10, 7.5))
plt.barh([f.replace(" p.", " ").replace(" c.", " c.") for f in top.feature], top.SHAP_mean, color="purple")
plt.axvline(0, color="black", lw=1); plt.xlabel("Mean SHAP value (predicted MIC class)")
plt.title(f"SHAP mutation importance, {drug} (binary-supported CNN)")
plt.tight_layout(); plt.savefig(f"{D}/tables/supp_fig4_shap_{drug}.png", dpi=150)
