"""Sensitivity analyses for the corrected re-run (Supplementary Table 13), using each drug's selected hyperparameters.
usage: python r08_sensitivity.py <DRUG> <perm|tbptrain>
  perm     : standard CNN with the feature columns randomly permuted (permutation seed 1), compared with genomic-position order
  tbptrain : binary-supported CNN with the TB-Profiler call (instead of the phenotype) as indicator in training and validation
Final training with seeds 42, 43, 44 (<=100 epochs, patience 10), as for the main models."""
import os
import sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r02_train import splits, fit_cnn, predict, D
drug, mode = sys.argv[1], sys.argv[2]
S = splits(drug); te = S["test"]
kind = "cnn_std" if mode == "perm" else "cnn_bin"
hp = json.load(open(f"{D}/results/{kind}_{drug}.json"))["best_params"]
if mode == "perm":
    perm = np.random.RandomState(1).permutation(S["X"].shape[1]); S["X"] = S["X"][:, perm]
    a_tr, a_va, a_te = np.zeros(len(S["train"])), np.zeros(len(S["val"])), np.zeros(len(te))
else:
    a_tr, a_va, a_te = S["tbp"][S["train"]], S["tbp"][S["val"]], S["tbp"][te]
out = {}
for seed in (42, 43, 44):
    m, vl, ep = fit_cnn(S, hp, a_tr, a_va, seed, 100, 10)
    out[f"s{seed}"] = predict(m, S["X"][te], a_te)
np.savez_compressed(f"{D}/results/sens_{mode}_{drug}.npz", y=S["y"][te], ids=S["z"]["ids"][te], **out)
y = S["y"][te]
print(drug, mode, [round(float(np.mean(np.abs(out[f"s{s}"].argmax(1) - y) <= 1)), 3) for s in (42, 43, 44)], flush=True)
