"""CNN retraining experiments (same architecture, loss, optimiser, early stopping
and drug-specific hyperparameters as the manuscript models).
  order     : standard CNN with the original vs randomly permuted feature order
  indicator : binary-supported CNN with alternative definitions of the training indicator
Usage: python 04_retrain_cnn.py <order|indicator> <DRUG> [DRUG ...]"""
import json, os, sys, time
import numpy as np, pandas as pd, torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from mic_common import *
ECOFF = dict(AMI=1, BDQ=0.25, CFZ=0.25, DLM=0.12, EMB=4, ETH=4, INH=0.1, KAN=4, LEV=1, LZD=1, MXF=1, RIF=0.5)
HP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperparams.json")))
dev = "cuda"

def train_eval(Xtr, ytr, atr, Xte, ate, hp, K, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    strat = np.array([f"{a}_{b}" for a, b in zip(ytr, atr)])
    cnt = pd.Series(strat).value_counts()
    strat = np.where(pd.Series(strat).map(cnt).values < 2, ytr.astype(str), strat)
    tr, va = train_test_split(np.arange(len(ytr)), test_size=0.1, random_state=42, shuffle=True, stratify=strat)
    vals, counts = np.unique(ytr, return_counts=True)
    w = 1 / (counts / len(ytr)); w = w / w.sum()
    wt = torch.zeros(K); wt[vals.astype(int)] = torch.tensor(w).float(); wt = wt.to(dev)
    X = torch.tensor(Xtr).float().unsqueeze(1).to(dev); Y = torch.tensor(ytr).long().to(dev); A = torch.tensor(atr).float().to(dev)
    model = Model(Xtr.shape[1], num_classes=K, num_filters=hp["num_filters"], num_conv_layers=hp["num_conv_layers"],
                  num_dense_neurons=hp["num_dense_neurons"], num_dense_layers=hp["num_dense_layers"],
                  conv_dropout_rate=hp["conv_dropout_rate"], dense_dropout_rate=hp["dense_dropout_rate"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit = lambda out, y: (wt[y] * torch.nn.functional.cross_entropy(out, y, reduction="none")).mean()
    g = torch.Generator().manual_seed(seed)
    tr_t, va_t = torch.tensor(tr), torch.tensor(va)
    best, pat, state, ep_best = float("inf"), 0, None, 0
    for ep in range(1, 101):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=g)]
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64].to(dev)
            if len(b) < 2: continue
            loss = crit(model(X[b], A[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vl = []
        with torch.no_grad():
            for i in range(0, len(va_t), 64):
                b = va_t[i:i + 64].to(dev); vl.append(crit(model(X[b], A[b]), Y[b]).item())
        v = float(np.mean(vl))
        if v < best:
            best, pat, ep_best = v, 0, ep; state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            pat += 1
            if pat >= 10: break
    model.load_state_dict(state)
    return predict(model, Xte, ate), ep_best, best

def summarise(P, y, rc):
    p = P.argmax(1); R = (y >= rc).astype(int)
    cw = {}
    for c in np.unique(y):
        try: cw[int(c)] = round(float(roc_auc_score((y == c).astype(int), P[:, c])), 3)
        except Exception: pass
    return dict(dda=round(dda(y, p), 4), auc_ovr=round(float(roc_auc_score(y, P, multi_class="ovr")), 4),
                bal_acc=round(float(balanced_accuracy_score(y, p)), 4),
                R_recall=round(float(recall_score(R, (p >= rc).astype(int), zero_division=0)), 4),
                dda_R=round(dda(y[R == 1], p[R == 1]), 4), dda_S=round(dda(y[R == 0], p[R == 0]), 4),
                classwise_auc=cw, pred_hist=np.bincount(p, minlength=P.shape[1]).tolist())

mode, drugs = sys.argv[1], sys.argv[2:]
log = open(f"{OUT}/04_{mode}_{'_'.join(drugs)}.jsonl", "a")
for drug in drugs:
    d = load_drug(drug, oversample=True)
    Xtr, ytr = d["X_train"], d["train_target"][d["mx"]].values.astype(int)
    Xte, yte, tbp_te = d["X_test"], d["y_test"], d["tbp_test"]
    K = len(d["mic_values"]); rc = int(np.argmax(d["mic_values"] > ECOFF[drug] * 1.05))
    # row indices of the oversampled training set, to recover TB-Profiler calls for training isolates
    idx = d["os_idx"]
    assert idx is not None and np.array_equal(d["raw"]["X_train"][idx], Xtr) and \
        np.array_equal(d["raw"]["train_target"][d["mx"]].values[idx].astype(int), ytr), "oversampling indices not recovered"
    tbp_tr = d["raw"]["train_target"]["tbp"].values[idx].astype(int)
    runs = []
    if mode == "order":
        hp = HP[drug]["std"]; z = np.zeros(len(ytr)); zt = np.zeros(len(yte))
        for s in (42, 43, 44): runs.append((f"original_seed{s}", Xtr, z, Xte, zt, hp, s))
        for ps in (1, 2, 3):
            perm = np.random.RandomState(ps).permutation(Xtr.shape[1])
            runs.append((f"permuted{ps}", Xtr[:, perm], z, Xte[:, perm], zt, hp, 42))
    else:
        hp = HP[drug]["bin"]
        a_code = d["train_target"][d["my"]].values.astype(int)   # indicator exactly as in the notebooks
        a_ecoff = (ytr >= rc).astype(int)
        runs.append(("phenotype_train(manuscript)", Xtr, a_code, Xte, tbp_te, hp, 42))
        if not np.array_equal(a_code, a_ecoff):
            runs.append(("phenotype_train_ECOFF", Xtr, a_ecoff, Xte, tbp_te, hp, 42))
        runs.append(("tbprofiler_train", Xtr, tbp_tr, Xte, tbp_te, hp, 42))
    import os
    keep = os.environ.get('RUNS')
    if keep: runs = [r for k in keep.split(',') for r in runs if r[0] == k]
    for name, a, at, b, bt, hp, s in runs:
        t0 = time.time()
        P, ep, vl = train_eval(a, ytr, at, b, bt, hp, K, s)
        r = dict(drug=drug, mode=mode, variant=name, best_epoch=ep, val_loss=round(vl, 4), secs=round(time.time() - t0), **summarise(P, yte, rc))
        print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        np.save(f"{OUT}/04_{mode}_{drug}_{name.replace('(', '_').replace(')', '')}.npy", P)
