"""Corrected re-run: train one model type for one drug.
usage: python r02_train.py <DRUG> <xgb|cnn_std|cnn_bin> [n_trials]
Splits: 10% stratified test set (seed 42); 10% of the remaining isolates as validation set (seed 42), taken BEFORE
oversampling. Oversampling: random duplication on the training part only, per-class targets keeping the ratios of the
original notebooks. Binary indicator: phenotypic resistance (MIC > ECOFF) for training/validation, TB-Profiler call for
the test set. CNN: Optuna (TPE, 20 trials, <=29 epochs, patience 8), then final training with seeds 42, 43, 44
(<=100 epochs, patience 10). Architecture, loss, optimiser and batch size as in the manuscript."""
import os
import json, os, sys, time, re
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "revision_analyses"))
from mic_common import Model
D = os.path.dirname(os.path.abspath(__file__))
# per-class oversampling targets from the original notebooks (class index -> count)
TC = {"AMI": {0: 6200, 1: 5084, 2: 4525, 6: 4340, 3: 3579, 4: 3530, 5: 3028}, "BDQ": {1: 3627, 2: 3386, 0: 3381, 3: 2705, 4: 2518, 5: 2557, 6: 2022, 7: 1504},
      "CFZ": {0: 6591, 1: 6412, 2: 6073, 3: 4303, 4: 4113, 5: 4033, 6: 4021}, "DLM": {2: 4822, 0: 4033, 1: 3204, 3: 3210, 6: 2581, 4: 2548, 5: 2536},
      "EMB": {2: 3571, 3: 3561, 4: 3344, 5: 3229, 1: 3005, 6: 2681, 0: 2149}, "ETH": {2: 3951, 3: 3538, 1: 2638, 4: 1819, 6: 1802, 5: 1637, 0: 1235},
      "INH": {1: 3047, 7: 2928, 0: 2188, 6: 2044, 3: 1326, 5: 1310, 4: 1274, 2: 1245}, "KAN": {1: 5085, 0: 5008, 2: 4479, 5: 4035, 3: 3522, 4: 3588},
      "LEV": {2: 5013, 1: 4750, 3: 4709, 5: 4588, 6: 4587, 0: 4386, 4: 4356, 7: 4257}, "LZD": {3: 5221, 2: 4864, 4: 4430, 1: 4744, 0: 4272, 6: 3571, 5: 3566},
      "MXF": {2: 3581, 3: 3381, 1: 2929, 0: 2854, 6: 2522, 4: 2501, 7: 2486, 5: 2420}, "RIF": {0: 3895, 7: 3625, 1: 2878, 2: 2645, 3: 2167, 6: 2145, 5: 2134, 4: 2103}}


def splits(drug):
    z = np.load(f"{D}/data/{drug}.npz", allow_pickle=True)
    X, y, tbp, R = z["X"].astype(np.float32), z["y"].astype(int), z["tbp"].astype(int), z["R"].astype(int)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.1, stratify=y, random_state=42)
    yt = y[tr]; strat = yt if np.bincount(yt).min() >= 2 else None
    tr2, va = train_test_split(tr, test_size=1 / 9, stratify=strat, random_state=42)   # 10% of all isolates
    # oversampling targets: keep the original ratios to the reference (most frequent) class
    cnt = pd.Series(y[tr2]).value_counts().to_dict(); ref = max(TC[drug].values()); top = max(cnt.values())
    target = {k: int(max(cnt.get(k, 0), round(TC[drug][k] / ref * top))) for k in cnt}
    ros = RandomOverSampler(sampling_strategy=target, random_state=42)
    ros.fit_resample(X[tr2], y[tr2]); os_idx = tr2[ros.sample_indices_]
    return dict(z=z, X=X, y=y, tbp=tbp, R=R, train=os_idx, train_orig=tr2, val=va, test=te, target=target, counts=cnt,
                K=len(z["mic_levels"]))


def class_weights(y, K):
    vals, counts = np.unique(y, return_counts=True)
    w = 1 / (counts / len(y)); w = w / w.sum()
    out = np.zeros(K, dtype=np.float32); out[vals] = w; return torch.tensor(out)


def fit_cnn(S, hp, aux_train, aux_val, seed, max_epochs, patience, dev="cuda", keep=True):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, K = S["X"], S["y"], S["K"]
    Xtr = torch.tensor(X[S["train"]]).unsqueeze(1).to(dev); ytr = torch.tensor(y[S["train"]]).long().to(dev)
    Xva = torch.tensor(X[S["val"]]).unsqueeze(1).to(dev); yva = torch.tensor(y[S["val"]]).long().to(dev)
    atr = torch.tensor(aux_train).float().to(dev); ava = torch.tensor(aux_val).float().to(dev)
    wt = class_weights(y[S["train"]], K).to(dev)
    crit = lambda o, t: (wt[t] * F.cross_entropy(o, t, reduction="none")).mean()
    model = Model(X.shape[1], num_classes=K, num_filters=hp["num_filters"], num_conv_layers=hp["num_conv_layers"],
                  num_dense_neurons=hp["num_dense_neurons"], num_dense_layers=hp["num_dense_layers"],
                  conv_dropout_rate=hp["conv_dropout_rate"], dense_dropout_rate=hp["dense_dropout_rate"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    g = torch.Generator().manual_seed(seed); n = len(ytr)
    best, bad, state, best_ep = float("inf"), 0, None, 0
    for ep in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            if len(b) < 2: continue
            loss = crit(model(Xtr[b], atr[b]), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for i in range(0, len(yva), 1024):
                tot += crit(model(Xva[i:i + 1024], ava[i:i + 1024]), yva[i:i + 1024]).item() * len(yva[i:i + 1024])
        vl = tot / len(yva)
        if vl < best:
            best, bad, best_ep = vl, 0, ep
            if keep: state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if keep: model.load_state_dict(state)
    return model, best, best_ep


def predict(model, X, aux, dev="cuda"):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(F.softmax(model(torch.tensor(X[i:i + 1024]).unsqueeze(1).to(dev), torch.tensor(aux[i:i + 1024]).float().to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    drug, kind = sys.argv[1], sys.argv[2]; n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    os.makedirs(f"{D}/results", exist_ok=True); os.makedirs(f"{D}/models", exist_ok=True)
    S = splits(drug); t0 = time.time()
    te = S["test"]; out = dict(test_idx=te, y=S["y"][te], tbp=S["tbp"][te], R=S["R"][te], ids=S["z"]["ids"][te])
    info = dict(drug=drug, kind=kind, n_train_orig=len(S["train_orig"]), n_train_os=len(S["train"]), n_val=len(S["val"]), n_test=len(te),
                counts={int(k): int(v) for k, v in S["counts"].items()}, target={int(k): int(v) for k, v in S["target"].items()}, n_features=int(S["X"].shape[1]))
    if kind.startswith("xgb"):
        import xgboost as xgb, pickle
        for name, binary in (("xgb", False), ("xgb_bin", True)):
            Xtr = S["X"][S["train"]]; Xte = S["X"][te]
            if binary:
                Xtr = np.column_stack([Xtr, S["R"][S["train"]]]); Xte = np.column_stack([Xte, S["tbp"][te]])
            m = xgb.XGBClassifier(random_state=42, n_jobs=16).fit(Xtr, S["y"][S["train"]])
            P = np.zeros((len(te), S["K"])); P[:, m.classes_] = m.predict_proba(Xte); out[name] = P
            pickle.dump(m, open(f"{D}/models/{name}_{drug}.pkl", "wb"))
    else:
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        binary = kind == "cnn_bin"
        a_tr = S["R"][S["train"]] if binary else np.zeros(len(S["train"])); a_va = S["R"][S["val"]] if binary else np.zeros(len(S["val"]))
        a_te = S["tbp"][te] if binary else np.zeros(len(te))

        def objective(trial):
            hp = dict(lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True), weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                      conv_dropout_rate=trial.suggest_float("conv_dropout_rate", 0.0, 0.6), dense_dropout_rate=trial.suggest_float("dense_dropout_rate", 0.0, 0.8),
                      num_filters=trial.suggest_int("num_filters", 32, 128), num_conv_layers=trial.suggest_int("num_conv_layers", 1, 3),
                      num_dense_neurons=trial.suggest_int("num_dense_neurons", 64, 256), num_dense_layers=trial.suggest_int("num_dense_layers", 1, 3))
            return fit_cnn(S, hp, a_tr, a_va, 42, 29, 8, keep=False)[1]
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42)); study.optimize(objective, n_trials=n_trials)
        hp = study.best_params; info["best_params"] = hp; info["optuna_secs"] = round(time.time() - t0)
        info["seeds"] = {}
        for seed in (42, 43, 44):
            m, vl, ep = fit_cnn(S, hp, a_tr, a_va, seed, 100, 10)
            out[f"{kind}_s{seed}"] = predict(m, S["X"][te], a_te); info["seeds"][seed] = dict(val_loss=vl, best_epoch=ep)
            torch.save(m.state_dict(), f"{D}/models/{kind}_{drug}_s{seed}.pt")
    info["secs"] = round(time.time() - t0)
    np.savez_compressed(f"{D}/results/{kind}_{drug}.npz", **out)
    json.dump(info, open(f"{D}/results/{kind}_{drug}.json", "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in info.items() if k not in ("counts", "target")}, default=int), flush=True)
