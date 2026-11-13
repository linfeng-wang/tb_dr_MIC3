"""Shared helpers for the revision 6 analyses.

Data loading re-executes the data and oversampling cells of the original
notebooks in tb_dr_MIC3/individual_models, so the train/test split, MIC
encoding and oversampling targets are identical to those used for the
manuscript. Nothing in tb_dr_MIC3 is modified.
"""
import json, os, re, io, contextlib, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IM = f"{REPO}/individual_models"
OUT = os.environ.get("MIC_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)
DRUGS = ["AMI", "BDQ", "CFZ", "DLM", "EMB", "ETH", "INH", "KAN", "LEV", "LZD", "MXF", "RIF"]
seed = 42


def data_split(aa_array, encoded_mic):
    np.random.seed(seed)
    return train_test_split(aa_array, encoded_mic, test_size=0.1,
                            stratify=encoded_mic.iloc[:, 0], random_state=42)


def _cells(drug):
    nb = json.load(open(f"{IM}/nn_{drug.lower()}_class_resFeed.ipynb"))
    data_cell = over_cell = None
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if data_cell is None and re.search(r"^drug\s*=", src, re.M) and "np.load" in src:
            data_cell = src
        if over_cell is None and "target_counts" in src and "RandomOverSampler" in src:
            over_cell = src
            continue
        # in the RIF and EMB notebooks the resampled arrays are assigned in the following cell
        if over_cell is not None and not re.search(r"^train_data = X_resampled", over_cell, re.M) \
                and re.search(r"^train_data = X_resampled", src, re.M) and len(src) < 400:
            over_cell += "\n" + src
    return data_cell, over_cell


def load_drug(drug, oversample=True):
    """Return a dict with the exact arrays used in the notebooks."""
    data_cell, over_cell = _cells(drug)
    ns = {"np": np, "pd": pd, "data_split": data_split, "seed": seed}
    cwd = os.getcwd()
    os.chdir(IM)
    try:
        if drug == "AMI":
            # The saved AMI models were trained on the 1,011-feature matrix in
            # generated_data30072025 (the TB-Profiler array is row-aligned with it).
            data_cell = data_cell.replace("./generated_data18122024/all_sample_snps_cryptic_{drug}.npy",
                                          "./generated_data30072025/all_sample_snps_cryptic_{drug}.npy")
            data_cell = data_cell.replace("./generated_data18122024/all_sample_drs_cryptic_{drug}.npy",
                                          "./generated_data30072025/all_sample_drs_cryptic_{drug}.npy")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(data_cell, ns)
            raw = dict(X_train=np.asarray(ns["train_data"]).copy(),
                       train_target=ns["train_target"].copy(),
                       X_test=np.asarray(ns["test_data"]).copy(),
                       test_target=ns["test_target"].copy())
            if oversample:
                exec(over_cell, ns)
    finally:
        os.chdir(cwd)
    mx, my = f"{drug}_MIC_x", f"{drug}_MIC_y"
    d = dict(drug=drug, mx=mx, my=my, cutoff=float(ns["cutoff"]),
             classes=np.sort(np.unique(np.asarray(ns["mic_series"]).ravel())),
             mic_values=np.sort(np.unique(np.asarray(ns.get("encoded_mic_", ns["encoded_mic"])).ravel())),
             target_counts=ns.get("target_counts"), raw=raw,
             X_test=raw["X_test"], test_target=raw["test_target"],
             y_test=raw["test_target"][mx].values.astype(int),
             tbp_test=raw["test_target"]["tbp"].values.astype(int),
             phen_test=raw["test_target"][my].values.astype(int))
    if oversample:
        d["X_train"] = np.asarray(ns["train_data"])
        d["train_target"] = ns["train_target"]
        ros = ns.get("ros")
        d["os_idx"] = getattr(ros, "sample_indices_", None)
    return d


class Model(nn.Module):
    """Identical to the notebook architecture."""
    def __init__(self, example_input_len, in_channels=1, num_classes=6, num_filters=64,
                 filter_length=25, num_conv_layers=2, filter_scaling_factor=1,
                 num_dense_neurons=256, num_dense_layers=2, conv_dropout_rate=0.0,
                 dense_dropout_rate=0.2, return_logits=False):
        super().__init__()
        self.conv_dropout_rate = conv_dropout_rate
        self.dense_dropout_rate = dense_dropout_rate
        self.feature_extraction_layer = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=filter_length),
            nn.BatchNorm1d(num_filters), nn.ReLU())
        c1 = num_filters
        self.conv_layers1 = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_layers1.append(self._conv_layer(c1, int(c1 * filter_scaling_factor), 3))
            c1 = int(c1 * filter_scaling_factor)
        c2 = 32
        self.conv_layers2 = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_layers2.append(self._conv_layer(c1, int(c2 * filter_scaling_factor), 3))
            c1 = c2
        self.m = nn.MaxPool1d(3, stride=1)
        with torch.no_grad():
            self.flat_dim = self._forward_conv_only(
                torch.zeros(1, in_channels, example_input_len)).view(1, -1).shape[1]
        self.dense_layers = nn.ModuleList(
            [self._dense_layer(self.flat_dim, num_dense_neurons)] +
            [self._dense_layer(num_dense_neurons, num_dense_neurons)
             for _ in range(num_dense_layers - 1)])
        t = num_dense_neurons + 1
        self.prediction_layer = nn.Sequential(
            nn.Linear(t, int(t * 0.7)), nn.ReLU(), nn.Dropout(p=dense_dropout_rate),
            nn.Linear(int(t * 0.7), int(t * 0.5)), nn.ReLU(), nn.Dropout(p=dense_dropout_rate),
            nn.Linear(int(t * 0.5), num_classes))
        self.apply(self.init_weights)

    def _forward_conv_only(self, x):
        x = self.feature_extraction_layer(x)
        for layer in self.conv_layers1:
            x = layer(x)
        x = self.m(x)
        for layer in self.conv_layers2:
            x = layer(x)
        return self.m(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _conv_layer(self, i, o, k):
        return nn.Sequential(nn.Dropout(p=self.conv_dropout_rate), nn.Conv1d(i, o, kernel_size=k),
                             nn.BatchNorm1d(o), nn.ReLU())

    def _dense_layer(self, i, o):
        return nn.Sequential(nn.Dropout(p=self.dense_dropout_rate), nn.Linear(i, o),
                             nn.BatchNorm1d(o), nn.ReLU())

    def forward(self, x, additional_input):
        x = self._forward_conv_only(x)
        x = x.view(x.size(0), -1)
        for layer in self.dense_layers:
            x = layer(x)
        x = torch.cat((x, additional_input.unsqueeze(1)), dim=1)
        return self.prediction_layer(x)


def arch_from_state(sd):
    return dict(
        num_filters=sd["feature_extraction_layer.0.weight"].shape[0],
        num_conv_layers=len([k for k in sd if re.match(r"conv_layers1\.\d+\.1\.weight", k)]),
        num_dense_layers=len([k for k in sd if re.match(r"dense_layers\.\d+\.1\.weight", k)]),
        num_dense_neurons=sd["dense_layers.0.1.weight"].shape[0],
        num_classes=sd["prediction_layer.6.weight"].shape[0])


def predict(model, X, aux, device="cuda", bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs]).float().unsqueeze(1).to(device)
            ab = torch.tensor(aux[i:i + bs]).float().to(device)
            out.append(F.softmax(model(xb, ab), dim=1).cpu().numpy())
    return np.concatenate(out)


def load_saved_cnn(drug, binary, input_len, device="cuda"):
    sd = torch.load(f"{IM}/saved_models/cnn_MIC_best_{drug}{'-bin' if binary else ''}.pt", map_location="cpu")
    a = arch_from_state(sd)
    m = Model(input_len, **a)
    m.load_state_dict(sd)
    return m.to(device), a


def dda(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))
