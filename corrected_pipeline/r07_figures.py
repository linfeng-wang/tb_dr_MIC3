"""Panels for Supplementary Figures 2 (stacked bars) and 3 (Sankey) of the corrected re-run: binary-supported CNN (seed 42)
and XGBoost for AMI, BDQ, RIF and EMB. Plotting code follows the original notebooks:
  stacked bars  -> individual_models/nn_<drug>_class_resFeed.ipynb (pandas .plot(kind='bar', stacked=True))
  Sankey        -> individual_models/sankey.ipynb, draw_sankey() (pysankey; run this part in the `sankey` conda env)
usage: python r07_figures.py stack     (ml-workshop env)
       python r07_figures.py sankey    (sankey env)"""
import os
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__)); FIG = f"{D}/figs"; TP = f"{D}/figs/targets_pred"
os.makedirs(TP, exist_ok=True)
DRUGS = ["AMI", "BDQ", "RIF", "EMB"]


def is_within_doubling_dilution(pred, target):
    return abs(pred - target) <= 1


def stacked(drug, y_test, y_pred, mic, cutoff, title, path, tight):
    # as in the notebooks (cell "Stacked Barplot of Correct vs. Erroneous MIC Predictions")
    mic_dict = {m: [0, 0] for m in np.unique(y_test)}
    for pred, true in zip(y_pred, y_test):
        if is_within_doubling_dilution(pred, true):
            mic_dict[true][0] += 1
        else:
            mic_dict[true][1] += 1
    mic_df = pd.DataFrame.from_dict(mic_dict, orient="index", columns=["Correct", "Erroneous"])
    mic_df.index = mic
    fig, ax = plt.subplots(figsize=(8, 6))
    mic_df.plot(kind='bar', stacked=True, ax=ax, color=["blue", "grey"])
    for i, (correct, erroneous) in enumerate(zip(mic_df["Correct"], mic_df["Erroneous"])):
        total = correct + erroneous
        if total > 0:
            ax.text(i, total + 8, f"{correct / total * 100:.1f}%", ha='center', fontsize=10)
    ax.set_xlabel("MIC Value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[handles[1], handles[0]], labels=["Erroneous", "Correct"], title="Prediction")
    plt.xticks(rotation=0)
    ax.axvline(x=cutoff, color='red', linestyle='--', linewidth=1.4)
    if tight:            # the XGBoost cell of the notebooks calls tight_layout, the CNN cell does not
        plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def draw_sankey(drug, model=''):
    # as in sankey.ipynb
    from pysankey import sankey
    with open(f'{TP}/mic_dic-{drug}.json', 'r') as file:
        mic_dic = json.load(file)
    file_path = f'{TP}/targets_pred{model}-{drug}.txt'
    df = pd.read_csv(file_path, sep=' ', names=['true', 'predicted'])
    df['true'] = df['true'].astype(int)
    df = df.astype(str)
    df_sorted = df.sort_values(by=['true', 'predicted'])
    df_sorted.replace(mic_dic, inplace=True)
    distinct_colors = ["#dfa39d", "#e4d69e", "#bce2a6", "#898ab5", "#a36ed8", "#6e80d8",
                       "#d8b3c4", "#f2cfa5", "#aed9c5", "#c9c4e4", "#b3bfe0", "#e8b9a6"]
    unique_values = pd.concat([df_sorted['true'], df_sorted['predicted']]).unique()
    colorDict = dict(zip(unique_values, distinct_colors[:len(unique_values)]))
    sankey(df_sorted['true'], df_sorted['predicted'], aspect=200, colorDict=colorDict, fontsize=7,
           figureName=f'{FIG}/targets_pred{model}-{drug}')
    plt.close("all")


mode = sys.argv[1]
for drug in (sys.argv[2:] or DRUGS):
    x = np.load(f"{D}/results/xgb_{drug}.npz", allow_pickle=True); c = np.load(f"{D}/results/cnn_bin_{drug}.npz", allow_pickle=True)
    z = np.load(f"{D}/data/{drug}.npz", allow_pickle=True); mic = z["mic_levels"].astype(float)
    cutoff = int(np.argmax(mic > float(z["ecoff"]) * 1.05))
    y = x["y"].astype(int)
    preds = {"": c["cnn_bin_s42"].argmax(1), "_XGB": x["xgb"].argmax(1)}
    if mode == "stack":
        stacked(drug, y, preds[""], mic, cutoff, f"Stacked Barplot of Correct vs. Erroneous MIC Predictions-{drug}", f"{FIG}/stack_cnn_bin_{drug}.png", False)
        stacked(drug, y, preds["_XGB"], mic, cutoff, f"Stacked Barplot of Correct vs. Erroneous MIC Predictions by XGB - {drug}", f"{FIG}/stack_xgb_{drug}.png", True)
        # files read by draw_sankey (class index as key, as the notebooks wrote target/prediction pairs)
        json.dump({str(k): str(v) for k, v in enumerate(mic)}, open(f"{TP}/mic_dic-{drug}.json", "w"))
        for model, p in preds.items():
            with open(f"{TP}/targets_pred{model}-{drug}.txt", "w") as f:
                for pred, target in zip(p, y):
                    f.write(f"{target} {pred}\n")
    else:
        draw_sankey(drug, '_XGB'); draw_sankey(drug)
print(mode, "figures written")
