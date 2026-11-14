"""Supplementary Figure 1 (MIC distributions) with the corrected delamanid categories.
Plotting code is cell 22 of individual_models/data_size_checks.ipynb, unchanged. The original figure read the MIC arrays in
generated_data18122024, in which delamanid readings of '<=0.015' had been placed at 0.0625 mg/L. Here the delamanid array
is replaced by the corrected one (rerun/data/DLM.npz); the other 11 drugs are read from the same files as before."""
import os
import numpy as np
import os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "individual_models", "generated_data18122024")

resistance_thresholds = {
    "AMI": 1,
    "RIF": 0.5,
    "DLM": 0.12,
    "MXF": 1,
    "EMB": 4,
    "BDQ": 0.25,
    "CFZ": 0.25,
    "LZD": 1,
    "LEV": 1,
    "ETH": 4,
    "KAN": 4,
    "INH": 0.1,
}

# panel order of the original figure
files = [f"all_sample_drs_cryptic_{d}.npy" for d in
         ["AMI", "RIF", "ETH", "DLM", "MXF", "EMB", "BDQ", "CFZ", "KAN", "INH", "LZD", "LEV"]]

color_palette = [
    "#dfa39d", "#e4d69e", "#bce2a6", "#898ab5", "#a36ed8", "#6e80d8",
    "#d8b3c4", "#f2cfa5", "#aed9c5", "#c9c4e4", "#b3bfe0", "#e8b9a6"
]

value_counts_dict = {}
n_col = 3
n_row = (len(files) + n_col - 1) // n_col

fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(15, n_row * 5))
axes = axes.flatten()

for i, (ax, file) in enumerate(zip(axes, files)):
    file_path = os.path.join(directory, file)
    if "DLM" in file:
        z = np.load(f"{D}/data/DLM.npz", allow_pickle=True)
        data = z["mic_levels"][z["y"]]
    else:
        data = np.load(file_path)

    # exact MIC bins
    unique_values, counts = np.unique(data, return_counts=True)
    labels = [str(v) for v in unique_values]
    x_pos = np.arange(len(unique_values))

    value_counts_dict[file] = dict(zip(unique_values, counts))

    color = color_palette[i % len(color_palette)]

    # bar plot
    ax.bar(x_pos, counts, width=0.6, color=color, edgecolor='black', alpha=1.0)

    # log y axis
    ax.set_yscale('log')

    # tick labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # axis labels with bigger font
    if i // n_col == n_row - 1:
        ax.set_xlabel("MIC Value", fontsize=14)

    if i % n_col == 0:
        ax.set_ylabel("Count", fontsize=14)

    # subplot title with bigger font
    file_clean = (
        file.replace("all_sample_drs_cryptic_", "")
            .replace("-tbp", "")
            .replace(".npy", "")
    )
    ax.set_title(f"{file_clean} MIC distribution", fontsize=16)

    ax.grid(False)

    # ECOFF line
    for drug in resistance_thresholds:
        if drug.lower() in file_clean.lower():
            if resistance_thresholds[drug] in unique_values:
                idx = np.where(unique_values == resistance_thresholds[drug])[0][0]
                ax.axvline(x=idx, color='red', linestyle='--', linewidth=1.5)
            break

for ax in axes[len(files):]:
    ax.axis('off')

plt.tight_layout()
fig.savefig(f"{D}/figs/supp_fig1_mic_distribution.png", dpi=144, bbox_inches="tight")
print({k.replace("all_sample_drs_cryptic_", "").replace(".npy", ""): {float(a): int(b) for a, b in v.items()}
       for k, v in value_counts_dict.items() if "DLM" in k})
