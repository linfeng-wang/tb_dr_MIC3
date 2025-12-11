# Quantitative Prediction of Tuberculosis Drug MICs Using Machine Learning  
*A reproducible framework for genome-to-phenotype modelling across 12 CRyPTIC antibiotics*

---

## Overview

This repository contains the complete machine learning workflow for predicting multi-level minimum inhibitory concentrations (MICs) for *Mycobacterium tuberculosis* (MTB) from whole-genome variation.  
The project aims to advance beyond binary resistance classification toward **quantitative MIC multi-classification prediction**, integrating:

- curated genomic preprocessing  
- multi-class XGBoost models  
- deep 1D CNNs with dynamic architecture  
- bootstrapped confidence intervals  
- multi-method interpretability (SHAP, Integrated Gradients, ΔAUC ablation)

The codebase is modular, reproducible, and designed for extensible antimicrobial resistance modelling.

---

## Data Sources

Primary data originate from the CRyPTIC consortium:

- `CRyPTIC_reuse_table_20231208.csv`  
- `CRyPTIC_reuse_table_20231208_cleaned+tbprofiler.csv`  

These include:

- quantitative MIC values for 12 antibiotics  
- TB-Profiler binary resistance predictions  
- curated SNP variant tables

All preprocessing steps needed to produce SNP × isolate matrices are included.

---

## Repository Structure

tb_dr_MIC3/
├── CRyPTIC_reuse_table_20231208.csv
├── CRyPTIC_reuse_table_20231208_cleaned+tbprofiler.csv
├── environment.yml
├── README.md
│
├── individual_models/
│ ├── all_snp_.npy # SNP matrices per drug
│ ├── generated_data30072025/ # processed SNP/MIC datasets
│ ├── saved_models/ # trained CNN + XGB models
│ ├── targets_pred/ # MIC predictions
│ ├── plots/ # evaluation plots
│ ├── nn_{drug}_class_resFeed.ipynb # CNN model training notebooks
│ ├── xgb_.ipynb # XGBoost training notebooks
│ └── shap/ and ablative analysis files
│
├── data_gen_new.ipynb # SNP + MIC preprocessing pipeline
├── data_size_checks.ipynb # dataset dimensionality/QC
└── sankey.ipynb # MIC distribution visualisation

---

## Environment Setup

Install all dependencies with Conda:

```bash
conda env create -f environment.yml
conda activate tb_mic

Environment includes:
- PyTorch
- XGBoost
- scikit-learn
- SHAP
- NumPy, pandas, SciPy
- seaborn, matplotlib
- JupyterLab
```

---

## Saved Model Weights

All final models used in the manuscript—including CNNs, binary-supported CNNs, and XGBoost baselines—are stored under:

individual_models/saved_models/


This directory contains:

- cnn_MIC_best_<drug>.pt (CNN full model)

- cnn_MIC_best_<drug>-bin.pt (CNN with TB-Profiler binary support)

- xgb_MIC_model_<drug>.pkl (XGBoost MIC classifier)

- xgb_MIC_model_<drug>_bin.pkl (binary-supported XGB model)

---

## Dataset Overview

All model-ready input arrays used for training and evaluation are stored in:

generated_data30072025/:

- all_sample_snps_cryptic_<drug>.npy # MIC labels aligned to the full CRyPTIC SNP panel.
- all_sample_snps_cryptic_<drug>.npy.gz #(compressed version of the same data)
- all_sample_snps_cryptic_{drug}.npy' # MIC labels
- all_sample_drs_cryptic_{drug}-tbp.npy # Optional TB-Profiler binary resistance indicator

individual_models/all_snp_{drug}.npy # list of SNP for each drug

---

## Citation

Please cite the accompanying manuscript if you use this repository in your research.

Wang, L., Campino, S., Clark, T.G. and Phelan, J.E. (2025c). Expanding Tuberculosis Drug Resistance Prediction beyond binary: Deep Learning for Minimum Inhibitory Concentration prediction. Research Square. doi:https://doi.org/10.21203/rs.3.rs-7621453/v1.
