# Revision analyses

Additional analyses run in response to reviewer comments (revision 6). They reuse the trained models in
`individual_models/saved_models/` and the per-drug data, train/test split and oversampling exactly as defined in the
`individual_models/nn_<drug>_class_resFeed.ipynb` notebooks (`mic_common.py` executes the data and oversampling cells
of each notebook, so nothing is redefined here).

| Script | Output (`results/`) | Manuscript |
|---|---|---|
| `01_eval_saved.py` | `01_reproduced_table2.csv`, `pred_<drug>.npz` | Re-evaluates all saved models on the held-out test set (reproduces Table 2) |
| `02_metrics.py` | `02_extra_metrics.csv`, `02_tbprofiler.csv`, `02_mcnemar.csv`, `02_thresholds.csv` | Supplementary Tables 10, 11 and 3 |
| `03_oversampling_xgb.py` | `03_oversampling_xgb.csv` | Supplementary Table 12 (none / random / SMOTE / ADASYN) |
| `04_retrain_cnn.py` | `04_*.jsonl` | Supplementary Table 13 (feature order, run-to-run variation, binary indicator definition) |
| `05_lineage_table1.py` | `05_table1_check.csv` | Lineage distribution of rpoB/rpoC background variants; Table 1 resistant counts |
| `06_feature_sets.py` | printed | Genes contributing to each feature matrix (Supplementary Table 6) |
| `07_geneset_sensitivity.py` | `07_geneset_sensitivity.csv` | XGBoost with the feature sets used vs the gene sets named in the Methods (BDQ, CFZ, INH) |

Run from this folder with the `tb_mic` environment (`environment.yml`), after generating the data files with
`individual_models/data_gen_new.ipynb`, in order 01 then 02; the others are independent. `04_retrain_cnn.py` takes a
mode and drug list, e.g. `python 04_retrain_cnn.py order RIF MXF` or `python 04_retrain_cnn.py indicator INH LEV`,
and `RUNS=<variant>,...` restricts the variants run. Set `MIC_OUT` to write results elsewhere.

Notes:
- The saved AMI models use the 1,011-feature matrix in `individual_models/generated_data30072025/`; all other drugs
  use `individual_models/generated_data18122024/`. `mic_common.py` loads these accordingly.
- Retrained CNNs differ from the saved models because CNN training is not deterministic on GPU.
