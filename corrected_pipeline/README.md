# Corrected pipeline (revision 6)

This folder contains the pipeline used for all model results in the revised manuscript. It replaces the per-drug
notebooks in `individual_models/` for data preparation, training and evaluation. The model architecture (`Model` in
`revision_analyses/mic_common.py`, taken from the notebooks), loss, optimiser, batch size and Optuna search space are
unchanged.

## What differs from the original notebooks

| Step | Original notebooks | This pipeline |
|---|---|---|
| Candidate genes | *Rv0678* (BDQ, CFZ) and *fabG1*, *ahpC* (INH) were not included; AMI used extra genes | Gene sets exactly as listed in the Methods |
| rRNA variants | removed (`non_coding_transcript_exon_variant`), which dropped *rrs* and *rrl* mutations | kept; only synonymous variants are removed |
| Feature order | order of first appearance, zero-padded | genomic position, one column per variant seen at allele frequency >= 0.8 |
| DLM MIC categories | readings of `<=0.015` placed at 0.0625 mg/L | lowest category (0.015625 mg/L), as in Supplementary Table 7 |
| Validation set | taken from the oversampled training data | 10% of isolates, stratified, taken before oversampling |
| Binary indicator (training) | cutoff above the ECOFF for EMB, INH and LEV | MIC > ECOFF for all drugs |
| CNN hyperparameters | not tuned for every drug and model | Optuna (TPE, 20 trials) for each drug and for the standard and binary-supported CNN |
| CNN training runs | one | three seeds (42, 43, 44); results are the mean |
| Feature attribution (RIF) | labels not aligned with the input columns | feature names stored with the input matrix; IG, ablation and SHAP on the input features |

## Scripts

| Script | Output | Manuscript |
|---|---|---|
| `r01_build_data.py` | `data/<drug>.npz`, `data/build_report.json` | per-drug input matrix, MIC class, TB-Profiler call, isolate IDs |
| `r02_train.py <drug> <xgb\|cnn_std\|cnn_bin>` | `results/`, `models/` | test-set predictions and trained models |
| `r03_evaluate.py` | `tables/table2.csv`, `s3_*`, `s4_*`, `s10_*`, `s11_*` | Table 2, Supplementary Tables 3, 4, 10, 11 |
| `r04_supp_tables.py` | `tables/s2_oversampling.csv`, `s6_features.csv` | Supplementary Tables 2, 6 |
| `r05_attribution.py`, `r05b_shap_plot.py` | `tables/s5_attribution_RIF.csv`, `figs/supp_fig4_shap_RIF.png` | Supplementary Table 5, Supplementary Figure 4 |
| `r06_oversampling_methods.py` | `tables/s12_oversampling_methods.csv` | Supplementary Table 12 |
| `r07_figures.py stack\|sankey` | `figs/stack_*`, `figs/targets_pred*` | Supplementary Figures 2, 3 |
| `r08_sensitivity.py <drug> <perm\|tbptrain>` | `results/sens_*` | Supplementary Table 13 |
| `r09_extra_tables.py` | `tables/s13_sensitivity.csv`, `s14_hyperparameters.csv` | Supplementary Tables 13, 14 |
| `r10_supp_fig1.py` | `figs/supp_fig1_mic_distribution.png` | Supplementary Figure 1 |
| `r11_lineage_markers.py` | `tables/s5_carrier_lineages_RIF.csv` | lineage distribution of the top-ranked rifampicin features |
| `r12_facts.py`, `r13_facts_extra.py` | printed | numbers quoted in the text |

Order: `r01`, then `r02 <drug> xgb` for each drug, `run_cnn.sh` (all CNNs), `run_all_sens.sh` (sensitivity runs),
`r05`, `r06`, and `finalize.sh` for the tables and figures. `r01` needs the variant table (`cache/variants_full.csv`)
and the CRyPTIC tables, which are not stored in the repository; the resulting inputs are provided in `data/`.

`models/` holds the XGBoost models and, for each drug and CNN type, the CNN of the first training run (seed 42); the
CNNs of the other two runs are not stored because of their size, but their test-set predictions are in `results/`.

`results/` holds the test-set predictions (class probabilities) of every model and seed, from which all tables can
be recomputed with `r03`, `r09` and `r12` without retraining. CNN training is not deterministic on GPU, so retrained
models will differ slightly from the stored predictions.

The analyses in `revision_analyses/` were run on the originally submitted models and are kept for reference; their
results are superseded by this folder.
