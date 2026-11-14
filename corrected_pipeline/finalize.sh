#!/bin/bash
# After all training has finished: tables, figures and the list of numbers quoted in the text.
set -e
cd "$(dirname "$0")"
mkdir -p logs results models tables figs
PY=${PY:-python}
SK=${SK:-python}   # environment with pysankey
$PY r03_evaluate.py > logs/r03.log 2>&1
$PY r04_supp_tables.py > logs/r04.log 2>&1
$PY r09_extra_tables.py > logs/r09.log 2>&1
$PY r07_figures.py stack > logs/r07.log 2>&1
$SK r07_figures.py sankey >> logs/r07.log 2>&1
$PY r05b_shap_plot.py RIF > logs/r05b.log 2>&1
$PY r10_supp_fig1.py > logs/r10.log 2>&1
$PY r11_lineage_markers.py > logs/r11.log 2>&1
$PY r12_facts.py > logs/r12_facts.txt 2>&1
$PY r13_facts_extra.py > logs/r13_facts_extra.txt 2>&1
echo FINALIZED
