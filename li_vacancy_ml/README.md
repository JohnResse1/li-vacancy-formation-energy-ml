# Li-vacancy formation-energy machine-learning study

This directory contains the reproducible analysis for the 24-material Li-vacancy dataset.

## Run

From this directory:

```bash
python scripts/run_all.py --config config.yaml
```

The configuration file is JSON syntax with a `.yaml` suffix, so the project does not
require PyYAML. All preprocessing, model selection, and scaling are performed inside
the relevant cross-validation folds. The original Excel workbook and PDF are copied
under `data/original/` and are never overwritten.

## Main outputs

- `outputs/data/feature_audit.csv`: field mapping, validity, duplicate/relation audit.
- `outputs/data/cleaned_dataset.csv`: canonical 24-row analysis table, with repaired
  derived fields explicitly recorded.
- `outputs/tables/`: correlations, model comparison, ablation, robustness, and OOF data.
- `outputs/predictions/oof_predictions.csv`: one out-of-fold prediction per material.
- `outputs/tables/feature_subset_comparison.csv`: leakage-safe nested feature-subset search.
- `outputs/tables/target_transform_comparison.csv`: original, square-root, and log-target checks.
- `outputs/tables/feature_selection_frequency.csv`: outer-fold feature-selection stability.
- `outputs/predictions/feature_subset_oof_predictions.csv` and
  `outputs/predictions/target_transform_oof_predictions.csv`: sensitivity OOF predictions.
- `outputs/figures/`: editable PDF and 300-dpi PNG figures.
- `outputs/reports/ML_research_report.md`: generated technical research report.

The two formula-linked workbook columns are audited rather than silently trusted. The
cached sigma-electronegativity values are independently reconstructed from the material
formula and Pauling electronegativities; framework oxygen fraction is reconstructed as
O atoms divided by all non-Li framework atoms because the workbook's external formula
for both columns points to the same external cell.
