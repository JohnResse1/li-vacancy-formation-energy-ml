"""Generate the technical Markdown report from saved analysis objects."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from .data_loader import (
    ALL_INTRINSIC_FEATURES,
    ALL_VALID_FEATURES,
    ELECTRONIC_FEATURES,
    TARGET_CANONICAL,
)


def _number(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    if value != 0 and abs(float(value)) < 10 ** (-digits):
        return f"{float(value):.2e}"
    return f"{float(value):.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    if frame.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(_number(value, digits))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _top_correlation(correlations: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return correlations.sort_values("coefficient", key=lambda values: values.abs(), ascending=False).head(n)


def generate_report(
    data: pd.DataFrame,
    audit: dict[str, Any],
    correlations: dict[str, Any],
    benchmark: dict[str, Any],
    ablation: dict[str, Any],
    robustness: dict[str, Any],
    output_root: str | Path,
    config: dict[str, Any],
    sensitivity: dict[str, Any] | None = None,
) -> Path:
    output_root = Path(output_root)
    report_path = output_root / "reports" / "ML_research_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = benchmark["comparison"].copy()
    primary = robustness["primary"]
    overall_non_dummy = comparison[comparison["model"] != "Dummy"].sort_values(["MAE", "RMSE"]).iloc[0]
    dummy = comparison[comparison["model"] == "Dummy"].sort_values("MAE").iloc[0]
    pearson = correlations["pearson"]
    spearman = correlations["spearman"]
    eg_p = correlations["eg_pearson"]
    eg_s = correlations["eg_spearman"]
    top_p = _top_correlation(pearson)
    top_s = _top_correlation(spearman)
    ridge_ablation = ablation["ablation"][ablation["ablation"]["model"] == "Ridge"].copy()
    selected_pairs = ridge_ablation[ridge_ablation[["baseline", "extended"]].apply(tuple, axis=1).isin([("M3", "M4"), ("M3", "M5"), ("M8", "M9"), ("M9", "M10")])]
    category = robustness["category"]
    pairs = robustness["pairs"]
    influence = robustness["influence"]
    material_errors = next(item for item in benchmark["results"] if item["experiment"].name == primary["experiment"] and item["model_name"] == primary["model"])["predictions"].sort_values("absolute_error", ascending=False).head(8)

    eg_without_extremes = influence.loc[influence["removed_material"].isin(["LiF", "LiC6"])]
    eg_removed_range = (influence["eg_pearson_r"].min(), influence["eg_pearson_r"].max())
    all_range = f"{data[TARGET_CANONICAL].min():.3f}–{data[TARGET_CANONICAL].max():.3f}"
    external_audit = audit["external_formula_audit"]
    if sensitivity is not None:
        subset_comparison = sensitivity["subset_comparison"].copy()
        fixed_comparison = sensitivity["fixed_comparison"].copy()
        selection_frequency = sensitivity["selection_frequency"].copy()
        subset_best = subset_comparison.iloc[0]
        transform_best = fixed_comparison.iloc[0]
        top_selected = selection_frequency.sort_values(
            ["selection_frequency", "feature"], ascending=[False, True]
        ).head(12)
        sensitivity_section = f"""
## 3.1 Controlled sensitivity experiments: feature subsets and target scale

Because the number of materials is fixed at 24, these experiments do not alter the
main M1–M6 benchmark. They test whether a smaller feature subset or a pre-specified
target transformation improves out-of-fold performance without leakage.

### Nested feature-subset search

`SelectKBest(f_regression)` was fitted inside every inner-CV training fold. The tested
subset sizes were {", ".join(str(value) for value in config.get("subset_k_values", [1, 2, 3, 4, 5, 6, 8, 10, "all"]))}. Only Ridge and linear SVR were used to keep the extension constrained.

The best nested subset-search result is **{subset_best['model']} / {subset_best['target_transform']}**,
with median selected feature count **{_number(subset_best['feature_count'])}**, MAE = **{_number(subset_best['MAE'])} eV**,
RMSE = **{_number(subset_best['RMSE'])} eV**, and R² = **{_number(subset_best['R2'])}**.
This is an exploratory nested result; it does not replace the prespecified primary model.

{_markdown_table(subset_comparison, ['model', 'target_transform', 'feature_count', 'MAE', 'RMSE', 'R2'])}

The most frequently selected features across outer folds were:

{_markdown_table(top_selected, ['experiment', 'model', 'target_transform', 'feature', 'selection_frequency'])}

Selection frequency is a stability diagnostic, not a causal importance measure. If the
selected subset changes substantially between outer folds, the data do not support a
unique feature ranking.

### Target transformations

The target was evaluated on the original, square-root, and logarithmic scales. All
metrics below are calculated after inverse-transforming predictions back to eV. The
transform was not selected using outer test predictions.

The best fixed-feature transform result is **{transform_best['experiment']} / {transform_best['model']} / {transform_best['target_transform']}**,
with MAE = **{_number(transform_best['MAE'])} eV**, RMSE = **{_number(transform_best['RMSE'])} eV**, and
R² = **{_number(transform_best['R2'])}**. The complete table is saved as
`outputs/tables/target_transform_comparison.csv`.

{_markdown_table(fixed_comparison.sort_values(['experiment', 'model', 'target_transform']), ['experiment', 'model', 'target_transform', 'MAE', 'RMSE', 'R2'])}

These tests can reveal whether the high-energy tail dominates the original-scale fit,
but a better transformed score is not automatically a better physical model.
"""
    else:
        sensitivity_section = ""
    report = f"""# Machine-learning research report: Li-vacancy formation energy

## Executive summary

This report analyzes the 24 material-level rows in the supplied workbook. The response
is the explicit `Li_vacancy_formation_energy` column, i.e. the minimum neutral Li-vacancy
formation energy over the 46 calculated symmetry-inequivalent sites. No site was
treated as an independent material sample.

- **Q1 — Predictability:** a material-level model can be fit and evaluated, but the
  sample is small and heterogeneous. The selected complexity-aware primary run is
  **{primary['experiment']} / {primary['model']}** with nested LOOCV MAE = **{_number(primary['MAE'])} eV**, RMSE = **{_number(primary['RMSE'])} eV**, R² = **{_number(primary['R2'])}**, median absolute error = **{_number(primary['MedianAE'])} eV**, and maximum absolute error = **{_number(primary['MaxAE'])} eV**. The numerical non-dummy minimum is {overall_non_dummy['experiment']} / {overall_non_dummy['model']} (MAE {_number(overall_non_dummy['MAE'])} eV); the primary rule favors a one-feature linear model when its error is practically close to the numerical winner.
- **Q2 — Eg:** Eg has Pearson r = **{_number(eg_p['coefficient'])}** (p = {_number(eg_p['p_value'])}, bootstrap 95% CI {_number(eg_p['ci_low'])} to {_number(eg_p['ci_high'])}) and Spearman rho = **{_number(eg_s['coefficient'])}** (p = {_number(eg_s['p_value'])}, bootstrap 95% CI {_number(eg_s['ci_low'])} to {_number(eg_s['ci_high'])}). This is an exploratory material-level association, not an independently validated law.
- **Q3 — independent electronic value:** the M3→M4 and M8→M9 comparisons below test Eg after composition/geometry and after thermodynamic descriptors. Their signs and paired intervals, rather than a single full-data coefficient, determine whether incremental value is credible.
- **Q4 — Bader/ELF complementarity:** Bader and ELF are not interchangeable measurements. Their incremental value is judged by M5/M10 and the Bader+ELF controls, while correlated coefficients are explicitly not interpreted as causal rankings.
- **Q5 — most testable hypothesis:** in matched oxygen-framework pairs, the energy cost of a neutral Li vacancy is influenced by the host's ability to redistribute/compensate the electrons and relax locally after Li removal; Eg may be a proxy for this response in this dataset, but not a direct Li-X bond-energy measure. The concrete next test is paired bulk/defect PDOS, charge-density difference, local Bader changes, and defect-relaxation energy decomposition.

## 1. Dataset and definitions

### Scope and target

- 24 distinct Materials Project-linked materials and 46 symmetry-inequivalent Li sites are
  represented in the DFT report.
- The target range is **{all_range} eV**. The report defines
  `E_f(V_Li^0) = E_defect - E_bulk + mu_Li`, with `mu_Li = -1.88547189 eV` from the same-parameter bcc Li reference.
- The PDF states that all 24 bulk and 46 defect calculations completed, with defect
  supercell vacancy concentrations spanning approximately 2.50–6.25%. This concentration
  is retained as an auxiliary control, not treated as an intrinsic electronic descriptor.
- The DFT report uses spin-polarized PBE/PAW VASP calculations, ENCUT = 450 eV,
  static `IBRION=-1`, `NSW=0`, and bulk Bader/ELF outputs. These facts contextualize
  the input features; this project does not rerun VASP.

### Feature groups frozen before modeling

- **E — electronic:** Eg, mean Li Bader charge loss, mean bulk ELF, ELF >= 0.8 fraction,
  and ELF <= 0.2 fraction. Eg, Bader, and ELF are retained as separate subgroups.
- **G — geometry:** Li coordination, normalized Li-X distance, volume per Li, density,
  volume per atom, Li-X bond statistics, and coordination-number dispersion.
- **C — composition:** Li fraction, Li-framework electronegativity difference, mean
  framework radius/atomic number/mass, framework electronegativity dispersion, and
  framework oxygen fraction.
- **T — thermodynamics:** bulk formation energy per atom and energy above hull. These
  are not treated as the same quantity as the defect target, although their DFT-energy
  provenance may create information overlap.
- **AUX:** nominal vacancy concentration, tested separately as a defect-model control.

The main matrix has **{len(ALL_INTRINSIC_FEATURES)} non-redundant intrinsic features** and
one auxiliary control. Exact/precision-duplicate representations and the affine
`mean_framework_electronegativity`/`li_framework_electronegativity_difference` pair are
kept in the audit but not simultaneously ranked in the main model.

### Data cleaning and audit

`outputs/data/data_quality_report.md` and `outputs/data/feature_audit.csv` contain the
full checks. The workbook has {len(external_audit)} audited external-formula fields. Both
formula fields point to the same external `W` cell by row. The cached sigma values are
consistent with an independent population SD using Pauling electronegativities; the
oxygen-fraction field is independently rebuilt as `N_O / N_framework` from the formula,
because trusting the duplicated external reference would be scientifically unsafe.
The original workbook is copied under `data/original/` and is not modified. No mean or
zero imputation is used to conceal an unavailable column.

## 2. Correlation analysis

All correlations use n = 24 material rows. p-values and Benjamini–Hochberg q-values are
exploratory under the small-sample setting; bootstrap intervals resample material rows
and do not substitute for external validation.

### Strongest Pearson associations

{_markdown_table(top_p, ['feature', 'group', 'coefficient', 'p_value', 'q_value_bh', 'ci_low', 'ci_high'])}

### Strongest Spearman associations

{_markdown_table(top_s, ['feature', 'group', 'coefficient', 'p_value', 'q_value_bh', 'ci_low', 'ci_high'])}

Eg is therefore checked independently in the scatter plot and in the nested model
comparisons. The full descriptor-by-descriptor matrices are in
`outputs/tables/descriptor_spearman_matrix.csv` and the Pearson counterpart. Strong
descriptor dependence is expected: volume per Li is algebraically related to volume per
atom and Li fraction, while the Bader duplicate and vacancy-concentration duplicate are
precision/identity variants. Coefficients should be read as correlated-set evidence.

## 3. Machine-learning prediction

Each material is the outer test case exactly once. Within each outer training set,
missing-value imputation and standardization are fitted only on training rows, and
Ridge/PLS/SVR parameters are selected by shuffled 5-fold inner CV with random seed 42.
No target-based feature selection is used in the prespecified M1–M6 benchmark. The Eg-only linear baseline is prespecified;
the M1–M6 models include Dummy, Ridge, PLS, linear SVR, and RBF SVR comparisons.

### Complete benchmark

The machine-readable full table is `outputs/tables/model_comparison.csv`. The table below
lists every evaluated non-dummy result; Dummy results are retained in the CSV as the
baseline.

{_markdown_table(comparison[comparison['model'] != 'Dummy'], ['experiment', 'model', 'feature_count', 'MAE', 'RMSE', 'R2', 'MedianAE', 'MaxAE'])}

The dummy baseline has MAE = **{_number(dummy['MAE'])} eV**, RMSE = **{_number(dummy['RMSE'])} eV**,
and R² = **{_number(dummy['R2'])}**. The best numerical score is not automatically treated
as a scientific discovery: model complexity, repeated coefficient behavior, and paired
error comparisons are considered together.

Per-material observed values, OOF predictions, residuals, and absolute errors are in
`outputs/predictions/oof_predictions.csv`; fold-specific selected parameters are in
`outputs/tables/outer_fold_parameters.csv`.

{sensitivity_section}

## 4. Feature contribution and independent value

### Group ablation

Positive `delta_MAE_baseline_minus_extended` means the extended feature set lowers MAE.
The paired bootstrap interval uses the same material-level OOF predictions for both
feature sets; it is a diagnostic uncertainty interval, not a formal hypothesis test.

{_markdown_table(selected_pairs, ['baseline', 'extended', 'model', 'delta_MAE_baseline_minus_extended', 'delta_RMSE_baseline_minus_extended', 'delta_R2_extended_minus_baseline', 'paired_delta_ci_low', 'paired_delta_ci_high', 'n_extended_better'])}

Interpretation must follow the actual signs in this table. In particular:

- M3→M4 asks whether Eg adds value to composition and geometry.
- M3→M5 asks whether the full electronic group adds value beyond composition and geometry.
- M8→M9 asks whether Eg remains useful after bulk thermodynamic descriptors.
- M9→M10 asks whether Bader/ELF information improves an Eg-plus-thermodynamics model.

The M6 Ridge coefficients and 500-resample coefficient bootstrap are in
`outputs/tables/ridge_coefficient_stability.csv`,
`outputs/tables/ridge_bootstrap_coefficient_summary.csv`, and the fold-level file. These
are standardized descriptive coefficients. Because correlated descriptors can share
the same signal, a larger absolute coefficient is not a causal effect and does not
prove that one electronic observable is physically primary.

## 5. Robustness and material dependence

### Category sensitivity

The mutually exclusive category rule is: carbon-containing first; then halide,
sulfide/selenide, phosphide/phosphate, nitride/nitrate, oxygen-containing, and other.
This is a transparent formula-based grouping, not a claim that each chemical label is a
single mechanistic family.

{_markdown_table(category, ['subset', 'n', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p'])}

The all-material Eg coefficient changes over **{_number(eg_removed_range[0])} to {_number(eg_removed_range[1])}** across
leave-one-material diagnostics. Removing LiF and LiC6 gives:

{_markdown_table(eg_without_extremes, ['removed_material', 'eg_pearson_r', 'eg_spearman_rho', 'slope_eV_per_eV', 'r2_in_sample'])}

These removals are not used to select a more favorable primary dataset. The optional
leave-one-category-out Ridge diagnostic is in `outputs/tables/leave_one_category_out.csv`;
it is not a replacement for material-level LOOCV because category coverage is uneven.

### Largest primary-model errors

{_markdown_table(material_errors, ['material', 'y_true', 'y_pred', 'absolute_error', 'residual'])}

The material influence table reports Eg coefficients, regression slopes, primary-model
error restricted to the remaining rows, and M3→M4 paired error changes for every removed
material. This makes LiF, LiC6, and any high-error material visible without deleting it
from the formal result.

## 6. Physical interpretation and candidate pairs

The observed association is compatible with several non-exclusive explanations:

1. A host gap can correlate with the availability and energetic cost of electronic
   compensation after neutral Li removal.
2. Removing Li changes charge distribution on anions and transition-metal or main-group
   framework sites; bulk Bader charge and ELF are only indirect pre-defect proxies.
3. Local Li-X coordination and structural relaxation can dominate the total energy and
   may be correlated with, but are not determined by, Eg.
4. A large gap is not a direct measure of Li-X bond strength, and the neutral-vacancy
   formation energy is not a Li migration barrier or a universal electrochemical voltage.

The matched-composition candidate contrasts are recorded in
`outputs/tables/material_pair_comparisons.csv`:

{_markdown_table(pairs, ['pair', 'feature', 'first_value', 'second_value', 'first_minus_second'])}

Li6CoO4/Li6ZnO4 and Li5FeO4/Li5AlO4 are useful next-step pairs because their nominal
oxygen-framework stoichiometries are closely related while the cation chemistry and
electronic response differ. The current descriptors cannot establish whether their
formation-energy gaps originate from band-edge alignment, transition-metal d-state
occupation, anion-hole localization, or relaxation. Those are hypotheses requiring
new DFT outputs: bulk and defect PDOS/band edges, charge-density difference maps,
site-resolved Bader changes before/after vacancy creation, ELF or localization changes,
and an explicit decomposition of defect relaxation and electronic contributions.

## 7. Research implications and limitations

### Findings suitable for Results

- The 24 material-level target is heterogeneous, spans {all_range} eV, and is not
  represented by the 46 sites as independent observations.
- Eg, ELF threshold volume, thermodynamic stability, composition, and geometry should
  be discussed jointly; the exact ranking is in the reproducible tables rather than
  selected visually.
- Nested LOOCV provides an honest material-level prediction estimate; the selected
  model's OOF predictions are available row by row.

### Discussion-level hypotheses

- Electronic descriptors may add value as proxies for post-defect charge compensation,
  but their apparent signal may partly encode chemical-family differences and overlap
  with thermodynamic/structural descriptors.
- Bader charge and ELF quantify different aspects—atomic charge partitioning versus
  spatial localization—so their physical complementarity should be tested in matched
  chemistry rather than inferred from a single global importance ranking.

### Limitations

- n = 24 is too small for strong confirmation, precise subgroup inference, or reliable
  high-dimensional feature ranking. Bootstrap intervals are themselves uncertain.
- The data span different chemistries and defect concentrations (approximately
  2.50–6.25%), and the target is a minimum over unequal numbers of inequivalent sites.
- Formation energy per atom and energy above hull are thermodynamic controls with DFT
  energy provenance; their inclusion can absorb variance related to host stability.
- Eg, Bader, and ELF are bulk descriptors while the target includes defect relaxation.
  The analysis therefore supports predictive association and testable hypotheses, not
  a verified microscopic mechanism or causal effect.

## 8. Reproducibility and outputs

Run from `li_vacancy_ml/`:

```bash
python scripts/run_all.py --config config.yaml
```

The run records the Python and library versions in `outputs/logs/run_manifest.json`.
The main deliverables are:

- `outputs/data/cleaned_dataset.csv`
- `outputs/data/feature_dictionary.csv`
- `outputs/data/feature_audit.csv`
- `outputs/reports/data_quality_report.md`
- `outputs/tables/pearson_correlations.csv`
- `outputs/tables/spearman_correlations.csv`
- `outputs/tables/feature_correlations.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/tables/feature_group_ablation.csv`
- `outputs/tables/outer_fold_parameters.csv`
- `outputs/predictions/oof_predictions.csv`
- `outputs/figures/Fig01_descriptor_target_relationships.pdf` and `.png`
- `outputs/figures/Fig02_machine_learning_benchmark.pdf` and `.png`
- `outputs/figures/Fig03_electronic_descriptor_contribution.pdf` and `.png`
- `outputs/figures/Fig04_robustness_material_dependence.pdf` and `.png`

Software versions for this report run: Python {platform.python_version()}, NumPy
{np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}, and
pandas {pd.__version__}.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path
