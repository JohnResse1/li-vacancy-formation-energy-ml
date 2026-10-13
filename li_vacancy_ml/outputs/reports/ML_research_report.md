# Machine-learning research report: Li-vacancy formation energy

## Executive summary

This report analyzes the 24 material-level rows in the supplied workbook. The response
is the explicit `Li_vacancy_formation_energy` column, i.e. the minimum neutral Li-vacancy
formation energy over the 46 calculated symmetry-inequivalent sites. No site was
treated as an independent material sample.

- **Q1 — Predictability:** a material-level model can be fit and evaluated, but the
  sample is small and heterogeneous. The selected complexity-aware primary run is
  **M1 / SVR_linear** with nested LOOCV MAE = **0.703 eV**, RMSE = **0.902 eV**, R² = **0.629**, median absolute error = **0.502 eV**, and maximum absolute error = **1.966 eV**. The numerical non-dummy minimum is M5 / SVR_linear (MAE 0.674 eV); the primary rule favors a one-feature linear model when its error is practically close to the numerical winner.
- **Q2 — Eg:** Eg has Pearson r = **0.806** (p = 1.97e-06, bootstrap 95% CI 0.627 to 0.915) and Spearman rho = **0.773** (p = 9.60e-06, bootstrap 95% CI 0.548 to 0.892). This is an exploratory material-level association, not an independently validated law.
- **Q3 — independent electronic value:** the M3→M4 and M8→M9 comparisons below test Eg after composition/geometry and after thermodynamic descriptors. Their signs and paired intervals, rather than a single full-data coefficient, determine whether incremental value is credible.
- **Q4 — Bader/ELF complementarity:** Bader and ELF are not interchangeable measurements. Their incremental value is judged by M5/M10 and the Bader+ELF controls, while correlated coefficients are explicitly not interpreted as causal rankings.
- **Q5 — most testable hypothesis:** in matched oxygen-framework pairs, the energy cost of a neutral Li vacancy is influenced by the host's ability to redistribute/compensate the electrons and relax locally after Li removal; Eg may be a proxy for this response in this dataset, but not a direct Li-X bond-energy measure. The concrete next test is paired bulk/defect PDOS, charge-density difference, local Bader changes, and defect-relaxation energy decomposition.

## 1. Dataset and definitions

### Scope and target

- 24 distinct Materials Project-linked materials and 46 symmetry-inequivalent Li sites are
  represented in the DFT report.
- The target range is **0.241–6.667 eV**. The report defines
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

The main matrix has **23 non-redundant intrinsic features** and
one auxiliary control. Exact/precision-duplicate representations and the affine
`mean_framework_electronegativity`/`li_framework_electronegativity_difference` pair are
kept in the audit but not simultaneously ranked in the main model.

### Data cleaning and audit

`outputs/data/data_quality_report.md` and `outputs/data/feature_audit.csv` contain the
full checks. The workbook has 2 audited external-formula fields. Both
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

| feature | group | coefficient | p_value | q_value_bh | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| band_gap | E | 0.806 | 1.97e-06 | 4.73e-05 | 0.627 | 0.915 |
| formation_energy_per_atom | T | -0.541 | 0.006 | 0.051 | -0.831 | -0.088 |
| li_framework_electronegativity_difference | C | 0.541 | 0.006 | 0.051 | 0.098 | 0.763 |
| mean_framework_atomic_radius | C | -0.419 | 0.041 | 0.248 | -0.725 | -0.104 |
| li_coordination_number_std | G | -0.358 | 0.086 | 0.414 | -0.642 | 0.066 |
| li_x_bond_length_std | G | -0.245 | 0.249 | 0.710 | -0.609 | 0.144 |
| mean_li_x_bond_length | G | -0.221 | 0.299 | 0.710 | -0.512 | 0.122 |
| li_atomic_fraction | C | -0.208 | 0.330 | 0.710 | -0.693 | 0.272 |

### Strongest Spearman associations

| feature | group | coefficient | p_value | q_value_bh | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| band_gap | E | 0.773 | 9.60e-06 | 2.30e-04 | 0.548 | 0.892 |
| mean_framework_atomic_radius | C | -0.564 | 0.004 | 0.049 | -0.793 | -0.224 |
| li_framework_electronegativity_difference | C | 0.524 | 0.009 | 0.069 | 0.159 | 0.780 |
| formation_energy_per_atom | T | -0.464 | 0.022 | 0.126 | -0.789 | -0.003 |
| elf_high_fraction | E | 0.453 | 0.026 | 0.126 | -0.027 | 0.768 |
| li_atomic_fraction | C | -0.334 | 0.111 | 0.442 | -0.725 | 0.122 |
| li_coordination_number_std | G | -0.306 | 0.146 | 0.502 | -0.636 | 0.140 |
| li_x_bond_length_std | G | -0.260 | 0.220 | 0.591 | -0.589 | 0.111 |

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

| experiment | model | feature_count | MAE | RMSE | R2 | MedianAE | MaxAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M5 | SVR_linear | 21 | 0.674 | 0.883 | 0.644 | 0.464 | 1.914 |
| M1 | SVR_linear | 1 | 0.703 | 0.902 | 0.629 | 0.502 | 1.966 |
| M1 | Ridge | 1 | 0.716 | 0.931 | 0.604 | 0.478 | 1.955 |
| M1 | PLS | 1 | 0.723 | 0.939 | 0.598 | 0.498 | 1.956 |
| M1 | Eg_linear | 1 | 0.723 | 0.939 | 0.598 | 0.498 | 1.956 |
| M1 | SVR_rbf | 1 | 0.731 | 0.916 | 0.617 | 0.636 | 1.993 |
| M2 | Ridge | 5 | 0.755 | 0.964 | 0.576 | 0.563 | 1.867 |
| M12 | Ridge | 5 | 0.755 | 0.964 | 0.576 | 0.563 | 1.867 |
| M2 | PLS | 5 | 0.801 | 1.030 | 0.516 | 0.555 | 1.895 |
| M12 | PLS | 5 | 0.801 | 1.030 | 0.516 | 0.555 | 1.895 |
| M5 | Ridge | 21 | 0.830 | 1.015 | 0.529 | 0.646 | 2.390 |
| M2 | SVR_linear | 5 | 0.831 | 0.978 | 0.563 | 0.674 | 1.819 |
| M2 | SVR_rbf | 5 | 0.870 | 1.049 | 0.498 | 0.624 | 1.897 |
| M12 | SVR_rbf | 5 | 0.870 | 1.049 | 0.498 | 0.624 | 1.897 |
| M11 | SVR_rbf | 4 | 0.932 | 1.250 | 0.287 | 0.569 | 3.280 |
| M5 | PLS | 21 | 0.938 | 1.311 | 0.216 | 0.661 | 4.683 |
| M5 | SVR_rbf | 21 | 0.950 | 1.217 | 0.324 | 0.806 | 2.906 |
| M3 | Ridge | 16 | 1.025 | 1.296 | 0.233 | 0.926 | 3.099 |
| M4 | SVR_rbf | 17 | 1.027 | 1.380 | 0.131 | 0.844 | 3.634 |
| M6_control | Ridge | 24 | 1.027 | 1.384 | 0.126 | 1.059 | 3.961 |
| M6 | Ridge | 23 | 1.043 | 1.400 | 0.106 | 0.825 | 4.024 |
| M10 | Ridge | 23 | 1.043 | 1.400 | 0.106 | 0.825 | 4.024 |
| M4 | Ridge | 17 | 1.050 | 1.421 | 0.079 | 0.901 | 4.735 |
| M6_control | SVR_rbf | 24 | 1.058 | 1.386 | 0.124 | 0.682 | 3.801 |
| M9 | SVR_rbf | 19 | 1.091 | 1.449 | 0.042 | 0.840 | 3.648 |
| M4 | SVR_linear | 17 | 1.092 | 1.475 | 0.008 | 0.843 | 4.610 |
| M3 | SVR_rbf | 16 | 1.099 | 1.403 | 0.102 | 0.835 | 3.569 |
| M10 | PLS | 23 | 1.115 | 1.688 | -0.300 | 0.753 | 5.762 |
| M6 | PLS | 23 | 1.115 | 1.688 | -0.300 | 0.753 | 5.762 |
| M8 | Ridge | 18 | 1.116 | 1.388 | 0.121 | 1.048 | 3.429 |
| M6_control | PLS | 24 | 1.134 | 1.597 | -0.164 | 0.828 | 5.393 |
| M8 | SVR_rbf | 18 | 1.153 | 1.459 | 0.029 | 0.972 | 3.412 |
| M4 | PLS | 17 | 1.167 | 1.498 | -0.024 | 1.068 | 4.517 |
| M6 | SVR_linear | 23 | 1.168 | 1.511 | -0.042 | 0.901 | 3.832 |
| M9 | Ridge | 19 | 1.173 | 1.516 | -0.049 | 1.038 | 4.489 |
| M10 | SVR_rbf | 23 | 1.176 | 1.519 | -0.053 | 0.870 | 3.603 |
| M6 | SVR_rbf | 23 | 1.176 | 1.519 | -0.053 | 0.870 | 3.603 |
| M8 | PLS | 18 | 1.190 | 1.511 | -0.042 | 0.992 | 4.449 |
| M7 | SVR_rbf | 2 | 1.213 | 1.714 | -0.340 | 0.820 | 4.768 |
| M7 | Ridge | 2 | 1.239 | 1.537 | -0.078 | 1.053 | 3.510 |
| M7 | PLS | 2 | 1.262 | 1.884 | -0.619 | 0.788 | 6.165 |
| M9 | PLS | 19 | 1.273 | 1.679 | -0.286 | 1.183 | 4.693 |
| M11 | Ridge | 4 | 1.326 | 1.573 | -0.129 | 1.465 | 3.536 |
| M11 | PLS | 4 | 1.329 | 1.558 | -0.108 | 1.218 | 3.151 |
| M3 | PLS | 16 | 1.535 | 2.150 | -1.110 | 0.989 | 6.554 |
| M3 | SVR_linear | 16 | 1.539 | 1.922 | -0.685 | 1.253 | 5.483 |

The dummy baseline has MAE = **1.243 eV**, RMSE = **1.545 eV**,
and R² = **-0.089**. The best numerical score is not automatically treated
as a scientific discovery: model complexity, repeated coefficient behavior, and paired
error comparisons are considered together.

Per-material observed values, OOF predictions, residuals, and absolute errors are in
`outputs/predictions/oof_predictions.csv`; fold-specific selected parameters are in
`outputs/tables/outer_fold_parameters.csv`.


## 3.1 Controlled sensitivity experiments: feature subsets and target scale

Because the number of materials is fixed at 24, these experiments do not alter the
main M1–M6 benchmark. They test whether a smaller feature subset or a pre-specified
target transformation improves out-of-fold performance without leakage.

### Nested feature-subset search

`SelectKBest(f_regression)` was fitted inside every inner-CV training fold. The tested
subset sizes were 1, 2, 3, 4, 5, 6, 8, 10, all. Only Ridge and linear SVR were used to keep the extension constrained.

The best nested subset-search result is **SVR_linear / identity**,
with median selected feature count **1.000**, MAE = **0.724 eV**,
RMSE = **0.929 eV**, and R² = **0.606**.
This is an exploratory nested result; it does not replace the prespecified primary model.

| model | target_transform | feature_count | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| SVR_linear | identity | 1.000 | 0.724 | 0.929 | 0.606 |
| Ridge | identity | 1.000 | 0.822 | 1.148 | 0.399 |
| SVR_linear | log | 2.000 | 0.958 | 1.264 | 0.271 |
| Ridge | sqrt | 1.000 | 0.985 | 1.555 | -0.103 |
| Ridge | log | 2.000 | 1.085 | 1.427 | 0.071 |
| SVR_linear | sqrt | 1.000 | 1.106 | 1.815 | -0.504 |

The most frequently selected features across outer folds were:

| experiment | model | target_transform | feature | selection_frequency |
| --- | --- | --- | --- | --- |
| FS1 | Ridge | identity | band_gap | 1.000 |
| FS1 | SVR_linear | identity | band_gap | 1.000 |
| FS1 | Ridge | sqrt | band_gap | 1.000 |
| FS1 | SVR_linear | sqrt | band_gap | 1.000 |
| FS1 | Ridge | log | band_gap | 1.000 |
| FS1 | SVR_linear | log | band_gap | 1.000 |
| FS1 | Ridge | log | formation_energy_per_atom | 0.792 |
| FS1 | SVR_linear | log | formation_energy_per_atom | 0.667 |
| FS1 | Ridge | log | li_mean_coordination_number | 0.375 |
| FS1 | Ridge | log | framework_oxygen_fraction | 0.250 |
| FS1 | Ridge | log | li_framework_electronegativity_difference | 0.250 |
| FS1 | Ridge | log | elf_low_fraction | 0.208 |

Selection frequency is a stability diagnostic, not a causal importance measure. If the
selected subset changes substantially between outer folds, the data do not support a
unique feature ranking.

### Target transformations

The target was evaluated on the original, square-root, and logarithmic scales. All
metrics below are calculated after inverse-transforming predictions back to eV. The
transform was not selected using outer test predictions.

The best fixed-feature transform result is **M5 / SVR_linear / identity**,
with MAE = **0.674 eV**, RMSE = **0.883 eV**, and
R² = **0.644**. The complete table is saved as
`outputs/tables/target_transform_comparison.csv`.

| experiment | model | target_transform | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| M1 | Ridge | identity | 0.716 | 0.931 | 0.604 |
| M1 | Ridge | log | 0.912 | 1.154 | 0.392 |
| M1 | Ridge | sqrt | 0.774 | 1.010 | 0.535 |
| M1 | SVR_linear | identity | 0.703 | 0.902 | 0.629 |
| M1 | SVR_linear | log | 0.895 | 1.189 | 0.355 |
| M1 | SVR_linear | sqrt | 0.769 | 0.983 | 0.559 |
| M3 | Ridge | identity | 1.025 | 1.296 | 0.233 |
| M3 | Ridge | log | 1.469 | 1.895 | -0.639 |
| M3 | Ridge | sqrt | 1.049 | 1.377 | 0.135 |
| M3 | SVR_linear | identity | 1.539 | 1.922 | -0.685 |
| M3 | SVR_linear | log | 1.394 | 1.778 | -0.442 |
| M3 | SVR_linear | sqrt | 1.281 | 1.741 | -0.383 |
| M4 | Ridge | identity | 1.050 | 1.421 | 0.079 |
| M4 | Ridge | log | 1.352 | 1.747 | -0.392 |
| M4 | Ridge | sqrt | 1.075 | 1.430 | 0.067 |
| M4 | SVR_linear | identity | 1.092 | 1.475 | 0.008 |
| M4 | SVR_linear | log | 1.226 | 1.627 | -0.207 |
| M4 | SVR_linear | sqrt | 1.195 | 1.541 | -0.083 |
| M5 | Ridge | identity | 0.830 | 1.015 | 0.529 |
| M5 | Ridge | log | 0.908 | 1.208 | 0.334 |
| M5 | Ridge | sqrt | 0.789 | 1.137 | 0.410 |
| M5 | SVR_linear | identity | 0.674 | 0.883 | 0.644 |
| M5 | SVR_linear | log | 0.975 | 1.199 | 0.344 |
| M5 | SVR_linear | sqrt | 0.753 | 0.944 | 0.593 |
| M6 | Ridge | identity | 1.043 | 1.400 | 0.106 |
| M6 | Ridge | log | 1.256 | 1.830 | -0.529 |
| M6 | Ridge | sqrt | 1.014 | 1.423 | 0.076 |
| M6 | SVR_linear | identity | 1.168 | 1.511 | -0.042 |
| M6 | SVR_linear | log | 1.161 | 1.535 | -0.076 |
| M6 | SVR_linear | sqrt | 1.071 | 1.400 | 0.105 |

These tests can reveal whether the high-energy tail dominates the original-scale fit,
but a better transformed score is not automatically a better physical model.


## 4. Feature contribution and independent value

### Group ablation

Positive `delta_MAE_baseline_minus_extended` means the extended feature set lowers MAE.
The paired bootstrap interval uses the same material-level OOF predictions for both
feature sets; it is a diagnostic uncertainty interval, not a formal hypothesis test.

| baseline | extended | model | delta_MAE_baseline_minus_extended | delta_RMSE_baseline_minus_extended | delta_R2_extended_minus_baseline | paired_delta_ci_low | paired_delta_ci_high | n_extended_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 | M4 | Ridge | -0.025 | -0.124 | -0.154 | -0.481 | 0.351 | 16 |
| M3 | M5 | Ridge | 0.195 | 0.281 | 0.296 | -0.130 | 0.519 | 15 |
| M8 | M9 | Ridge | -0.058 | -0.128 | -0.170 | -0.404 | 0.243 | 15 |
| M9 | M10 | Ridge | 0.130 | 0.117 | 0.155 | -0.003 | 0.263 | 13 |

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

| subset | n | pearson_r | pearson_p | spearman_rho | spearman_p |
| --- | --- | --- | --- | --- | --- |
| all_24 | 24 | 0.806 | 1.97e-06 | 0.773 | 9.60e-06 |
| without_LiF_LiC6 | 22 | 0.688 | 4.06e-04 | 0.709 | 2.23e-04 |
| oxygen-bearing subset | 17 | 0.613 | 0.009 | 0.620 | 0.008 |
| category:carbon-containing | 5 | 0.908 | 0.033 | 0.900 | 0.037 |
| category:halide | 1 | NA | NA | NA | NA |
| category:nitride/nitrate | 2 | NA | NA | NA | NA |
| category:oxygen-containing | 13 | 0.730 | 0.005 | 0.736 | 0.004 |
| category:phosphide/phosphate | 1 | NA | NA | NA | NA |
| category:sulfide-selenide | 2 | NA | NA | NA | NA |

The all-material Eg coefficient changes over **0.740 to 0.837** across
leave-one-material diagnostics. Removing LiF and LiC6 gives:

| removed_material | eg_pearson_r | eg_spearman_rho | slope_eV_per_eV | r2_in_sample |
| --- | --- | --- | --- | --- |
| LiF | 0.740 | 0.742 | 0.617 | 0.547 |
| LiC6 | 0.780 | 0.745 | 0.566 | 0.608 |

These removals are not used to select a more favorable primary dataset. The optional
leave-one-category-out Ridge diagnostic is in `outputs/tables/leave_one_category_out.csv`;
it is not a replacement for material-level LOOCV because category coverage is uneven.

### Largest primary-model errors

| material | y_true | y_pred | absolute_error | residual |
| --- | --- | --- | --- | --- |
| LiNO3 | 5.067 | 3.102 | 1.966 | -1.966 |
| Li2C2O4 | 5.271 | 3.483 | 1.789 | -1.789 |
| Li2CuO2 | 2.592 | 1.191 | 1.401 | -1.401 |
| Li6MnO4 | 1.499 | 2.897 | 1.397 | 1.397 |
| Li6CoO4 | 1.368 | 2.760 | 1.393 | 1.393 |
| Li2C4O4 | 3.876 | 2.675 | 1.201 | -1.201 |
| LiC6 | 0.241 | 1.409 | 1.168 | 1.168 |
| Li2C2 | 4.195 | 3.289 | 0.906 | -0.906 |

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

| pair | feature | first_value | second_value | first_minus_second |
| --- | --- | --- | --- | --- |
| Li6CoO4 vs Li6ZnO4 | band_gap | 2.400 | 3.510 | -1.110 |
| Li6CoO4 vs Li6ZnO4 | Li_vacancy_formation_energy | 1.368 | 3.603 | -2.235 |
| Li6CoO4 vs Li6ZnO4 | li_mean_bader_charge | 0.840 | 0.902 | -0.062 |
| Li6CoO4 vs Li6ZnO4 | bulk_mean_elf | 0.275 | 0.280 | -0.005 |
| Li6CoO4 vs Li6ZnO4 | elf_high_fraction | 6.290 | 6.380 | -0.090 |
| Li6CoO4 vs Li6ZnO4 | elf_low_fraction | 58.426 | 57.585 | 0.841 |
| Li6CoO4 vs Li6ZnO4 | li_mean_coordination_number | 4.000 | 4.000 | 0.000 |
| Li6CoO4 vs Li6ZnO4 | normalized_li_x_distance | 1.022 | 1.021 | 1.62e-04 |
| Li6CoO4 vs Li6ZnO4 | volume_per_li | 16.093 | 16.116 | -0.023 |
| Li6CoO4 vs Li6ZnO4 | formation_energy_per_atom | -1.939 | -2.036 | 0.097 |
| Li6CoO4 vs Li6ZnO4 | energy_above_hull | 0.000 | 0.000 | 0.000 |
| Li5FeO4 vs Li5AlO4 | band_gap | 3.050 | 4.660 | -1.610 |
| Li5FeO4 vs Li5AlO4 | Li_vacancy_formation_energy | 2.912 | 4.303 | -1.391 |
| Li5FeO4 vs Li5AlO4 | li_mean_bader_charge | 0.652 | 0.667 | -0.016 |
| Li5FeO4 vs Li5AlO4 | bulk_mean_elf | 0.274 | 0.301 | -0.027 |
| Li5FeO4 vs Li5AlO4 | elf_high_fraction | 6.520 | 7.660 | -1.140 |
| Li5FeO4 vs Li5AlO4 | elf_low_fraction | 56.564 | 53.009 | 3.555 |
| Li5FeO4 vs Li5AlO4 | li_mean_coordination_number | 4.000 | 3.800 | 0.200 |
| Li5FeO4 vs Li5AlO4 | normalized_li_x_distance | 1.018 | 1.013 | 0.005 |
| Li5FeO4 vs Li5AlO4 | volume_per_li | 18.800 | 18.073 | 0.727 |
| Li5FeO4 vs Li5AlO4 | formation_energy_per_atom | -2.042 | -2.496 | 0.454 |
| Li5FeO4 vs Li5AlO4 | energy_above_hull | 0.000 | 0.000 | 0.000 |

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

- The 24 material-level target is heterogeneous, spans 0.241–6.667 eV, and is not
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

Software versions for this report run: Python 3.13.13, NumPy
2.4.6, SciPy 1.17.1, scikit-learn 1.8.0, and
pandas 3.0.3.
