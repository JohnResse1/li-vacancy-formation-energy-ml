# Data quality report

## Scope

- Source workbook: `/Users/srp_group/Desktop/bts2/li_vacancy_ml/data/original/CIJ-24种补锂材料_核心候选描述符矩阵.xlsx`.
- Sheets found: 24种材料特征矩阵, 描述符说明与来源.
- Main sheet rows/columns: 24 / 30.
- Target: `Li_vacancy_formation_energy`, taken only from `Y 最低中性Li空位形成能 E_f(V_Li^0) (eV)`.
- Overall automated status: **PASS**.

## Completeness and physical checks

The checks are written to `outputs/data/quality_checks.csv`. A nonzero `WARN` is retained
as a scientific data-quality flag rather than silently changing a value.

- `sample_count`: PASS (24); Expected 24 material rows.
- `unique_material_names`: PASS (24); Material labels are unique.
- `duplicate_material_records`: PASS (0); Duplicate material labels checked.
- `duplicate_mp_ids`: PASS (0); MP-ID uniqueness checked.
- `duplicate_full_records`: PASS (0); Full canonical records checked.
- `target_numeric_complete`: PASS (24); Target is parsed from the explicit Y column.
- `numeric_formation_energy_per_atom`: PASS (0); Missing or non-numeric values in formation_energy_per_atom: 0.
- `numeric_li_mean_bader_charge`: PASS (0); Missing or non-numeric values in li_mean_bader_charge: 0.
- `numeric_li_mean_coordination_number`: PASS (0); Missing or non-numeric values in li_mean_coordination_number: 0.
- `numeric_normalized_li_x_distance`: PASS (0); Missing or non-numeric values in normalized_li_x_distance: 0.
- `numeric_volume_per_li`: PASS (0); Missing or non-numeric values in volume_per_li: 0.
- `numeric_li_framework_electronegativity_difference`: PASS (0); Missing or non-numeric values in li_framework_electronegativity_difference: 0.
- `numeric_band_gap`: PASS (0); Missing or non-numeric values in band_gap: 0.
- `numeric_energy_above_hull`: PASS (0); Missing or non-numeric values in energy_above_hull: 0.
- `numeric_li_atomic_fraction`: PASS (0); Missing or non-numeric values in li_atomic_fraction: 0.
- `numeric_density`: PASS (0); Missing or non-numeric values in density: 0.
- `numeric_volume_per_atom`: PASS (0); Missing or non-numeric values in volume_per_atom: 0.
- `numeric_mean_li_x_bond_length`: PASS (0); Missing or non-numeric values in mean_li_x_bond_length: 0.
- `numeric_li_x_bond_length_std`: PASS (0); Missing or non-numeric values in li_x_bond_length_std: 0.
- `numeric_li_x_bond_length_range`: PASS (0); Missing or non-numeric values in li_x_bond_length_range: 0.
- `numeric_li_coordination_number_std`: PASS (0); Missing or non-numeric values in li_coordination_number_std: 0.
- `numeric_mean_framework_electronegativity`: PASS (0); Missing or non-numeric values in mean_framework_electronegativity: 0.
- `numeric_mean_framework_atomic_radius`: PASS (0); Missing or non-numeric values in mean_framework_atomic_radius: 0.
- `numeric_mean_atomic_number`: PASS (0); Missing or non-numeric values in mean_atomic_number: 0.
- `numeric_mean_atomic_mass`: PASS (0); Missing or non-numeric values in mean_atomic_mass: 0.
- `numeric_bulk_mean_elf`: PASS (0); Missing or non-numeric values in bulk_mean_elf: 0.
- `numeric_elf_high_fraction`: PASS (0); Missing or non-numeric values in elf_high_fraction: 0.
- `numeric_elf_low_fraction`: PASS (0); Missing or non-numeric values in elf_low_fraction: 0.
- `numeric_framework_electronegativity_std`: PASS (0); Missing or non-numeric values in framework_electronegativity_std: 0.
- `numeric_framework_oxygen_fraction`: PASS (0); Missing or non-numeric values in framework_oxygen_fraction: 0.
- `numeric_vacancy_concentration`: PASS (0); Missing or non-numeric values in vacancy_concentration: 0.
- `numeric_li_mean_charge_loss`: PASS (0); Missing or non-numeric values in li_mean_charge_loss: 0.
- `numeric_vacancy_concentration_percent`: PASS (0); Missing or non-numeric values in vacancy_concentration_percent: 0.
- `numeric_Li_vacancy_formation_energy`: PASS (0); Missing or non-numeric values in Li_vacancy_formation_energy: 0.
- `constant_formation_energy_per_atom`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_li_mean_bader_charge`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_li_mean_coordination_number`: PASS (8); A constant descriptor cannot contribute predictive information.
- `constant_normalized_li_x_distance`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_volume_per_li`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_li_framework_electronegativity_difference`: PASS (21); A constant descriptor cannot contribute predictive information.
- `constant_band_gap`: PASS (23); A constant descriptor cannot contribute predictive information.
- `constant_energy_above_hull`: PASS (7); A constant descriptor cannot contribute predictive information.
- `constant_li_atomic_fraction`: PASS (12); A constant descriptor cannot contribute predictive information.
- `constant_density`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_volume_per_atom`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_mean_li_x_bond_length`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_li_x_bond_length_std`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_li_x_bond_length_range`: PASS (22); A constant descriptor cannot contribute predictive information.
- `constant_li_coordination_number_std`: PASS (5); A constant descriptor cannot contribute predictive information.
- `constant_mean_framework_electronegativity`: PASS (21); A constant descriptor cannot contribute predictive information.
- `constant_mean_framework_atomic_radius`: PASS (21); A constant descriptor cannot contribute predictive information.
- `constant_mean_atomic_number`: PASS (21); A constant descriptor cannot contribute predictive information.
- `constant_mean_atomic_mass`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_bulk_mean_elf`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_elf_high_fraction`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_elf_low_fraction`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_framework_electronegativity_std`: PASS (17); A constant descriptor cannot contribute predictive information.
- `constant_framework_oxygen_fraction`: PASS (7); A constant descriptor cannot contribute predictive information.
- `constant_vacancy_concentration`: PASS (4); A constant descriptor cannot contribute predictive information.
- `constant_li_mean_charge_loss`: PASS (24); A constant descriptor cannot contribute predictive information.
- `constant_vacancy_concentration_percent`: PASS (4); A constant descriptor cannot contribute predictive information.
- `physical_target`: PASS (0); neutral vacancy formation energy >= 0 eV in this dataset; failed materials: none.
- `physical_band_gap`: PASS (0); band gap >= 0 eV; failed materials: none.
- `physical_li_mean_bader_charge`: PASS (0); positive Li charge loss; failed materials: none.
- `physical_li_atomic_fraction`: PASS (0); 0 < Li atomic fraction < 1; failed materials: none.
- `physical_density`: PASS (0); density > 0; failed materials: none.
- `physical_volume_per_atom`: PASS (0); volume per atom > 0; failed materials: none.
- `physical_volume_per_li`: PASS (0); volume per Li > 0; failed materials: none.
- `physical_bulk_mean_elf`: PASS (0); 0 <= mean ELF <= 1; failed materials: none.
- `physical_elf_high_fraction`: PASS (0); 0 <= ELF high fraction <= 100 percent; failed materials: none.
- `physical_elf_low_fraction`: PASS (0); 0 <= ELF low fraction <= 100 percent; failed materials: none.
- `physical_framework_oxygen_fraction`: PASS (0); 0 <= framework O fraction <= 1; failed materials: none.
- `physical_vacancy_concentration`: PASS (0); 0 <= nominal vacancy concentration <= 100 percent; failed materials: none.

## External formula audit and repairs

The workbook contains 1 external-link package(s). The formulas in both
`候选16 骨架电负性标准差 σ_χ` and `候选17 框架含氧原子分数 xO_fw` point to the same
external `W` cell by row. The cached values are therefore not independently trustworthy
for the oxygen-fraction field, even though the current file has cached numeric values
rather than literal `#NAME?` strings. The original workbook is unchanged.

- `σ_χ`: independently reconstructed as the population standard deviation of non-Li
  framework Pauling electronegativities, weighted by framework atom counts. The
  reconstructed values agree with the cached values within floating-point precision.
- `xO_fw`: independently reconstructed as the number of O atoms divided by all
  non-Li framework atoms using the chemical formula in `Material`. This repair is
  necessary because the workbook formula duplicates the sigma-chi external reference.
- The formulas, cached examples, and name-error counts are in
  `outputs/data/external_formula_audit.csv`; row-level differences are in
  `outputs/data/cleaned_dataset.csv`.
- Formula reconstruction uses the Pauling scale with the following elements present in
  this dataset: Li, B, C, N, O, F, Si, P, S, Se, Fe, Co, Ni, Cu, Mn, Zn, Mo, and Zr.

## Descriptor relationships

The full relation audit is in `outputs/data/feature_audit.csv`.

- `li_mean_bader_charge` vs `li_mean_charge_loss`: **same_quantity_different_precision**; maximum residual `4.34e-06`. Same physical quantity at different numerical precision; use one in the main matrix.
- `li_framework_electronegativity_difference` vs `mean_framework_electronegativity`: **deterministic_linear**; maximum residual `1.33e-15`. Affine redundancy; retain one representation in the main matrix.
- `mean_atomic_number` vs `mean_atomic_mass`: **very_high_statistical_correlation**; maximum residual `1.39`. Potentially unstable split importance; interpret jointly.
- `vacancy_concentration` vs `vacancy_concentration_percent`: **exact_duplicate**; maximum residual `0`. Do not include both in the main explanatory model.
- `volume_per_atom` vs `li_atomic_fraction`: **deterministic_nonlinear**; maximum residual `3.55e-15`. volume_per_li = volume_per_atom / li_atomic_fraction; physically distinct but algebraically linked.

The main explanatory matrix uses one representation of exact duplicates and affine
duplicates. `volume_per_li`, `volume_per_atom`, and `li_atomic_fraction` are retained
because their relation is deterministic but nonlinear (`volume_per_li = volume_per_atom /
li_atomic_fraction`); their joint importance is interpreted as a correlated descriptor
set, not as independent causal effects.

## Traceability

`outputs/data/feature_dictionary.csv` contains the original column name, canonical name,
unit, physical meaning, source, group, and whether the field is available to the main
feature matrix. No imputation is used to conceal a fully unavailable descriptor.
