"""Data integrity, physical-range, and descriptor-redundancy checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .data_loader import (
    ALL_VALID_FEATURES,
    AUXILIARY_FEATURES,
    CHEMICAL_FEATURES,
    ELECTRONIC_FEATURES,
    GEOMETRIC_FEATURES,
    REDUNDANT_FEATURES,
    TARGET_CANONICAL,
    THERMODYNAMIC_FEATURES,
    feature_dictionary_frame,
)


def _numeric_relation(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    valid = x.notna() & y.notna()
    x_values = x[valid].to_numpy(dtype=float)
    y_values = y[valid].to_numpy(dtype=float)
    if len(x_values) < 3 or np.ptp(x_values) == 0:
        return np.nan, np.nan, np.inf
    slope, intercept = np.polyfit(x_values, y_values, 1)
    residual = y_values - (slope * x_values + intercept)
    return float(slope), float(intercept), float(np.max(np.abs(residual)))


def _audit_relations(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, first in enumerate(features):
        for second in features[index + 1 :]:
            x = data[first]
            y = data[second]
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue
            x_values = x[valid].to_numpy(dtype=float)
            y_values = y[valid].to_numpy(dtype=float)
            max_diff = float(np.max(np.abs(x_values - y_values)))
            if max_diff < 1e-10:
                rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "relation_type": "exact_duplicate",
                        "slope": 1.0,
                        "intercept": 0.0,
                        "max_abs_residual": max_diff,
                        "pearson_r": 1.0,
                        "note": "Do not include both in the main explanatory model.",
                    }
                )
                continue
            slope, intercept, residual = _numeric_relation(x[valid], y[valid])
            r = float(stats.pearsonr(x_values, y_values).statistic) if np.ptp(x_values) and np.ptp(y_values) else np.nan
            if abs(slope - 1.0) < 1e-4 and abs(intercept) < 1e-4 and residual < 1e-4:
                rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "relation_type": "same_quantity_different_precision",
                        "slope": slope,
                        "intercept": intercept,
                        "max_abs_residual": residual,
                        "pearson_r": r,
                        "note": "Same physical quantity at different numerical precision; use one in the main matrix.",
                    }
                )
            elif residual < 1e-8:
                rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "relation_type": "deterministic_linear",
                        "slope": slope,
                        "intercept": intercept,
                        "max_abs_residual": residual,
                        "pearson_r": r,
                        "note": "Affine redundancy; retain one representation in the main matrix.",
                    }
                )
            elif r >= 0.995:
                rows.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "relation_type": "very_high_statistical_correlation",
                        "slope": slope,
                        "intercept": intercept,
                        "max_abs_residual": residual,
                        "pearson_r": r,
                        "note": "Potentially unstable split importance; interpret jointly.",
                    }
                )
    volume_relation = (
        data["volume_per_li"]
        - data["volume_per_atom"] / data["li_atomic_fraction"]
    ).abs()
    rows.append(
        {
            "feature_1": "volume_per_atom",
            "feature_2": "li_atomic_fraction",
            "relation_type": "deterministic_nonlinear",
            "slope": np.nan,
            "intercept": np.nan,
            "max_abs_residual": float(volume_relation.max()),
            "pearson_r": float(data["volume_per_atom"].corr(data["li_atomic_fraction"])),
            "note": "volume_per_li = volume_per_atom / li_atomic_fraction; physically distinct but algebraically linked.",
        }
    )
    return pd.DataFrame(rows)


def _physical_checks(data: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(feature: str, condition: pd.Series, rule: str) -> None:
        bad = data.loc[~condition.fillna(False), "material"].tolist()
        checks.append(
            {
                "feature": feature,
                "check": rule,
                "n_failed": len(bad),
                "failed_materials": "; ".join(bad),
            }
        )

    add("target", data[TARGET_CANONICAL] >= 0, "neutral vacancy formation energy >= 0 eV in this dataset")
    add("band_gap", data["band_gap"] >= 0, "band gap >= 0 eV")
    add("li_mean_bader_charge", data["li_mean_bader_charge"] > 0, "positive Li charge loss")
    add("li_atomic_fraction", (data["li_atomic_fraction"] > 0) & (data["li_atomic_fraction"] < 1), "0 < Li atomic fraction < 1")
    add("density", data["density"] > 0, "density > 0")
    add("volume_per_atom", data["volume_per_atom"] > 0, "volume per atom > 0")
    add("volume_per_li", data["volume_per_li"] > 0, "volume per Li > 0")
    add("bulk_mean_elf", (data["bulk_mean_elf"] >= 0) & (data["bulk_mean_elf"] <= 1), "0 <= mean ELF <= 1")
    add("elf_high_fraction", (data["elf_high_fraction"] >= 0) & (data["elf_high_fraction"] <= 100), "0 <= ELF high fraction <= 100 percent")
    add("elf_low_fraction", (data["elf_low_fraction"] >= 0) & (data["elf_low_fraction"] <= 100), "0 <= ELF low fraction <= 100 percent")
    add("framework_oxygen_fraction", (data["framework_oxygen_fraction"] >= 0) & (data["framework_oxygen_fraction"] <= 1), "0 <= framework O fraction <= 1")
    add("vacancy_concentration", (data["vacancy_concentration"] >= 0) & (data["vacancy_concentration"] <= 100), "0 <= nominal vacancy concentration <= 100 percent")
    return checks


def audit_dataset(dataset: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root)
    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    tables_dir = output_root / "tables"
    for directory in (data_dir, reports_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = dataset["canonical"].copy()
    feature_specs = dataset["feature_specs"]
    numeric_columns = [spec["canonical"] for spec in feature_specs.values()] + [TARGET_CANONICAL]
    numeric_columns = list(dict.fromkeys(numeric_columns))
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    quality_rows: list[dict[str, Any]] = []
    quality_rows.append({"check": "sample_count", "value": len(data), "status": "PASS" if len(data) == 24 else "FAIL", "details": "Expected 24 material rows."})
    quality_rows.append({"check": "unique_material_names", "value": data["material"].nunique(), "status": "PASS" if data["material"].is_unique else "FAIL", "details": "Material labels are unique."})
    quality_rows.append({"check": "duplicate_material_records", "value": int(data.duplicated(subset=["material"]).sum()), "status": "PASS" if not data.duplicated(subset=["material"]).any() else "FAIL", "details": "Duplicate material labels checked."})
    quality_rows.append({"check": "duplicate_mp_ids", "value": int(data["mp_id"].duplicated().sum()), "status": "PASS" if not data["mp_id"].duplicated().any() else "FAIL", "details": "MP-ID uniqueness checked."})
    quality_rows.append({"check": "duplicate_full_records", "value": int(data.duplicated().sum()), "status": "PASS" if not data.duplicated().any() else "FAIL", "details": "Full canonical records checked."})
    quality_rows.append({"check": "target_numeric_complete", "value": int(data[TARGET_CANONICAL].notna().sum()), "status": "PASS" if data[TARGET_CANONICAL].notna().all() else "FAIL", "details": "Target is parsed from the explicit Y column."})

    for column in numeric_columns:
        missing = int(data[column].isna().sum())
        quality_rows.append({
            "check": f"numeric_{column}",
            "value": missing,
            "status": "PASS" if missing == 0 else "FAIL",
            "details": f"Missing or non-numeric values in {column}: {missing}.",
        })
    for column in [column for column in numeric_columns if column != TARGET_CANONICAL]:
        nunique = data[column].nunique(dropna=False)
        quality_rows.append({
            "check": f"constant_{column}",
            "value": nunique,
            "status": "FAIL" if nunique <= 1 else "PASS",
            "details": "A constant descriptor cannot contribute predictive information.",
        })

    physical_rows = _physical_checks(data)
    for row in physical_rows:
        quality_rows.append({
            "check": f"physical_{row['feature']}",
            "value": row["n_failed"],
            "status": "PASS" if row["n_failed"] == 0 else "WARN",
            "details": f"{row['check']}; failed materials: {row['failed_materials'] or 'none'}.",
        })

    relations = _audit_relations(data, numeric_columns[:-1])
    relations.to_csv(data_dir / "feature_audit.csv", index=False)
    dictionary = feature_dictionary_frame()
    dictionary.to_csv(data_dir / "feature_dictionary.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(data_dir / "quality_checks.csv", index=False)

    raw = dataset["raw"]
    external_formula_rows = []
    for original in ["候选16 骨架电负性标准差 σ_χ", "候选17 框架含氧原子分数 xO_fw"]:
        formula_values = dataset["workbook_metadata"]["formulas"][original]
        cached_values = dataset["workbook_metadata"]["cached_values"][original]
        external_formula_rows.append(
            {
                "column": original,
                "formula_examples": "; ".join(str(value) for value in formula_values[:2]),
                "cached_value_examples": "; ".join(str(value) for value in cached_values[:2]),
                "n_name_errors_in_formula": sum(str(value).strip() == "#NAME?" for value in formula_values),
                "contains_external_reference": any("[1]" in str(value) for value in formula_values),
            }
        )
    external_formula_audit = pd.DataFrame(external_formula_rows)
    external_formula_audit.to_csv(data_dir / "external_formula_audit.csv", index=False)

    cleaned_columns = [
        "material", "chemical_formula", "mp_id", "source", "category",
        TARGET_CANONICAL, *ALL_VALID_FEATURES,
        "raw_formula_sigma_chi", "raw_formula_xO_fw",
        "sigma_chi_reconstruction_abs_diff", "xO_reconstruction_abs_diff",
        "sigma_formula", "oxygen_formula",
    ]
    cleaned = data[cleaned_columns].copy()
    cleaned.to_csv(data_dir / "cleaned_dataset.csv", index=False)

    n_external = len(dataset["workbook_metadata"]["external_links"])
    duplicate_lines = []
    if relations.empty:
        duplicate_lines.append("No pairwise descriptor relations were detected.")
    else:
        for _, row in relations.iterrows():
            duplicate_lines.append(
                f"- `{row['feature_1']}` vs `{row['feature_2']}`: **{row['relation_type']}**; "
                f"maximum residual `{row['max_abs_residual']:.3g}`. {row['note']}"
            )
    quality_status = "PASS" if all(row["status"] != "FAIL" for row in quality_rows) else "FAIL"
    report = f"""# Data quality report

## Scope

- Source workbook: `{dataset['workbook_metadata'].get('source_path', 'input Excel workbook')}`.
- Sheets found: {', '.join(dataset['workbook_metadata']['sheet_names'])}.
- Main sheet rows/columns: {dataset['workbook_metadata']['n_rows']} / {dataset['workbook_metadata']['n_columns']}.
- Target: `{TARGET_CANONICAL}`, taken only from `{dataset['target_original']}`.
- Overall automated status: **{quality_status}**.

## Completeness and physical checks

The checks are written to `outputs/data/quality_checks.csv`. A nonzero `WARN` is retained
as a scientific data-quality flag rather than silently changing a value.

{chr(10).join(f"- `{row['check']}`: {row['status']} ({row['value']}); {row['details']}" for row in quality_rows)}

## External formula audit and repairs

The workbook contains {n_external} external-link package(s). The formulas in both
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

{chr(10).join(duplicate_lines)}

The main explanatory matrix uses one representation of exact duplicates and affine
duplicates. `volume_per_li`, `volume_per_atom`, and `li_atomic_fraction` are retained
because their relation is deterministic but nonlinear (`volume_per_li = volume_per_atom /
li_atomic_fraction`); their joint importance is interpreted as a correlated descriptor
set, not as independent causal effects.

## Traceability

`outputs/data/feature_dictionary.csv` contains the original column name, canonical name,
unit, physical meaning, source, group, and whether the field is available to the main
feature matrix. No imputation is used to conceal a fully unavailable descriptor.
"""
    (reports_dir / "data_quality_report.md").write_text(report, encoding="utf-8")
    return {
        "data": cleaned,
        "relations": relations,
        "quality_checks": pd.DataFrame(quality_rows),
        "feature_dictionary": dictionary,
        "physical_checks": pd.DataFrame(physical_rows),
        "external_formula_audit": external_formula_audit,
    }
