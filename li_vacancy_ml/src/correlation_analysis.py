"""Correlation statistics with small-sample bootstrap intervals and FDR control."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .data_loader import ALL_VALID_FEATURES, TARGET_CANONICAL


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return q_values
    indices = np.flatnonzero(valid)
    order = indices[np.argsort(p_values[valid])]
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values[order] = np.clip(adjusted, 0, 1)
    return q_values


def _coefficient(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    replicates: int,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(x)
    for _ in range(replicates):
        sample = rng.integers(0, n, size=n)
        value = _coefficient(x[sample], y[sample], method)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan, 0
    interval = np.percentile(values, [2.5, 97.5])
    return float(interval[0]), float(interval[1]), len(values)


def _single_correlation(
    x: pd.Series,
    y: pd.Series,
    method: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    valid = x.notna() & y.notna()
    x_values = x.loc[valid].to_numpy(dtype=float)
    y_values = y.loc[valid].to_numpy(dtype=float)
    if len(x_values) < 3 or np.std(x_values) == 0 or np.std(y_values) == 0:
        return {
            "n": int(len(x_values)),
            "coefficient": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_valid": 0,
        }
    if method == "pearson":
        result = stats.pearsonr(x_values, y_values)
    else:
        result = stats.spearmanr(x_values, y_values)
    ci_low, ci_high, bootstrap_valid = bootstrap_correlation(
        x_values, y_values, method, replicates, seed
    )
    return {
        "n": int(len(x_values)),
        "coefficient": float(result.statistic),
        "p_value": float(result.pvalue),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_valid": int(bootstrap_valid),
    }


def run_correlation_analysis(
    data: pd.DataFrame,
    output_root: str | Path,
    feature_dictionary: pd.DataFrame,
    bootstrap_replicates: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dictionary = feature_dictionary.set_index("canonical_name")
    rows: list[dict[str, object]] = []
    for method in ("pearson", "spearman"):
        method_rows = []
        for position, feature in enumerate(ALL_VALID_FEATURES):
            result = _single_correlation(
                data[feature],
                data[TARGET_CANONICAL],
                method,
                bootstrap_replicates,
                seed + position * 17 + (0 if method == "pearson" else 10000),
            )
            row = {
                "feature": feature,
                "group": dictionary.loc[feature, "group"],
                "subgroup": dictionary.loc[feature, "subgroup"],
                "unit": dictionary.loc[feature, "unit"],
                "method": method,
                **result,
            }
            method_rows.append(row)
        q_values = benjamini_hochberg(
            np.array([row["p_value"] for row in method_rows], dtype=float)
        )
        for row, q_value in zip(method_rows, q_values):
            row["q_value_bh"] = float(q_value) if np.isfinite(q_value) else np.nan
            rows.append(row)

    long_table = pd.DataFrame(rows)
    long_table.to_csv(tables_dir / "feature_correlations.csv", index=False)
    pearson = long_table[long_table["method"] == "pearson"].copy()
    spearman = long_table[long_table["method"] == "spearman"].copy()
    pearson.to_csv(tables_dir / "pearson_correlations.csv", index=False)
    spearman.to_csv(tables_dir / "spearman_correlations.csv", index=False)

    x_x = data[ALL_VALID_FEATURES].corr(method="spearman")
    x_x.to_csv(tables_dir / "descriptor_spearman_matrix.csv")
    x_x_pearson = data[ALL_VALID_FEATURES].corr(method="pearson")
    x_x_pearson.to_csv(tables_dir / "descriptor_pearson_matrix.csv")

    eg = pearson.loc[pearson["feature"] == "band_gap"].iloc[0]
    eg_spearman = spearman.loc[spearman["feature"] == "band_gap"].iloc[0]
    return {
        "long": long_table,
        "pearson": pearson,
        "spearman": spearman,
        "x_x_spearman": x_x,
        "x_x_pearson": x_x_pearson,
        "eg_pearson": eg.to_dict(),
        "eg_spearman": eg_spearman.to_dict(),
    }

