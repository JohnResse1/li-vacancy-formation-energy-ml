"""Feature-group ablations, paired error comparisons, and Ridge stability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import (
    CHEMICAL_FEATURES,
    ELECTRONIC_FEATURES,
    GEOMETRIC_FEATURES,
    TARGET_CANONICAL,
    THERMODYNAMIC_FEATURES,
)
from .models import Experiment, experiment_specs


def _bootstrap_mean_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=float)
    for index in range(replicates):
        means[index] = np.mean(values[rng.integers(0, len(values), size=len(values))])
    interval = np.percentile(means, [2.5, 97.5])
    return float(interval[0]), float(interval[1])


def _result_lookup(benchmark: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(result["experiment"].name, result["model_name"]): result for result in benchmark["results"]}


def _paired_comparison(
    lookup: dict[tuple[str, str], dict[str, Any]],
    baseline: str,
    extended: str,
    model: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    base = lookup.get((baseline, model))
    ext = lookup.get((extended, model))
    if base is None or ext is None:
        return None
    base_frame = base["predictions"][["row_index", "material", "y_true", "y_pred", "absolute_error"]].copy()
    ext_frame = ext["predictions"][["row_index", "y_pred", "absolute_error"]].copy()
    merged = base_frame.merge(ext_frame, on="row_index", suffixes=("_baseline", "_extended"))
    difference = merged["absolute_error_baseline"] - merged["absolute_error_extended"]
    ci_low, ci_high = _bootstrap_mean_interval(difference.to_numpy(), bootstrap_replicates, seed)
    return {
        "baseline": baseline,
        "extended": extended,
        "model": model,
        "baseline_MAE": float(merged["absolute_error_baseline"].mean()),
        "extended_MAE": float(merged["absolute_error_extended"].mean()),
        "delta_MAE_baseline_minus_extended": float(difference.mean()),
        "paired_delta_ci_low": ci_low,
        "paired_delta_ci_high": ci_high,
        "median_paired_delta": float(difference.median()),
        "n_extended_better": int((difference > 0).sum()),
        "n_baseline_better": int((difference < 0).sum()),
        "n_ties": int((difference == 0).sum()),
    }


def _parse_alpha(result: dict[str, Any]) -> float:
    values = result["fold_parameters"]["best_params"].tolist()
    alphas = []
    for value in values:
        params = json.loads(value)
        if "model__alpha" in params:
            alphas.append(float(params["model__alpha"]))
    return float(pd.Series(alphas).mode().iloc[0]) if alphas else 1.0


def _ridge_coefficients(benchmark: dict[str, Any], output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = _result_lookup(benchmark)
    experiments = {experiment.name: experiment for experiment in experiment_specs()}
    for (experiment_name, model_name), result in lookup.items():
        if model_name != "Ridge":
            continue
        experiment = experiments[experiment_name]
        for fold, fitted in enumerate(result["fold_models"]):
            coefficients = np.asarray(fitted.named_steps["model"].coef_).ravel()
            for feature, coefficient in zip(experiment.features, coefficients):
                rows.append(
                    {
                        "experiment": experiment_name,
                        "model": model_name,
                        "fold": fold,
                        "material_left_out": result["predictions"].iloc[fold]["material"],
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                        "absolute_standardized_coefficient": float(abs(coefficient)),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "tables" / "ridge_coefficients_by_fold.csv", index=False)
    return frame


def _coefficient_summary(coefficients: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    group_map = {}
    for feature in ELECTRONIC_FEATURES:
        group_map[feature] = "E"
    for feature in GEOMETRIC_FEATURES:
        group_map[feature] = "G"
    for feature in CHEMICAL_FEATURES:
        group_map[feature] = "C"
    for feature in THERMODYNAMIC_FEATURES:
        group_map[feature] = "T"
    grouped = coefficients.groupby(["experiment", "feature"])["standardized_coefficient"]
    summary = grouped.agg(["mean", "median", "std", "min", "max"]).reset_index()
    summary["group"] = summary["feature"].map(group_map).fillna("AUX")
    sign_consistency = grouped.apply(
        lambda values: float(np.mean(np.sign(values) == np.sign(np.median(values))))
    ).rename("sign_consistency").reset_index()
    summary = summary.merge(sign_consistency, on=["experiment", "feature"], how="left")
    summary.to_csv(output_root / "tables" / "ridge_coefficient_stability.csv", index=False)
    group_summary = (
        coefficients.assign(group=coefficients["feature"].map(group_map).fillna("AUX"), abs_coefficient=coefficients["absolute_standardized_coefficient"])
        .groupby(["experiment", "group"], as_index=False)["abs_coefficient"]
        .agg(mean_abs_coefficient="mean", median_abs_coefficient="median")
    )
    group_summary.to_csv(output_root / "tables" / "ridge_group_coefficient_stability.csv", index=False)
    return summary


def _bootstrap_ridge_coefficients(
    data: pd.DataFrame,
    experiment: Experiment,
    alpha: float,
    replicates: int,
    seed: int,
    output_root: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = data[list(experiment.features)].to_numpy(dtype=float)
    y = data[TARGET_CANONICAL].to_numpy(dtype=float)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        indices = rng.integers(0, len(data), size=len(data))
        pipeline.fit(X[indices], y[indices])
        coefficients = np.asarray(pipeline.named_steps["model"].coef_).ravel()
        for feature, coefficient in zip(experiment.features, coefficients):
            rows.append(
                {
                    "replicate": replicate,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "tables" / "ridge_bootstrap_coefficients.csv", index=False)
    summary = frame.groupby("feature", as_index=False)["standardized_coefficient"].agg(
        mean="mean",
        median="median",
        ci_low=lambda values: np.percentile(values, 2.5),
        ci_high=lambda values: np.percentile(values, 97.5),
        positive_fraction=lambda values: np.mean(values > 0),
    )
    summary["alpha_used"] = alpha
    summary.to_csv(output_root / "tables" / "ridge_bootstrap_coefficient_summary.csv", index=False)
    return summary


def run_feature_ablation(
    data: pd.DataFrame,
    benchmark: dict[str, Any],
    output_root: str | Path,
    bootstrap_replicates: int = 2000,
    coefficient_bootstrap_replicates: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    output_root = Path(output_root)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    comparison = benchmark["comparison"]
    pairs = [
        ("M3", "M4", "Eg increment over composition+geometry"),
        ("M3", "M5", "All electronic increment over composition+geometry"),
        ("M8", "M9", "Eg increment after thermodynamics"),
        ("M9", "M10", "Bader/ELF increment after Eg and thermodynamics"),
        ("M1", "M4", "Eg only versus composition+geometry+Eg"),
        ("M7", "M8", "Composition+geometry increment over thermodynamics"),
    ]
    ablation_rows: list[dict[str, Any]] = []
    for baseline, extended, question in pairs:
        for model in sorted(set(comparison.loc[comparison["experiment"].isin([baseline, extended]), "model"])):
            base = comparison[(comparison["experiment"] == baseline) & (comparison["model"] == model)]
            ext = comparison[(comparison["experiment"] == extended) & (comparison["model"] == model)]
            if base.empty or ext.empty:
                continue
            base_row = base.iloc[0]
            ext_row = ext.iloc[0]
            paired = _paired_comparison(
                _result_lookup(benchmark), baseline, extended, model,
                bootstrap_replicates, seed + len(ablation_rows),
            )
            ablation_rows.append(
                {
                    "question": question,
                    "baseline": baseline,
                    "extended": extended,
                    "model": model,
                    "baseline_MAE": float(base_row["MAE"]),
                    "extended_MAE": float(ext_row["MAE"]),
                    "delta_MAE_baseline_minus_extended": float(base_row["MAE"] - ext_row["MAE"]),
                    "delta_RMSE_baseline_minus_extended": float(base_row["RMSE"] - ext_row["RMSE"]),
                    "delta_R2_extended_minus_baseline": float(ext_row["R2"] - base_row["R2"]),
                    "paired_delta_ci_low": paired["paired_delta_ci_low"] if paired else np.nan,
                    "paired_delta_ci_high": paired["paired_delta_ci_high"] if paired else np.nan,
                    "n_extended_better": paired["n_extended_better"] if paired else np.nan,
                }
            )
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(output_root / "tables" / "feature_group_ablation.csv", index=False)
    paired_rows = [
        row for row in (
            _paired_comparison(_result_lookup(benchmark), base, ext, model, bootstrap_replicates, seed + i)
            for i, (base, ext, model) in enumerate(
                (pair[0], pair[1], "Ridge") for pair in pairs
            )
        ) if row is not None
    ]
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_root / "tables" / "paired_error_comparisons.csv", index=False)

    coefficients = _ridge_coefficients(benchmark, output_root)
    coefficient_summary = _coefficient_summary(coefficients, output_root)
    lookup = _result_lookup(benchmark)
    m6_ridge = lookup.get(("M6", "Ridge"))
    bootstrap_summary = pd.DataFrame()
    if m6_ridge is not None:
        alpha = _parse_alpha(m6_ridge)
        bootstrap_summary = _bootstrap_ridge_coefficients(
            data,
            next(experiment for experiment in experiment_specs() if experiment.name == "M6"),
            alpha,
            coefficient_bootstrap_replicates,
            seed,
            output_root,
        )
    return {
        "ablation": ablation,
        "paired": paired,
        "coefficients": coefficients,
        "coefficient_summary": coefficient_summary,
        "bootstrap_summary": bootstrap_summary,
    }
