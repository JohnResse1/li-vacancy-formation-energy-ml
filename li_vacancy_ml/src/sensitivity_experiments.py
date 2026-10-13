"""Leakage-safe feature-subset and target-scale sensitivity experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import TransformedTargetRegressor

from .data_loader import ALL_INTRINSIC_FEATURES, TARGET_CANONICAL
from .models import (
    _jsonable,
    _param_grid,
    _pipeline,
    evaluate_predictions,
    experiment_specs,
)


TARGET_TRANSFORMS: dict[str, tuple[Callable[[np.ndarray], np.ndarray] | None, Callable[[np.ndarray], np.ndarray] | None]] = {
    "identity": (None, None),
    "sqrt": (np.sqrt, np.square),
    "log": (np.log, np.exp),
}


def _with_target_transform(estimator: Any, target_transform: str) -> Any:
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError(f"Unknown target transform: {target_transform}")
    function, inverse_function = TARGET_TRANSFORMS[target_transform]
    if function is None or inverse_function is None:
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator,
        func=function,
        inverse_func=inverse_function,
        check_inverse=True,
    )


def _subset_pipeline(model_name: str, target_transform: str) -> Any:
    base = _pipeline(model_name)
    # Insert feature selection after imputation and scaling. Both operations are
    # therefore fitted inside every inner-CV training fold.
    base.steps.insert(-1, ("selector", SelectKBest(score_func=f_regression)))
    return _with_target_transform(base, target_transform)


def _subset_param_grid(
    model_name: str,
    n_features: int,
    n_train: int,
    inner_folds: int,
    target_transform: str,
    k_values: list[int | str],
) -> dict[str, list[Any]]:
    valid_k: list[int | str] = []
    for value in k_values:
        if value == "all":
            valid_k.append(value)
        elif isinstance(value, int) and 1 <= value <= n_features:
            valid_k.append(value)
    if not valid_k:
        valid_k = [min(3, n_features)]
    prefix = "" if target_transform == "identity" else "regressor__"
    grid: dict[str, list[Any]] = {f"{prefix}selector__k": list(dict.fromkeys(valid_k))}
    if model_name == "Ridge":
        grid[f"{prefix}model__alpha"] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    elif model_name == "SVR_linear":
        grid[f"{prefix}model__C"] = [0.01, 0.1, 1.0, 10.0, 100.0]
        grid[f"{prefix}model__epsilon"] = [0.01, 0.1, 0.2]
    else:
        raise ValueError("Subset search currently supports Ridge and SVR_linear only")
    return grid


def _fixed_param_grid(model_name: str, target_transform: str, n_features: int, n_train: int, inner_folds: int) -> dict[str, list[Any]]:
    grid = _param_grid(model_name, n_features, n_train, inner_folds)
    if target_transform == "identity":
        return grid
    return {f"regressor__{key}": values for key, values in grid.items()}


def _unwrap_regressor(fitted: Any, target_transform: str) -> Any:
    return fitted.regressor_ if target_transform != "identity" else fitted


def _selected_features(fitted: Any, feature_pool: list[str], target_transform: str) -> list[str]:
    regressor = _unwrap_regressor(fitted, target_transform)
    selector = regressor.named_steps["selector"]
    return [feature for feature, selected in zip(feature_pool, selector.get_support()) if selected]


def _nested_fit_predict(
    data: pd.DataFrame,
    feature_pool: list[str],
    model_name: str,
    target_transform: str,
    inner_folds: int,
    seed: int,
    k_values: list[int | str],
    subset_search: bool,
    experiment_name: str,
    experiment_label: str,
) -> dict[str, Any]:
    X = data[feature_pool].to_numpy(dtype=float)
    y = data[TARGET_CANONICAL].to_numpy(dtype=float)
    predictions: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    selected_sets: list[dict[str, Any]] = []

    for test_index in range(len(data)):
        train_indices = np.array([index for index in range(len(data)) if index != test_index])
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[[test_index]], y[[test_index]]
        if subset_search:
            estimator = _subset_pipeline(model_name, target_transform)
            parameter_grid = _subset_param_grid(
                model_name,
                len(feature_pool),
                len(train_indices),
                inner_folds,
                target_transform,
                k_values,
            )
        else:
            estimator = _with_target_transform(_pipeline(model_name), target_transform)
            parameter_grid = _fixed_param_grid(
                model_name,
                target_transform,
                len(feature_pool),
                len(train_indices),
                inner_folds,
            )
        folds = min(inner_folds, len(train_indices))
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="neg_mean_absolute_error",
            cv=KFold(n_splits=folds, shuffle=True, random_state=seed),
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        prediction = float(np.asarray(fitted.predict(X_test)).ravel()[0])
        material = str(data.iloc[test_index]["material"])
        selected = _selected_features(fitted, feature_pool, target_transform) if subset_search else list(feature_pool)
        best_params = _jsonable(search.best_params_)
        predictions.append(
            {
                "row_index": int(test_index),
                "material": material,
                "y_true": float(y_test[0]),
                "y_pred": prediction,
                "residual": prediction - float(y_test[0]),
                "absolute_error": abs(prediction - float(y_test[0])),
                "experiment": experiment_name,
                "experiment_label": experiment_label,
                "model": model_name,
                "target_transform": target_transform,
                "selected_features": ";".join(selected),
                "selected_feature_count": len(selected),
            }
        )
        parameters.append(
            {
                "row_index": int(test_index),
                "material": material,
                "experiment": experiment_name,
                "model": model_name,
                "target_transform": target_transform,
                "inner_cv_mae": float(-search.best_score_),
                "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
                "selected_features": ";".join(selected),
                "selected_feature_count": len(selected),
            }
        )
        selected_sets.append(
            {
                "experiment": experiment_name,
                "model": model_name,
                "target_transform": target_transform,
                "outer_row_index": int(test_index),
                "material": material,
                "selected_features": selected,
            }
        )

    prediction_frame = pd.DataFrame(predictions)
    metrics = evaluate_predictions(prediction_frame["y_true"].to_numpy(), prediction_frame["y_pred"].to_numpy())
    return {
        "experiment": experiment_name,
        "experiment_label": experiment_label,
        "model": model_name,
        "target_transform": target_transform,
        "feature_pool": feature_pool,
        "predictions": prediction_frame,
        "parameters": pd.DataFrame(parameters),
        "selected_sets": selected_sets,
        "metrics": metrics,
    }


def _selection_frequency(selected_sets: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not selected_sets:
        return pd.DataFrame(columns=["experiment", "model", "target_transform", "feature", "selected_count", "selection_frequency"])
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in selected_sets:
        key = (item["experiment"], item["model"], item["target_transform"])
        groups.setdefault(key, []).append(item)
    for (experiment, model, transform), items in groups.items():
        n_outer = len(items)
        counts: dict[str, int] = {}
        for item in items:
            for feature in item["selected_features"]:
                counts[feature] = counts.get(feature, 0) + 1
        for feature, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
            rows.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "target_transform": transform,
                    "feature": feature,
                    "selected_count": count,
                    "selection_frequency": count / n_outer,
                }
            )
    return pd.DataFrame(rows)


def run_sensitivity_experiments(
    data: pd.DataFrame,
    output_root: str | Path,
    inner_folds: int = 5,
    seed: int = 42,
    target_transforms: list[str] | None = None,
    subset_k_values: list[int | str] | None = None,
) -> dict[str, Any]:
    """Run constrained feature selection and target-scale sensitivity checks.

    All target-dependent selection occurs inside the inner CV of each outer LOOCV
    fold. Results are intentionally reported as exploratory extensions rather
    than replacing the prespecified M1-M6 benchmark.
    """

    output_root = Path(output_root)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    (output_root / "predictions").mkdir(parents=True, exist_ok=True)
    (output_root / "models").mkdir(parents=True, exist_ok=True)
    target_transforms = target_transforms or ["identity", "sqrt", "log"]
    subset_k_values = subset_k_values or [1, 2, 3, 4, 5, 6, 8, 10, "all"]
    invalid = set(target_transforms) - set(TARGET_TRANSFORMS)
    if invalid:
        raise ValueError(f"Unknown target transforms: {sorted(invalid)}")

    subset_results: list[dict[str, Any]] = []
    fixed_results: list[dict[str, Any]] = []
    all_selected_sets: list[dict[str, Any]] = []
    subset_oof: list[pd.DataFrame] = []
    fixed_oof: list[pd.DataFrame] = []
    subset_parameters: list[pd.DataFrame] = []
    fixed_parameters: list[pd.DataFrame] = []

    feature_pool = list(ALL_INTRINSIC_FEATURES)
    for transform in target_transforms:
        for model_name in ["Ridge", "SVR_linear"]:
            result = _nested_fit_predict(
                data,
                feature_pool,
                model_name,
                transform,
                inner_folds,
                seed,
                subset_k_values,
                subset_search=True,
                experiment_name="FS1",
                experiment_label="Nested SelectKBest subset search over all intrinsic features",
            )
            subset_results.append(result)
            subset_oof.append(result["predictions"])
            subset_parameters.append(result["parameters"])
            all_selected_sets.extend(result["selected_sets"])

    # A small, prespecified set of existing feature groups is used for the target
    # transform check. This avoids selecting the transform and feature set from
    # the same outer test predictions.
    fixed_names = {"M1", "M3", "M4", "M5", "M6"}
    fixed_experiments = [experiment for experiment in experiment_specs() if experiment.name in fixed_names]
    for experiment in fixed_experiments:
        for transform in target_transforms:
            for model_name in ["Ridge", "SVR_linear"]:
                result = _nested_fit_predict(
                    data,
                    list(experiment.features),
                    model_name,
                    transform,
                    inner_folds,
                    seed,
                    subset_k_values,
                    subset_search=False,
                    experiment_name=experiment.name,
                    experiment_label=experiment.label,
                )
                fixed_results.append(result)
                frame = result["predictions"].copy()
                fixed_oof.append(frame)
                fixed_parameters.append(result["parameters"])

    def comparison_frame(results: list[dict[str, Any]], search_type: str) -> pd.DataFrame:
        rows = []
        for result in results:
            selected_counts = result["predictions"]["selected_feature_count"] if "selected_feature_count" in result["predictions"] else pd.Series(dtype=float)
            selected_series = result["predictions"].get("selected_features", pd.Series(dtype=str))
            selected_mode = selected_series.mode().iloc[0] if not selected_series.empty else ";".join(result["feature_pool"])
            selected_mode_frequency = float((selected_series == selected_mode).mean()) if not selected_series.empty else 1.0
            rows.append(
                {
                    "search_type": search_type,
                    "experiment": result["experiment"],
                    "experiment_label": result["experiment_label"],
                    "model": result["model"],
                    "target_transform": result["target_transform"],
                    "feature_pool_count": len(result["feature_pool"]),
                    "feature_count": float(selected_counts.median()) if not selected_counts.empty else len(result["feature_pool"]),
                    "feature_pool": ";".join(result["feature_pool"]),
                    "selected_feature_mode": selected_mode,
                    "selected_mode_frequency": selected_mode_frequency,
                    **result["metrics"],
                }
            )
        return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)

    subset_comparison = comparison_frame(subset_results, "nested_select_k_best")
    fixed_comparison = comparison_frame(fixed_results, "fixed_feature_set")
    subset_prediction_frame = pd.concat(subset_oof, ignore_index=True)
    fixed_prediction_frame = pd.concat(fixed_oof, ignore_index=True)
    subset_parameter_frame = pd.concat(subset_parameters, ignore_index=True)
    fixed_parameter_frame = pd.concat(fixed_parameters, ignore_index=True)
    selection_frequency = _selection_frequency(all_selected_sets)

    subset_comparison.to_csv(output_root / "tables" / "feature_subset_comparison.csv", index=False)
    fixed_comparison.to_csv(output_root / "tables" / "target_transform_comparison.csv", index=False)
    subset_prediction_frame.to_csv(output_root / "predictions" / "feature_subset_oof_predictions.csv", index=False)
    fixed_prediction_frame.to_csv(output_root / "predictions" / "target_transform_oof_predictions.csv", index=False)
    subset_parameter_frame.to_csv(output_root / "tables" / "feature_subset_outer_parameters.csv", index=False)
    fixed_parameter_frame.to_csv(output_root / "tables" / "target_transform_outer_parameters.csv", index=False)
    selection_frequency.to_csv(output_root / "tables" / "feature_selection_frequency.csv", index=False)
    metadata = {
        "outer_cv": "Leave-One-Material-Out",
        "inner_cv": f"KFold(n_splits={inner_folds}, shuffle=True, random_state={seed})",
        "feature_pool": feature_pool,
        "target_transforms": target_transforms,
        "subset_k_values": subset_k_values,
        "subset_models": ["Ridge", "SVR_linear"],
        "fixed_transform_experiments": [experiment.name for experiment in fixed_experiments],
        "selection_is_nested": True,
    }
    (output_root / "models" / "sensitivity_experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "subset_results": subset_results,
        "fixed_results": fixed_results,
        "subset_comparison": subset_comparison,
        "fixed_comparison": fixed_comparison,
        "subset_oof": subset_prediction_frame,
        "fixed_oof": fixed_prediction_frame,
        "subset_parameters": subset_parameter_frame,
        "fixed_parameters": fixed_parameter_frame,
        "selection_frequency": selection_frequency,
    }
