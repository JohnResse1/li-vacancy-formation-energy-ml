"""Nested LOOCV model benchmark for small-sample regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data_loader import (
    ALL_INTRINSIC_FEATURES,
    AUXILIARY_FEATURES,
    CHEMICAL_FEATURES,
    ELECTRONIC_FEATURES,
    GEOMETRIC_FEATURES,
    TARGET_CANONICAL,
    THERMODYNAMIC_FEATURES,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    label: str
    features: tuple[str, ...]


def _ordered_unique(features: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(features))


def experiment_specs() -> list[Experiment]:
    cg = CHEMICAL_FEATURES + GEOMETRIC_FEATURES
    cgt = cg + THERMODYNAMIC_FEATURES
    return [
        Experiment("M1", "Eg only", ("band_gap",)),
        Experiment("M2", "All electronic", tuple(ELECTRONIC_FEATURES)),
        Experiment("M3", "Composition + geometry", _ordered_unique(cg)),
        Experiment("M4", "Composition + geometry + Eg", _ordered_unique(cg + ["band_gap"])),
        Experiment("M5", "Composition + geometry + all electronic", _ordered_unique(cg + ELECTRONIC_FEATURES)),
        Experiment("M6", "All intrinsic features", _ordered_unique(ALL_INTRINSIC_FEATURES)),
        Experiment("M7", "Thermodynamic only", tuple(THERMODYNAMIC_FEATURES)),
        Experiment("M8", "Composition + geometry + thermodynamics", _ordered_unique(cgt)),
        Experiment("M9", "M8 + Eg", _ordered_unique(cgt + ["band_gap"])),
        Experiment("M10", "M8 + all electronic", _ordered_unique(cgt + ELECTRONIC_FEATURES)),
        Experiment("M11", "Bader + ELF", ("li_mean_bader_charge", "bulk_mean_elf", "elf_high_fraction", "elf_low_fraction")),
        Experiment("M12", "Eg + Bader + ELF", ("band_gap", "li_mean_bader_charge", "bulk_mean_elf", "elf_high_fraction", "elf_low_fraction")),
        Experiment("M6_control", "All intrinsic + vacancy concentration control", _ordered_unique(ALL_INTRINSIC_FEATURES + AUXILIARY_FEATURES)),
    ]


def _pipeline(model_name: str) -> Pipeline:
    if model_name == "Dummy":
        estimator = DummyRegressor(strategy="mean")
    elif model_name == "Ridge":
        estimator = Ridge()
    elif model_name == "PLS":
        estimator = PLSRegression(scale=False, max_iter=1000)
    elif model_name == "SVR_linear":
        estimator = SVR(kernel="linear")
    elif model_name == "SVR_rbf":
        estimator = SVR(kernel="rbf")
    elif model_name == "Eg_linear":
        estimator = LinearRegression()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _param_grid(model_name: str, n_features: int, n_train: int, inner_folds: int) -> dict[str, list[Any]]:
    if model_name == "Dummy" or model_name == "Eg_linear":
        return {}
    if model_name == "Ridge":
        return {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    if model_name == "PLS":
        # The smallest inner training fold is approximately n_train - ceil(n_train/k).
        smallest_inner_train = max(2, n_train - int(np.ceil(n_train / inner_folds)))
        max_components = max(1, min(8, n_features, smallest_inner_train))
        return {"model__n_components": list(range(1, max_components + 1))}
    if model_name == "SVR_linear":
        return {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__epsilon": [0.01, 0.1, 0.2],
        }
    if model_name == "SVR_rbf":
        return {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": ["scale", 0.01, 0.1, 1.0],
            "model__epsilon": [0.01, 0.1, 0.2],
        }
    raise ValueError(f"Unknown model: {model_name}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MaxAE": float(np.max(np.abs(residual))),
    }


def nested_loocv(
    data: pd.DataFrame,
    experiment: Experiment,
    model_name: str,
    inner_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    X = data[list(experiment.features)].to_numpy(dtype=float)
    y = data[TARGET_CANONICAL].to_numpy(dtype=float)
    n_samples = len(data)
    predictions: list[dict[str, Any]] = []
    fold_parameters: list[dict[str, Any]] = []
    fold_models: list[Pipeline] = []
    for test_index in range(n_samples):
        train_indices = np.array([index for index in range(n_samples) if index != test_index])
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[[test_index]]
        y_test = y[[test_index]]
        pipeline = _pipeline(model_name)
        parameter_grid = _param_grid(model_name, len(experiment.features), len(train_indices), inner_folds)
        if parameter_grid:
            n_splits = min(inner_folds, len(train_indices))
            inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = _jsonable(search.best_params_)
            inner_score = float(-search.best_score_)
        else:
            pipeline.fit(X_train, y_train)
            fitted = pipeline
            best_params = {}
            inner_score = np.nan
        prediction = float(np.asarray(fitted.predict(X_test)).ravel()[0])
        material = str(data.iloc[test_index]["material"])
        predictions.append(
            {
                "row_index": int(test_index),
                "material": material,
                "y_true": float(y_test[0]),
                "y_pred": prediction,
                "residual": prediction - float(y_test[0]),
                "absolute_error": abs(prediction - float(y_test[0])),
                "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            }
        )
        fold_parameters.append(
            {
                "row_index": int(test_index),
                "material": material,
                "inner_cv_mae": inner_score,
                "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            }
        )
        fold_models.append(fitted)
    prediction_frame = pd.DataFrame(predictions)
    metrics = evaluate_predictions(prediction_frame["y_true"].to_numpy(), prediction_frame["y_pred"].to_numpy())
    return {
        "experiment": experiment,
        "model_name": model_name,
        "predictions": prediction_frame,
        "fold_parameters": pd.DataFrame(fold_parameters),
        "fold_models": fold_models,
        "metrics": metrics,
    }


def benchmark_models(
    data: pd.DataFrame,
    output_root: str | Path,
    inner_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    output_root = Path(output_root)
    for directory in (output_root / "tables", output_root / "predictions", output_root / "models"):
        directory.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    parameter_rows: list[pd.DataFrame] = []
    primary_models = ["Dummy", "Ridge", "PLS", "SVR_linear", "SVR_rbf"]
    supplementary_models = ["Ridge", "PLS", "SVR_rbf"]
    for experiment in experiment_specs():
        model_names = primary_models if experiment.name in {f"M{i}" for i in range(1, 7)} else supplementary_models
        if experiment.name == "M1":
            model_names = primary_models + ["Eg_linear"]
        for model_name in model_names:
            result = nested_loocv(data, experiment, model_name, inner_folds=inner_folds, seed=seed)
            result["predictions"]["experiment"] = experiment.name
            result["predictions"]["experiment_label"] = experiment.label
            result["predictions"]["model"] = model_name
            result["fold_parameters"]["experiment"] = experiment.name
            result["fold_parameters"]["model"] = model_name
            result["fold_parameters"]["feature_count"] = len(experiment.features)
            comparison_rows.append(
                {
                    "experiment": experiment.name,
                    "experiment_label": experiment.label,
                    "model": model_name,
                    "feature_count": len(experiment.features),
                    "features": ";".join(experiment.features),
                    **result["metrics"],
                }
            )
            oof_rows.append(result["predictions"])
            parameter_rows.append(result["fold_parameters"])
            all_results.append(result)
    comparison = pd.DataFrame(comparison_rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    oof = pd.concat(oof_rows, ignore_index=True)
    parameters = pd.concat(parameter_rows, ignore_index=True)
    comparison.to_csv(output_root / "tables" / "model_comparison.csv", index=False)
    oof.to_csv(output_root / "predictions" / "oof_predictions.csv", index=False)
    parameters.to_csv(output_root / "tables" / "outer_fold_parameters.csv", index=False)
    with (output_root / "models" / "model_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "random_seed": seed,
                "inner_cv": f"KFold(n_splits={inner_folds}, shuffle=True, random_state={seed})",
                "outer_cv": "LeaveOneOut over material rows",
                "target": TARGET_CANONICAL,
                "n_materials": len(data),
                "experiments": [
                    {"name": exp.name, "label": exp.label, "features": list(exp.features)}
                    for exp in experiment_specs()
                ],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "results": all_results,
        "comparison": comparison,
        "oof": oof,
        "parameters": parameters,
    }

