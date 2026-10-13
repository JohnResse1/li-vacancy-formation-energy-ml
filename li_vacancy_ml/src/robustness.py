"""Material influence, category sensitivity, and exploratory group tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from .data_loader import (
    ALL_INTRINSIC_FEATURES,
    CHEMICAL_FEATURES,
    ELECTRONIC_FEATURES,
    GEOMETRIC_FEATURES,
    TARGET_CANONICAL,
    THERMODYNAMIC_FEATURES,
)
from .models import _param_grid, _pipeline, experiment_specs


def select_primary_model(comparison: pd.DataFrame) -> dict[str, Any]:
    candidates = comparison[comparison["model"] != "Dummy"].copy()
    candidates = candidates.sort_values(["MAE", "RMSE", "feature_count"]).reset_index(drop=True)
    numerical_winner = candidates.iloc[0]
    best = numerical_winner.to_dict()
    # With 24 rows, a 21-feature model is a useful benchmark but a poor sole
    # scientific reference. Prefer a one-to-five-feature linear candidate when
    # its nested-LOOCV MAE is within 0.05 eV of the numerical winner.
    simple_candidates = candidates[
        (candidates["feature_count"] <= 5)
        & candidates["model"].isin(["Ridge", "SVR_linear", "Eg_linear"])
    ]
    if not simple_candidates.empty:
        simple = simple_candidates.sort_values(["MAE", "RMSE", "feature_count"]).iloc[0]
        if float(simple["MAE"]) <= float(numerical_winner["MAE"]) + 0.05:
            best = simple.to_dict()
    best["numerical_winner_experiment"] = numerical_winner["experiment"]
    best["numerical_winner_model"] = numerical_winner["model"]
    best["numerical_winner_MAE"] = float(numerical_winner["MAE"])
    return best


def _corr(values_x: pd.Series, values_y: pd.Series) -> dict[str, float]:
    valid = values_x.notna() & values_y.notna()
    x = values_x[valid].to_numpy(dtype=float)
    y = values_y[valid].to_numpy(dtype=float)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": len(x),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _eg_fit(data: pd.DataFrame) -> dict[str, float]:
    x = data["band_gap"].to_numpy(dtype=float)
    y = data[TARGET_CANONICAL].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    return {
        "slope_eV_per_eV": float(slope),
        "intercept_eV": float(intercept),
        "r2_in_sample": float(r2_score(y, prediction)),
    }


def _restricted_metrics(predictions: pd.DataFrame, keep_materials: set[str]) -> dict[str, float]:
    restricted = predictions[predictions["material"].isin(keep_materials)]
    if restricted.empty:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    return {
        "n": int(len(restricted)),
        "MAE": float(mean_absolute_error(restricted["y_true"], restricted["y_pred"])),
        "RMSE": float(np.sqrt(mean_squared_error(restricted["y_true"], restricted["y_pred"]))),
        "R2": float(r2_score(restricted["y_true"], restricted["y_pred"])),
    }


def material_influence(data: pd.DataFrame, benchmark: dict[str, Any], primary: dict[str, Any]) -> pd.DataFrame:
    lookup = {(item["experiment"].name, item["model_name"]): item for item in benchmark["results"]}
    primary_result = lookup[(primary["experiment"], primary["model"])]
    m3 = lookup.get(("M3", primary["model"]))
    m4 = lookup.get(("M4", primary["model"]))
    all_materials = set(data["material"])
    rows: list[dict[str, Any]] = []
    for removed in data["material"]:
        keep = all_materials - {removed}
        subset = data[data["material"].isin(keep)]
        correlation = _corr(subset["band_gap"], subset[TARGET_CANONICAL])
        fit = _eg_fit(subset)
        metrics = _restricted_metrics(primary_result["predictions"], keep)
        if m3 is not None and m4 is not None:
            p3 = m3["predictions"].set_index("material").loc[sorted(keep)]
            p4 = m4["predictions"].set_index("material").loc[sorted(keep)]
            delta = float((p3["absolute_error"] - p4["absolute_error"]).mean())
        else:
            delta = np.nan
        rows.append(
            {
                "removed_material": removed,
                "n_remaining": len(subset),
                "eg_n": correlation["n"],
                "eg_pearson_r": correlation["pearson_r"],
                "eg_pearson_p": correlation["pearson_p"],
                "eg_spearman_rho": correlation["spearman_rho"],
                "eg_spearman_p": correlation["spearman_p"],
                **fit,
                "primary_experiment": primary["experiment"],
                "primary_model": primary["model"],
                "primary_MAE_remaining": metrics["MAE"],
                "primary_RMSE_remaining": metrics["RMSE"],
                "M3_to_M4_paired_delta_MAE_remaining": delta,
            }
        )
    return pd.DataFrame(rows)


def category_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    subsets: list[tuple[str, pd.Series]] = [("all_24", pd.Series(True, index=data.index))]
    subsets.append(("without_LiF_LiC6", ~data["material"].isin(["LiF", "LiC6"])))
    # This is a chemically defined subset rather than the mutually exclusive
    # category label: carbonates and nitrates with O must also be included.
    subsets.append(("oxygen-bearing subset", data["framework_oxygen_fraction"] > 0))
    for category in sorted(data["category"].unique()):
        subsets.append((f"category:{category}", data["category"] == category))
    rows = []
    for label, mask in subsets:
        subset = data.loc[mask]
        row = {"subset": label, "category": label, **_corr(subset["band_gap"], subset[TARGET_CANONICAL])}
        row["materials"] = ";".join(subset["material"])
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_category_out(
    data: pd.DataFrame,
    feature_set: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    categories = data["category"].to_numpy()
    for category in sorted(set(categories)):
        train = data[data["category"] != category]
        test = data[data["category"] == category]
        if len(test) == 0 or len(train) < 6:
            continue
        model_name = "Ridge"
        pipeline = _pipeline(model_name)
        folds = min(5, len(train))
        search = GridSearchCV(
            pipeline,
            _param_grid(model_name, len(feature_set), len(train), folds),
            scoring="neg_mean_absolute_error",
            cv=KFold(n_splits=folds, shuffle=True, random_state=seed),
            n_jobs=1,
        )
        search.fit(train[feature_set], train[TARGET_CANONICAL])
        prediction = search.predict(test[feature_set])
        rows.append(
            {
                "held_out_category": category,
                "n_train": len(train),
                "n_test": len(test),
                "test_materials": ";".join(test["material"]),
                "MAE": float(mean_absolute_error(test[TARGET_CANONICAL], prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(test[TARGET_CANONICAL], prediction))),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def pair_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Li6CoO4", "Li6ZnO4", "same Li6MO4 oxygen framework; Co/Zn contrast"),
        ("Li5FeO4", "Li5AlO4", "same Li5MO4 oxygen framework; Fe/Al contrast"),
    ]
    features = [
        "band_gap", "Li_vacancy_formation_energy", "li_mean_bader_charge",
        "bulk_mean_elf", "elf_high_fraction", "elf_low_fraction",
        "li_mean_coordination_number", "normalized_li_x_distance", "volume_per_li",
        "formation_energy_per_atom", "energy_above_hull",
    ]
    rows = []
    for first, second, rationale in pairs:
        first_row = data[data["material"] == first].iloc[0]
        second_row = data[data["material"] == second].iloc[0]
        for feature in features:
            rows.append(
                {
                    "pair": f"{first} vs {second}",
                    "first_material": first,
                    "second_material": second,
                    "rationale": rationale,
                    "feature": feature,
                    "first_value": first_row[feature],
                    "second_value": second_row[feature],
                    "first_minus_second": first_row[feature] - second_row[feature],
                }
            )
    return pd.DataFrame(rows)


def run_robustness(
    data: pd.DataFrame,
    benchmark: dict[str, Any],
    output_root: str | Path,
    seed: int = 42,
    run_group_cv: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    primary = select_primary_model(benchmark["comparison"])
    influence = material_influence(data, benchmark, primary)
    category = category_sensitivity(data)
    pairs = pair_comparisons(data)
    influence.to_csv(output_root / "tables" / "material_influence.csv", index=False)
    category.to_csv(output_root / "tables" / "category_sensitivity.csv", index=False)
    pairs.to_csv(output_root / "tables" / "material_pair_comparisons.csv", index=False)
    group_cv = pd.DataFrame()
    if run_group_cv:
        # Exploratory LOGO uses the compact M4 Ridge model so that held-out categories
        # are not evaluated with a feature count comparable to the sample count.
        m4 = next(experiment for experiment in experiment_specs() if experiment.name == "M4")
        group_cv = leave_one_category_out(data, list(m4.features), seed=seed)
        group_cv.to_csv(output_root / "tables" / "leave_one_category_out.csv", index=False)
    with (output_root / "models" / "primary_model_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(primary, handle, ensure_ascii=False, indent=2, default=str)
    return {
        "primary": primary,
        "influence": influence,
        "category": category,
        "pairs": pairs,
        "group_cv": group_cv,
    }
