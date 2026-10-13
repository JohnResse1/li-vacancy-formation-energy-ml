"""Publication-oriented Matplotlib figures for the completed analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


GROUP_COLORS = {
    "E": "#2f6690",
    "G": "#4d908e",
    "C": "#f4a261",
    "T": "#9b5de5",
    "AUX": "#777777",
}
GROUP_LABELS = {"E": "Electronic", "G": "Geometry", "C": "Composition", "T": "Thermodynamic", "AUX": "Auxiliary"}
CAT_COLORS = {
    "oxygen-containing": "#4575b4",
    "carbon-containing": "#d73027",
    "sulfide-selenide": "#66bd63",
    "phosphide/phosphate": "#984ea3",
    "nitride/nitrate": "#fdae61",
    "halide": "#666666",
    "other": "#aaaaaa",
}
FEATURE_LABELS = {
    "band_gap": "Band gap, Eg (eV)",
    "li_mean_bader_charge": "Mean Li Bader charge loss (e)",
    "bulk_mean_elf": "Mean bulk ELF",
    "elf_high_fraction": "ELF >= 0.8 volume (%)",
    "elf_low_fraction": "ELF <= 0.2 volume (%)",
    "li_mean_coordination_number": "Mean Li coordination",
    "normalized_li_x_distance": "Normalized Li-X distance",
    "volume_per_li": "Volume per Li (A3/Li)",
    "density": "Density (g/cm3)",
    "volume_per_atom": "Volume per atom (A3/atom)",
    "mean_li_x_bond_length": "Mean Li-X bond length (A)",
    "li_x_bond_length_std": "Li-X bond length SD (A)",
    "li_x_bond_length_range": "Li-X bond length range (A)",
    "li_coordination_number_std": "Li coordination SD",
    "li_atomic_fraction": "Li atomic fraction",
    "li_framework_electronegativity_difference": "Li-framework electronegativity difference",
    "mean_framework_atomic_radius": "Mean framework radius (A)",
    "mean_atomic_number": "Mean atomic number",
    "mean_atomic_mass": "Mean atomic mass (g/mol-atom)",
    "framework_electronegativity_std": "Framework electronegativity SD",
    "framework_oxygen_fraction": "Framework O fraction",
    "formation_energy_per_atom": "Bulk formation energy (eV/atom)",
    "energy_above_hull": "Energy above hull (eV/atom)",
    "vacancy_concentration": "Vacancy concentration (%)",
}


def _short_label(feature: str) -> str:
    replacements = {
        "band_gap": "Eg",
        "li_mean_bader_charge": "Bader",
        "bulk_mean_elf": "ELF mean",
        "elf_high_fraction": "ELF >= 0.8",
        "elf_low_fraction": "ELF <= 0.2",
        "li_mean_coordination_number": "CN mean",
        "normalized_li_x_distance": "d_norm",
        "volume_per_li": "V/Li",
        "density": "Density",
        "volume_per_atom": "V/atom",
        "mean_li_x_bond_length": "d Li-X",
        "li_x_bond_length_std": "SD d",
        "li_x_bond_length_range": "range d",
        "li_coordination_number_std": "SD CN",
        "li_atomic_fraction": "x Li",
        "li_framework_electronegativity_difference": "Delta chi",
        "mean_framework_atomic_radius": "r framework",
        "mean_atomic_number": "Z mean",
        "mean_atomic_mass": "M mean",
        "framework_electronegativity_std": "SD chi fw",
        "framework_oxygen_fraction": "xO fw",
        "formation_energy_per_atom": "Ef bulk",
        "energy_above_hull": "E hull",
        "vacancy_concentration": "VLi %",
    }
    return replacements.get(feature, feature)


def _save_figure(figure: plt.Figure, figures_dir: Path, stem: str) -> None:
    figure.savefig(figures_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    figure.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _eg_regression_band(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    n = len(x)
    s_err = np.sqrt(np.sum(residual**2) / max(1, n - 2))
    x_bar = np.mean(x)
    sxx = np.sum((x - x_bar) ** 2)
    if sxx == 0:
        band = np.full_like(x_grid, np.nan)
    else:
        band = stats.t.ppf(0.975, max(1, n - 2)) * s_err * np.sqrt(1 / n + (x_grid - x_bar) ** 2 / sxx)
    center = slope * x_grid + intercept
    return center, center - band, center + band


def _legend_for_groups(ax: plt.Axes) -> None:
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=GROUP_LABELS[group], markersize=7) for group, color in GROUP_COLORS.items() if group != "AUX"]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")


def make_fig01(
    data: pd.DataFrame,
    correlations: dict[str, Any],
    output_root: str | Path,
) -> None:
    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pearson = correlations["pearson"].copy()
    spearman = correlations["spearman"].copy()
    merged = pearson[["feature", "group", "coefficient"]].rename(columns={"coefficient": "pearson"}).merge(
        spearman[["feature", "coefficient"]].rename(columns={"coefficient": "spearman"}), on="feature"
    )
    merged["rank"] = merged[["pearson", "spearman"]].abs().max(axis=1)
    merged = merged.sort_values("rank")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), gridspec_kw={"height_ratios": [1.35, 1]})
    ax = axes[0, 0]
    y_positions = np.arange(len(merged))
    for position, (_, row) in enumerate(merged.iterrows()):
        color = GROUP_COLORS.get(row["group"], "#777777")
        ax.plot([row["pearson"], row["spearman"]], [position, position], color=color, alpha=0.35, lw=2)
        ax.scatter(row["pearson"], position, color=color, marker="o", s=22, zorder=3)
        ax.scatter(row["spearman"], position, color=color, marker="s", s=22, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_short_label(value) for value in merged["feature"]], fontsize=7)
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_xlabel("Correlation with Li-vacancy formation energy")
    ax.set_title("(a) Descriptor-target correlations", loc="left", fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="#333333", label="Pearson", linestyle="None"), Line2D([0], [0], marker="s", color="#333333", label="Spearman", linestyle="None")], frameon=False, fontsize=8, loc="lower right")

    ax = axes[0, 1]
    matrix = correlations["x_x_spearman"]
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    labels = [_short_label(value) for value in matrix.columns]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("(b) Descriptor Spearman correlation", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")

    ax = axes[1, 0]
    for category, group in data.groupby("category"):
        ax.scatter(group["band_gap"], group["Li_vacancy_formation_energy"], s=46, color=CAT_COLORS.get(category, "#777777"), label=category, edgecolor="white", linewidth=0.5)
        for _, row in group.iterrows():
            ax.annotate(row["material"], (row["band_gap"], row["Li_vacancy_formation_energy"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
    x = data["band_gap"].to_numpy(dtype=float)
    y = data["Li_vacancy_formation_energy"].to_numpy(dtype=float)
    x_grid = np.linspace(x.min(), x.max(), 200)
    center, lower, upper = _eg_regression_band(x, y, x_grid)
    ax.plot(x_grid, center, color="#222222", lw=1.2)
    ax.fill_between(x_grid, lower, upper, color="#222222", alpha=0.12, label="95% CI for mean fit")
    eg_p = correlations["eg_pearson"]
    eg_s = correlations["eg_spearman"]
    ax.text(0.04, 0.96, f"Pearson r = {eg_p['coefficient']:.3f}\nSpearman rho = {eg_s['coefficient']:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.set_xlabel("Band gap, Eg (eV)")
    ax.set_ylabel("Minimum neutral Li-vacancy formation energy (eV)")
    ax.set_title("(c) Eg-target relationship", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")

    ax = axes[1, 1]
    top_features = merged.sort_values("rank", ascending=False).head(4)["feature"].tolist()
    # The panel compares selected descriptors with the target; selection is based on
    # the predeclared correlation ranking, not on visual preference for E features.
    for feature in top_features:
        ax.scatter(data[feature], data["Li_vacancy_formation_energy"], s=25, alpha=0.65, label=_short_label(feature))
    ax.set_xlabel("Selected descriptor value (native units)")
    ax.set_ylabel("Li-vacancy formation energy (eV)")
    ax.set_title("(d) Strongest descriptor relationships", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Fig. 01 | Descriptor-target relationships", fontsize=15, y=1.01)
    fig.text(0.5, -0.015, "Panels (a-d) use the same 24 material rows; correlations are exploratory and not causal estimates.", ha="center", fontsize=8)
    _save_figure(fig, figures_dir, "Fig01_descriptor_target_relationships")


def make_fig02(
    benchmark: dict[str, Any],
    robustness: dict[str, Any],
    output_root: str | Path,
) -> None:
    figures_dir = Path(output_root) / "figures"
    comparison = benchmark["comparison"].copy()
    comparison = comparison[comparison["model"] != "Dummy"]
    best_by_experiment = comparison.loc[comparison.groupby("experiment")["MAE"].idxmin()].sort_values("experiment")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x = np.arange(len(best_by_experiment))
    width = 0.36
    ax.bar(x - width / 2, best_by_experiment["MAE"], width, label="MAE", color="#2f6690")
    ax.bar(x + width / 2, best_by_experiment["RMSE"], width, label="RMSE", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(best_by_experiment["experiment"], rotation=45)
    ax.set_ylabel("Cross-validated error (eV)")
    ax.set_title("(a) Best nested-LOOCV result per feature set", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    for position, (_, row) in enumerate(best_by_experiment.iterrows()):
        ax.text(position, max(row["MAE"], row["RMSE"]) + 0.03, row["model"], ha="center", fontsize=7)

    primary = robustness["primary"]
    result = next(item for item in benchmark["results"] if item["experiment"].name == primary["experiment"] and item["model_name"] == primary["model"])
    prediction = result["predictions"]
    ax = axes[1]
    ax.scatter(prediction["y_true"], prediction["y_pred"], s=45, color="#2f6690", edgecolor="white", linewidth=0.5)
    bounds = [min(prediction["y_true"].min(), prediction["y_pred"].min()), max(prediction["y_true"].max(), prediction["y_pred"].max())]
    ax.plot(bounds, bounds, color="#222222", lw=1, linestyle="--")
    for _, row in prediction.iterrows():
        ax.annotate(row["material"], (row["y_true"], row["y_pred"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.set_xlabel("Observed formation energy (eV)")
    ax.set_ylabel("OOF predicted formation energy (eV)")
    ax.set_title(f"(b) {primary['experiment']} {primary['model']} parity", loc="left", fontweight="bold")
    ax.text(0.04, 0.96, f"MAE = {primary['MAE']:.3f} eV\nRMSE = {primary['RMSE']:.3f} eV\nR2 = {primary['R2']:.3f}", transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    fig.suptitle("Fig. 02 | Machine-learning benchmark", fontsize=15, y=1.02)
    _save_figure(fig, figures_dir, "Fig02_machine_learning_benchmark")


def make_fig03(
    ablation: dict[str, Any],
    output_root: str | Path,
) -> None:
    figures_dir = Path(output_root) / "figures"
    ablation_frame = ablation["ablation"]
    ridge_ablation = ablation_frame[ablation_frame["model"] == "Ridge"].copy()
    desired = [("M3", "M4"), ("M3", "M5"), ("M8", "M9"), ("M9", "M10")]
    labels = [f"{first}->{second}" for first, second in desired]
    values = []
    ci_low = []
    ci_high = []
    for first, second in desired:
        rows = ridge_ablation[(ridge_ablation["baseline"] == first) & (ridge_ablation["extended"] == second)]
        if rows.empty:
            values.append(np.nan); ci_low.append(np.nan); ci_high.append(np.nan)
        else:
            row = rows.iloc[0]
            values.append(row["delta_MAE_baseline_minus_extended"])
            ci_low.append(row["paired_delta_ci_low"])
            ci_high.append(row["paired_delta_ci_high"])
    coefficients = ablation["coefficient_summary"]
    if not coefficients.empty:
        coeff_m6 = coefficients[coefficients["experiment"] == "M6"].sort_values("median", key=lambda x: x.abs(), ascending=False).head(12)
    else:
        coeff_m6 = pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=["#2f6690", "#4d908e", "#9b5de5", "#f4a261"])
    for position, low, high in zip(positions, ci_low, ci_high):
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([position, position], [low, high], color="#222222", lw=1.2)
            ax.scatter([position], [(low + high) / 2], color="#222222", s=10)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35)
    ax.set_ylabel("Paired MAE improvement (eV)\npositive = extended model better")
    ax.set_title("(a) Feature-group increments", loc="left", fontweight="bold")

    ax = axes[1]
    if not coeff_m6.empty:
        y_positions = np.arange(len(coeff_m6))
        colors = [GROUP_COLORS.get(group, "#777777") for group in coeff_m6["group"]]
        ax.barh(y_positions, coeff_m6["median"], color=colors, alpha=0.85)
        ax.errorbar(coeff_m6["median"], y_positions, xerr=[coeff_m6["median"] - coeff_m6["min"], coeff_m6["max"] - coeff_m6["median"]], fmt="none", ecolor="#222222", capsize=2, lw=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([_short_label(value) for value in coeff_m6["feature"]], fontsize=8)
        ax.axvline(0, color="#222222", lw=0.8)
        ax.set_xlabel("Standardized Ridge coefficient (eV)")
        ax.set_title("(b) M6 coefficient stability across outer folds", loc="left", fontweight="bold")
        _legend_for_groups(ax)
    else:
        ax.text(0.5, 0.5, "No Ridge coefficient output", ha="center", va="center")
    fig.suptitle("Fig. 03 | Electronic descriptor contribution", fontsize=15, y=1.02)
    fig.text(0.5, -0.02, "Error bars in panel (a) are bootstrap intervals for paired OOF-error differences; coefficients are descriptive under correlation.", ha="center", fontsize=8)
    _save_figure(fig, figures_dir, "Fig03_electronic_descriptor_contribution")


def make_fig04(
    data: pd.DataFrame,
    benchmark: dict[str, Any],
    robustness: dict[str, Any],
    output_root: str | Path,
) -> None:
    figures_dir = Path(output_root) / "figures"
    category = robustness["category"]
    influence = robustness["influence"]
    primary = robustness["primary"]
    primary_result = next(item for item in benchmark["results"] if item["experiment"].name == primary["experiment"] and item["model_name"] == primary["model"])
    errors = primary_result["predictions"].sort_values("absolute_error")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1.1, 1.15, 1.25]})
    ax = axes[0]
    category_plot = category[category["subset"].str.startswith("category:") | category["subset"].isin(["all_24", "without_LiF_LiC6", "oxygen-bearing subset"])].copy()
    category_plot["label"] = category_plot["subset"].str.replace("category:", "", regex=False)
    category_plot = category_plot.sort_values("n")
    y_positions = np.arange(len(category_plot))
    ax.axvline(0, color="#222222", lw=0.7)
    ax.errorbar(category_plot["pearson_r"], y_positions, fmt="o", color="#2f6690", label="Pearson r")
    ax.scatter(category_plot["spearman_rho"], y_positions, marker="s", color="#f4a261", label="Spearman rho")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{label} (n={n})" for label, n in zip(category_plot["label"], category_plot["n"])], fontsize=7)
    ax.set_xlabel("Eg-target correlation")
    ax.set_title("(a) Category sensitivity", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    influence_plot = influence.sort_values("eg_pearson_r")
    y_positions = np.arange(len(influence_plot))
    ax.plot(influence_plot["eg_pearson_r"], y_positions, "o", color="#2f6690", markersize=3)
    ax.axvline(influence_plot["eg_pearson_r"].iloc[-1], color="#aaaaaa", lw=0.7, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(influence_plot["removed_material"], fontsize=6)
    ax.set_xlabel("Pearson r after removing material")
    ax.set_title("(b) Leave-one-material Eg sensitivity", loc="left", fontweight="bold")
    for material in ["LiF", "LiC6"]:
        match = influence_plot[influence_plot["removed_material"] == material]
        if not match.empty:
            position = int(np.flatnonzero(influence_plot["removed_material"].to_numpy() == material)[0])
            ax.annotate(material, (match["eg_pearson_r"].iloc[0], position), xytext=(4, 0), textcoords="offset points", fontsize=7)

    ax = axes[2]
    colors = [CAT_COLORS.get(data.set_index("material").loc[material, "category"], "#777777") for material in errors["material"]]
    ax.barh(np.arange(len(errors)), errors["absolute_error"], color=colors)
    ax.set_yticks(np.arange(len(errors)))
    ax.set_yticklabels(errors["material"], fontsize=7)
    ax.set_xlabel("Absolute OOF error (eV)")
    ax.set_title(f"(c) {primary['experiment']} {primary['model']} material errors", loc="left", fontweight="bold")
    fig.suptitle("Fig. 04 | Robustness and material dependence", fontsize=15, y=1.02)
    fig.text(0.5, -0.02, "Material removal is diagnostic only; the 24-row primary result is not filtered by these analyses.", ha="center", fontsize=8)
    _save_figure(fig, figures_dir, "Fig04_robustness_material_dependence")


def make_all_figures(
    data: pd.DataFrame,
    correlations: dict[str, Any],
    benchmark: dict[str, Any],
    ablation: dict[str, Any],
    robustness: dict[str, Any],
    output_root: str | Path,
) -> None:
    make_fig01(data, correlations, output_root)
    make_fig02(benchmark, robustness, output_root)
    make_fig03(ablation, output_root)
    make_fig04(data, benchmark, robustness, output_root)
