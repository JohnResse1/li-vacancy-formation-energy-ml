#!/usr/bin/env python
"""Run the complete Li-vacancy descriptor audit and ML study."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.correlation_analysis import run_correlation_analysis
from src.data_loader import load_dataset
from src.data_validation import audit_dataset
from src.feature_ablation import run_feature_ablation
from src.models import benchmark_models
from src.reporting import generate_report
from src.robustness import run_robustness
from src.sensitivity_experiments import run_sensitivity_experiments
from src.visualization import make_all_figures


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as error:
            raise RuntimeError("config.yaml is not JSON and PyYAML is unavailable") from error
        return yaml.safe_load(text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf_text(pdf_path: Path, output_root: Path) -> None:
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "dft_report_extracted.txt"
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(output_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            logging.warning("pdftotext stderr: %s", result.stderr.strip())
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        output_path.write_text(
            "PDF text extraction was unavailable in this environment. The PDF was still preserved as an input artifact.\n"
            f"Error: {error}\n",
            encoding="utf-8",
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="JSON-compatible YAML configuration path")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)
    output_root = PROJECT_ROOT / config.get("output_root", "outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_root / "logs" / "run.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    input_excel = Path(config["input_excel"])
    input_pdf = Path(config["input_pdf"])
    if not input_excel.is_absolute():
        input_excel = PROJECT_ROOT / input_excel
    if not input_pdf.is_absolute():
        input_pdf = PROJECT_ROOT / input_pdf
    if not input_excel.exists():
        raise FileNotFoundError(f"Excel input not found: {input_excel}")
    if not input_pdf.exists():
        raise FileNotFoundError(f"PDF input not found: {input_pdf}")

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "config": config,
        "input_excel": str(input_excel),
        "input_excel_sha256": sha256(input_excel),
        "input_pdf": str(input_pdf),
        "input_pdf_sha256": sha256(input_pdf),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (output_root / "logs" / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    extract_pdf_text(input_pdf, output_root)

    logging.info("Loading workbook and reconstructing auditable canonical fields")
    dataset = load_dataset(input_excel)
    dataset["workbook_metadata"]["source_path"] = str(input_excel)
    audit = audit_dataset(dataset, output_root)
    data = audit["data"]
    logging.info("Data audit complete: %d material rows, canonical feature fields are available", len(data))

    logging.info("Running Pearson/Spearman correlations and bootstrap intervals")
    correlations = run_correlation_analysis(
        data,
        output_root,
        audit["feature_dictionary"],
        bootstrap_replicates=int(config.get("bootstrap_replicates", 2000)),
        seed=seed,
    )
    logging.info("Running nested LOOCV benchmark")
    benchmark = benchmark_models(
        data,
        output_root,
        inner_folds=int(config.get("inner_cv_folds", 5)),
        seed=seed,
    )
    sensitivity = None
    if bool(config.get("run_sensitivity_experiments", True)):
        logging.info("Running nested feature-subset and target-transform sensitivity experiments")
        sensitivity = run_sensitivity_experiments(
            data,
            output_root,
            inner_folds=int(config.get("inner_cv_folds", 5)),
            seed=seed,
            target_transforms=list(config.get("target_transforms", ["identity", "sqrt", "log"])),
            subset_k_values=list(config.get("subset_k_values", [1, 2, 3, 4, 5, 6, 8, 10, "all"])),
        )
    logging.info("Running feature-group ablation and coefficient stability")
    ablation = run_feature_ablation(
        data,
        benchmark,
        output_root,
        bootstrap_replicates=int(config.get("bootstrap_replicates", 2000)),
        coefficient_bootstrap_replicates=int(config.get("coefficient_bootstrap_replicates", 500)),
        seed=seed,
    )
    logging.info("Running material and category robustness diagnostics")
    robustness = run_robustness(
        data,
        benchmark,
        output_root,
        seed=seed,
        run_group_cv=bool(config.get("run_group_cv", True)),
    )
    logging.info("Rendering figures")
    make_all_figures(data, correlations, benchmark, ablation, robustness, output_root)
    logging.info("Writing technical report")
    report_path = generate_report(data, audit, correlations, benchmark, ablation, robustness, output_root, config, sensitivity)
    sensitivity_summary = {}
    if sensitivity is not None:
        subset_best = sensitivity["subset_comparison"].iloc[0].to_dict()
        transform_best = sensitivity["fixed_comparison"].iloc[0].to_dict()
        sensitivity_summary = {
            "feature_subset_best": subset_best,
            "target_transform_best": transform_best,
        }
    summary = {
        "report": str(report_path),
        "n_materials": len(data),
        "n_features_intrinsic": 23,
        "best_numeric_model": benchmark["comparison"].query("model != 'Dummy'").sort_values(["MAE", "RMSE"]).iloc[0].to_dict(),
        "primary_model": robustness["primary"],
        "eg_pearson": correlations["eg_pearson"],
        "eg_spearman": correlations["eg_spearman"],
        "sensitivity": sensitivity_summary,
        "output_files": sorted(str(path.relative_to(PROJECT_ROOT)) for path in output_root.rglob("*") if path.is_file()),
    }
    (output_root / "logs" / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logging.info("Complete. Report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
