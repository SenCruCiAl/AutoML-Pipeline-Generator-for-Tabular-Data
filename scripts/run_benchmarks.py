"""Run the pipeline on several datasets and write docs/results.json for the project page."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import DataLoader  # noqa: E402
from evaluator import ModelEvaluator  # noqa: E402
from model_generator import get_model_configs  # noqa: E402


def synthetic_mixed(seed: int = 0) -> pd.DataFrame:
    """Mixed numeric/categorical data with missing values and string labels."""
    rng = np.random.default_rng(seed)
    n = 600
    income = rng.lognormal(10, 0.5, n)
    age = rng.integers(18, 70, n).astype(float)
    region = rng.choice(["north", "south", "east", "west"], n)
    plan = rng.choice(["basic", "plus", "pro"], n, p=[0.5, 0.3, 0.2])
    logit = 0.8 * (np.log(income) - 10) / 0.5 - 0.03 * (age - 40) + (plan == "pro") * 1.2 - (region == "south") * 0.7
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "stay", "churn")
    age[rng.random(n) < 0.08] = np.nan
    region = np.where(rng.random(n) < 0.05, None, region)
    return pd.DataFrame({"income": income, "age": age, "region": region, "plan": plan, "outcome": churn})


def sk(loader) -> pd.DataFrame:
    return loader(as_frame=True).frame


DATASETS = {
    "breast_cancer": ("Breast Cancer Wisconsin (Diagnostic)", lambda: sk(load_breast_cancer), "target"),
    "wine": ("Wine (3 cultivars)", lambda: sk(load_wine), "target"),
    "iris": ("Iris", lambda: sk(load_iris), "target"),
    "digits": ("Optical digits (8x8, 10 classes)", lambda: sk(load_digits), "target"),
    "mixed": ("Synthetic churn: mixed types, missing values, string labels", synthetic_mixed, "outcome"),
}


def main() -> None:
    out = []
    configs = get_model_configs(random_state=42)
    with tempfile.TemporaryDirectory() as tmp:
        for key, (title, make, target) in DATASETS.items():
            df = make()
            path = Path(tmp) / f"{key}.csv"
            df.to_csv(path, index=False)
            bundle = DataLoader(str(path), target, random_state=42).prepare_data()
            results, _ = ModelEvaluator(cv_folds=5, random_state=42).evaluate_models(
                configs, bundle.preprocessor, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test
            )
            test_best = results.loc[results["f1_score"].idxmax(), "model"]
            print(f"{key}: selected={results.loc[0, 'model']} (test-best would be {test_best})")
            out.append({
                "key": key,
                "title": title,
                "rows": int(len(df)),
                "features": int(df.shape[1] - 1),
                "categorical": int(df.drop(columns=[target]).select_dtypes(exclude="number").shape[1]),
                "missing_cells": int(df.drop(columns=[target]).isna().sum().sum()),
                "classes": bundle.class_names,
                "metric": "F1" if len(bundle.class_names) == 2 else "macro F1",
                "results": [
                    {k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item())
                     for k, v in {**r, "best_params": {p.replace("classifier__", "").replace("estimator__", ""): str(val)
                                                       for p, val in r["best_params"].items()}}.items()}
                    for r in results.to_dict(orient="records")
                ],
            })
    (ROOT / "docs" / "results.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
