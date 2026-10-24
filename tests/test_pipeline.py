import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import DataLoader  # noqa: E402
from evaluator import ModelEvaluator  # noqa: E402
from model_generator import get_model_configs  # noqa: E402


def _small_configs():
    configs = get_model_configs()
    return {k: configs[k] for k in ("LogisticRegression", "KNN")}


def _run(csv_path, target):
    bundle = DataLoader(str(csv_path), target).prepare_data()
    results, model = ModelEvaluator(cv_folds=3, n_jobs=1).evaluate_models(
        _small_configs(), bundle.preprocessor, bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test
    )
    return bundle, results, model


def test_multiclass_target_runs(tmp_path):
    df = load_iris(as_frame=True).frame
    path = tmp_path / "iris.csv"
    df.to_csv(path, index=False)
    bundle, results, _ = _run(path, "target")
    assert len(bundle.class_names) == 3
    assert results["f1_score"].between(0, 1).all()


def test_string_labels_categoricals_and_missing_values(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "num": np.where(rng.random(n) < 0.1, np.nan, x),
        "colour": rng.choice(["red", "green", None], size=n),
        "label": np.where(x > 0, "yes", "no"),
    })
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    bundle, results, model = _run(path, "label")
    assert bundle.class_names == ["no", "yes"]
    assert results.loc[0, "f1_score"] > 0.8
    assert model.predict(bundle.X_test).shape == (len(bundle.X_test),)


def test_winner_is_chosen_by_cv_not_test(tmp_path):
    df = load_iris(as_frame=True).frame
    path = tmp_path / "iris.csv"
    df.to_csv(path, index=False)
    _, results, _ = _run(path, "target")
    assert results.loc[0, "selected"]
    assert results.loc[0, "cv_mean_f1"] == results["cv_mean_f1"].max()


def test_single_class_target_is_rejected(tmp_path):
    path = tmp_path / "one.csv"
    pd.DataFrame({"a": [1, 2, 3, 4], "y": [1, 1, 1, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="two classes"):
        DataLoader(str(path), "y").prepare_data()
