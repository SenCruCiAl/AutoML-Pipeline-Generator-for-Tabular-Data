"""Model training, tuning, and evaluation routines."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


class ModelEvaluator:
    """Run model selection workflows with grid search and ranking."""

    def __init__(self, cv_folds: int = 5, n_jobs: int = -1):
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs

    @staticmethod
    def _safe_roc_auc(y_true: Any, y_proba: np.ndarray) -> float:
        """Compute ROC-AUC robustly for binary and multiclass outputs."""
        if y_proba.ndim == 1 or y_proba.shape[1] == 1:
            return float("nan")
        if y_proba.shape[1] == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    def evaluate_models(
        self,
        model_configs: dict,
        preprocessor,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[pd.DataFrame, Pipeline]:
        """Train/evaluate all model candidates and return ranked results."""
        records: list[dict] = []
        best_model: Pipeline | None = None
        best_f1 = -np.inf

        for model_name, config in model_configs.items():
            LOGGER.info("Training %s", model_name)
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", config["estimator"])]
            )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=config["param_grid"],
                scoring="f1",
                cv=self.cv_folds,
                n_jobs=self.n_jobs,
            )
            search.fit(X_train, y_train)

            best_estimator = search.best_estimator_
            y_pred = best_estimator.predict(X_test)

            if hasattr(best_estimator, "predict_proba"):
                y_proba = best_estimator.predict_proba(X_test)
            else:
                y_proba = np.array([])

            cv_scores = cross_val_score(
                best_estimator,
                X_train,
                y_train,
                cv=self.cv_folds,
                scoring="f1",
                n_jobs=self.n_jobs,
            )

            record = {
                "model": model_name,
                "best_params": search.best_params_,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": self._safe_roc_auc(y_test, y_proba) if y_proba.size else float("nan"),
                "cv_mean_f1": float(np.mean(cv_scores)),
                "cv_std_f1": float(np.std(cv_scores)),
            }
            records.append(record)

            if record["f1_score"] > best_f1:
                best_f1 = record["f1_score"]
                best_model = best_estimator

        if best_model is None:
            raise RuntimeError("No model was trained successfully.")

        results_df = pd.DataFrame(records).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        return results_df, best_model
