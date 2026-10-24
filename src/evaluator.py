"""Model training, tuning, and evaluation routines."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


class ModelEvaluator:
    """Run model selection workflows with grid search and ranking."""

    def __init__(self, cv_folds: int = 5, n_jobs: int = -1, random_state: int = 42):
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state

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
        """Tune every candidate by CV, pick the winner by CV score, then score all on the test set.

        The winner is chosen by mean cross-validated F1 on the training split only.
        Choosing it by test-set F1 would make the reported test score of the winner
        optimistically biased, because the test set would then have been used for
        model selection (Cawley & Talbot, JMLR 11, 2010).

        Labels must already be integer-encoded (0..k-1). Binary problems use F1 on
        class 1; multiclass problems use macro-averaged F1.
        """
        n_classes = int(pd.Series(y_train).nunique())
        binary = n_classes == 2
        scoring = "f1" if binary else "f1_macro"
        average = "binary" if binary else "macro"
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        records: list[dict] = []
        estimators: dict[str, Pipeline] = {}

        for model_name, config in model_configs.items():
            LOGGER.info("Training %s", model_name)
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", config["estimator"])]
            )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=config["param_grid"],
                scoring=scoring,
                cv=cv,
                n_jobs=self.n_jobs,
            )
            search.fit(X_train, y_train)

            best_estimator = search.best_estimator_
            estimators[model_name] = best_estimator
            y_pred = best_estimator.predict(X_test)
            y_proba = best_estimator.predict_proba(X_test) if hasattr(best_estimator, "predict_proba") else None

            records.append({
                "model": model_name,
                "best_params": search.best_params_,
                "cv_mean_f1": float(search.best_score_),
                "cv_std_f1": float(search.cv_results_["std_test_score"][search.best_index_]),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
                "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
                "roc_auc": self._safe_roc_auc(y_test, y_proba) if y_proba is not None else float("nan"),
            })

        if not records:
            raise RuntimeError("No model was trained successfully.")

        results_df = (
            pd.DataFrame(records).sort_values(by="cv_mean_f1", ascending=False).reset_index(drop=True)
        )
        results_df.insert(1, "selected", results_df.index == 0)
        return results_df, estimators[results_df.loc[0, "model"]]
