"""Generate model candidates and associated parameter grids."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_model_configs(random_state: int = 42) -> dict:
    """Return model dictionary with estimator and grid-search parameters."""
    return {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=random_state),
            "param_grid": {
                # lbfgs handles both binary and multiclass; liblinear is binary-only.
                "classifier__C": [0.1, 1.0, 10.0],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        },
        "SVM": {
            # SVC(probability=True) is deprecated since scikit-learn 1.9; this is its replacement.
            "estimator": CalibratedClassifierCV(SVC(random_state=random_state), ensemble=False),
            "param_grid": {
                "classifier__estimator__C": [0.5, 1.0, 2.0],
                "classifier__estimator__kernel": ["linear", "rbf"],
                "classifier__estimator__gamma": ["scale", "auto"],
            },
        },
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "classifier__n_neighbors": [3, 5, 7, 9],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["minkowski", "manhattan"],
            },
        },
    }
