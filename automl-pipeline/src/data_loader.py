"""Data loading and preprocessing utilities for tabular datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container for split data and preprocessing pipeline."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


class DataLoader:
    """Load datasets and create preprocessing pipeline."""

    def __init__(self, data_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def _load_dataframe(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        LOGGER.info("Loading dataset from %s", self.data_path)
        dataframe = pd.read_csv(self.data_path)
        if dataframe.empty:
            raise ValueError("Input dataset is empty.")
        if self.target_column not in dataframe.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
        return dataframe

    @staticmethod
    def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

        LOGGER.info(
            "Detected %s numeric and %s categorical features.",
            len(numeric_features),
            len(categorical_features),
        )

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

    def prepare_data(self) -> DatasetBundle:
        """Load, split, and prepare preprocessing config for model training."""
        df = self._load_dataframe()
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if y.nunique() > 1 else None,
        )

        preprocessor = self._build_preprocessor(X_train)
        return DatasetBundle(X_train, X_test, y_train, y_test, preprocessor)
