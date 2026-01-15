"""Preprocessing pipelines for tabular data."""

from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class TitanicPreprocessor:
    """Preprocessing pipeline for Titanic dataset."""

    DEFAULT_NUMERICAL = ["Age", "SibSp", "Parch", "Fare"]
    DEFAULT_CATEGORICAL = ["Pclass", "Sex", "Embarked"]

    def __init__(
        self,
        numerical_features: List[str] | None = None,
        categorical_features: List[str] | None = None,
    ):
        """
        Initialize preprocessor.

        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
        """
        self.numerical_features = numerical_features or self.DEFAULT_NUMERICAL
        self.categorical_features = categorical_features or self.DEFAULT_CATEGORICAL
        self._pipeline: ColumnTransformer | None = None
        self._fitted = False

    def _build_pipeline(self) -> ColumnTransformer:
        """Build the preprocessing pipeline."""
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])

        return ColumnTransformer([
            ("num", numerical_pipeline, self.numerical_features),
            ("cat", categorical_pipeline, self.categorical_features),
        ])

    @property
    def pipeline(self) -> ColumnTransformer:
        """Get or create the preprocessing pipeline."""
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def fit(self, X) -> "TitanicPreprocessor":
        """Fit the preprocessing pipeline."""
        self.pipeline.fit(X)
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Transform data using fitted pipeline."""
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        return self.pipeline.transform(X)

    def fit_transform(self, X) -> np.ndarray:
        """Fit and transform in one step."""
        self._pipeline = self._build_pipeline()
        self._fitted = True
        return self.pipeline.fit_transform(X)

    def get_feature_names(self) -> List[str]:
        """Get output feature names after transformation."""
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        return list(self.pipeline.get_feature_names_out())
