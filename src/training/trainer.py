"""Model training and evaluation utilities."""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


class Trainer:
    """Train and evaluate ML models."""

    def __init__(self, model, random_state: int = 42):
        """
        Initialize trainer.

        Args:
            model: Scikit-learn compatible model
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Trainer":
        """
        Train the model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data.

        Args:
            X: Test features
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "predictions": y_pred,
            "report": classification_report(y, y_pred, output_dict=True),
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 10,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Dictionary with CV results
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std(),
            "cv_folds": cv,
            "scoring": scoring,
        }

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted
