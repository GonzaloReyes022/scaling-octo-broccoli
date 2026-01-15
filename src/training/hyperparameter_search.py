"""Hyperparameter search utilities."""

from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HyperparameterSearcher:
    """Hyperparameter tuning with grid or random search."""

    def __init__(
        self,
        model,
        param_grid: Dict[str, List[Any]] | List[Dict[str, List[Any]]],
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1,
    ):
        """
        Initialize searcher.

        Args:
            model: Scikit-learn compatible model
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self._search_results = None

    def grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "HyperparameterSearcher":
        """
        Perform grid search.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        searcher = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        searcher.fit(X, y)
        self._search_results = searcher
        return self

    def random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 50,
        random_state: int = 42,
    ) -> "HyperparameterSearcher":
        """
        Perform randomized search.

        Args:
            X: Training features
            y: Training labels
            n_iter: Number of parameter combinations to try
            random_state: Random seed

        Returns:
            Self for method chaining
        """
        searcher = RandomizedSearchCV(
            self.model,
            self.param_grid,
            n_iter=n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=random_state,
        )
        searcher.fit(X, y)
        self._search_results = searcher
        return self

    @property
    def best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        if self._search_results is None:
            raise ValueError("No search performed yet.")
        return self._search_results.best_params_

    @property
    def best_score(self) -> float:
        """Get best cross-validation score."""
        if self._search_results is None:
            raise ValueError("No search performed yet.")
        return self._search_results.best_score_

    @property
    def best_estimator(self):
        """Get the best fitted estimator."""
        if self._search_results is None:
            raise ValueError("No search performed yet.")
        return self._search_results.best_estimator_

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of search results."""
        if self._search_results is None:
            raise ValueError("No search performed yet.")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_folds": self.cv,
            "scoring": self.scoring,
        }
