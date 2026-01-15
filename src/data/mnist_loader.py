"""MNIST dataset loader with stratified splitting."""

import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class MNISTLoader:
    """Load and split MNIST dataset with stratified sampling."""

    def __init__(self, data_path: str | Path):
        """
        Initialize MNIST loader.

        Args:
            data_path: Path to mnist_784.csv or mnist_784.csv.zip
        """
        self.data_path = Path(data_path)
        self._data: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Load MNIST data from CSV or ZIP file."""
        if self._data is not None:
            return self._data

        if self.data_path.suffix == ".zip":
            with zipfile.ZipFile(self.data_path, "r") as zip_ref:
                csv_name = self.data_path.stem  # mnist_784.csv
                with zip_ref.open(csv_name) as f:
                    self._data = pd.read_csv(f)
        else:
            self._data = pd.read_csv(self.data_path)

        return self._data

    def get_features_and_labels(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return features (X) and labels (y) separately."""
        data = self.load()
        X = data.drop("class", axis=1)
        y = data["class"]
        return X, y

    def stratified_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified train/test split.

        Args:
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test as numpy arrays
        """
        X, y = self.get_features_and_labels()

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )

        for train_idx, test_idx in splitter.split(X, y):
            X_train = X.iloc[train_idx].to_numpy()
            X_test = X.iloc[test_idx].to_numpy()
            y_train = y.iloc[train_idx].to_numpy()
            y_test = y.iloc[test_idx].to_numpy()

        return X_train, X_test, y_train, y_test

    @property
    def shape(self) -> Tuple[int, int]:
        """Return shape of loaded data."""
        return self.load().shape
