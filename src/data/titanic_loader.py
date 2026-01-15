"""Titanic dataset loader."""

import os
import tarfile
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

import pandas as pd


class TitanicLoader:
    """Load and prepare Titanic dataset."""

    TITANIC_URL = "https://homl.info/titanic.tgz"

    def __init__(self, data_dir: str | Path):
        """
        Initialize Titanic loader.

        Args:
            data_dir: Directory containing or to download titanic data
        """
        self.data_dir = Path(data_dir)
        self._train_data: pd.DataFrame | None = None
        self._test_data: pd.DataFrame | None = None

    def download(self, force: bool = False) -> None:
        """
        Download and extract Titanic dataset.

        Args:
            force: Re-download even if files exist
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tgz_path = self.data_dir / "titanic.tgz"

        # Check if already extracted
        train_path = self._find_train_csv()
        if train_path and not force:
            return

        # Download
        urlretrieve(self.TITANIC_URL, tgz_path)

        # Extract
        with tarfile.open(tgz_path) as tar:
            tar.extractall(self.data_dir, filter="data")

    def _find_train_csv(self) -> Path | None:
        """Find train.csv in data directory."""
        # Check in titanic subfolder first
        titanic_dir = self.data_dir / "titanic"
        if (titanic_dir / "train.csv").exists():
            return titanic_dir / "train.csv"
        if (self.data_dir / "train.csv").exists():
            return self.data_dir / "train.csv"
        return None

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test dataframes.

        Returns:
            (train_df, test_df) tuple
        """
        if self._train_data is not None:
            return self._train_data, self._test_data

        train_path = self._find_train_csv()
        if train_path is None:
            self.download()
            train_path = self._find_train_csv()

        test_path = train_path.parent / "test.csv"

        self._train_data = pd.read_csv(train_path, index_col="PassengerId")
        self._test_data = pd.read_csv(test_path, index_col="PassengerId")

        return self._train_data, self._test_data

    def get_features_and_labels(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return training features and labels."""
        train_df, _ = self.load()
        X = train_df.drop("Survived", axis=1)
        y = train_df["Survived"]
        return X, y

    @property
    def feature_names(self) -> list:
        """Return list of feature column names."""
        train_df, _ = self.load()
        return [col for col in train_df.columns if col != "Survived"]
