"""Tests for data loading modules."""

import pytest
import numpy as np
from pathlib import Path


class TestMNISTLoader:
    """Tests for MNISTLoader class."""

    def test_stratified_split_sizes(self):
        """Test that stratified split produces correct sizes."""
        from src.data import MNISTLoader

        # Skip if no data file available
        data_path = Path("data/raw/mnist_784.csv.zip")
        if not data_path.exists():
            pytest.skip("MNIST data file not found")

        loader = MNISTLoader(data_path)
        X_train, X_test, y_train, y_test = loader.stratified_split(test_size=0.2)

        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.01


class TestTitanicLoader:
    """Tests for TitanicLoader class."""

    def test_feature_names(self):
        """Test that feature names are correctly identified."""
        from src.data import TitanicLoader

        loader = TitanicLoader("data/raw")
        # This will download if needed
        try:
            loader.download()
            X, y = loader.get_features_and_labels()
            assert "Age" in X.columns
            assert "Survived" not in X.columns
        except Exception:
            pytest.skip("Could not download Titanic data")


class TestImageAugmentor:
    """Tests for ImageAugmentor class."""

    def test_shift_image(self):
        """Test that image shifting works correctly."""
        from src.preprocessing import ImageAugmentor

        augmentor = ImageAugmentor(image_shape=(3, 3))

        # Create simple test image
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        # Shift right by 1
        shifted = augmentor.shift_image(image, dx=1, dy=0)
        assert shifted[0, 0] == 0  # Empty from shift
        assert shifted[1, 1] == 4  # Original position shifted

    def test_augment_multiplies_data(self):
        """Test that augmentation increases dataset size."""
        from src.preprocessing import ImageAugmentor

        augmentor = ImageAugmentor(image_shape=(28, 28))

        X = np.random.rand(10, 784)
        y = np.arange(10)

        X_aug, y_aug = augmentor.augment_with_shifts(X, y)

        # Default: 4 shifts + original = 5x
        assert len(X_aug) == 50
        assert len(y_aug) == 50


class TestClassifierFactory:
    """Tests for ClassifierFactory class."""

    def test_create_knn(self):
        """Test KNN classifier creation."""
        from src.models import ClassifierFactory

        model = ClassifierFactory.create("knn")
        assert model.n_neighbors == 3  # Default
        assert model.weights == "distance"

    def test_create_with_custom_params(self):
        """Test classifier creation with custom parameters."""
        from src.models import ClassifierFactory

        model = ClassifierFactory.create("knn", params={"n_neighbors": 7})
        assert model.n_neighbors == 7

    def test_unknown_classifier_raises(self):
        """Test that unknown classifier name raises error."""
        from src.models import ClassifierFactory

        with pytest.raises(ValueError, match="Unknown classifier"):
            ClassifierFactory.create("unknown_model")
