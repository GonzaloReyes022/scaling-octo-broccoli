"""Image augmentation utilities for MNIST and similar datasets."""

from typing import List, Tuple

import numpy as np
from scipy.ndimage import shift


class ImageAugmentor:
    """Augment images with various transformations."""

    def __init__(self, image_shape: Tuple[int, int] = (28, 28)):
        """
        Initialize augmentor.

        Args:
            image_shape: Shape of images (height, width)
        """
        self.image_shape = image_shape

    def shift_image(
        self,
        image: np.ndarray,
        dx: int,
        dy: int,
        cval: float = 0,
    ) -> np.ndarray:
        """
        Shift an image by dx, dy pixels.

        Args:
            image: 2D image array or flattened 1D array
            dx: Horizontal shift (positive = right)
            dy: Vertical shift (positive = down)
            cval: Fill value for empty pixels

        Returns:
            Shifted image in same shape as input
        """
        was_flat = image.ndim == 1
        if was_flat:
            image = image.reshape(self.image_shape)

        shifted = shift(image, [dy, dx], cval=cval, mode="constant")

        if was_flat:
            shifted = shifted.flatten()

        return shifted

    def augment_with_shifts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shifts: List[Tuple[int, int]] | None = None,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset by creating shifted copies of each image.

        Args:
            X: Features array (n_samples, n_features) or (n_samples, h, w)
            y: Labels array
            shifts: List of (dx, dy) shifts. Default: 4-directional 1-pixel shifts
            shuffle: Whether to shuffle the augmented dataset
            random_state: Random seed for shuffling

        Returns:
            (X_augmented, y_augmented) tuple
        """
        if shifts is None:
            shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Start with original data
        X_augmented = [x for x in X]
        y_augmented = [label for label in y]

        # Add shifted versions
        for dx, dy in shifts:
            for image, label in zip(X, y):
                shifted = self.shift_image(image, dx, dy)
                X_augmented.append(shifted)
                y_augmented.append(label)

        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)

        if shuffle:
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(len(X_augmented))
            X_augmented = X_augmented[indices]
            y_augmented = y_augmented[indices]

        return X_augmented, y_augmented

    def augment_factor(self, shifts: List[Tuple[int, int]] | None = None) -> int:
        """Return the multiplication factor for dataset size."""
        if shifts is None:
            shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return 1 + len(shifts)  # Original + shifted copies
