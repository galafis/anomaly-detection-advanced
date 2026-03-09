"""
Statistical anomaly detection methods.

Implements classical statistical approaches for identifying outliers:
- Z-Score detector
- Modified Z-Score detector (MAD-based)
- Grubbs test detector
- Interquartile Range (IQR) detector

Author: Gabriel Demetrios Lafis
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors."""

    def __init__(self, contamination: float = 0.05):
        """
        Initialize the base detector.

        Args:
            contamination: Expected proportion of anomalies in the dataset.
                           Must be in (0, 0.5].
        """
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.contamination = contamination
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        """Fit the detector on training data."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 for anomaly, 0 for normal."""
        ...

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in a single call."""
        self.fit(X)
        return self.predict(X)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert input to numpy array."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D")
        if np.any(np.isnan(X)):
            raise ValueError("Input contains NaN values")
        return X

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has been fitted."""
        return self._is_fitted


class ZScoreDetector(BaseDetector):
    """
    Z-Score based anomaly detector.

    Detects anomalies by measuring how many standard deviations
    a data point is from the mean. Points beyond the threshold
    are flagged as anomalies.
    """

    def __init__(self, threshold: float = 3.0, contamination: float = 0.05):
        """
        Initialize Z-Score detector.

        Args:
            threshold: Number of standard deviations for anomaly boundary.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        """
        Compute mean and standard deviation from training data.

        Args:
            X: Training data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_input(X)
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0, ddof=1)
        # Prevent division by zero for constant features
        self._std[self._std == 0] = 1.0
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Z-scores for each sample.

        Args:
            X: Data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            Array of maximum absolute Z-scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        z_scores = np.abs((X - self._mean) / self._std)
        return np.max(z_scores, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using Z-score threshold.

        Args:
            X: Data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)


class ModifiedZScoreDetector(BaseDetector):
    """
    Modified Z-Score detector using Median Absolute Deviation (MAD).

    More robust than standard Z-Score for datasets with heavy tails
    or extreme outliers, since it uses the median instead of the mean.
    """

    MAD_SCALE = 0.6745  # Consistency constant for normal distribution

    def __init__(self, threshold: float = 3.5, contamination: float = 0.05):
        """
        Initialize Modified Z-Score detector.

        Args:
            threshold: Modified Z-score threshold for anomaly classification.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._median: Optional[np.ndarray] = None
        self._mad: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ModifiedZScoreDetector":
        """
        Compute median and MAD from training data.

        Args:
            X: Training data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_input(X)
        self._median = np.median(X, axis=0)
        self._mad = np.median(np.abs(X - self._median), axis=0)
        # Prevent division by zero
        self._mad[self._mad == 0] = 1.0
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute modified Z-scores for each sample.

        Args:
            X: Data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            Array of maximum modified Z-scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        modified_z = self.MAD_SCALE * np.abs(X - self._median) / self._mad
        return np.max(modified_z, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using modified Z-score threshold.

        Args:
            X: Data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)


class GrubbsTestDetector(BaseDetector):
    """
    Grubbs test for outlier detection.

    Applies the Grubbs statistical test to determine if the most extreme
    value in a dataset is an outlier. Supports iterative application
    to detect multiple outliers.
    """

    def __init__(self, alpha: float = 0.05, contamination: float = 0.05):
        """
        Initialize Grubbs test detector.

        Args:
            alpha: Significance level for the test.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._n: int = 0
        self._critical_value: Optional[float] = None

    def _compute_critical_value(self, n: int) -> float:
        """
        Compute the Grubbs critical value.

        Args:
            n: Number of samples.

        Returns:
            Critical value for the Grubbs test.
        """
        t_dist = sp_stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
        numerator = (n - 1) * t_dist
        denominator = np.sqrt(n) * np.sqrt(n - 2 + t_dist ** 2)
        return numerator / denominator

    def fit(self, X: np.ndarray) -> "GrubbsTestDetector":
        """
        Fit the Grubbs test on training data.

        Args:
            X: Training data of shape (n_samples,) or (n_samples, 1).

        Returns:
            self
        """
        X = self._validate_input(X)
        if X.shape[1] != 1:
            raise ValueError("Grubbs test only supports univariate data (1 feature)")
        x = X.ravel()
        self._n = len(x)
        if self._n < 3:
            raise ValueError("Need at least 3 samples for Grubbs test")
        self._mean = np.mean(x)
        self._std = np.std(x, ddof=1)
        if self._std == 0:
            self._std = 1.0
        self._critical_value = self._compute_critical_value(self._n)
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Grubbs test statistic for each sample.

        Args:
            X: Data of shape (n_samples,) or (n_samples, 1).

        Returns:
            Array of Grubbs statistics per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        x = X.ravel()
        return np.abs(x - self._mean) / self._std

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using Grubbs critical value.

        Args:
            X: Data of shape (n_samples,) or (n_samples, 1).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        scores = self.score_samples(X)
        return (scores > self._critical_value).astype(int)


class IQRDetector(BaseDetector):
    """
    Interquartile Range (IQR) based anomaly detector.

    Uses the IQR method to define fences beyond which data points
    are considered anomalies. Robust to non-normal distributions.
    """

    def __init__(self, factor: float = 1.5, contamination: float = 0.05):
        """
        Initialize IQR detector.

        Args:
            factor: Multiplier for IQR to set fence width.
                    1.5 detects mild outliers; 3.0 detects extreme outliers.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.factor = factor
        self._q1: Optional[np.ndarray] = None
        self._q3: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None
        self._lower_fence: Optional[np.ndarray] = None
        self._upper_fence: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "IQRDetector":
        """
        Compute quartiles and fences from training data.

        Args:
            X: Training data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_input(X)
        self._q1 = np.percentile(X, 25, axis=0)
        self._q3 = np.percentile(X, 75, axis=0)
        self._iqr = self._q3 - self._q1
        # Prevent zero IQR
        self._iqr[self._iqr == 0] = 1.0
        self._lower_fence = self._q1 - self.factor * self._iqr
        self._upper_fence = self._q3 + self.factor * self._iqr
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute IQR-based anomaly scores.

        The score is the maximum over features of the distance beyond
        the fences, normalized by the IQR.

        Args:
            X: Data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        lower_dist = np.maximum(0, self._lower_fence - X) / self._iqr
        upper_dist = np.maximum(0, X - self._upper_fence) / self._iqr
        combined = np.maximum(lower_dist, upper_dist)
        return np.max(combined, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using IQR fences.

        A point is anomalous if any feature falls outside the fences.

        Args:
            X: Data of shape (n_samples,) or (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        below = np.any(X < self._lower_fence, axis=1)
        above = np.any(X > self._upper_fence, axis=1)
        return (below | above).astype(int)

    def get_fences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the lower and upper fences.

        Returns:
            Tuple of (lower_fence, upper_fence) arrays.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        return self._lower_fence.copy(), self._upper_fence.copy()
