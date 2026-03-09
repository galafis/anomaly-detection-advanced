"""
Machine Learning based anomaly detection methods.

Implements ML approaches for identifying anomalies:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- DBSCAN-based detector

Author: Gabriel Demetrios Lafis
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.detectors.statistical import BaseDetector


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest anomaly detector.

    Isolates anomalies using random binary partitions. Anomalous points
    tend to have shorter average path lengths in the isolation trees,
    as they are easier to separate from the rest of the data.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        contamination: float = 0.05,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            n_estimators: Number of isolation trees.
            max_samples: Number of samples to draw for each tree.
            contamination: Expected proportion of anomalies.
            random_state: Random seed for reproducibility.
        """
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest on training data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_input(X)
        X_scaled = self._scaler.fit_transform(X)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (negated decision function).

        Higher values indicate more anomalous points.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        # Negate so higher = more anomalous
        return -self._model.decision_function(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        # sklearn returns -1 for anomaly, 1 for normal
        return (preds == -1).astype(int)


class LOFDetector(BaseDetector):
    """
    Local Outlier Factor (LOF) anomaly detector.

    Measures the local density deviation of a data point relative
    to its neighbors. Points with substantially lower density than
    their neighbors are classified as anomalies.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        metric: str = "minkowski",
        novelty: bool = True,
    ):
        """
        Initialize LOF detector.

        Args:
            n_neighbors: Number of neighbors to consider.
            contamination: Expected proportion of anomalies.
            metric: Distance metric to use.
            novelty: If True, can predict on new unseen data.
        """
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.novelty = novelty
        self._model: Optional[LocalOutlierFactor] = None
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "LOFDetector":
        """
        Fit the LOF model on training data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_input(X)
        X_scaled = self._scaler.fit_transform(X)
        self._model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X_scaled) - 1),
            contamination=self.contamination,
            metric=self.metric,
            novelty=self.novelty,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using LOF.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        # Negate so higher = more anomalous
        return -self._model.decision_function(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return (preds == -1).astype(int)


class OneClassSVMDetector(BaseDetector):
    """
    One-Class SVM anomaly detector.

    Learns a boundary around the normal data distribution using
    a support vector machine. Points outside the learned boundary
    are classified as anomalies.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str = "scale",
        nu: float = 0.05,
        contamination: float = 0.05,
    ):
        """
        Initialize One-Class SVM detector.

        Args:
            kernel: Kernel type for SVM.
            gamma: Kernel coefficient.
            nu: Upper bound on the fraction of training errors.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self._model: Optional[OneClassSVM] = None
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """
        Fit the One-Class SVM on training data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_input(X)
        X_scaled = self._scaler.fit_transform(X)
        self._model = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu,
        )
        self._model.fit(X_scaled)
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using the SVM decision function.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        return -self._model.decision_function(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return (preds == -1).astype(int)


class DBSCANDetector(BaseDetector):
    """
    DBSCAN-based anomaly detector.

    Uses density-based spatial clustering to identify noise points
    (samples not assigned to any cluster) as anomalies.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        contamination: float = 0.05,
        metric: str = "euclidean",
    ):
        """
        Initialize DBSCAN detector.

        Args:
            eps: Maximum distance between two samples in a neighborhood.
            min_samples: Minimum number of samples in a neighborhood
                        for a point to be a core point.
            contamination: Expected proportion of anomalies.
            metric: Distance metric to use.
        """
        super().__init__(contamination=contamination)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._model: Optional[DBSCAN] = None
        self._scaler = StandardScaler()
        self._train_labels: Optional[np.ndarray] = None
        self._train_data: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCANDetector":
        """
        Fit DBSCAN on training data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        X = self._validate_input(X)
        X_scaled = self._scaler.fit_transform(X)
        self._model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=-1,
        )
        self._train_labels = self._model.fit_predict(X_scaled)
        self._train_data = X_scaled.copy()
        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores based on distance to nearest core point.

        Points farther from core samples receive higher anomaly scores.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)

        # Find core sample indices
        core_mask = self._train_labels != -1
        if not np.any(core_mask):
            return np.ones(len(X_scaled))

        core_samples = self._train_data[core_mask]

        # Compute distance to nearest core point
        scores = np.zeros(len(X_scaled))
        for i, point in enumerate(X_scaled):
            distances = np.linalg.norm(core_samples - point, axis=1)
            scores[i] = np.min(distances)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using DBSCAN clustering.

        Points that would be classified as noise are anomalies.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        X_scaled = self._scaler.transform(X)

        core_mask = self._train_labels != -1
        if not np.any(core_mask):
            return np.ones(len(X_scaled), dtype=int)

        core_samples = self._train_data[core_mask]

        predictions = np.zeros(len(X_scaled), dtype=int)
        for i, point in enumerate(X_scaled):
            distances = np.linalg.norm(core_samples - point, axis=1)
            min_dist = np.min(distances)
            if min_dist > self.eps:
                predictions[i] = 1

        return predictions
