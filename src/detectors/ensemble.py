"""
Ensemble anomaly detection methods.

Combines multiple anomaly detectors using voting, stacking,
or weighted combination strategies for improved detection.

Author: Gabriel Demetrios Lafis
"""

from typing import Dict, List, Optional, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.detectors.statistical import BaseDetector


class EnsembleDetector(BaseDetector):
    """
    Ensemble anomaly detector combining multiple detection methods.

    Supports three combination strategies:
    - voting: Majority vote (or threshold-based) among detectors.
    - averaging: Weighted average of normalized anomaly scores.
    - stacking: Meta-learner trained on base detector outputs.
    """

    VALID_STRATEGIES = ("voting", "averaging", "stacking")

    def __init__(
        self,
        detectors: List[BaseDetector],
        strategy: Literal["voting", "averaging", "stacking"] = "averaging",
        weights: Optional[List[float]] = None,
        voting_threshold: float = 0.5,
        contamination: float = 0.05,
    ):
        """
        Initialize Ensemble detector.

        Args:
            detectors: List of base anomaly detectors.
            strategy: Combination strategy - 'voting', 'averaging', or 'stacking'.
            weights: Weights for each detector (used with 'averaging' and 'voting').
                     If None, uniform weights are used.
            voting_threshold: Fraction of detectors that must agree for voting.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        if not detectors:
            raise ValueError("At least one detector is required")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.VALID_STRATEGIES}, got '{strategy}'"
            )

        self.detectors = detectors
        self.strategy = strategy
        self.voting_threshold = voting_threshold

        if weights is not None:
            if len(weights) != len(detectors):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"detectors length ({len(detectors)})"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            n = len(detectors)
            self.weights = [1.0 / n] * n

        self._meta_learner: Optional[LogisticRegression] = None
        self._score_mins: Optional[np.ndarray] = None
        self._score_ranges: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "EnsembleDetector":
        """
        Fit all base detectors and optionally the meta-learner.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Labels for stacking strategy (1 = anomaly, 0 = normal).
               If None and strategy='stacking', uses base detector consensus.

        Returns:
            self
        """
        X = self._validate_input(X)

        # Fit all base detectors
        for detector in self.detectors:
            detector.fit(X)

        if self.strategy == "stacking":
            # Generate meta-features from base detector scores
            meta_features = self._get_meta_features(X)

            if y is None:
                # Use averaging as pseudo-labels
                avg_scores = np.average(meta_features, axis=1, weights=self.weights)
                threshold = np.percentile(avg_scores, 100 * (1 - self.contamination))
                y = (avg_scores > threshold).astype(int)

            self._meta_learner = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
            self._meta_learner.fit(meta_features, y)

        elif self.strategy == "averaging":
            # Compute normalization parameters from training scores
            scores_matrix = np.column_stack(
                [det.score_samples(X) for det in self.detectors]
            )
            self._score_mins = scores_matrix.min(axis=0)
            self._score_ranges = scores_matrix.max(axis=0) - self._score_mins
            self._score_ranges[self._score_ranges == 0] = 1.0

        self._is_fitted = True
        return self

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate meta-features from base detector scores.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Meta-feature matrix of shape (n_samples, n_detectors).
        """
        return np.column_stack(
            [det.score_samples(X) for det in self.detectors]
        )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ensemble anomaly scores.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Array of ensemble anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)

        if self.strategy == "stacking":
            meta_features = self._get_meta_features(X)
            # Use probability of anomaly class
            proba = self._meta_learner.predict_proba(meta_features)
            # Return probability of being anomalous
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba[:, 0]

        elif self.strategy == "averaging":
            scores_matrix = np.column_stack(
                [det.score_samples(X) for det in self.detectors]
            )
            # Min-max normalize each detector's scores
            normalized = (scores_matrix - self._score_mins) / self._score_ranges
            normalized = np.clip(normalized, 0, None)
            return np.average(normalized, axis=1, weights=self.weights)

        else:  # voting
            predictions = np.column_stack(
                [det.predict(X) for det in self.detectors]
            )
            return np.average(predictions, axis=1, weights=self.weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using ensemble.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)

        if self.strategy == "voting":
            predictions = np.column_stack(
                [det.predict(X) for det in self.detectors]
            )
            weighted_votes = np.average(predictions, axis=1, weights=self.weights)
            return (weighted_votes >= self.voting_threshold).astype(int)

        elif self.strategy == "stacking":
            meta_features = self._get_meta_features(X)
            return self._meta_learner.predict(meta_features).astype(int)

        else:  # averaging
            scores = self.score_samples(X)
            threshold = np.percentile(scores, 100 * (1 - self.contamination))
            return (scores > threshold).astype(int)

    def get_detector_weights(self) -> Dict[str, float]:
        """
        Return the weight assigned to each detector.

        Returns:
            Dictionary mapping detector class name to its weight.
        """
        return {
            type(det).__name__: w
            for det, w in zip(self.detectors, self.weights)
        }
