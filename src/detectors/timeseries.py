"""
Time-series anomaly detection methods.

Implements temporal-aware approaches for identifying anomalies:
- Seasonal Decomposition detector (STL-based residual analysis)
- CUSUM (Cumulative Sum) change-point detector
- Exponential Smoothing residual detector

Author: Gabriel Demetrios Lafis
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from src.detectors.statistical import BaseDetector


class SeasonalDecompositionDetector(BaseDetector):
    """
    Seasonal decomposition-based anomaly detector.

    Decomposes a time series into trend, seasonal, and residual
    components. Anomalies are detected in the residual component
    using statistical thresholds.
    """

    def __init__(
        self,
        period: int = 24,
        threshold_sigma: float = 3.0,
        contamination: float = 0.05,
    ):
        """
        Initialize Seasonal Decomposition detector.

        Args:
            period: Expected seasonal period (e.g., 24 for hourly data with daily cycle).
            threshold_sigma: Number of standard deviations in residuals for anomaly.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        if period < 2:
            raise ValueError("period must be at least 2")
        self.period = period
        self.threshold_sigma = threshold_sigma
        self._seasonal_pattern: Optional[np.ndarray] = None
        self._residual_std: Optional[float] = None
        self._residual_mean: Optional[float] = None

    def _moving_average(self, x: np.ndarray, window: int) -> np.ndarray:
        """Compute centered moving average."""
        if window % 2 == 0:
            # Two-pass for even window (convolution approach)
            kernel = np.ones(window) / window
            ma = np.convolve(x, kernel, mode="same")
        else:
            kernel = np.ones(window) / window
            ma = np.convolve(x, kernel, mode="same")
        return ma

    def _decompose(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose time series into trend, seasonal, and residual.

        Uses a classical additive decomposition approach.
        """
        n = len(x)

        # Trend via moving average
        trend = self._moving_average(x, self.period)

        # Detrended series
        detrended = x - trend

        # Seasonal component: average of detrended values at each position
        seasonal = np.zeros(n)
        for i in range(self.period):
            indices = np.arange(i, n, self.period)
            seasonal_mean = np.mean(detrended[indices])
            seasonal[indices] = seasonal_mean

        # Normalize seasonal component
        seasonal -= np.mean(seasonal[:self.period])

        # Residual
        residual = x - trend - seasonal

        return trend, seasonal, residual

    def fit(self, X: np.ndarray) -> "SeasonalDecompositionDetector":
        """
        Fit the seasonal decomposition detector.

        Args:
            X: Training time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            self
        """
        X = self._validate_input(X)
        if X.shape[1] != 1:
            raise ValueError("Seasonal decomposition only supports univariate data")
        x = X.ravel()

        if len(x) < 2 * self.period:
            raise ValueError(
                f"Need at least {2 * self.period} samples for period={self.period}"
            )

        _, seasonal, residual = self._decompose(x)

        # Store seasonal pattern (one full cycle)
        self._seasonal_pattern = np.zeros(self.period)
        for i in range(self.period):
            indices = np.arange(i, len(x), self.period)
            self._seasonal_pattern[i] = np.mean(seasonal[indices])

        # Statistics of residuals (exclude edges affected by moving average)
        margin = self.period
        clean_residual = residual[margin:-margin]
        self._residual_mean = np.mean(clean_residual)
        self._residual_std = np.std(clean_residual, ddof=1)
        if self._residual_std == 0:
            self._residual_std = 1.0

        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores from residual analysis.

        Args:
            X: Time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        x = X.ravel()

        _, _, residual = self._decompose(x)
        scores = np.abs(residual - self._residual_mean) / self._residual_std
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the time series.

        Args:
            X: Time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        scores = self.score_samples(X)
        return (scores > self.threshold_sigma).astype(int)


class CUSUMDetector(BaseDetector):
    """
    Cumulative Sum (CUSUM) change-point and anomaly detector.

    Detects shifts in the mean of a process by accumulating
    deviations from a target value. When the cumulative sum
    exceeds a threshold, an anomaly is signaled.
    """

    def __init__(
        self,
        drift: float = 0.5,
        threshold: float = 5.0,
        contamination: float = 0.05,
    ):
        """
        Initialize CUSUM detector.

        Args:
            drift: Allowable drift (slack) before accumulation.
                   Typically set to half the expected shift.
            threshold: Decision threshold for cumulative sum.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        self.drift = drift
        self.threshold = threshold
        self._target: Optional[float] = None
        self._std: Optional[float] = None

    def fit(self, X: np.ndarray) -> "CUSUMDetector":
        """
        Fit CUSUM on training data to establish baseline.

        Args:
            X: Training time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            self
        """
        X = self._validate_input(X)
        if X.shape[1] != 1:
            raise ValueError("CUSUM only supports univariate data")
        x = X.ravel()
        self._target = np.mean(x)
        self._std = np.std(x, ddof=1)
        if self._std == 0:
            self._std = 1.0
        self._is_fitted = True
        return self

    def _compute_cusum(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute positive and negative CUSUM statistics.

        Args:
            x: Input time series array.

        Returns:
            Tuple of (cusum_pos, cusum_neg) arrays.
        """
        normalized = (x - self._target) / self._std
        n = len(normalized)
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(1, n):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + normalized[i] - self.drift)
            cusum_neg[i] = max(0, cusum_neg[i - 1] - normalized[i] - self.drift)

        return cusum_pos, cusum_neg

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute CUSUM-based anomaly scores.

        The score is the maximum of positive and negative CUSUM values.

        Args:
            X: Time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        x = X.ravel()
        cusum_pos, cusum_neg = self._compute_cusum(x)
        return np.maximum(cusum_pos, cusum_neg)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using CUSUM threshold.

        Args:
            X: Time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)


class ExponentialSmoothingDetector(BaseDetector):
    """
    Exponential Smoothing residual-based anomaly detector.

    Applies simple or double exponential smoothing to produce
    forecasts, then detects anomalies based on the magnitude
    of forecast residuals.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: Optional[float] = None,
        threshold_sigma: float = 3.0,
        contamination: float = 0.05,
    ):
        """
        Initialize Exponential Smoothing detector.

        Args:
            alpha: Smoothing factor for the level (0 < alpha < 1).
            beta: Smoothing factor for the trend. If None, uses simple
                  exponential smoothing (no trend).
            threshold_sigma: Number of residual standard deviations for anomaly.
            contamination: Expected proportion of anomalies.
        """
        super().__init__(contamination=contamination)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if beta is not None and not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")
        self.alpha = alpha
        self.beta = beta
        self.threshold_sigma = threshold_sigma
        self._residual_std: Optional[float] = None
        self._residual_mean: Optional[float] = None
        self._last_level: Optional[float] = None
        self._last_trend: Optional[float] = None

    def _smooth(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply exponential smoothing and compute residuals.

        Args:
            x: Time series array.

        Returns:
            Tuple of (smoothed values, residuals).
        """
        n = len(x)
        smoothed = np.zeros(n)
        smoothed[0] = x[0]

        if self.beta is not None:
            # Double exponential smoothing (Holt's method)
            level = x[0]
            trend = x[1] - x[0] if n > 1 else 0.0

            for i in range(1, n):
                new_level = self.alpha * x[i] + (1 - self.alpha) * (level + trend)
                new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
                smoothed[i] = new_level + new_trend
                level = new_level
                trend = new_trend

            self._last_level = level
            self._last_trend = trend
        else:
            # Simple exponential smoothing
            level = x[0]
            for i in range(1, n):
                level = self.alpha * x[i] + (1 - self.alpha) * level
                smoothed[i] = level
            self._last_level = level

        # Residuals: difference between actual and one-step-ahead forecast
        # For forecasting, the smoothed value at time t is the forecast for t+1
        forecasts = np.zeros(n)
        forecasts[0] = x[0]
        forecasts[1:] = smoothed[:-1]
        residuals = x - forecasts

        return smoothed, residuals

    def fit(self, X: np.ndarray) -> "ExponentialSmoothingDetector":
        """
        Fit the exponential smoothing detector.

        Args:
            X: Training time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            self
        """
        X = self._validate_input(X)
        if X.shape[1] != 1:
            raise ValueError("Exponential smoothing only supports univariate data")
        x = X.ravel()

        _, residuals = self._smooth(x)

        # Skip first few residuals (warm-up)
        warmup = max(3, len(x) // 10)
        stable_residuals = residuals[warmup:]

        self._residual_mean = np.mean(stable_residuals)
        self._residual_std = np.std(stable_residuals, ddof=1)
        if self._residual_std == 0:
            self._residual_std = 1.0

        self._is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores from smoothing residuals.

        Args:
            X: Time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            Array of anomaly scores per sample.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        X = self._validate_input(X)
        x = X.ravel()

        _, residuals = self._smooth(x)
        scores = np.abs(residuals - self._residual_mean) / self._residual_std
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on smoothing residuals.

        Args:
            X: Time series of shape (n_samples,) or (n_samples, 1).

        Returns:
            Binary array: 1 for anomaly, 0 for normal.
        """
        scores = self.score_samples(X)
        return (scores > self.threshold_sigma).astype(int)
