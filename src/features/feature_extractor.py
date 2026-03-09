"""
Feature engineering for anomaly detection.

Extracts informative features from raw time-series or tabular data:
- Rolling window statistics (mean, std, min, max, skew, kurtosis)
- Lag features (auto-regressive predictors)
- Frequency domain features (FFT-based spectral analysis)
- Rate-of-change and difference features

Author: Gabriel Demetrios Lafis
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureExtractor:
    """
    Feature extractor for anomaly detection pipelines.

    Transforms raw univariate or multivariate data into a rich
    feature representation suitable for downstream anomaly detectors.
    """

    def __init__(
        self,
        rolling_windows: Optional[List[int]] = None,
        lag_orders: Optional[List[int]] = None,
        fft_n_components: int = 10,
        include_rolling: bool = True,
        include_lags: bool = True,
        include_fft: bool = True,
        include_diff: bool = True,
    ):
        """
        Initialize the feature extractor.

        Args:
            rolling_windows: Window sizes for rolling statistics.
                             Defaults to [5, 10, 20].
            lag_orders: Lag orders for auto-regressive features.
                        Defaults to [1, 2, 3, 5, 10].
            fft_n_components: Number of top FFT components to retain.
            include_rolling: Whether to compute rolling statistics.
            include_lags: Whether to compute lag features.
            include_fft: Whether to compute FFT features.
            include_diff: Whether to compute difference features.
        """
        self.rolling_windows = rolling_windows or [5, 10, 20]
        self.lag_orders = lag_orders or [1, 2, 3, 5, 10]
        self.fft_n_components = fft_n_components
        self.include_rolling = include_rolling
        self.include_lags = include_lags
        self.include_fft = include_fft
        self.include_diff = include_diff

    def extract_rolling_features(
        self, data: np.ndarray, column_name: str = "value"
    ) -> pd.DataFrame:
        """
        Compute rolling window statistics.

        For each window size, computes: mean, std, min, max, skewness, kurtosis.

        Args:
            data: 1D array of values.
            column_name: Name prefix for the generated columns.

        Returns:
            DataFrame with rolling statistics features.
        """
        series = pd.Series(data, name=column_name)
        features = {}

        for w in self.rolling_windows:
            rolling = series.rolling(window=w, min_periods=1)
            features[f"{column_name}_roll_mean_{w}"] = rolling.mean()
            features[f"{column_name}_roll_std_{w}"] = rolling.std().fillna(0)
            features[f"{column_name}_roll_min_{w}"] = rolling.min()
            features[f"{column_name}_roll_max_{w}"] = rolling.max()
            features[f"{column_name}_roll_range_{w}"] = (
                rolling.max() - rolling.min()
            )
            features[f"{column_name}_roll_skew_{w}"] = rolling.skew().fillna(0)
            features[f"{column_name}_roll_kurt_{w}"] = (
                rolling.kurt().fillna(0)
            )
            # Z-score within rolling window
            roll_mean = rolling.mean()
            roll_std = rolling.std().fillna(1).replace(0, 1)
            features[f"{column_name}_roll_zscore_{w}"] = (
                (series - roll_mean) / roll_std
            ).fillna(0)

        return pd.DataFrame(features)

    def extract_lag_features(
        self, data: np.ndarray, column_name: str = "value"
    ) -> pd.DataFrame:
        """
        Compute lag features for auto-regressive analysis.

        Args:
            data: 1D array of values.
            column_name: Name prefix for the generated columns.

        Returns:
            DataFrame with lag features.
        """
        series = pd.Series(data, name=column_name)
        features = {}

        for lag in self.lag_orders:
            features[f"{column_name}_lag_{lag}"] = series.shift(lag).fillna(
                method="bfill"
            )
            features[f"{column_name}_diff_{lag}"] = (
                series - series.shift(lag)
            ).fillna(0)
            features[f"{column_name}_pct_change_{lag}"] = (
                series.pct_change(periods=lag).fillna(0).replace(
                    [np.inf, -np.inf], 0
                )
            )

        return pd.DataFrame(features)

    def extract_fft_features(
        self, data: np.ndarray, column_name: str = "value"
    ) -> pd.DataFrame:
        """
        Extract frequency domain features using Fast Fourier Transform.

        Computes the dominant frequencies and their magnitudes,
        plus aggregate spectral statistics.

        Args:
            data: 1D array of values.
            column_name: Name prefix for the generated columns.

        Returns:
            DataFrame with FFT-based features (single row per input).
        """
        n = len(data)
        fft_vals = np.fft.rfft(data)
        fft_magnitudes = np.abs(fft_vals)
        fft_phases = np.angle(fft_vals)
        freqs = np.fft.rfftfreq(n)

        features = {}

        # Top-k components by magnitude (skip DC component)
        if len(fft_magnitudes) > 1:
            mag_no_dc = fft_magnitudes[1:]
            top_k = min(self.fft_n_components, len(mag_no_dc))
            top_indices = np.argsort(mag_no_dc)[::-1][:top_k]

            for i, idx in enumerate(top_indices):
                features[f"{column_name}_fft_mag_{i}"] = mag_no_dc[idx]
                if idx + 1 < len(freqs):
                    features[f"{column_name}_fft_freq_{i}"] = freqs[idx + 1]

        # Spectral statistics
        if len(fft_magnitudes) > 1:
            features[f"{column_name}_fft_total_power"] = np.sum(
                fft_magnitudes[1:] ** 2
            )
            features[f"{column_name}_fft_mean_mag"] = np.mean(fft_magnitudes[1:])
            features[f"{column_name}_fft_std_mag"] = np.std(fft_magnitudes[1:])
            features[f"{column_name}_fft_max_mag"] = np.max(fft_magnitudes[1:])

            # Spectral entropy
            psd = fft_magnitudes[1:] ** 2
            psd_norm = psd / (np.sum(psd) + 1e-10)
            spectral_entropy = -np.sum(
                psd_norm * np.log2(psd_norm + 1e-10)
            )
            features[f"{column_name}_fft_spectral_entropy"] = spectral_entropy

            # Spectral centroid
            if np.sum(fft_magnitudes[1:]) > 0:
                centroid = np.sum(
                    freqs[1:len(fft_magnitudes)] * fft_magnitudes[1:]
                ) / np.sum(fft_magnitudes[1:])
                features[f"{column_name}_fft_spectral_centroid"] = centroid

        return pd.DataFrame([features])

    def extract_diff_features(
        self, data: np.ndarray, column_name: str = "value"
    ) -> pd.DataFrame:
        """
        Compute difference and rate-of-change features.

        Args:
            data: 1D array of values.
            column_name: Name prefix for the generated columns.

        Returns:
            DataFrame with difference features.
        """
        series = pd.Series(data, name=column_name)
        features = {}

        # First and second order differences
        features[f"{column_name}_diff1"] = series.diff().fillna(0)
        features[f"{column_name}_diff2"] = series.diff().diff().fillna(0)

        # Absolute differences
        features[f"{column_name}_abs_diff1"] = series.diff().abs().fillna(0)

        # Cumulative sum
        features[f"{column_name}_cumsum"] = series.cumsum()

        # Expanding statistics
        expanding = series.expanding(min_periods=1)
        features[f"{column_name}_expanding_mean"] = expanding.mean()
        features[f"{column_name}_expanding_std"] = expanding.std().fillna(0)

        # Deviation from expanding mean
        exp_mean = expanding.mean()
        exp_std = expanding.std().fillna(1).replace(0, 1)
        features[f"{column_name}_deviation_from_mean"] = (
            (series - exp_mean) / exp_std
        ).fillna(0)

        return pd.DataFrame(features)

    def transform(
        self,
        data: np.ndarray,
        column_name: str = "value",
    ) -> pd.DataFrame:
        """
        Extract all configured features from the input data.

        Args:
            data: 1D array of values.
            column_name: Name prefix for generated feature columns.

        Returns:
            DataFrame with all extracted features.
        """
        data = np.asarray(data, dtype=np.float64).ravel()
        parts = [pd.DataFrame({column_name: data})]

        if self.include_rolling:
            parts.append(self.extract_rolling_features(data, column_name))

        if self.include_lags:
            parts.append(self.extract_lag_features(data, column_name))

        if self.include_diff:
            parts.append(self.extract_diff_features(data, column_name))

        result = pd.concat(parts, axis=1)

        # Replace any remaining NaN/inf
        result = result.replace([np.inf, -np.inf], 0).fillna(0)

        return result

    def transform_multivariate(
        self,
        data: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract features from multivariate data.

        Applies feature extraction to each column independently
        and concatenates the results.

        Args:
            data: 2D array of shape (n_samples, n_features).
            column_names: Optional names for each feature column.

        Returns:
            DataFrame with all extracted features for all columns.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_features = data.shape[1]
        if column_names is None:
            column_names = [f"feature_{i}" for i in range(n_features)]

        if len(column_names) != n_features:
            raise ValueError(
                f"column_names length ({len(column_names)}) must match "
                f"data columns ({n_features})"
            )

        parts = []
        for i, name in enumerate(column_names):
            parts.append(self.transform(data[:, i], name))

        return pd.concat(parts, axis=1)

    def get_feature_names(
        self, column_name: str = "value"
    ) -> List[str]:
        """
        Get the list of feature names that would be generated.

        Args:
            column_name: Base column name.

        Returns:
            List of feature name strings.
        """
        # Generate features for a small dummy array to get names
        dummy = np.random.randn(max(self.rolling_windows + self.lag_orders) + 5)
        features = self.transform(dummy, column_name)
        return list(features.columns)
