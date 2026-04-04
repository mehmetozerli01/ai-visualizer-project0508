"""
Machine learning analysis core: scaling, clustering, PCA, and anomaly detection.

This module is intentionally free of Streamlit or plotting dependencies.
Numeric column selection aligns with :class:`processor.DataLoader` inference
so the same features drive statistics, cleaning, and modeling.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from exceptions import AIModelError
from processor import DataLoader


class AIEngine:
    """Run distance-based ML models on tabular data with consistent preprocessing.

    Public methods expect a non-empty ``pandas.DataFrame`` and scale numeric
    features with ``StandardScaler`` before fitting, which stabilizes K-Means,
    PCA, and Isolation Forest in feature space.

    Attributes:
        _random_state: Integer seed passed to stochastic estimators.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Create an engine with a reproducible RNG seed for all estimators.

        Args:
            random_state: Integer seed used by K-Means, PCA, and Isolation
                Forest.
        """
        self._random_state: Final[int] = random_state

    def _prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Select numeric columns, coerce values, impute missing values, and scale.

        Uses ``DataLoader.infer_column_types`` so numeric detection matches the
        ingestion layer (including numeric-like strings).

        Returns:
            A tuple ``(X_scaled, numeric_column_names)`` where ``X_scaled`` is a
            ``float64`` array of shape ``(n_samples, n_numeric_features)``.

        Raises:
            AIModelError: If the frame is empty, has no usable numeric columns,
                or scaling fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise AIModelError("Input must be a pandas DataFrame.")
        if df.empty:
            raise AIModelError("Input DataFrame is empty.")

        try:
            numeric_cols, _ = DataLoader.infer_column_types(df)
            if not numeric_cols:
                raise AIModelError(
                    "No numeric columns found. Ensure the dataset has numeric "
                    "features or numeric-like text columns after cleaning."
                )

            X_frame = df[numeric_cols].copy()
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(X_frame[col]):
                    X_frame[col] = pd.to_numeric(X_frame[col], errors="coerce")

            col_means = X_frame.mean()
            X_frame = X_frame.fillna(col_means).fillna(0.0)

            X = np.asarray(X_frame, dtype=np.float64)
            if X.shape[0] == 0 or X.shape[1] == 0:
                raise AIModelError("Prepared matrix has no rows or no features.")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(f"Data preparation failed: {exc}") from exc

        return X_scaled, numeric_cols

    def perform_clustering(
        self, df: pd.DataFrame, n_clusters: int = 3
    ) -> np.ndarray:
        """Partition rows into ``n_clusters`` groups using K-Means on scaled data.

        Args:
            df: Input table; only inferred numeric columns are used.
            n_clusters: Number of clusters (>= 1 and at most the row count).

        Returns:
            Integer cluster label per row, shape ``(n_samples,)``, aligned with
            ``df.index``.

        Raises:
            AIModelError: If arguments are invalid or K-Means fails to fit.
        """
        if n_clusters < 1:
            raise AIModelError("n_clusters must be at least 1.")

        try:
            X, _ = self._prepare_data(df)
            n_samples = int(X.shape[0])
            if n_clusters > n_samples:
                raise AIModelError(
                    f"n_clusters ({n_clusters}) cannot exceed the number of "
                    f"rows ({n_samples})."
                )

            model = KMeans(
                n_clusters=n_clusters,
                random_state=self._random_state,
                n_init=10,
            )
            labels = model.fit_predict(X)
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(f"Clustering failed: {exc}") from exc

        return np.asarray(labels, dtype=np.int32)

    def perform_pca(
        self, df: pd.DataFrame, n_components: int = 2
    ) -> pd.DataFrame:
        """Linearly project scaled numeric data onto principal components.

        Args:
            df: Input table; only inferred numeric columns are used.
            n_components: Number of components; cannot exceed
                ``min(n_samples, n_features)``.

        Returns:
            Data frame with columns ``PC1`` … ``PC{n}``, index aligned with
            ``df.index``.

        Raises:
            AIModelError: If ``n_components`` is invalid or PCA fails.
        """
        if n_components < 1:
            raise AIModelError("n_components must be at least 1.")

        try:
            X, _ = self._prepare_data(df)
            n_samples, n_features = X.shape
            max_components = int(min(n_samples, n_features))
            if n_components > max_components:
                raise AIModelError(
                    f"n_components ({n_components}) cannot exceed "
                    f"min(n_samples, n_features) ({max_components})."
                )

            model = PCA(
                n_components=n_components,
                random_state=self._random_state,
            )
            coordinates = model.fit_transform(X)
            column_names = [f"PC{i + 1}" for i in range(n_components)]
            result = pd.DataFrame(
                coordinates,
                columns=column_names,
                index=df.index,
            )
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(f"PCA failed: {exc}") from exc

        return result

    def detect_anomalies(
        self, df: pd.DataFrame, contamination: float = 0.1
    ) -> np.ndarray:
        """Flag outliers using Isolation Forest on scaled numeric features.

        Args:
            df: Input table; only inferred numeric columns are used.
            contamination: Expected proportion of anomalies in ``(0.0, 0.5]``
                as required by scikit-learn.

        Returns:
            Per-row predictions: ``-1`` anomaly (outlier), ``1`` normal
            (inlier), same length and order as ``df``.

        Raises:
            AIModelError: If ``contamination`` is outside the allowed interval
                or fitting fails.
        """
        if contamination <= 0.0 or contamination > 0.5:
            raise AIModelError(
                "contamination must be in the interval (0.0, 0.5] for "
                "IsolationForest."
            )

        try:
            X, _ = self._prepare_data(df)
            model = IsolationForest(
                contamination=contamination,
                random_state=self._random_state,
            )
            predictions = model.fit_predict(X)
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(f"Anomaly detection failed: {exc}") from exc

        return np.asarray(predictions, dtype=np.int8)
