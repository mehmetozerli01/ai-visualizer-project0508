"""
Plotly-based interactive charts for clustering, anomalies, and feature views.

This module stays independent of Streamlit; callers choose ``theme`` to match
the host application's light or dark UI. Correlation views use
:class:`processor.DataLoader` so numeric columns match the ingestion layer.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from processor import DataLoader


class DataVisualizer:
    """Build consistent Plotly figures with shared layout and theming.

    Args:
        theme: ``"light"`` uses the ``plotly_white`` template; ``"dark"`` uses
            ``plotly_dark`` for axes, grid, and background harmony with dark UIs.
    """

    def __init__(self, theme: Literal["light", "dark"] = "light") -> None:
        self._theme: Literal["light", "dark"] = theme
        self._template: str = "plotly_dark" if theme == "dark" else "plotly_white"

    def _base_layout(
        self,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        *,
        height: int = 520,
        legend: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return layout kwargs shared by all figures."""
        leg = {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
        if legend:
            leg.update(legend)
        return {
            "title": {"text": title, "x": 0.5, "xanchor": "center"},
            "xaxis_title": xaxis_title,
            "yaxis_title": yaxis_title,
            "template": self._template,
            "height": height,
            "margin": {"l": 60, "r": 40, "t": 80, "b": 60},
            "font": {"family": "system-ui, Segoe UI, sans-serif", "size": 13},
            "legend": leg,
            "hovermode": "closest",
        }

    @staticmethod
    def _resolve_pc_columns(df_pca: pd.DataFrame) -> tuple[str, str]:
        """Pick x/y PCA column names (PC1/PC2 preferred, else first two columns)."""
        if df_pca.shape[1] < 2:
            raise ValueError("df_pca must contain at least two coordinate columns.")
        cols = list(df_pca.columns)
        if "PC1" in df_pca.columns and "PC2" in df_pca.columns:
            return "PC1", "PC2"
        return str(cols[0]), str(cols[1])

    def plot_clustering(self, df_pca: pd.DataFrame, labels: np.ndarray | pd.Series) -> Figure:
        """Scatter PCA coordinates colored by cluster with rich hover.

        Extra columns in ``df_pca`` (beyond the two PC axes) are included in the
        hover tooltip as row-level detail. The ``labels`` array must align with
        ``df_pca.index`` row order.

        Args:
            df_pca: At least two columns for 2D coordinates (ideally ``PC1``,
                ``PC2``). Additional columns appear in hover.
            labels: Integer (or string) cluster id per row, same length as
                ``df_pca``.

        Returns:
            A ``plotly.graph_objects.Figure`` ready for ``st.plotly_chart``.

        Raises:
            ValueError: If shapes mismatch or PCA columns cannot be resolved.
        """
        if not isinstance(df_pca, pd.DataFrame):
            raise ValueError("df_pca must be a pandas DataFrame.")
        lab = np.asarray(labels)
        if lab.shape[0] != len(df_pca):
            raise ValueError(
                f"labels length ({lab.shape[0]}) must match df_pca rows ({len(df_pca)})."
            )

        pc_x, pc_y = self._resolve_pc_columns(df_pca)
        work = df_pca.copy()
        work["_cluster"] = lab.astype(str)

        hover_cols = [
            c
            for c in work.columns
            if c not in (pc_x, pc_y, "_cluster")
        ]
        hover_data = ["_cluster"] + hover_cols if hover_cols else ["_cluster"]

        fig = px.scatter(
            work,
            x=pc_x,
            y=pc_y,
            color="_cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={
                pc_x: pc_x,
                pc_y: pc_y,
                "_cluster": "Küme",
            },
            hover_data=hover_data,
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")))
        fig.update_layout(
            **self._base_layout(
                "K-Means kümeleme (PCA düzlemi)",
                xaxis_title=pc_x,
                yaxis_title=pc_y,
            )
        )
        return fig

    def plot_anomalies(
        self, df_pca: pd.DataFrame, anomaly_labels: np.ndarray | pd.Series
    ) -> Figure:
        """Plot PCA plane: normal points in blue, anomalies (-1) larger and red.

        Args:
            df_pca: Two-dimensional PCA coordinates (e.g. ``PC1``, ``PC2``).
            anomaly_labels: Per-row labels; ``-1`` anomaly, ``1`` normal
                (scikit-learn ``IsolationForest`` convention).

        Returns:
            Interactive ``Figure`` with two traces (normal vs anomali).

        Raises:
            ValueError: If lengths mismatch or PCA columns cannot be resolved.
        """
        if not isinstance(df_pca, pd.DataFrame):
            raise ValueError("df_pca must be a pandas DataFrame.")
        preds = np.asarray(anomaly_labels).ravel()
        if preds.shape[0] != len(df_pca):
            raise ValueError(
                "anomaly_labels length must match df_pca rows."
            )

        pc_x, pc_y = self._resolve_pc_columns(df_pca)
        x = df_pca[pc_x].to_numpy()
        y = df_pca[pc_y].to_numpy()
        is_anomaly = preds == -1
        normal = ~is_anomaly

        # Hover: index + coords + label
        idx = df_pca.index.astype(str)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x[normal],
                y=y[normal],
                mode="markers",
                name="Normal (1)",
                marker=dict(
                    color="#2563EB",
                    size=9,
                    opacity=0.82,
                    line=dict(width=0.5, color="rgba(255,255,255,0.35)"),
                ),
                text=[f"idx={i}<br>{pc_x}={xv:.4f}<br>{pc_y}={yv:.4f}<br>label=1" for i, xv, yv in zip(idx[normal], x[normal], y[normal])],
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x[is_anomaly],
                y=y[is_anomaly],
                mode="markers",
                name="Anomali (-1)",
                marker=dict(
                    color="#FF2D2D",
                    size=16,
                    opacity=0.95,
                    symbol="circle",
                    line=dict(width=2, color="#FFFFFF"),
                ),
                text=[f"idx={i}<br>{pc_x}={xv:.4f}<br>{pc_y}={yv:.4f}<br>label=-1" for i, xv, yv in zip(idx[is_anomaly], x[is_anomaly], y[is_anomaly])],
                hoverinfo="text",
            )
        )
        fig.update_layout(**self._base_layout("Anomali tespiti (PCA düzlemi)", pc_x, pc_y))
        return fig

    def plot_feature_distribution(self, df: pd.DataFrame, column: str) -> Figure:
        """Show distribution of a single column (histogram or bar).

        Numeric columns use a histogram; non-numeric columns use a bar chart of
        category frequencies.

        Args:
            df: Source table.
            column: Column name present in ``df``.

        Returns:
            ``Figure`` with titled axes and theme-aware styling.

        Raises:
            ValueError: If ``column`` is missing from ``df``.
        """
        if column not in df.columns:
            raise ValueError(f"Column {column!r} not found in DataFrame.")
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            fig = px.histogram(
                df,
                x=column,
                nbins=min(50, max(10, int(np.sqrt(len(df))))),
                color_discrete_sequence=["#6366F1"],
            )
            fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
            fig.update_layout(
                **self._base_layout(
                    f"Dağılım: {column}",
                    xaxis_title=str(column),
                    yaxis_title="Frekans",
                )
            )
        else:
            vc = (
                df[column]
                .astype(str)
                .replace({"nan": "(eksik)", "<NA>": "(eksik)"})
                .fillna("(eksik)")
                .value_counts()
                .reset_index()
            )
            vc.columns = ["category", "count"]
            fig = px.bar(
                vc,
                x="category",
                y="count",
                color_discrete_sequence=["#0EA5E9"],
            )
            fig.update_layout(
                **self._base_layout(
                    f"Kategori frekansı: {column}",
                    xaxis_title=str(column),
                    yaxis_title="Adet",
                )
            )
            fig.update_xaxes(tickangle=-35)

        return fig

    def plot_elbow_curve(
        self,
        elbow_df: pd.DataFrame,
        *,
        selected_k: int | None = None,
    ) -> Figure:
        """Line chart of K-Means inertia vs k (elbow method).

        Args:
            elbow_df: Data frame with columns ``k`` and ``inertia`` (e.g. from
                :meth:`ai_engine.AIEngine.elbow_inertia_scan`).
            selected_k: When set, draws a vertical dashed line at this cluster
                count to relate the plot to the user's K-Means setting.

        Returns:
            Themed Plotly figure.

        Raises:
            ValueError: If required columns are missing or the frame is empty.
        """
        if not isinstance(elbow_df, pd.DataFrame) or elbow_df.empty:
            raise ValueError("elbow_df must be a non-empty DataFrame.")
        if not {"k", "inertia"}.issubset(set(elbow_df.columns)):
            raise ValueError("elbow_df must contain columns 'k' and 'inertia'.")

        fig = px.line(
            elbow_df,
            x="k",
            y="inertia",
            markers=True,
            color_discrete_sequence=["#7C3AED"],
        )
        fig.update_traces(line=dict(width=2), marker=dict(size=8))
        fig.update_layout(
            **self._base_layout(
                "Dirsek yöntemi: K vs inertia (WCSS, ölçeklenmiş uzayda)",
                xaxis_title="k (küme sayısı)",
                yaxis_title="Inertia (küme içi kare uzaklık toplamı)",
            )
        )
        if selected_k is not None and selected_k >= 1:
            fig.add_vline(
                x=float(selected_k),
                line_width=2,
                line_dash="dash",
                line_color="#EA580C",
                annotation_text=f"Seçilen k = {selected_k}",
                annotation_position="top right",
            )
        return fig

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str] | None = None,
    ) -> Figure:
        """Pearson correlation heatmap for numeric columns.

        Args:
            df: Source table.
            numeric_columns: Columns to include; if ``None``, all columns that
                :meth:`processor.DataLoader.infer_column_types` marks as numeric
                are used. Non-numeric names are dropped so the heatmap uses only
                model-aligned numeric features.

        Returns:
            ``Figure`` with diverging colors around zero correlation.

        Raises:
            ValueError: If fewer than two numeric columns are available after
                selection.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")

        inferred_numeric, _ = DataLoader.infer_column_types(df)
        inferred_set = set(inferred_numeric)

        if numeric_columns is not None:
            missing = set(numeric_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Unknown columns: {sorted(missing)}")
            cols = [c for c in numeric_columns if c in inferred_set]
        else:
            cols = [c for c in inferred_numeric if c in df.columns]

        if len(cols) < 2:
            raise ValueError(
                "At least two inferred numeric columns are required for a "
                "correlation heatmap."
            )

        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        corr = sub.corr(numeric_only=True, method="pearson")
        if corr.shape[0] < 2:
            raise ValueError("Correlation matrix is too small to display.")

        fig = px.imshow(
            corr,
            aspect="equal",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            labels=dict(color="Pearson r"),
        )
        fig.update_traces(
            texttemplate="%{z:.2f}",
            textfont={"size": 11},
            hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.2f}<extra></extra>",
        )
        fig.update_layout(
            **self._base_layout(
                "Korelasyon matrisi (Pearson)",
                xaxis_title="",
                yaxis_title="",
                height=560,
            )
        )
        fig.update_xaxes(side="bottom")
        return fig
