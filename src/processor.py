"""
Structured data loading, column inference, summarization, and cleaning.

This module exposes :class:`DataLoader` for CSV/Excel ingestion and preprocessing
steps used before machine learning in the visualization app.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import PurePath
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from exceptions import DataLoadError, PreprocessingError

# Minimum ratio of non-null values that must parse as numeric for an object
# column to be treated as numeric (avoids misclassifying IDs with few digits).
_NUMERIC_STRING_RATIO: float = 0.95


class DataLoader:
    """
    Load tabular files and prepare pandas DataFrames for downstream analysis.

    Responsibilities include format detection, parsing, schema inference
    (numeric vs categorical columns), descriptive summaries, and missing-value
    imputation tailored by column type.
    """

    @staticmethod
    def load_file(file_obj: BinaryIO) -> pd.DataFrame:
        """
        Read a CSV or Excel workbook from a binary file-like object.

        Parameters
        ----------
        file_obj
            Readable binary stream (e.g. Streamlit ``UploadedFile``). Must expose
            ``read()`` and the caller should provide ``name`` on the object when
            available for extension detection.

        Returns
        -------
        pd.DataFrame
            Parsed tabular data with original column names preserved.

        Raises
        ------
        DataLoadError
            If the format is unsupported, the file is empty, or parsing fails.
        """
        try:
            raw = file_obj.read()
        except OSError as exc:
            raise DataLoadError("Could not read uploaded file.") from exc

        if not raw:
            raise DataLoadError("Uploaded file is empty.")

        name = getattr(file_obj, "name", "") or ""
        suffix = PurePath(name).suffix.lower()

        buffer = BytesIO(raw)

        try:
            if suffix == ".csv" or suffix == "":
                df = pd.read_csv(buffer)
            elif suffix in (".xlsx", ".xlsm"):
                buffer.seek(0)
                df = pd.read_excel(buffer, engine="openpyxl")
            else:
                raise DataLoadError(
                    f"Unsupported file type {suffix!r}. Use .csv or .xlsx."
                )
        except DataLoadError:
            raise
        except ImportError as exc:
            raise DataLoadError(
                "Excel support requires the 'openpyxl' package to be installed."
            ) from exc
        except Exception as exc:
            raise DataLoadError(f"Failed to parse file: {exc}") from exc

        if df.empty:
            raise DataLoadError("File contains no rows.")

        return df

    @staticmethod
    def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Split columns into numeric and categorical groups for preprocessing.

        Boolean and datetime columns are treated as categorical for imputation
        rules (see :meth:`clean_data`). Object columns that are mostly numeric
        strings are classified as numeric after coercion.

        Parameters
        ----------
        df
            Input frame.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(numeric_columns, categorical_columns)`` with no duplicate names.
        """
        if df.empty:
            return [], []

        numeric: list[str] = []
        categorical: list[str] = []

        for col in df.columns:
            series = df[col]

            if DataLoader._is_numeric_feature_column(series):
                numeric.append(col)
            else:
                categorical.append(col)

        return numeric, categorical

    @staticmethod
    def _is_numeric_feature_column(series: pd.Series) -> bool:
        """Return True if the series should be imputed and scaled as numeric."""
        if pd.api.types.is_bool_dtype(series):
            return False
        if pd.api.types.is_datetime64_any_dtype(series):
            return False
        if pd.api.types.is_numeric_dtype(series):
            return True

        if series.dtype == object or pd.api.types.is_string_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            mask = series.notna()
            if not bool(mask.any()):
                return False
            ok = coerced.notna() & mask
            ratio = float(ok.sum()) / float(mask.sum())
            return ratio >= _NUMERIC_STRING_RATIO

        return False

    @staticmethod
    def get_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
        """
        Build a structured summary for UI or logging.

        Includes shape, inferred numeric/categorical column lists, per-column
        missing counts, ``describe`` output for numeric columns, and small
        frequency tables for categorical columns.

        Parameters
        ----------
        df
            Data to summarize.

        Returns
        -------
        dict[str, Any]
            Keys: ``row_count``, ``column_count``, ``columns``, ``numeric_columns``,
            ``categorical_columns``, ``missing_per_column``, ``numeric_describe``,
            ``categorical_top_values``.

        Raises
        ------
        PreprocessingError
            If ``df`` is empty or not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError("Expected a pandas DataFrame.")
        if df.empty:
            raise PreprocessingError("Cannot summarize an empty DataFrame.")

        numeric_cols, categorical_cols = DataLoader.infer_column_types(df)
        missing = df.isna().sum().astype(int).to_dict()

        numeric_describe: dict[str, dict[str, float]] = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().T
            numeric_describe = desc.replace({np.nan: None}).to_dict(orient="index")

        categorical_top: dict[str, dict[str, int]] = {}
        for col in categorical_cols:
            vc = df[col].value_counts(dropna=True).head(10)
            categorical_top[col] = {str(k): int(v) for k, v in vc.items()}

        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns.astype(str)),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_per_column": {str(k): int(v) for k, v in missing.items()},
            "numeric_describe": numeric_describe,
            "categorical_top_values": categorical_top,
        }

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of ``df`` with missing values filled by column type.

        Numeric columns (including object columns inferred as numeric) are filled
        with the column mean; if all values are missing, fills with ``0.0``.
        Categorical-like columns (object, string, category, bool, datetime) use
        ``'Unknown'`` where a string placeholder is appropriate; datetimes use
        the column median timestamp; booleans use the mode or ``False``.

        Parameters
        ----------
        df
            Raw frame produced by :meth:`load_file` or compatible source.

        Returns
        -------
        pd.DataFrame
            Cleaned copy; original frame is not modified.

        Raises
        ------
        PreprocessingError
            If ``df`` is empty or invalid.
        """
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError("Expected a pandas DataFrame.")
        if df.empty:
            raise PreprocessingError("Cannot clean an empty DataFrame.")

        out = df.copy()
        numeric_cols, categorical_cols = DataLoader.infer_column_types(out)

        for col in numeric_cols:
            s = out[col]
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                s = pd.to_numeric(s, errors="coerce")
            if pd.api.types.is_numeric_dtype(s):
                mean_val = s.mean()
                fill = 0.0 if pd.isna(mean_val) else float(mean_val)
                out[col] = s.fillna(fill)
            else:
                out[col] = s

        for col in categorical_cols:
            series = out[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                med = series.median()
                out[col] = series.fillna(med)
            elif pd.api.types.is_bool_dtype(series):
                mode = series.mode()
                fill_bool = bool(mode.iloc[0]) if len(mode) else False
                out[col] = series.fillna(fill_bool)
            else:
                out[col] = series.astype("object").fillna("Unknown")

        return out

    @staticmethod
    def compute_fill_quality_metrics(
        raw: pd.DataFrame,
        cleaned: pd.DataFrame | None = None,
    ) -> dict[str, float | int]:
        """
        Measure overall missingness and how many cells were filled by cleaning.

        Doluluk oranı: payda tüm hücre sayısı (satır × sütun). Eksik hücreler
        ham verideki NaN sayımıdır. ``cleaned`` verildiğinde, hamda eksik olup
        temizlenmiş tabloda dolu olan hücre sayısı otomatik doldurma (ortalama,
        mod, ``Unknown`` vb.) ile kapatılmış kabul edilir.

        Parameters
        ----------
        raw
            Ham yükleme çıktısı.
        cleaned
            :meth:`clean_data` çıktısı; ``None`` ise sadece ham eksiklik raporu
            üretilir.

        Returns
        -------
        dict[str, float | int]
            ``total_cells``, ``missing_cells``, ``fill_ratio`` (0–1),
            ``missing_ratio`` (0–1), ``pct_missing``, ``pct_filled`` (doluluk %),
            ``imputed_cells`` (temizleme ile doldurulan), ``pct_imputed_of_total``.
        """
        if not isinstance(raw, pd.DataFrame):
            raise PreprocessingError("raw must be a pandas DataFrame.")
        if raw.empty:
            raise PreprocessingError("Cannot score quality on an empty DataFrame.")

        n_rows, n_cols = raw.shape
        total_cells = int(n_rows * n_cols)
        if total_cells == 0:
            raise PreprocessingError("DataFrame has no cells.")

        missing_cells = int(raw.isna().sum().sum())
        fill_ratio = (total_cells - missing_cells) / float(total_cells)
        missing_ratio = missing_cells / float(total_cells)

        imputed_cells = 0
        if cleaned is not None:
            if cleaned.shape != raw.shape:
                raise PreprocessingError(
                    "cleaned DataFrame must match raw shape for imputation counting."
                )
            imputed_cells = int((raw.isna() & cleaned.notna()).sum().sum())

        return {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "fill_ratio": float(fill_ratio),
            "missing_ratio": float(missing_ratio),
            "pct_missing": float(100.0 * missing_ratio),
            "pct_filled": float(100.0 * fill_ratio),
            "imputed_cells": imputed_cells,
            "pct_imputed_of_total": float(
                100.0 * imputed_cells / float(total_cells) if total_cells else 0.0
            ),
        }
