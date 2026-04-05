"""
Yapılandırılmış veri yükleme, şema çıkarımı, özet ve eksik değer temizliği.

:class:`DataLoader` CSV/Excel okuma ve ML öncesi ön işlemi kapsar. İmputation
kuralları istatistiksel (ortalama, mod, medyan) ve semantiktir; **eksik verinin
tamamen kayıpsız giderilmesi mümkün değildir** — amaç tutarlı ve analiz edilebilir
bir tablo üretmektir.
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
    Tabular dosyaları okur ve pandas ``DataFrame`` olarak ML boru hattına hazırlar.

    **Rol:** Biçim algılama, ayrıştırma, sayısal/kategorik şema çıkarımı, tanımlayıcı
    özetler ve sütun tipine göre eksik değer doldurma (imputation). Çıkarım
    kuralları, sayısal benzeri metin sütunlarını oran eşiği ile ayırt eder.
    """

    @staticmethod
    def load_file(file_obj: BinaryIO) -> pd.DataFrame:
        """
        İkili akıştan CSV veya Excel çalışma kitabını okur (ayrıştırma problemi).

        **Girdi:** ``read()`` destekleyen dosya benzeri nesne; uzantı ``name``
        üzerinden seçilir. **Çıktı:** Ham tablo; şema çıkarımı :meth:`infer_column_types`
        ile ayrı yapılır.

        Parameters
        ----------
        file_obj
            Örn. Streamlit ``UploadedFile``; ``name`` ile ``.csv`` / ``.xlsx`` seçimi.

        Returns
        -------
        pd.DataFrame
            Orijinal sütun adları korunmuş tablo.

        Raises
        ------
        DataLoadError
            Desteklenmeyen biçim, boş dosya veya ayrıştırma hatası.
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
        Sütunları ön işlem kuralları için sayısal ve kategorik kümeye ayırır.

        **Heuristik problemi:** Metin/object sütunlarında ``to_numeric`` başarı
        oranı eşik (:data:`_NUMERIC_STRING_RATIO`) üzerindeyse sayısal kabul edilir;
        bool ve datetime kategorik imputation kuralına alınır (:meth:`clean_data`).

        Parameters
        ----------
        df
            Girdi çerçevesi.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(numeric_columns, categorical_columns)``, tekrarsız isimler.
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
        """
        Serinin sayısal özellik olarak ölçeklenip doldurulup doldurulmayacağını döner.

        **Kriter:** Yerleşik sayısal dtype, veya metin sütununda yüksek oranda
        başarılı sayısal zorlama (eşik tabanlı ikili sınıflandırma).
        """
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
        Tanımlayıcı istatistik ve şema özetini tek sözlükte toplar (EDA görünümü).

        **İçerik:** Boyut, çıkarılan sayısal/kategorik listeler, sütun başına eksik
        sayımı, sayısal ``describe`` ve kategorik frekans üstleri — jüri / rapor
        için yapılandırılmış özet.

        Parameters
        ----------
        df
            Özetlenecek veri.

        Returns
        -------
        dict[str, Any]
            ``row_count``, ``column_count``, ``columns``, ``numeric_columns``,
            ``categorical_columns``, ``missing_per_column``, ``numeric_describe``,
            ``categorical_top_values``.

        Raises
        ------
        PreprocessingError
            Boş veya geçersiz ``DataFrame``.
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
        Eksik değerleri sütun tipine göre doldurarak kopya üretir (imputation).

        **Matematiksel anlam:** Sayısal sütunlarda sütun ortalaması
        :math:`\\hat{\\mu}_j` ile eksikleri doldurma (tamamı eksikse ``0.0``);
        kategorikte moda / ``Unknown``; zamanda medyan zaman damgası — bu, ML
        öncesi **basit tek değer imputation** stratejisidir (varyansı küçükte olsa
        değiştirir).

        Parameters
        ----------
        df
            :meth:`load_file` veya uyumlu kaynaktan ham çerçeve.

        Returns
        -------
        pd.DataFrame
            Temizlenmiş kopya; orijinal değişmez.

        Raises
        ------
        PreprocessingError
            Boş veya geçersiz çerçeve.
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
    def apply_log1p_to_numeric_columns(
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Seçili sayısal sütunlarda ``log1p`` ile varyans dengeleme (sağ çarpık dağılımlar).

        **Matematiksel işlem:** :math:`x \\mapsto \\log(1 + \\max(x, 0))` (``numpy.log1p``),
        yalnızca sütunun tüm gözlemlenebilir değerleri :math:`\\geq 0` ise uygulanır;
        aksi halde sütun olduğu gibi bırakılır ve isim ``skipped`` listesine eklenir
        (negatif veya karışık işaretli sütunlarda log tanımı sorunludur).

        Parameters
        ----------
        df
            Kaynak çerçeve (kopyalanır).
        columns
            Dönüştürülecek sütunlar; ``None`` ise :meth:`infer_column_types` ile
            sayısal sütunların tümü denenir.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            Dönüştürülmüş kopya ve atlanan sütun adları.

        Raises
        ------
        PreprocessingError
            Boş çerçeve.
        """
        if not isinstance(df, pd.DataFrame):
            raise PreprocessingError("Expected a pandas DataFrame.")
        if df.empty:
            raise PreprocessingError("Cannot transform an empty DataFrame.")

        inferred_numeric, _ = DataLoader.infer_column_types(df)
        inferred_set = set(inferred_numeric)
        if columns is None:
            use_cols = [c for c in inferred_numeric if c in df.columns]
        else:
            use_cols = [c for c in columns if c in df.columns and c in inferred_set]

        out = df.copy()
        skipped: list[str] = []

        for col in use_cols:
            s = pd.to_numeric(out[col], errors="coerce")
            finite = s.dropna()
            if finite.empty:
                skipped.append(col)
                continue
            if float(finite.min()) < 0.0:
                skipped.append(col)
                continue
            out[col] = np.log1p(s.clip(lower=0.0))

        return out, skipped

    @staticmethod
    def compute_fill_quality_metrics(
        raw: pd.DataFrame,
        cleaned: pd.DataFrame | None = None,
    ) -> dict[str, float | int]:
        """
        Ham veri doluluk oranı ve temizlikle kapatılan hücre sayısını ölçer.

        **Tanımlar:** Payda :math:`n \\times p` toplam hücre; eksiklik ham ``NaN``
        sayısıdır. ``cleaned`` ile birlikte, hamda ``NaN`` iken temizde dolu olan
        hücreler *imputation ile doldurulmuş* sayılır (kalite göstergesi, gerçek
        değer bilgisi değildir).

        Parameters
        ----------
        raw
            Ham tablo.
        cleaned
            :meth:`clean_data` çıktısı veya ``None`` (yalnızca ham özet).

        Returns
        -------
        dict[str, float | int]
            ``total_cells``, ``missing_cells``, ``fill_ratio``, ``missing_ratio``,
            ``pct_missing``, ``pct_filled``, ``imputed_cells``, ``pct_imputed_of_total``.
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
