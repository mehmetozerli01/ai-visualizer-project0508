"""
Yapılandırılmış veri yükleme, şema çıkarımı, özet ve eksik değer temizliği;
isteğe bağlı metin derlemi (NLP Lite) ve simüle ses özellik tablosu.

:class:`DataLoader` CSV/Excel okuma ve ML öncesi ön işlemi kapsar. İmputation
kuralları istatistiksel (ortalama, mod, medyan) ve semantiktir; **eksik verinin
tamamen kayıpsız giderilmesi mümkün değildir** — amaç tutarlı ve analiz edilebilir
bir tablo üretmektir.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import PurePath
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from exceptions import DataLoadError, PreprocessingError

# Minimum ratio of non-null values that must parse as numeric for an object
# column to be treated as numeric (avoids misclassifying IDs with few digits).
_NUMERIC_STRING_RATIO: float = 0.95

# Basit duygu sözlüğü (İngilizce + Türkçe); yoğunluk skoru keşifseldir, model değildir.
_SENTIMENT_POSITIVE: frozenset[str] = frozenset(
    {
        "good",
        "great",
        "excellent",
        "happy",
        "love",
        "positive",
        "best",
        "amazing",
        "wonderful",
        "nice",
        "iyi",
        "güzel",
        "harika",
        "mutlu",
        "mükemmel",
        "süper",
    }
)
_SENTIMENT_NEGATIVE: frozenset[str] = frozenset(
    {
        "bad",
        "terrible",
        "awful",
        "hate",
        "negative",
        "worst",
        "sad",
        "angry",
        "poor",
        "ugly",
        "kötü",
        "berbat",
        "üzgün",
        "korkunç",
        "nefret",
        "fena",
    }
)


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
    def _tokenize_words(text: str) -> list[str]:
        """Unicode kelimeleri küçük harfe indirger (basit tokenizer)."""
        return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _text_document_features(doc: str) -> dict[str, float]:
        """
        Tek belge için sayısal özellikler: uzunluk, çeşitlilik, cümle başına kelime,
        sözlük tabanlı duygu skoru (yaklaşık −1…1).
        """
        doc = (doc or "").strip()
        words = DataLoader._tokenize_words(doc)
        n_w = len(words)
        if n_w == 0:
            return {
                "word_count": 0.0,
                "unique_word_ratio": 0.0,
                "avg_sentence_length": 0.0,
                "sentiment_score": 0.0,
            }
        uniq_ratio = float(len(set(words))) / float(n_w)
        parts = re.split(r"[.!?]+", doc)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            avg_sent = float(n_w)
        else:
            lens = [len(DataLoader._tokenize_words(p)) for p in parts]
            avg_sent = float(np.mean(lens)) if lens else float(n_w)
        pos_hits = sum(1 for w in words if w in _SENTIMENT_POSITIVE)
        neg_hits = sum(1 for w in words if w in _SENTIMENT_NEGATIVE)
        sentiment = (pos_hits - neg_hits) / float(max(n_w, 1))
        sentiment = float(max(-1.0, min(1.0, sentiment * 5.0)))
        return {
            "word_count": float(n_w),
            "unique_word_ratio": float(uniq_ratio),
            "avg_sentence_length": float(avg_sent),
            "sentiment_score": float(sentiment),
        }

    @staticmethod
    def process_text_file(
        raw: bytes,
        *,
        encoding: str = "utf-8",
        document_separator_pattern: str | None = r"\n{3,}",
    ) -> tuple[pd.DataFrame, str]:
        """
        Ham metin dosyasını belge satırlarına ve sayısal özelliklere dönüştürür (NLP Lite).

        Çoklu belge: varsayılan olarak üç veya daha fazla ardışık satım sonu ile ayrılır.
        Her belge bir satır; ``doc_id`` indeks olur. İkinci dönüş, kelime bulutu vb. için
        birleştirilmiş metin (üst sınırlı).

        Parameters
        ----------
        raw
            Dosya içeriği (bayt).
        encoding
            Metin kodlaması.
        document_separator_pattern
            ``re.split`` deseni; ``None`` ise tüm dosya tek belge.

        Returns
        -------
        tuple[pd.DataFrame, str]
            Özellik çerçevesi (yalnızca sayısal sütunlar) ve önizleme metni.

        Raises
        ------
        DataLoadError
            Boş veya ayrıştırılamaz içerik.
        """
        if not raw:
            raise DataLoadError("Text file is empty.")
        try:
            text = raw.decode(encoding, errors="replace")
        except LookupError as exc:
            raise DataLoadError(f"Unknown text encoding: {encoding!r}.") from exc

        text = text.strip()
        if not text:
            raise DataLoadError("Text file contains no usable characters.")

        if document_separator_pattern:
            parts = re.split(document_separator_pattern, text)
            docs = [p.strip() for p in parts if p and p.strip()]
        else:
            docs = [text]

        if not docs:
            raise DataLoadError("No text documents found after splitting.")

        rows: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            feats = DataLoader._text_document_features(doc)
            feats["doc_id"] = i
            rows.append(feats)

        df = pd.DataFrame(rows).set_index("doc_id")
        preview = "\n\n".join(docs)[:500_000]
        return df, preview

    @staticmethod
    def simulate_audio_features(
        *,
        n_rows: int = 100,
        seed: int | None = None,
        name_hint: str = "",
    ) -> pd.DataFrame:
        """
        Gerçek WAV ayrıştırması yerine tutarlı rastgele ‘ses özelliği’ tablosu üretir.

        Dosya adı veya ``seed`` ile tekrarlanabilirlik; sütunlar kümeleme / PCA için
        sayısal vektör olarak kullanılabilir.

        Parameters
        ----------
        n_rows
            Örnek (satır) sayısı.
        seed
            RNG tohumu; ``None`` ise ``name_hint`` karması kullanılır.
        name_hint
            Dosya adı gibi dize; tohum türetmek için.

        Returns
        -------
        pd.DataFrame
            ``freq_centroid_hz``, ``amplitude_rms``, ``tempo_bpm``,
            ``zero_crossing_rate``, ``spectral_spread_sim`` sütunları.
        """
        n = max(5, min(5000, int(n_rows)))
        if seed is None:
            seed = abs(hash(name_hint)) % (2**31 - 1) if name_hint else 42
        rng = np.random.default_rng(int(seed))
        idx = pd.RangeIndex(start=0, stop=n, name="sample_id")
        return pd.DataFrame(
            {
                "freq_centroid_hz": rng.uniform(120.0, 7800.0, n),
                "amplitude_rms": rng.lognormal(mean=0.0, sigma=0.45, size=n),
                "tempo_bpm": rng.uniform(62.0, 178.0, n),
                "zero_crossing_rate": rng.uniform(0.02, 0.22, n),
                "spectral_spread_sim": rng.uniform(0.15, 0.98, n),
            },
            index=idx,
        )

    @staticmethod
    def simulate_image_features(
        *,
        n_rows: int = 100,
        seed: int | None = None,
        name_hint: str = "",
    ) -> pd.DataFrame:
        """
        Görüntü koleksiyonunu taklit eden sayısal özellik tablosu (OpenCV/CNN yok).

        Parlaklık, kontrast, renk doygunluğu, kenar yoğunluğu ve tahmini nesne sayısı
        üretilir; gerçek piksellerden çıkarım yerine savunma ve boru hattı gösterimi
        içindir (OpenCV veya evrişimli ağ ile değiştirilebilir).

        Parameters
        ----------
        n_rows
            Görüntü / örnek sayısı.
        seed
            RNG tohumu.
        name_hint
            Dosya adından türetilebilen tekrarlanabilirlik ipucu.

        Returns
        -------
        pd.DataFrame
            ``brightness``, ``contrast``, ``color_saturation``, ``edge_density``,
            ``object_count``.
        """
        n = max(5, min(5000, int(n_rows)))
        if seed is None:
            seed = abs(hash(name_hint)) % (2**31 - 1) if name_hint else 43
        rng = np.random.default_rng(int(seed))
        idx = pd.RangeIndex(start=0, stop=n, name="image_id")
        return pd.DataFrame(
            {
                "brightness": rng.uniform(0.08, 0.98, n),
                "contrast": rng.uniform(0.12, 0.92, n),
                "color_saturation": rng.uniform(0.0, 1.0, n),
                "edge_density": rng.uniform(0.05, 0.95, n),
                "object_count": rng.integers(1, 15, size=n).astype(np.float64),
            },
            index=idx,
        )

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
