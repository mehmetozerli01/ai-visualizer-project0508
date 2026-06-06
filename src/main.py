"""
Streamlit tabanlı veri görselleştirme ve makine öğrenmesi paneli.

Uçtan uca boru hattı: dosya yükleme, şema çıkarımı, eksik değer temizliği,
ölçeklenmiş özellik uzayında K-Means / PCA / Isolation Forest ve Plotly
grafikleri. Bu modül yalnızca sunum (UI) ve oturum yönetiminden sorumludur;
matematiksel modeller :mod:`ai_engine` içindedir.

Çalıştırma::

    streamlit run src/main.py
"""

from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ai_engine import AIEngine
from exceptions import AIModelError, DataLoadError, PreprocessingError
from processor import DataLoader
from visualizer import (
    DataVisualizer,
    StatisticalCommentator,
    build_bivariate_decision_banner,
    count_structure_variables,
)

# --- Merkezi ayarlar (bakım / tema / varsayılanlar) -----------------------------
CONFIG: dict[str, Any] = {
    "VERSION": "1.2.0",
    "UI": {
        "PAGE_TITLE": "AI Visualizer",
        "PAGE_LAYOUT": "wide",
        "APP_TITLE": "AI Visualizer — Fortune 500 BI Dashboard v1.2.0",
        "APP_CAPTION": (
            "Executive KPI kartları, veri sağlığı ısı haritası, doğal dil Q&A "
            "asistanı ve jüri sunum modu."
        ),
        "FILE_UPLOADER_LABEL": "CSV / Excel dosyası",
        "FILE_EXTENSIONS": ["csv", "xlsx", "xlsm"],
        "FILE_UPLOADER_HELP": "CSV veya Excel (.xlsx). Excel için openpyxl gerekir.",
        "INFO_NO_FILE": "Başlamak için sol menüden bir veri dosyası seçin veya örnek veriyi yükleyin.",
        "BTN_DEMO_DATA": "Örnek veriyi yükle",
        "BTN_DEMO_HELP": (
            "Iris, Wine, Kaliforniya ev fiyatları, Diyabet (sklearn) veya sentetik anomali; "
            "dosya seçmeden yüklenir."
        ),
        "DEMO_LABEL": "Örnek veri seti",
        "SUCCESS_DEMO": "Örnek veri yüklendi: **{dataset}**.",
        "BTN_DOWNLOAD_REPORT": "Akademik analiz raporunu indir",
        "BTN_DOWNLOAD_CODE": "💻 Analiz Kodunu İndir (.py)",
        "BTN_DOWNLOAD_EXCEL": "📊 Temizlenmiş Veriyi İndir (.xlsx)",
        "BTN_DOWNLOAD_MODELS": "🧠 Eğitilmiş Modelleri İndir (.zip)",
        "BTN_DOWNLOAD_PDF": "📄 Yönetici Özetini İndir (.pdf)",
        "PDF_FILENAME_PREFIX": "yonetici_ozeti",
        "CODE_FILENAME_PREFIX": "ai_visualizer_reproducible_analysis",
        "EXCEL_FILENAME_PREFIX": "ai_visualizer_cleaned_export",
        "MODELS_ZIP_PREFIX": "ai_visualizer_trained_models",
        "REPORT_FILENAME_PREFIX": "akademik_analiz_raporu",
        "APP_TITLE_ICON": "◆",
        "TAB_ICONS": ["📋", "📑", "✨", "🧪", "🎯"],
        "TABS": [
            "Raw preview",
            "Summary stats",
            "Cleaned preview",
            "Analiz Paneli",
            "Yönetici Özeti (Executive Summary)",
        ],
        "SPINNER_ANALYSIS": "AI Analiz Yapıyor...",
        "CHECKBOX_ACADEMIC_TIPS": "Akademik Sunum İpuçlarını Göster",
        "CHECKBOX_ACADEMIC_TIPS_HELP": (
            "Her analiz altında jüriye okuyabileceğiniz kısa teknik notlar (sunum modu)."
        ),
        "REPORT_PREVIEW_EXPANDER": "📄 İndirilecek rapor önizlemesi",
        "REPORT_STRATEGY_SECTION_MD": "## 🤖 AI Stratejik Tavsiyeler",
        "FOOTER_TEXT": (
            "v1.2.0 | Mehmet Özerli — Executive BI dashboard, veri sağlığı haritası, "
            "AI Q&A ve jüri sunum modu"
        ),
        "DATA_QA_PLACEHOLDER": "🔍 Verinize Soru Sorun (Örn: En yüksek değerli küme hangisi?)",
        "DATA_QA_HINT": (
            "Küme, ortalama, anomali, silhouette veya hedef önemi hakkında Türkçe soru yazın."
        ),
        "JURY_MODE_LABEL": "🎬 Jüri Sunum Modu",
        "JURY_MODE_HELP": (
            "Açıkken sidebar ve kalabalık paneller gizlenir; 3D PCA, karar bantları ve "
            "stratejik AI yorumları slayt düzeninde kalır."
        ),
        "DATA_MODE_LABEL": "Veri Türü Seçin",
        "DATA_MODES": {
            "tabular": "📊 Tabüler Veri (Excel/CSV)",
            "text_corpus": "📝 Metin Derlemi (NLP)",
            "audio_features": "🔊 Ses Sinyalleri (Frekans)",
            "image_features": "🔬 Medikal Görüntü (Hücre Analizi)",
        },
        "UPLOADER_HELP_BY_MODE": {
            "tabular": "CSV veya Excel (.xlsx). Excel için openpyxl gerekir.",
            "text_corpus": ".txt — birden fazla belge için paragraflar arasında boş satırlar (3+ newline) kullanın.",
            "audio_features": "Yer tutucu dosya (.txt/.csv). Frekans özellikleri simüle edilir; dosya adında 'audio' geçerse tohum farklılaşır.",
            "image_features": (
                "Kan hücresi mikroskopi simülasyonu: çap, çekirdek yoğunluğu ve "
                "şekil bozukluğu özellikleri üretilir (CNN/OpenCV entegrasyonu için hazır)."
            ),
        },
    },
    "MODEL": {
        "DEFAULT_K_MEANS_CAP": 3,
        "ELBOW_K_MAX_UPPER": 20,
        "ELBOW_K_DEFAULT_CAP": 10,
        "CONTAMINATION_DEFAULT": 0.1,
        "CONTAMINATION_MIN": 0.01,
        "CONTAMINATION_MAX": 0.5,
        "CONTAMINATION_STEP": 0.01,
        "PCA_COMPONENTS_UI": 2,
        "PCA_COMPONENTS_3D": 3,
        "MIN_NUMERIC_FEATURES": 2,
    },
    "PREVIEW": {
        "RAW_CLEAN_HEAD": 50,
        "ANALYSIS_TABLE_HEAD": 100,
    },
    "FORMATTING": {
        "TABLE_FLOAT_DECIMALS": 2,
    },
    "MESSAGES": {
        "WARN_MIN_NUMERIC": "Analiz için en az 2 sayısal sütun gereklidir.",
        "WARN_NO_INFERRED_NUMERIC": (
            "Sayısal sütun çıkarılamadı; kümeleme / PCA / anomali devre dışı."
        ),
        "CAPTION_PICK_TWO_COLUMNS": "Analiz için en az iki sayısal sütun seçin.",
        "CAPTION_ANALYSIS_EMPTY": "Sonuçları görmek için **Analizi Çalıştır** düğmesine basın.",
        "MSG_VECTORIZED_PIPELINE": (
            "**Not:** Bu veriler analiz için vektörleştirilmiştir (sayısal özellik satırları). "
            "Ham metin, ses dalga formu veya ham görüntü pikseli doğrudan kullanılmaz; türetilmiş "
            "sütunlar üzerinden kümeleme / PCA / anomali çalışır."
        ),
    },
    "PRESENTATION_TIPS": {
        "correlation": (
            "Pearson korelasyonu doğrusal birlikte değişimi ölçer; yüksek |r| nedensellik "
            "kanıtı değildir. Jüride hangi değişken çiftlerinin eş yönlü hareket ettiğini "
            "söyleyip gizli faktörlere dikkat çekebilirsiniz."
        ),
        "elbow": (
            "Dirsek eğrisi, k arttıkça WCSS’in nerede yavaşladığını sezgisel gösterir; "
            "istatistiksel olarak tek doğru k yoktur. Turuncu çizgi, sizin seçtiğiniz k ile "
            "hizalanır."
        ),
        "clustering": (
            "Silhouette −1…1 aralığında küme ayrışmasını özetler; ölçeklenmiş uzayda "
            "hesaplanır. Inertia ise k ile genelde düştüğü için tek başına kalite ölçütü "
            "sayılmamalıdır."
        ),
        "manual_scatter": (
            "Bu grafik PCA’dan daha yorumlanabilir eksenler sunar; renkler yine K-Means’ten "
            "gelir. Log dönüşümü açıksa kümeler dönüşümlü uzayda, eksenler orijinal ölçektedir."
        ),
        "pca": (
            "PC1 ve PC2 toplam açıklanan varyans yüzdesi, 2B projeksiyonda ne kadar bilgi "
            "kaldığını gösterir; düşükse tablo ve manuel eksenlere de bakın."
        ),
        "interpreter": (
            "Otomatik yorumlar eşik tabanlıdır; alan bilgisi ve veri kalitesi her zaman "
            "model çıktısından önce gelmelidir."
        ),
        "anomaly": (
            "Kırmızı noktalar Isolation Forest’un düşük yoğunluk / kısa yol skoruna dayalı "
            "adaylarıdır; domain uzmanı onayı ve iş kuralı olmadan tek başına karar vermez."
        ),
        "feature_dist": (
            "Marjinal dağılım tek değişkeni gösterir; çok modlu veya çarpık dağılımlarda "
            "log dönüşümü veya segmentasyon düşünün."
        ),
        "rf_importance": (
            "Bu grafik, seçilen hedef değişkeni tahmin etmek için hangi sütunların "
            "en yüksek marjinal katkıyı verdiğini gösterir (Random Forest)."
        ),
        "smart_chart": (
            "Akıllı seçici: iki sürekli sayısal sütunda saçılım; biri kategorik/düşük "
            "varyanslıysa kutu veya çubuk grafiği otomatik seçilir."
        ),
        "anomaly_character": (
            "Bu tablo, aykırı değerlerin normal veriden hangi istatistiksel sapmalarla "
            "ayrıldığını kanıtlar."
        ),
    },
    "FILE_TYPES_BY_MODE": {
        "tabular": ["csv", "xlsx", "xlsm"],
        "text_corpus": ["txt"],
        "audio_features": ["txt", "csv", "xlsx"],
        "image_features": ["txt", "csv", "xlsx"],
    },
    "SESSION_KEYS": {
        "FILE_UPLOADER": "wv_dataset_upload",
        "FEATURE_DIST_COLUMN": "feature_dist_column",
        "DEMO_SAMPLE": "wv_demo_sample",
        "TARGET_VARIABLE": "wv_target_variable",
        "TIME_COLUMN": "wv_time_column",
        "TS_VALUE_COLUMN": "wv_ts_value_column",
        "DATA_MODE": "wv_data_mode",
        "TEXT_CORPUS_BLOB": "_wv_text_corpus_display",
    },
    "PLOTLY": {
        "CHART_CONFIG": {
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["lasso2d", "select2d"],
        },
        "CAPTION_SELECTION": (
            "Grafik araç çubuğundan **Kutu seç** veya **Lasso** ile alan seçin; "
            "seçilen satırlar aşağıda listelenir."
        ),
    },
    "INTERPRETER": {
        "SILHOUETTE_STRONG": 0.5,
        "SILHOUETTE_WEAK": 0.2,
        "PCA_LOW_VARIANCE_PCT": 60.0,
        "MSG_SIL_HIGH": (
            "Veri setindeki kümeler belirgin ve başarılı şekilde ayrışmıştır."
        ),
        "MSG_SIL_LOW": (
            "Kümeler iç içe geçmiş görünüyor; farklı sütunlar seçmeyi veya k "
            "sayısını değiştirmeyi deneyin."
        ),
        "MSG_PCA_LOW": (
            "2D görselleştirme bilgi kaybı yaşıyor, sonuçları tablodan da teyit edin."
        ),
    },
    "DEVELOPER": {
        "NAME": "Ad Soyad",
        "SCHOOL": "Üniversite / Bölüm — bitirme projesi",
        "TECH_STACK": (
            "Python · Streamlit · scikit-learn · Plotly · pandas · NumPy · openpyxl"
        ),
    },
    "VISION_NOTE": (
        "HealthTech vizyonu: kan hücresi morfometrisi, sunucu loglarında zaman "
        "serisi anomali ve klasik tabüler analiz tek platformda birleşir."
    ),
    # Plotly / UI ile hizalı referans palet (grafik modülü ayrı; burada tek kaynak).
    "COLORS": {
        "PRIMARY": "#6366F1",
        "ACCENT": "#0EA5E9",
        "ELBOW_LINE": "#7C3AED",
        "ELBOW_MARKER": "#7C3AED",
        "VLINE_SELECTED_K": "#EA580C",
        "CLUSTER_NORMAL": "#2563EB",
        "ANOMALY": "#FF2D2D",
    },
    "EXEC_SUMMARY_CHART_HEIGHT": 620,
}

_DEMO_DATASET_LABELS: dict[str, str] = {
    "iris": "Iris (botanik)",
    "wine": "Wine (şarap kalitesi)",
    "california_housing": "Kaliforniya Ev Fiyatları (regresyon)",
    "diabetes": "Diyabet (regresyon)",
    "anomaly_synthetic": "Anomali Test Seti (Sentetik)",
}

_TARGET_COLUMN_HINTS: tuple[str, ...] = (
    "target",
    "irregularity_score",
    "nucleus_density",
    "medhouseval",
    "price",
    "label",
    "y",
)

_MODE_TARGET_HINTS: dict[str, str] = {
    "image_features": "irregularity_score",
    "audio_features": "amplitude_rms",
}

_EXPORT_INDEX_SKIP: frozenset[str] = frozenset(
    {"cell_id", "sample_id", "doc_id", "image_id"}
)

_DATETIME_NAME_HINTS: tuple[str, ...] = (
    "date",
    "time",
    "timestamp",
    "datetime",
    "tarih",
    "zaman",
    "created",
    "logged",
    "recorded",
)

_DATA_MODE_ORDER: tuple[str, ...] = (
    "tabular",
    "text_corpus",
    "audio_features",
    "image_features",
)

# Örnek veri yüklemede modu tabüler yap: selectbox oluşmadan önce işlenir (widget anahtarı yazılamaz).
_DEMO_FORCE_TABULAR_KEY = "_wv_demo_force_tabular"


def _normalize_data_mode_index(sess_key: str) -> None:
    """``wv_data_mode`` artık 0…n-1 indeks tutar; eski oturumlardaki string değeri dönüştür."""
    raw = st.session_state.get(sess_key)
    if raw is None:
        st.session_state[sess_key] = 0
        return
    if isinstance(raw, str):
        try:
            st.session_state[sess_key] = _DATA_MODE_ORDER.index(raw)
        except ValueError:
            st.session_state[sess_key] = 0
        return
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        st.session_state[sess_key] = 0
        return
    if not (0 <= idx < len(_DATA_MODE_ORDER)):
        st.session_state[sess_key] = 0


def _data_mode_id(sess_key: str) -> str:
    """Seçili veri modu kimliği (``tabular``, ``text_corpus``, …)."""
    _normalize_data_mode_index(sess_key)
    return _DATA_MODE_ORDER[int(st.session_state[sess_key])]


def _extensions_for_data_mode(mode: str) -> list[str]:
    m = CONFIG["FILE_TYPES_BY_MODE"].get(
        mode, CONFIG["FILE_TYPES_BY_MODE"]["tabular"]
    )
    return list(m)


def _render_wordcloud_section(text_blob: str) -> None:
    """Metin derlemi için kelime bulutu (wordcloud + matplotlib)."""
    blob = (text_blob or "").strip()
    if len(blob) < 12:
        st.caption("Kelime bulutu için metin çok kısa.")
        return
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except ImportError:
        st.info(
            "Kelime bulutu için `wordcloud` paketini yükleyin: `pip install wordcloud`."
        )
        return
    try:
        wc = WordCloud(
            width=900,
            height=420,
            background_color="white",
            colormap="viridis",
            max_words=120,
        ).generate(blob)
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except ValueError:
        st.caption("Kelime bulutu üretilemedi.")


def _streamlit_theme() -> Literal["light", "dark"]:
    """Map Streamlit host theme to a simple light/dark flag for Plotly."""
    try:
        ctx = getattr(st, "context", None)
        theme = getattr(ctx, "theme", None) if ctx is not None else None
        base = getattr(theme, "base", None) if theme is not None else None
        if base == "dark":
            return "dark"
    except Exception:
        pass
    return "light"


def _academic_tip_if(enabled: bool, text: str) -> None:
    """Sunum modu: bölüm altında kısa jüri notu (``st.success``)."""
    if enabled and text.strip():
        st.success(text.strip())


def _render_ai_comment(text: str) -> None:
    """Grafik altı istatistiksel yorum (AI Commentator)."""
    if text and str(text).strip():
        st.info(f"🤖 **AI Yorumu:** {text.strip()}")


def _resolve_deploy_target(
    df: pd.DataFrame,
    target_variable: str,
    data_mode_id: str,
) -> str:
    """Multimodal modlarda RF/ZIP için anlamlı hedef sütun seçer."""
    mode_hint = _MODE_TARGET_HINTS.get(data_mode_id)
    if mode_hint and mode_hint in df.columns:
        return mode_hint
    if target_variable in df.columns:
        return str(target_variable)
    numeric, _ = DataLoader.infer_column_types(df)
    candidates = [c for c in numeric if c not in _EXPORT_INDEX_SKIP]
    if candidates:
        return str(candidates[-1])
    cols = list(df.columns.astype(str))
    return cols[0] if cols else str(target_variable)


def _export_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Excel/ZIP dışa aktarımı için indeks ve datetime uyumluluğu sağlar."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    if out.index.name or not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    return out


def _default_target_column_index(columns: list[str]) -> int:
    """Hedef değişken için varsayılan selectbox indeksi."""
    lower_map = {str(c).lower(): i for i, c in enumerate(columns)}
    for hint in _TARGET_COLUMN_HINTS:
        if hint in lower_map:
            return int(lower_map[hint])
    return max(0, len(columns) - 1)


def _apply_row_filters(
    df: pd.DataFrame,
    *,
    cat_filters: dict[str, list[str]],
    numeric_ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Kategorik çoklu seçim ve sayısal aralık filtrelerini satır bazında uygular."""
    out = df
    for col, selected in cat_filters.items():
        if not selected or col not in out.columns:
            continue
        allowed = {str(v) for v in selected}
        mask = out[col].astype(str).isin(allowed)
        out = out.loc[mask]
    for col, bounds in numeric_ranges.items():
        if col not in out.columns:
            continue
        lo, hi = bounds
        s = pd.to_numeric(out[col], errors="coerce")
        out = out.loc[s.between(lo, hi, inclusive="both")]
    return out


def _whatif_session_key(upload_fingerprint: str) -> str:
    """What-If düzenleyici için oturum anahtarı."""
    return f"whatif_df_{upload_fingerprint}"


def _dataframe_edit_fingerprint(df: pd.DataFrame) -> str:
    """``st.data_editor`` değişikliklerini güvenilir şekilde izlemek için özet."""
    try:
        h = pd.util.hash_pandas_object(df, index=True).sum()
        return f"{len(df)}:{int(h)}"
    except Exception:
        return f"{len(df)}:{df.shape[1]}:{df.columns.tolist()}"


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Tarih/zaman sütunlarını dtype veya parse başarı oranıyla tespit eder."""
    if df.empty:
        return []
    found: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(str(col))
            continue
        col_l = str(col).lower()
        name_hit = any(h in col_l for h in _DATETIME_NAME_HINTS)
        parsed = pd.to_datetime(series, errors="coerce")
        ratio = float(parsed.notna().mean()) if len(series) else 0.0
        if ratio >= 0.85 and (name_hit or ratio >= 0.95):
            found.append(str(col))
    return list(dict.fromkeys(found))


def _run_analysis_pipeline(
    analysis_df: pd.DataFrame,
    *,
    n_clusters: int,
    contamination: float,
    elbow_k_max: int,
    selected_numeric: list[str],
    target_variable: str,
    pca_components_2d: int,
    pca_components_3d: int,
    data_mode_id: str = "tabular",
) -> dict[str, Any]:
    """K-Means, PCA (2D+3D), anomali, dirsek ve RF önemlerini tek çağrıda çalıştırır."""
    labels: pd.Series | None = None
    pca_coords: pd.DataFrame | None = None
    pca_coords_3d: pd.DataFrame | None = None
    pred: pd.Series | None = None
    err_cluster: str | None = None
    err_pca: str | None = None
    err_pca_3d: str | None = None
    err_anomaly: str | None = None
    inertia_val: float | None = None
    silhouette_val: float | None = None
    pca_variance_pct: float | None = None
    pca_variance_pct_3d: float | None = None
    elbow_df: pd.DataFrame | None = None
    err_elbow: str | None = None
    cluster_imp_df: pd.DataFrame | None = None
    err_importance: str | None = None
    target_imp_df: pd.DataFrame | None = None
    err_target_importance: str | None = None

    engine = AIEngine()
    cols_arg = selected_numeric
    deploy_target = _resolve_deploy_target(
        analysis_df, str(target_variable), data_mode_id
    )

    try:
        raw_labels, inertia_val, silhouette_val = engine.perform_clustering(
            analysis_df,
            n_clusters,
            numeric_columns=cols_arg,
        )
        labels = pd.Series(raw_labels, index=analysis_df.index)
    except AIModelError as exc:
        err_cluster = str(exc)
        inertia_val = None
        silhouette_val = None

    try:
        pca_coords, pca_var_info = engine.perform_pca(
            analysis_df,
            n_components=int(pca_components_2d),
            numeric_columns=cols_arg,
        )
        pca_variance_pct = float(pca_var_info["variance_explained_pct"])
    except AIModelError as exc:
        err_pca = str(exc)
        pca_variance_pct = None

    try:
        pca_coords_3d, pca_var_3d = engine.perform_pca(
            analysis_df,
            n_components=int(pca_components_3d),
            numeric_columns=cols_arg,
        )
        pca_variance_pct_3d = float(pca_var_3d["variance_explained_pct"])
    except AIModelError as exc:
        err_pca_3d = str(exc)
        pca_variance_pct_3d = None

    try:
        raw_pred = engine.detect_anomalies(
            analysis_df,
            contamination=contamination,
            numeric_columns=cols_arg,
        )
        pred = pd.Series(raw_pred, index=analysis_df.index)
    except AIModelError as exc:
        err_anomaly = str(exc)

    try:
        elbow_df = engine.elbow_inertia_scan(
            analysis_df,
            k_max=elbow_k_max,
            numeric_columns=cols_arg,
        )
    except AIModelError as exc:
        err_elbow = str(exc)
        elbow_df = None

    if err_cluster is None and labels is not None:
        try:
            if len(np.unique(labels.to_numpy())) >= 2:
                cluster_imp_df = engine.cluster_feature_importance_rf(
                    analysis_df,
                    labels.to_numpy(),
                    numeric_columns=cols_arg,
                )
        except AIModelError as exc:
            err_importance = str(exc)

    if deploy_target and deploy_target in analysis_df.columns:
        try:
            target_imp_df = engine.target_feature_importance_rf(
                analysis_df,
                deploy_target,
                numeric_columns=cols_arg,
            )
        except AIModelError as exc:
            err_target_importance = str(exc)

    models_zip_bytes: bytes | None = None
    err_models_zip: str | None = None
    try:
        bundle = engine.fit_deployable_models(
            analysis_df,
            n_clusters=int(n_clusters),
            contamination=float(contamination),
            numeric_columns=cols_arg,
            target_column=str(deploy_target),
        )
        models_zip_bytes = _build_models_zip_bytes(bundle, app_version=CONFIG["VERSION"])
    except AIModelError as exc:
        err_models_zip = str(exc)

    return {
        "labels": labels,
        "pca_coords": pca_coords,
        "pca_coords_3d": pca_coords_3d,
        "pred": pred,
        "err_cluster": err_cluster,
        "err_pca": err_pca,
        "err_pca_3d": err_pca_3d,
        "err_anomaly": err_anomaly,
        "n_clusters": n_clusters,
        "inertia": inertia_val,
        "silhouette": silhouette_val,
        "pca_variance_pct": pca_variance_pct,
        "pca_variance_pct_3d": pca_variance_pct_3d,
        "elbow_df": elbow_df,
        "err_elbow": err_elbow,
        "feature_columns": list(selected_numeric),
        "cluster_importance_df": cluster_imp_df,
        "err_importance": err_importance,
        "target_column": str(deploy_target),
        "target_importance_df": target_imp_df,
        "err_target_importance": err_target_importance,
        "models_zip_bytes": models_zip_bytes,
        "err_models_zip": err_models_zip,
    }


def _build_multi_sheet_excel_bytes(
    analysis_base: pd.DataFrame,
    result: dict[str, Any],
) -> bytes:
    """3 sekmeli Excel: Cleaned Data, Anomalies, Summary (describe)."""
    buf = BytesIO()
    export_base = _export_safe_dataframe(analysis_base)
    cleaned = export_base.copy()
    labels = result.get("labels")
    pred = result.get("pred")
    if labels is not None:
        lab = np.asarray(labels).ravel()
        if len(lab) == len(cleaned):
            cleaned.insert(0, "Cluster_ID", lab)
    if pred is not None:
        pr = np.asarray(pred).ravel()
        if len(pr) == len(cleaned):
            cleaned["Anomaly_Label"] = pr

    numeric_cols, _ = DataLoader.infer_column_types(export_base)
    if numeric_cols:
        summary = (
            export_base[numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .describe()
            .T
        )
    else:
        summary = pd.DataFrame({"note": ["Sayısal sütun yok"]})

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        cleaned.to_excel(writer, sheet_name="Cleaned Data", index=False)
        if pred is not None:
            pr = np.asarray(pred).ravel()
            if len(pr) == len(cleaned):
                anomalies = cleaned.iloc[np.where(pr == -1)[0]]
            else:
                anomalies = cleaned.head(0)
        else:
            anomalies = cleaned.head(0)
        anomalies.to_excel(writer, sheet_name="Anomalies", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=True)
    return buf.getvalue()


def _strip_md_for_pdf(text: str) -> str:
    """PDF metni için basit Markdown sadeleştirme."""
    import re

    out = str(text)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"^#+\s*", "", out, flags=re.MULTILINE)
    out = out.replace("## ", "").replace("### ", "")
    return out.strip()


def _pdf_text(text: str) -> str:
    """Helvetica uyumu için Türkçe karakterleri güvenli ASCII'ye indirger."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _build_executive_summary_pdf_bytes(
    result: dict[str, Any],
    *,
    app_version: str,
    n_obs: int,
    n_clusters: int,
    strategy_md: str,
    feat_cols: list[str],
) -> bytes:
    """Tek sayfalık yönetici özeti PDF (fpdf2)."""
    from fpdf import FPDF

    labels = result.get("labels")
    pred = result.get("pred")
    cluster_lines: list[str] = ["K-Means kume dagilimi:"]
    if labels is not None:
        counts = (
            pd.Series(np.asarray(labels).ravel())
            .value_counts()
            .sort_index()
        )
        for cid, cnt in counts.items():
            pct = 100.0 * float(cnt) / max(1, n_obs)
            cluster_lines.append(f"  - Küme {cid}: {int(cnt)} gözlem (%{pct:.1f})")
    else:
        cluster_lines.append("  (Kümeleme sonucu yok)")

    n_anom = 0
    if pred is not None:
        n_anom = int((np.asarray(pred).ravel() == -1).sum())

    sil = result.get("silhouette")
    pv = result.get("pca_variance_pct")
    strategy_plain = _strip_md_for_pdf(strategy_md)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    epw_title = float(pdf.epw)
    pdf.cell(
        epw_title,
        9,
        _pdf_text(f"AI Visualizer - Yonetici Ozeti v{app_version}"),
        ln=True,
    )
    pdf.set_font("Helvetica", size=10)
    pdf.ln(2)
    epw = float(pdf.epw)
    pdf.multi_cell(epw, 5, _pdf_text(f"Gozlem sayisi: {n_obs}  |  k: {n_clusters}"))
    pdf.multi_cell(
        epw,
        5,
        _pdf_text(
            f"Silhouette: {float(sil):.3f}" if sil is not None else "Silhouette: -"
        ),
    )
    pdf.multi_cell(
        epw,
        5,
        _pdf_text(
            f"PCA PC1+PC2 varyans: %{float(pv):.1f}"
            if pv is not None
            else "PCA varyans: -"
        ),
    )
    pdf.multi_cell(epw, 5, _pdf_text(f"Tespit edilen anomali: {n_anom}"))
    pdf.multi_cell(
        epw,
        5,
        _pdf_text(
            "Ozellikler: " + ", ".join(feat_cols[:8])
            + (" ..." if len(feat_cols) > 8 else "")
        ),
    )
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(epw, 6, _pdf_text("Kume dagilimi"), ln=True)
    pdf.set_font("Helvetica", size=10)
    for line in cluster_lines:
        pdf.multi_cell(epw, 5, _pdf_text(line))
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(epw, 6, _pdf_text("AI Stratejik Tavsiyeler"), ln=True)
    pdf.set_font("Helvetica", size=10)
    for para in strategy_plain.split("\n"):
        if para.strip():
            pdf.multi_cell(epw, 5, _pdf_text(para.strip()))
    raw_out = pdf.output()
    return raw_out if isinstance(raw_out, bytes) else bytes(raw_out)


def _build_models_zip_bytes(
    bundle: dict[str, Any],
    *,
    app_version: str,
) -> bytes:
    """joblib ile eğitilmiş modelleri metadata ile ZIP arşivine paketler."""
    meta = {
        "app_version": app_version,
        "feature_columns": bundle.get("feature_columns", []),
        "target_feature_columns": bundle.get("target_feature_columns", []),
        "target_column": bundle.get("target_column"),
        "rf_task": bundle.get("rf_task"),
        "n_clusters": bundle.get("n_clusters"),
        "contamination": bundle.get("contamination"),
        "random_state": bundle.get("random_state"),
        "load_hint": (
            "joblib.load('kmeans.pkl'); aynı scaler ile ölçeklenmiş özellikler "
            "gönderin."
        ),
    }
    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(meta, indent=2, ensure_ascii=False))
        zf.writestr("README.txt", (
            "AI Visualizer — eğitilmiş model paketi\n"
            f"Sürüm: {app_version}\n\n"
            "Canlı ortamda: joblib.load + metadata.json sütun listesi ile kullanın.\n"
        ))

        def _add_pkl(name: str, obj: Any) -> None:
            blob = BytesIO()
            joblib.dump(obj, blob)
            zf.writestr(name, blob.getvalue())

        _add_pkl("kmeans.pkl", bundle["kmeans"])
        _add_pkl("isolation_forest.pkl", bundle["isolation_forest"])
        _add_pkl("standard_scaler.pkl", bundle["scaler"])
        rf = bundle.get("random_forest")
        if rf is not None:
            _add_pkl("random_forest.pkl", rf)
        t_scaler = bundle.get("target_scaler")
        if t_scaler is not None:
            _add_pkl("target_scaler.pkl", t_scaler)
    return buf.getvalue()


def _build_reproducible_analysis_script(
    *,
    feature_columns: list[str],
    target_column: str,
    n_clusters: int,
    contamination: float,
    elbow_k_max: int,
    use_log_transform: bool,
    app_version: str,
    time_column: str | None = None,
    ts_value_column: str | None = None,
) -> str:
    """Panel ayarlarından yerel ortamda tekrarlanabilir sklearn/pandas betiği üretir."""
    cols_repr = repr(list(feature_columns))
    log_block = ""
    if use_log_transform:
        log_block = (
            "for _c in FEATURE_COLUMNS:\n"
            "    s = pd.to_numeric(work[_c], errors='coerce')\n"
            "    if s.dropna().empty or float(s.min()) < 0:\n"
            "        continue\n"
            "    work[_c] = np.log1p(s.clip(lower=0.0))\n"
        )
    ts_block = ""
    if time_column and ts_value_column:
        ts_block = (
            f"\n# Zaman serisi modu\n"
            f"TIME_COL = {time_column!r}\n"
            f"TS_VALUE_COL = {ts_value_column!r}\n"
            f"work[TIME_COL] = pd.to_datetime(work[TIME_COL], errors='coerce')\n"
            f"work = work.sort_values(TIME_COL)\n"
        )
    return f'''#!/usr/bin/env python3
"""
AI Visualizer v{app_version} — yeniden üretilebilir analiz betiği.
Bu dosya uygulama panelindeki ayarlardan otomatik üretilmiştir (kara kutu değildir).
Gereksinimler: pandas, numpy, scikit-learn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Kullanıcı: veri yolunu güncelleyin ---
DATA_PATH = "veri_setiniz.csv"  # veya .xlsx

FEATURE_COLUMNS = {cols_repr}
TARGET_COLUMN = {target_column!r}
N_CLUSTERS = {int(n_clusters)}
CONTAMINATION = {float(contamination)}
ELBOW_K_MAX = {int(elbow_k_max)}
USE_LOG_TRANSFORM = {bool(use_log_transform)}
RANDOM_STATE = 42
{ts_block}

def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xlsm")):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def prepare_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    frame = df[columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.mean()).fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(frame.to_numpy(dtype=float)), frame


def main() -> None:
    raw = load_data(DATA_PATH)
    work = raw.copy()
    {log_block}
    X, frame = prepare_matrix(work, FEATURE_COLUMNS)

    # K-Means
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    print("K-Means inertia:", float(km.inertia_))
    work["cluster"] = labels

    # PCA (2D + 3D)
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    coords2 = pca2.fit_transform(X)
    print("PCA PC1+PC2 varyans %:", float(pca2.explained_variance_ratio_.sum() * 100))
    pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
    coords3 = pca3.fit_transform(X)
    print("PCA PC1+PC2+PC3 varyans %:", float(pca3.explained_variance_ratio_.sum() * 100))

    # Isolation Forest
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    work["anomaly_label"] = iso.fit_predict(X)
    n_anom = int((work["anomaly_label"] == -1).sum())
    print(f"Anomali sayısı (-1): {{n_anom}}")

    # Hedef değişken — Random Forest önemi
    feat_for_target = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]
    if TARGET_COLUMN in work.columns and feat_for_target:
        X_t, _ = prepare_matrix(work, feat_for_target)
        y_raw = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")
        if y_raw.nunique(dropna=True) > 10:
            rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
            rf.fit(X_t, y_raw.fillna(y_raw.mean()))
        else:
            rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
            rf.fit(X_t, work[TARGET_COLUMN].astype(str))
        imp = pd.DataFrame({{
            "feature": feat_for_target,
            "importance": rf.feature_importances_,
        }}).sort_values("importance", ascending=False)
        print("\\nHedef değişken özellik önemi:\\n", imp.to_string(index=False))

    # Dirsek taraması
    print("\\nDirsek (k, inertia):")
    for k in range(1, min(ELBOW_K_MAX, len(work) - 1) + 1):
        m = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        m.fit(X)
        print(k, float(m.inertia_))

    work.to_csv("analiz_sonucu_etiketli.csv", index=False)
    print("\\nEtiketli çıktı: analiz_sonucu_etiketli.csv")


if __name__ == "__main__":
    main()
'''


def _filter_state_fingerprint(
    cat_filters: dict[str, list[str]],
    numeric_ranges: dict[str, tuple[float, float]],
) -> tuple[Any, ...]:
    """Oturum anahtarı: filtre değişince analiz önbelleğini geçersiz kılmak için."""
    cat_part = tuple(
        sorted((k, tuple(sorted(str(x) for x in v))) for k, v in cat_filters.items())
    )
    num_part = tuple(
        sorted((k, (float(lo), float(hi))) for k, (lo, hi) in numeric_ranges.items())
    )
    return (cat_part, num_part)


def _style_analysis_dataframe(df: pd.DataFrame, *, float_decimals: int) -> Any:
    """Sayıları sabit ondalıkla göster; tamsayı sütunları düzgün biçimle."""
    spec: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            spec[col] = "{:d}"
        elif pd.api.types.is_float_dtype(s) or (
            pd.api.types.is_numeric_dtype(s)
            and not pd.api.types.is_integer_dtype(s)
        ):
            spec[col] = f"{{:.{float_decimals}f}}"
    styled = df.style
    if spec:
        styled = styled.format(spec, na_rep="—")
    return styled


def _reset_app_session() -> None:
    """Analiz durumunu ve yüklenen dosyayı temizleyip başa döndür."""
    keys_to_drop = (
        "analiz_result",
        "_upload_fingerprint",
        "_feature_selection_key",
        CONFIG["SESSION_KEYS"]["FILE_UPLOADER"],
        CONFIG["SESSION_KEYS"]["FEATURE_DIST_COLUMN"],
        CONFIG["SESSION_KEYS"]["DEMO_SAMPLE"],
        CONFIG["SESSION_KEYS"]["TEXT_CORPUS_BLOB"],
        CONFIG["SESSION_KEYS"]["DATA_MODE"],
        "_whatif_fp",
        "whatif_auto_run",
    )
    for k in keys_to_drop:
        st.session_state.pop(k, None)
    for k in list(st.session_state.keys()):
        if str(k).startswith("whatif_df_") or str(k).startswith("whatif_snap_"):
            st.session_state.pop(k, None)
    st.rerun()


def _load_sklearn_demo(name: str) -> pd.DataFrame:
    """sklearn ``datasets`` içinden çerçeve olarak örnek veri döndürür (sunum / demo).

    Iris ve Wine çok sınıflı, sayısal özellikli tablolardır; kümeleme ve PCA ile
    gösterim için uygundur (etiket sütunu da sayısal çıkarımda yer alabilir).
    """
    if name == "iris":
        from sklearn.datasets import load_iris

        bundle = load_iris(as_frame=True)
        return bundle.frame.copy()
    if name == "wine":
        from sklearn.datasets import load_wine

        bundle = load_wine(as_frame=True)
        return bundle.frame.copy()
    if name == "california_housing":
        from sklearn.datasets import fetch_california_housing

        bundle = fetch_california_housing(as_frame=True)
        return bundle.frame.copy()
    if name == "diabetes":
        from sklearn.datasets import load_diabetes

        bundle = load_diabetes(as_frame=True)
        return bundle.frame.copy()
    raise ValueError(f"Bilinmeyen örnek veri: {name!r}")


def _load_synthetic_anomaly_demo(*, rng_seed: int = 42) -> pd.DataFrame:
    """İki boyutta 100 normal + 5 uç aykırı nokta (anomali sunumu için)."""
    rng = np.random.default_rng(rng_seed)
    normal = rng.normal(0.0, 1.0, size=(100, 2))
    outliers = np.array(
        [
            [12.0, 11.0],
            [-11.0, 12.0],
            [13.0, -10.0],
            [-12.0, -11.0],
            [15.0, 15.0],
        ],
        dtype=np.float64,
    )
    stacked = np.vstack([normal, outliers])
    return pd.DataFrame(stacked, columns=["x1", "x2"])


def _load_demo_dataset(name: str) -> pd.DataFrame:
    """Örnek veri anahtarına göre sklearn veya sentetik tablo döndürür."""
    if name == "anomaly_synthetic":
        return _load_synthetic_anomaly_demo()
    return _load_sklearn_demo(name)


def _cluster_profile_means(
    analysis_df: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Küme başına seçili özelliklerde koşullu beklenen değer (örnek ortalaması).

    **Tanım:** Her küme :math:`c` için :math:`\\hat{\\mu}_{c,j} = \\text{mean}(x_{ij} \\mid i \\in c)`.
    Segment karakterizasyonu ve jüriye sözlü yorum için kullanılır.
    """
    block = analysis_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    grp = block.assign(_cluster=labels.values).groupby("_cluster", sort=True).mean()
    grp.index = [f"Cluster {i}" for i in grp.index]
    grp.index.name = "Küme"
    return grp


def _anomaly_vs_normal_mean_table(
    analysis_base: pd.DataFrame,
    pred: pd.Series,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, str]:
    """Anomali (-1) ile normal (1) gruplarında sütun ortalamalarını kıyaslar (keşifsel)."""
    normal_mask = pred == 1
    anomaly_mask = pred == -1
    rows_out: list[dict[str, Any]] = []
    best: tuple[str, float] | None = None

    for col in feature_columns:
        if col not in analysis_base.columns:
            continue
        s = pd.to_numeric(analysis_base[col], errors="coerce")
        mn = float(s[normal_mask].mean())
        ma = float(s[anomaly_mask].mean())
        if np.isnan(mn) and np.isnan(ma):
            continue
        if np.isnan(mn):
            mn = 0.0
        if np.isnan(ma):
            ma = 0.0
        denom = max(abs(mn), 1e-12)
        pct = (ma - mn) / denom * 100.0
        if abs(ma - mn) < 1e-9:
            direction = "yakın (aynı)"
        elif ma > mn:
            direction = "anomali daha yüksek"
        else:
            direction = "anomali daha düşük"
        rows_out.append(
            {
                "Sütun": col,
                "Normal ort.": mn,
                "Anomali ort.": ma,
                "Göreli fark (%)": pct,
                "Yön": direction,
            }
        )
        if best is None or abs(pct) > abs(best[1]):
            best = (col, pct)

    out = pd.DataFrame(rows_out)
    if out.empty or best is None:
        summary = (
            "Karşılaştırma için yeterli sayısal sütun veya gözlem yok "
            "(veya tüm değerler eksik)."
        )
    else:
        c, p = best
        summary = (
            f"En belirgin fark **{c}** sütununda: anomali grubu ortalaması, normal gruba "
            f"göre yaklaşık **%{abs(p):.1f}** oranında **{'yüksek' if p > 0 else 'düşük'}**. "
            "Bu tablo keşifseldir; Isolation Forest çıktısının **nedensel** açıklaması değildir."
        )
    return out, summary


def _resolve_plotly_selection_rows(
    plotly_state: Any,
    base: pd.DataFrame,
) -> pd.DataFrame | None:
    """Lasso/kutu seçimindeki noktaları `customdata` / indeks ile tablo satırlarına bağla."""
    if plotly_state is None:
        return None
    sel = getattr(plotly_state, "selection", None)
    if sel is None:
        return None
    if not isinstance(sel, dict):
        try:
            sel = dict(sel)
        except (TypeError, ValueError):
            return None
    points = sel.get("points") or []
    if not points:
        return None
    index_set = set(base.index)
    str_to_idx = {str(i): i for i in base.index}
    keys: list[Any] = []
    for p in points:
        cd = p.get("customdata")
        if cd is None:
            continue
        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) > 0:
            cell = cd[0]
        else:
            cell = cd
        if cell in index_set:
            keys.append(cell)
        elif str(cell) in str_to_idx:
            keys.append(str_to_idx[str(cell)])
        elif str(cell) in index_set:
            keys.append(str(cell))
    if not keys:
        return None
    uniq = pd.unique(pd.Series(keys))
    try:
        return base.loc[uniq]
    except (KeyError, TypeError):
        return None


def _render_plotly_interactive(
    fig: Any,
    *,
    key: str,
    selection_base: pd.DataFrame,
    float_decimals: int,
    show_selection_hint: bool = True,
) -> None:
    """Scatter vb. için lasso/kutu seçimi ve seçilen satır önizlemesi."""
    cfg = CONFIG["PLOTLY"]["CHART_CONFIG"]
    if show_selection_hint:
        st.caption(CONFIG["PLOTLY"]["CAPTION_SELECTION"])
    ps = st.plotly_chart(
        fig,
        use_container_width=True,
        config=cfg,
        key=key,
        on_select="rerun",
        selection_mode=("lasso", "box"),
    )
    picked = _resolve_plotly_selection_rows(ps, selection_base)
    if picked is not None and not picked.empty:
        st.markdown(f"**Seçilen gözlemler** ({len(picked)} satır)")
        st.dataframe(
            _style_analysis_dataframe(
                picked.head(200),
                float_decimals=float_decimals,
            ),
            use_container_width=True,
        )


def _render_decision_banner(markdown_text: str) -> None:
    """Akıllı grafik seçim gerekçesini grafik üstünde gösterir."""
    st.markdown(
        f'<div class="wv-decision-banner">{markdown_text}</div>',
        unsafe_allow_html=True,
    )


def _render_filter_impact_summary(
    full_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
) -> None:
    """Sidebar filtre alanı altında veri yapısı ve filtre oranı özetini gösterir."""
    n_full = max(1, len(full_df))
    n_filt = len(filtered_df)
    pct_filtered = max(0.0, min(100.0, (1.0 - n_filt / n_full) * 100.0))
    n_continuous, n_categorical = count_structure_variables(filtered_df)
    st.markdown(
        (
            '<div class="wv-filter-impact">'
            f"Şu an toplam verinin <strong>%{pct_filtered:.1f}</strong> kadarı filtrelendi. "
            f"Filtreleme sonrası sürekli değişken sayısı: <strong>{n_continuous}</strong>, "
            f"Kategorik değişken sayısı: <strong>{n_categorical}</strong>."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_plotly_static(fig: Any, *, key: str | None = None) -> None:
    """Isı haritası, dirsek, histogram: araç çubuğu + yakınlaştırma; seçim tablosu yok."""
    cfg = CONFIG["PLOTLY"]["CHART_CONFIG"]
    kw: dict[str, Any] = {
        "use_container_width": True,
        "config": cfg,
    }
    if key is not None:
        kw["key"] = key
    st.plotly_chart(fig, **kw)


def _model_interpreter_lines(result: dict[str, Any]) -> list[str]:
    """Eşik tabanlı model yorumları (UI ve rapor için ortak)."""
    thr = CONFIG["INTERPRETER"]
    lines: list[str] = []
    sil = result.get("silhouette")
    if sil is not None:
        if sil > float(thr["SILHOUETTE_STRONG"]):
            lines.append(thr["MSG_SIL_HIGH"])
        elif sil < float(thr["SILHOUETTE_WEAK"]):
            lines.append(thr["MSG_SIL_LOW"])
    pv = result.get("pca_variance_pct")
    if pv is not None and float(pv) < float(thr["PCA_LOW_VARIANCE_PCT"]):
        lines.append(thr["MSG_PCA_LOW"])
    return lines


def _render_model_interpreter(result: dict[str, Any]) -> None:
    """Silhouette ve PCA varyansına göre otomatik karar desteği notları."""
    lines = _model_interpreter_lines(result)
    if lines:
        st.info("**Model yorumlayıcı**\n\n" + "\n\n".join(lines))


def _inject_premium_enterprise_css() -> None:
    """Kurumsal kimlik: metrik kartları, sidebar degrade, buton hover, kod blokları."""
    st.markdown(
        """
        <style>
        /* Kod blokları — tema uyumu */
        .stCodeBlock, .stCodeBlock pre, .stCodeBlock code,
        div[data-testid="stCode"] pre, div[data-testid="stCode"] code {
            background-color: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
        }

        /* Metrik kartları — gölge ve yuvarlatılmış köşe */
        div[data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid rgba(99, 102, 241, 0.12);
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            padding: 14px 18px;
            transition: box-shadow 0.2s ease, transform 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.14);
            transform: translateY(-1px);
        }
        div[data-testid="stMetric"] label {
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        /* Sidebar — hafif kurumsal degrade */
        section[data-testid="stSidebar"] > div {
            background: linear-gradient(
                165deg,
                rgba(79, 70, 229, 0.07) 0%,
                rgba(14, 165, 233, 0.05) 45%,
                var(--background-color) 100%
            ) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown h1,
        section[data-testid="stSidebar"] .stMarkdown h2,
        section[data-testid="stSidebar"] .stMarkdown h3 {
            letter-spacing: -0.02em;
        }

        /* Birincil / indirme butonları — hover belirginleşme */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease,
                filter 0.18s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(79, 70, 229, 0.28) !important;
            filter: brightness(1.04);
        }
        .stButton > button:active {
            transform: translateY(0);
        }
        div[data-testid="stDownloadButton"] > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease,
                filter 0.18s ease !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(14, 165, 233, 0.28) !important;
            filter: brightness(1.05);
        }

        /* Sekmeler — ince üst çizgi vurgusu */
        button[data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            transition: background 0.15s ease;
        }

        /* Akıllı karar bandı — grafik üstü bilgi şeridi */
        .wv-decision-banner {
            background: linear-gradient(
                90deg,
                rgba(79, 70, 229, 0.1) 0%,
                rgba(14, 165, 233, 0.08) 100%
            );
            border-left: 4px solid rgba(79, 70, 229, 0.55);
            border-radius: 10px;
            padding: 10px 14px;
            margin: 6px 0 12px 0;
            font-size: 0.92rem;
            line-height: 1.45;
            color: var(--text-color);
        }

        /* Sidebar filtre etkisi özeti */
        .wv-filter-impact {
            background: var(--secondary-background-color);
            border: 1px solid rgba(14, 165, 233, 0.18);
            border-radius: 10px;
            padding: 10px 12px;
            margin: 4px 0 10px 0;
            font-size: 0.82rem;
            line-height: 1.4;
            color: var(--text-color);
        }
        .wv-filter-impact strong {
            color: rgba(79, 70, 229, 0.95);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _column_uniqueness_score(raw_df: pd.DataFrame) -> float:
    """Sütun başına benzersiz değer oranının ortalaması (0–100)."""
    if raw_df.empty or raw_df.shape[1] == 0:
        return 50.0
    n = len(raw_df)
    ratios: list[float] = []
    for c in raw_df.columns:
        nu = int(raw_df[c].nunique(dropna=True))
        ratios.append(min(1.0, nu / max(1, n)))
    return float(100.0 * (sum(ratios) / len(ratios)))


def _variance_balance_score(raw_df: pd.DataFrame) -> float:
    """Sayısal sütunların varyanslarının göreli dengesi (düşük CV → yüksek skor, 0–100)."""
    num_cols, _ = DataLoader.infer_column_types(raw_df)
    if len(num_cols) < 2:
        return 65.0
    vars_list: list[float] = []
    for c in num_cols:
        s = pd.to_numeric(raw_df[c], errors="coerce").dropna()
        if len(s) < 2:
            continue
        v = float(s.var(ddof=0))
        if np.isfinite(v) and v >= 0.0:
            vars_list.append(max(v, 1e-18))
    if len(vars_list) < 2:
        return 60.0
    arr = np.asarray(vars_list, dtype=float)
    mean_v = float(np.mean(arr))
    if mean_v < 1e-15:
        return 50.0
    cv = float(np.std(arr, ddof=0) / mean_v)
    return float(max(0.0, min(100.0, 100.0 * (1.0 - min(cv / 2.5, 1.0)))))


def _numeric_feature_richness_score(n_numeric: int, *, sweet: int = 12) -> float:
    """Modele girebilecek sayısal sütun sayısı için 0–100 zenginlik skoru."""
    if n_numeric < 1:
        return 25.0
    if n_numeric >= sweet:
        return 100.0
    return float(25.0 + 75.0 * (n_numeric - 1) / max(1, sweet - 1))


def _data_quality_radar_axes(
    qm: dict[str, Any],
    raw_df: pd.DataFrame,
    result: dict[str, Any] | None,
    n_numeric_features: int,
) -> list[tuple[str, float]]:
    """Kalite radarı: altı eksen (0–100)."""
    uniq = _column_uniqueness_score(raw_df)
    var_bal = _variance_balance_score(raw_df)
    feat_rich = _numeric_feature_richness_score(n_numeric_features)
    anomaly_scarcity = 72.0
    if result is not None:
        pred = result.get("pred")
        if pred is not None:
            p = np.asarray(pred).ravel()
            n = int(len(p))
            if n > 0:
                n_an = int((p == -1).sum())
                pct_an = 100.0 * n_an / float(n)
                anomaly_scarcity = max(0.0, min(100.0, 100.0 - pct_an))
    imput_score = max(
        0.0, 100.0 - float(qm.get("pct_imputed_of_total", 0.0))
    )
    return [
        ("Doluluk oranı", float(qm["pct_filled"])),
        ("Benzersizlik", uniq),
        ("Anomali azlığı", anomaly_scarcity),
        ("Varyans dengesi", var_bal),
        ("Özellik sayısı", feat_rich),
        ("Düşük imputasyon yükü", imput_score),
    ]


def _build_ai_strategy_advice_md(
    result: dict[str, Any],
    n_obs: int,
    *,
    data_mode_id: str = "tabular",
) -> str:
    """Kural tabanlı strateji metni (Insight Engine — rapor sonu ve analiz sekmesi)."""
    thr = CONFIG["INTERPRETER"]
    bullets: list[str] = []
    sil = result.get("silhouette")
    if sil is not None and float(sil) > float(thr["SILHOUETTE_STRONG"]):
        bullets.append(
            "Segmentasyon kalitesi yüksek, **hedef kitle operasyonlarına** başlanabilir."
        )
    elif sil is not None and float(sil) < float(thr["SILHOUETTE_WEAK"]):
        bullets.append(
            "Küme ayrışması zayıf; **k**, özellik seçimi veya ön işlemi (ölçekleme / log) "
            "gözden geçirilmeli."
        )
    pred = result.get("pred")
    if pred is not None and n_obs > 0:
        p = np.asarray(pred).ravel()
        pct_an = 100.0 * float((p == -1).sum()) / float(len(p))
        if pct_an > 10.0:
            bullets.append(
                "Veri giriş süreçlerinde **yüksek gürültü** tespit edildi; "
                "**sensör kalibrasyonu** veya kayıt hattı kontrolü gerekebilir "
                f"(aykırı oranı ~%{pct_an:.1f})."
            )
        elif pct_an > 5.0:
            bullets.append(
                f"Aykırı oranı **%{pct_an:.1f}** — kaynak kalitesini izlemek faydalıdır."
            )
    if data_mode_id == "image_features" and n_obs > 0:
        bullets.append(
            "Kan hücresi morfometrisinde kümeler ayrışmış görünüyor; **anormal hücre** "
            "adayları irregularity_score ve nucleus_density ile klinik ön tarama "
            "iş akışına alınabilir."
        )
    pv = result.get("pca_variance_pct")
    if pv is not None and float(pv) < float(thr["PCA_LOW_VARIANCE_PCT"]):
        bullets.append(
            "İlk iki PCA bileşeni düşük varyans açıklıyor; ek bileşen veya doğrusal olmayan "
            "yöntemler tartışılabilir."
        )
    if not bullets:
        bullets.append(
            "Bu çalıştırmada güçlü uyarı üretilmedi; sonuçları domain bilgisi ve iş hedefleriyle "
            "birlikte yorumlayın."
        )
    body = "\n".join(f"- {b}" for b in bullets)
    return (
        body
        + "\n\n*Kural tabanlı önerilerdir; üretken yapay zeka (LLM) çıktısı değildir.*"
    )


def _dataframe_to_markdown_table(df: pd.DataFrame, *, float_decimals: int) -> str:
    """tabulate bağımlılığı olmadan basit Markdown tablo (UTF-8 rapor için)."""
    if df.empty:
        return "(Boş tablo)"
    idx_name = str(df.index.name) if df.index.name else "Küme"
    cols = [idx_name] + [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows: list[str] = []
    for idx, row in df.iterrows():
        cells: list[str] = [str(idx)]
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                cells.append("—")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):.{float_decimals}f}")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _build_academic_report_markdown(
    *,
    result: dict[str, Any],
    analysis_df: pd.DataFrame,
    feat_cols: list[str],
    n_clusters: int,
    contamination: float,
    app_version: str,
    float_decimals: int,
    cluster_profile_df: pd.DataFrame | None = None,
    use_log_transform: bool = False,
    log_skipped_columns: list[str] | None = None,
    data_mode_id: str = "tabular",
) -> str:
    """Savunma / jüri için Markdown metin raporu (indirme .txt)."""
    parts: list[str] = [
        "# Akademik analiz raporu",
        "",
        f"**Uygulama sürümü:** {app_version}",
        f"**Analiz edilen gözlem sayısı:** {len(analysis_df)}",
        "",
        "## Model parametreleri",
        "",
        f"- **K-Means küme sayısı (k):** {n_clusters}",
        f"- **Isolation Forest contamination:** {contamination}",
        f"- **Modele dahil sayısal sütunlar:** {', '.join(feat_cols) if feat_cols else '(yok)'}",
        f"- **Hedef değişken (RF):** {result.get('target_column') or '(seçilmedi)'}",
        "",
    ]
    if use_log_transform:
        parts.extend(
            [
                "- **Ön işlem:** Sayısal modele giren sütunlarda `log1p` dönüşümü "
                "uygulandı (sağ çarpıklığı azaltmak, kümeleme için).",
                "",
            ]
        )
        sk = log_skipped_columns or []
        if sk:
            parts.append(
                f"- **Log dönüşümü atlanan sütunlar (negatif değer vb.):** "
                f"{', '.join(sk)}"
            )
            parts.append("")
    parts.extend(
        [
            "## Özet metrikler",
            "",
        ]
    )
    sil = result.get("silhouette")
    if sil is not None:
        parts.append(f"- **Silhouette skoru (ortalama, öklidyen):** {sil:.4f}")
    else:
        parts.append(
            "- **Silhouette skoru:** tanımsız veya hesaplanmadı (ör. k = 1 veya tek küme)."
        )
    pv = result.get("pca_variance_pct")
    if pv is not None:
        parts.append(
            f"- **PCA — ilk iki bileşenin açıkladığı varyans oranı:** {pv:.2f}% "
            "(ölçeklenmiş özellik uzayı)"
        )
    inertia = result.get("inertia")
    if inertia is not None:
        parts.append(
            f"- **K-Means inertia (WCSS, ölçeklenmiş uzay):** {inertia:.6f}"
        )
    parts.extend(["", "## Model yorumlayıcı (otomatik)", ""])
    interp = _model_interpreter_lines(result)
    if interp:
        for line in interp:
            parts.append(f"- {line}")
    else:
        parts.append(
            "- Bu çalıştırmada eşik tabanlı uyarı üretilmedi (Silhouette ve PCA "
            "eşikleri dışında)."
        )
    tgt_imp = result.get("target_importance_df")
    tgt_name = str(result.get("target_column") or "")
    parts.extend(["", "## Hedef değişken özellik önemi (Random Forest)", ""])
    if isinstance(tgt_imp, pd.DataFrame) and not tgt_imp.empty:
        parts.append(
            StatisticalCommentator.feature_importance(
                tgt_imp, target_name=tgt_name or None
            )
        )
        parts.append("")
        parts.append(
            _dataframe_to_markdown_table(
                tgt_imp.head(10), float_decimals=float_decimals
            )
        )
    elif result.get("err_target_importance"):
        parts.append(f"- Atlandı: {result['err_target_importance']}")
    else:
        parts.append("- Hedef önemi hesaplanmadı.")
    parts.extend(["", "## Küme profilleri (seçilen sütunlarda ortalama)", ""])
    prof_df = cluster_profile_df if cluster_profile_df is not None else analysis_df
    if result.get("labels") is not None and feat_cols:
        try:
            cmeans = _cluster_profile_means(
                prof_df, result["labels"], feat_cols
            )
            parts.append(_dataframe_to_markdown_table(cmeans, float_decimals=float_decimals))
        except (KeyError, TypeError, ValueError) as exc:
            parts.append(f"(Küme ortalama tablosu üretilemedi: {exc})")
    else:
        parts.append("(Küme etiketleri veya sütun seti yetersiz.)")
    parts.extend(["", "## İşlem durumu ve hatalar", ""])
    err_parts = []
    if result.get("err_cluster"):
        err_parts.append(f"- **Kümeleme:** {result['err_cluster']}")
    if result.get("err_pca"):
        err_parts.append(f"- **PCA:** {result['err_pca']}")
    if result.get("err_anomaly"):
        err_parts.append(f"- **Anomali:** {result['err_anomaly']}")
    if result.get("err_elbow"):
        err_parts.append(f"- **Dirsek taraması:** {result['err_elbow']}")
    parts.extend(err_parts if err_parts else ["- Kayıtlı hata yok."])
    strat_title = str(
        CONFIG.get("UI", {}).get(
            "REPORT_STRATEGY_SECTION_MD",
            "## 🤖 AI Stratejik Tavsiyeler",
        )
    )
    parts.extend(
        [
            "",
            strat_title,
            "",
            _build_ai_strategy_advice_md(
                result,
                len(analysis_df),
                data_mode_id=data_mode_id,
            ),
            "",
            "---",
            "*Bu rapor, analiz sekmesindeki o anki oturum çıktılarından üretilmiştir.*",
        ]
    )
    return "\n".join(parts)


def _compute_system_health_score(
    qm: dict[str, Any],
    result: dict[str, Any] | None,
) -> float:
    """Kayıp veri doluluğu ve anomali oranından 0–100 sistem sağlığı skoru."""
    fill_pct = float(qm.get("pct_filled", 100.0))
    anomaly_pct = 0.0
    if result is not None:
        pred = result.get("pred")
        if pred is not None:
            p = np.asarray(pred).ravel()
            if len(p) > 0:
                anomaly_pct = 100.0 * float((p == -1).sum()) / float(len(p))
    score = 0.55 * fill_pct + 0.45 * max(0.0, 100.0 - anomaly_pct)
    return float(max(0.0, min(100.0, score)))


def _dominant_cluster_info(labels: pd.Series | None) -> tuple[str, str | None]:
    """En kalabalık küme adı ve delta metni."""
    if labels is None or len(labels) == 0:
        return "—", None
    vc = labels.astype(int).value_counts().sort_index()
    dom = int(vc.idxmax())
    count = int(vc.loc[dom])
    pct = 100.0 * count / max(1, len(labels))
    delta = f"%{pct:.0f} pay"
    return f"Küme {dom}", delta


def _top_target_correlation_info(
    target_imp_df: pd.DataFrame | None,
) -> tuple[str, str | None]:
    """Hedefi en çok etkileyen özellik adı ve etki yüzdesi."""
    if not isinstance(target_imp_df, pd.DataFrame) or target_imp_df.empty:
        return "—", None
    feat_col = str(target_imp_df.columns[0])
    imp_col = str(target_imp_df.columns[1])
    top = target_imp_df.sort_values(imp_col, ascending=False).iloc[0]
    name = str(top[feat_col])
    pct = float(top[imp_col]) * 100.0
    return name, f"%{pct:.1f} etki"


def _primary_imputation_method_label(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame | None,
) -> str:
    """Temizlemede baskın imputation yöntemi (Ortalama / Moda / Medyan)."""
    if cleaned_df is None:
        return "Ortalama"
    numeric_cols, cat_cols = DataLoader.infer_column_types(raw_df)
    counts = {"Ortalama": 0, "Moda": 0, "Medyan": 0}
    for col in numeric_cols:
        if col not in raw_df.columns or col not in cleaned_df.columns:
            continue
        counts["Ortalama"] += int((raw_df[col].isna() & cleaned_df[col].notna()).sum())
    for col in cat_cols:
        if col not in raw_df.columns or col not in cleaned_df.columns:
            continue
        n_imp = int((raw_df[col].isna() & cleaned_df[col].notna()).sum())
        if pd.api.types.is_datetime64_any_dtype(raw_df[col]):
            counts["Medyan"] += n_imp
        else:
            counts["Moda"] += n_imp
    best = max(counts, key=lambda k: counts[k])
    if counts[best] <= 0:
        return "Ortalama"
    used = [k for k, v in counts.items() if v > 0]
    if len(used) == 1:
        return used[0]
    return " / ".join(used)


def _render_executive_kpi_row(
    *,
    n_obs: int,
    n_base: int,
    health_score: float,
    dominant_cluster: str,
    cluster_delta: str | None,
    top_feature: str,
    feature_delta: str | None,
) -> None:
    """Analiz ekranı üstü dört Executive KPI kartı."""
    obs_delta = n_obs - n_base
    delta_obs: str | int | None = obs_delta if obs_delta != 0 else None
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Toplam Gözlem Sayısı",
            f"{n_obs:,}",
            delta=delta_obs,
            help="Filtre sonrası analize giren satır; delta, filtre öncesi tabloya göre fark.",
        )
    with k2:
        st.metric(
            "Sistem Sağlığı Skoru",
            f"{health_score:.0f}/100",
            help="Doluluk oranı (%55) ve anomali azlığı (%45) birleşik göstergesi.",
        )
    with k3:
        st.metric(
            "Baskın Küme",
            dominant_cluster,
            delta=cluster_delta,
            help="En çok gözlemi barındıran K-Means kümesi.",
        )
    with k4:
        st.metric(
            "Hedef Korelasyonu",
            top_feature,
            delta=feature_delta,
            help="Random Forest özellik öneminde hedefi en çok açıklayan sütun.",
        )


def _normalize_turkish_query(text: str) -> str:
    """Regex eşleştirme için basit Türkçe karakter normalizasyonu."""
    repl = str.maketrans(
        "çğıöşüÇĞİÖŞÜ",
        "cgiosucgiosu",
    )
    return text.translate(repl).lower().strip()


def _answer_data_question(
    query: str,
    *,
    result: dict[str, Any] | None,
    analysis_df: pd.DataFrame,
    analysis_base: pd.DataFrame,
    target_col: str,
    qm: dict[str, Any] | None,
    n_clusters: int,
) -> str | None:
    """Doğal dil veri Q&A simülatörü (kural + regex tabanlı)."""
    q = _normalize_turkish_query(query)
    if not q:
        return None

    labels = result.get("labels") if result else None
    pred = result.get("pred") if result else None

    if result is None:
        if re.search(r"(kac|kadar|sayi|satir|gozlem)", q):
            return (
                f"Şu an **{len(analysis_df):,}** gözlem yüklü. Küme ve anomali yanıtları için "
                "**Analizi Çalıştır** düğmesine basın."
            )
        return (
            "Henüz model çıktısı yok. Analizi çalıştırdıktan sonra küme, anomali ve hedef "
            "sorularını yanıtlayabilirim."
        )

    if re.search(r"(en yuksek|en buyuk|baskin|dominant|kalabalik)", q) and re.search(
        r"kume", q
    ):
        if labels is None:
            return "Küme etiketleri üretilemedi; analiz hatasını kontrol edin."
        vc = labels.astype(int).value_counts()
        dom = int(vc.idxmax())
        return (
            f"En yüksek değerli küme **Küme {dom}**; toplam **{int(vc[dom]):,}** gözlem "
            f"(**%{100.0 * vc[dom] / len(labels):.1f}** pay). "
            f"Silhouette: **{result.get('silhouette', '—')}**."
        )

    if re.search(r"ortalama|mean|medyan", q):
        num_cols, _ = DataLoader.infer_column_types(analysis_base)
        if not num_cols:
            return "Sayısal sütun bulunamadı; ortalama hesaplanamadı."
        col = num_cols[0]
        for c in num_cols:
            if c.lower() in q or _normalize_turkish_query(c) in q:
                col = c
                break
        s = pd.to_numeric(analysis_base[col], errors="coerce").dropna()
        if s.empty:
            return f"**{col}** sütununda yeterli sayısal veri yok."
        return (
            f"**{col}** için örnek ortalama **{float(s.mean()):.4g}**, "
            f"medyan **{float(s.median()):.4g}** "
            f"({len(s):,} gözlem)."
        )

    if re.search(r"anomali|aykiri|outlier", q):
        if pred is None:
            return "Anomali modeli çalıştırılamadı."
        p = np.asarray(pred).ravel()
        n_an = int((p == -1).sum())
        return (
            f"Isolation Forest **{n_an}** anomali işaretledi "
            f"(toplam **{len(p):,}** gözlemin **%{100.0 * n_an / len(p):.1f}**'i)."
        )

    if re.search(r"silhouette|ayrisma|ayrism", q):
        sil = result.get("silhouette")
        if sil is None:
            return "Silhouette skoru hesaplanamadı."
        return (
            f"Silhouette skoru **{float(sil):.3f}** "
            f"({int(result.get('n_clusters', n_clusters))} küme). "
            "−1…1 aralığında; yüksek değer daha net segmentasyonu destekler."
        )

    if re.search(r"hedef|onem|etki|korelasyon|feature", q):
        imp = result.get("target_importance_df")
        name, delta = _top_target_correlation_info(
            imp if isinstance(imp, pd.DataFrame) else None
        )
        if name == "—":
            return f"**{target_col}** için özellik önemi tablosu üretilemedi."
        return (
            f"**{target_col}** hedefini en çok etkileyen sütun **{name}** "
            f"({delta or 'etki bilinmiyor'})."
        )

    if re.search(r"saglik|kalite|skor", q):
        if qm is None:
            return "Veri kalitesi metrikleri henüz hesaplanmadı."
        hs = _compute_system_health_score(qm, result)
        return (
            f"Sistem sağlığı skoru **{hs:.0f}/100**. "
            f"Ham doluluk **%{float(qm.get('pct_filled', 0)):.1f}**, "
            f"eksiklik **%{float(qm.get('pct_missing', 0)):.1f}**."
        )

    if re.search(r"kume", q):
        if labels is None:
            return "Küme bilgisi yok."
        vc = labels.astype(int).value_counts().sort_index()
        parts = [f"Küme {int(k)}: {int(v)} gözlem" for k, v in vc.items()]
        return "K-Means dağılımı — " + "; ".join(parts) + "."

    return (
        "Sorunuzu tam eşleştiremedim. Şunları deneyin: "
        "'En yüksek küme hangisi?', 'Anomali sayısı?', 'Silhouette skoru?', "
        "'Hedef önemi?' veya 'Ortalama değer?'."
    )


def _render_data_qa_bar(
    *,
    result: dict[str, Any] | None,
    analysis_df: pd.DataFrame,
    analysis_base: pd.DataFrame,
    target_col: str,
    qm: dict[str, Any] | None,
    n_clusters: int,
    ui: dict[str, Any],
) -> None:
    """Dashboard üstü doğal dil veri sorgu alanı."""
    st.markdown("#### 💬 Veri Asistanı (Q&A)")
    question = st.text_input(
        ui.get("DATA_QA_PLACEHOLDER", "Verinize soru sorun"),
        key="wv_data_qa_input",
        placeholder=ui.get("DATA_QA_HINT", ""),
    )
    if question.strip():
        answer = _answer_data_question(
            question,
            result=result,
            analysis_df=analysis_df,
            analysis_base=analysis_base,
            target_col=target_col,
            qm=qm,
            n_clusters=n_clusters,
        )
        if answer:
            st.success(answer)


def _inject_zen_mode_css(*, full: bool = True) -> None:
    """Jüri sunum modu: sidebar ve kalabalık UI öğelerini gizler / soluklaştırır."""
    hide_sidebar = (
        "section[data-testid='stSidebar'] { display: none !important; }"
        if full
        else ""
    )
    st.markdown(
        f"""
        <style>
        {hide_sidebar}
        .main .block-container {{
            padding-top: 1.5rem;
            max-width: 100%;
        }}
        .wv-zen-dim {{
            opacity: 0.12 !important;
            pointer-events: none !important;
            max-height: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        div[data-testid="stTabs"] {{
            opacity: 0.08;
            pointer-events: none;
            max-height: 2.5rem;
            overflow: hidden;
        }}
        .wv-jury-stage {{
            background: linear-gradient(
                180deg,
                rgba(79, 70, 229, 0.06) 0%,
                transparent 35%
            );
            border-radius: 16px;
            padding: 8px 4px 24px 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_jury_presentation_view(
    result: dict[str, Any],
    *,
    analysis_df: pd.DataFrame,
    analysis_base: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    n_clusters: int,
    dm_sess: str,
) -> None:
    """Jüri modu: 3D PCA, karar bandı ve stratejik AI yorumları."""
    viz = DataVisualizer(theme=_streamlit_theme())
    labels = result.get("labels")
    st.markdown('<div class="wv-jury-stage">', unsafe_allow_html=True)
    st.markdown("# 🎬 Jüri Sunum Görünümü")
    st.caption("Odak: 3D segmentasyon, akıllı karar matrisi ve stratejik öneriler.")

    pca_3d = result.get("pca_coords_3d")
    if pca_3d is not None and labels is not None:
        st.markdown("### 3D PCA Explorer")
        try:
            pca3_enriched = pca_3d.join(analysis_df, how="left")
            pv3 = result.get("pca_variance_pct_3d")
            fig_3d = viz.plot_3d_pca_clusters(
                pca3_enriched,
                labels.values,
                variance_explained_pct=float(pv3) if pv3 is not None else None,
                chart_height=860,
            )
            _render_plotly_static(fig_3d, key="wv_jury_plot_pca_3d")
            if pv3 is not None:
                _render_ai_comment(
                    f"PC1+PC2+PC3 ile açıklanan varyans **%{float(pv3):.1f}** "
                    "(ölçeklenmiş uzay)."
                )
        except ValueError as exc:
            st.warning(str(exc))
    else:
        st.info("3D PCA için analiz sonuçları eksik.")

    smart_cols = list(feat_cols) if feat_cols else list(analysis_base.columns.astype(str))
    if len(smart_cols) >= 2:
        st.markdown("### Akıllı Karar Matrisi")
        sa, sb = smart_cols[0], smart_cols[1]
        try:
            fig_smart, smart_kind, smart_comment = viz.plot_smart_bivariate(
                analysis_base,
                sa,
                sb,
            )
            _render_decision_banner(
                build_bivariate_decision_banner(analysis_base, sa, sb, smart_kind)
            )
            _render_plotly_static(fig_smart, key="wv_jury_plot_smart")
            _render_ai_comment(smart_comment)
        except ValueError as exc:
            st.warning(str(exc))

    st.markdown(
        CONFIG.get("UI", {}).get(
            "REPORT_STRATEGY_SECTION_MD",
            "## 🤖 AI Stratejik Tavsiyeler",
        )
    )
    st.markdown(
        _build_ai_strategy_advice_md(
            result,
            len(analysis_df),
            data_mode_id=_data_mode_id(dm_sess),
        )
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    ui = CONFIG["UI"]
    model = CONFIG["MODEL"]
    prev = CONFIG["PREVIEW"]
    fmt = CONFIG["FORMATTING"]
    msg = CONFIG["MESSAGES"]
    sess = CONFIG["SESSION_KEYS"]
    dev = CONFIG["DEVELOPER"]

    st.set_page_config(
        page_title=f'{ui["PAGE_TITLE"]} v{CONFIG["VERSION"]}',
        layout=ui["PAGE_LAYOUT"],
    )
    _inject_premium_enterprise_css()
    icon = ui.get("APP_TITLE_ICON", "")
    st.title(f"{icon} {ui['APP_TITLE']}".strip() if icon else ui["APP_TITLE"])
    st.caption(ui["APP_CAPTION"])

    demo_key = sess["DEMO_SAMPLE"]
    loader = DataLoader()
    uploaded = None

    dm_sess = sess["DATA_MODE"]
    if st.session_state.pop(_DEMO_FORCE_TABULAR_KEY, False):
        st.session_state[dm_sess] = 0
    _normalize_data_mode_index(dm_sess)

    with st.sidebar:
        st.markdown("**Hızlı başlangıç**")
        st.selectbox(
            ui["DATA_MODE_LABEL"],
            options=list(range(len(_DATA_MODE_ORDER))),
            format_func=lambda i: ui["DATA_MODES"][_DATA_MODE_ORDER[int(i)]],
            key=dm_sess,
        )
        demo_choice = st.selectbox(
            ui["DEMO_LABEL"],
            options=list(_DEMO_DATASET_LABELS.keys()),
            format_func=lambda k: _DEMO_DATASET_LABELS[k],
            key="wv_demo_dataset_pick",
        )
        if st.button(
            ui["BTN_DEMO_DATA"],
            help=ui.get("BTN_DEMO_HELP", ""),
            use_container_width=True,
        ):
            st.session_state[demo_key] = demo_choice
            st.session_state[_DEMO_FORCE_TABULAR_KEY] = True
            st.session_state.pop(sess["TEXT_CORPUS_BLOB"], None)
            st.session_state.pop(sess["FILE_UPLOADER"], None)
            st.rerun()

        _mode_now = _data_mode_id(dm_sess)
        uploaded = st.file_uploader(
            ui["FILE_UPLOADER_LABEL"],
            type=_extensions_for_data_mode(_mode_now),
            help=ui["UPLOADER_HELP_BY_MODE"].get(
                _mode_now, ui["FILE_UPLOADER_HELP"]
            ),
            key=sess["FILE_UPLOADER"],
        )

    mode_load = _data_mode_id(dm_sess)

    if uploaded is not None:
        st.session_state.pop(demo_key, None)
        try:
            if mode_load == "text_corpus":
                raw_bytes = uploaded.read()
                df, text_preview = DataLoader.process_text_file(raw_bytes)
                st.session_state[sess["TEXT_CORPUS_BLOB"]] = text_preview
                upload_fingerprint = (
                    f"text_corpus:{uploaded.name}:"
                    f"{getattr(uploaded, 'size', 0)}"
                )
            elif mode_load == "audio_features":
                st.session_state.pop(sess["TEXT_CORPUS_BLOB"], None)
                name_l = (uploaded.name or "").lower()
                nm_audio = uploaded.name or "audio_placeholder"
                if "audio" in name_l:
                    st.caption(
                        "Dosya adında **audio** geçti — simülasyon tohumu buna göre "
                        "ayarlanır (gerçek WAV ayrıştırması yok)."
                    )
                _seed_a = abs(hash(nm_audio.lower())) % (2**31 - 1)
                if "audio" in name_l:
                    _seed_a ^= 0x1A2B3C4D
                df = DataLoader.simulate_audio_features(
                    n_rows=120,
                    seed=_seed_a,
                    name_hint=nm_audio,
                )
                upload_fingerprint = (
                    f"audio_sim:{uploaded.name}:"
                    f"{getattr(uploaded, 'size', 0)}"
                )
            elif mode_load == "image_features":
                st.session_state.pop(sess["TEXT_CORPUS_BLOB"], None)
                name_li = (uploaded.name or "").lower()
                nm_img = uploaded.name or "image_placeholder"
                if "image" in name_li or "img" in name_li:
                    st.caption(
                        "Dosya adında **image** / **img** geçti — simülasyon tohumu "
                        "buna göre ayarlanır (gerçek CNN/OpenCV çıkarımı yok)."
                    )
                _seed_i = abs(hash(nm_img.lower())) % (2**31 - 1)
                if "image" in name_li:
                    _seed_i ^= 0x5EEDFACE
                elif "img" in name_li:
                    _seed_i ^= 0x10CA6001
                df = DataLoader.simulate_image_features(
                    n_rows=100,
                    seed=_seed_i,
                    name_hint=nm_img,
                )
                upload_fingerprint = (
                    f"image_sim:{uploaded.name}:"
                    f"{getattr(uploaded, 'size', 0)}"
                )
            else:
                st.session_state.pop(sess["TEXT_CORPUS_BLOB"], None)
                df = loader.load_file(uploaded)
                upload_fingerprint = (
                    f"tabular:{uploaded.name}:"
                    f"{getattr(uploaded, 'size', 0)}"
                )
        except DataLoadError as exc:
            st.error(str(exc))
            return
        st.success(
            f"Yüklendi: **{len(df)}** satır, **{len(df.columns)}** sütun."
        )
        if mode_load == "image_features":
            st.caption(
                "HealthTech modu: mikroskopi görüntülerinden çıkarılan **hücre çapı**, "
                "**çekirdek yoğunluğu** ve **şekil bozukluğu** özellikleri ile sağlıklı / "
                "anormal hücre kümelemesi simüle edilir."
            )
    elif st.session_state.get(demo_key):
        try:
            dkey = str(st.session_state[demo_key])
            df = _load_demo_dataset(dkey)
        except (ImportError, OSError, ValueError) as exc:
            st.error(f"Örnek veri yüklenemedi: {exc}")
            return
        st.session_state.pop(sess["TEXT_CORPUS_BLOB"], None)
        if dkey == "anomaly_synthetic":
            upload_fingerprint = "demo:anomaly_synthetic:numpy"
            demo_src = "100 normal + 5 aykırı nokta (NumPy)"
        else:
            upload_fingerprint = f"demo:{dkey}:sklearn"
            demo_src = "sklearn.datasets"
        ds_name = _DEMO_DATASET_LABELS.get(dkey, "Örnek")
        st.success(
            f'{ui["SUCCESS_DEMO"].format(dataset=ds_name)} '
            f"**{len(df)}** satır, **{len(df.columns)}** sütun — _{demo_src}_"
        )
    else:
        st.info(ui["INFO_NO_FILE"])
        return

    if st.session_state.get("_upload_fingerprint") != upload_fingerprint:
        st.session_state["_upload_fingerprint"] = upload_fingerprint
        st.session_state.pop("analiz_result", None)
        st.session_state.pop("_feature_selection_key", None)

    cleaned: pd.DataFrame | None = None
    clean_error: str | None = None
    try:
        cleaned = loader.clean_data(df)
    except PreprocessingError as exc:
        clean_error = str(exc)

    analysis_base = cleaned if cleaned is not None else df

    numeric_for_model, categorical_for_model = DataLoader.infer_column_types(
        analysis_base
    )

    cat_filters: dict[str, list[str]] = {}
    numeric_ranges: dict[str, tuple[float, float]] = {}
    target_variable: str = str(analysis_base.columns[0])
    datetime_candidates = _detect_datetime_columns(analysis_base)
    time_column: str | None = None
    ts_value_column: str | None = None
    time_series_mode = len(datetime_candidates) > 0
    whatif_key = _whatif_session_key(upload_fingerprint)

    with st.sidebar:
        with st.expander("🗂️ Veri Kaynağı & Filtreleme", expanded=True):
            src_lbl = (
                uploaded.name
                if uploaded is not None
                else f"Örnek ({st.session_state.get(demo_key, '—')})"
            )
            st.caption(f"Aktif: **{src_lbl}** · {len(df)} satır")
            if not numeric_for_model:
                st.warning(msg["WARN_NO_INFERRED_NUMERIC"])
                selected_numeric: list[str] = []
            else:
                selected_numeric = st.multiselect(
                    "Analize dahil sayısal sütunlar",
                    options=numeric_for_model,
                    default=numeric_for_model,
                    help="K-Means, PCA ve Isolation Forest yalnızca seçilen sütunlarla çalışır.",
                )
                if len(selected_numeric) < int(model["MIN_NUMERIC_FEATURES"]):
                    st.caption(msg["CAPTION_PICK_TWO_COLUMNS"])

        with st.expander("🔎 Veri Filtresi", expanded=True):
            st.caption(
                "Kategori ve sayısal aralık filtreleri seçildiğinde tüm grafikler "
                "anında güncellenir."
            )
            for cat_col in categorical_for_model[:8]:
                opts = sorted(
                    analysis_base[cat_col].astype(str).dropna().unique().tolist()
                )
                if len(opts) <= 1:
                    continue
                picked = st.multiselect(
                    f"{cat_col}",
                    options=opts,
                    default=opts,
                    key=f"wv_filter_cat_{cat_col}",
                )
                cat_filters[cat_col] = picked
            num_filter_options = ["—"] + list(numeric_for_model)
            num_filter_col = st.selectbox(
                "Sayısal aralık filtresi",
                options=num_filter_options,
                index=0,
                key="wv_filter_numeric_col",
            )
            if num_filter_col != "—":
                num_s = pd.to_numeric(
                    analysis_base[num_filter_col], errors="coerce"
                ).dropna()
                if not num_s.empty:
                    lo_raw, hi_raw = float(num_s.min()), float(num_s.max())
                    if lo_raw < hi_raw:
                        rng = st.slider(
                            f"{num_filter_col} aralığı",
                            min_value=lo_raw,
                            max_value=hi_raw,
                            value=(lo_raw, hi_raw),
                            key="wv_filter_numeric_rng",
                        )
                        numeric_ranges[num_filter_col] = (
                            float(rng[0]),
                            float(rng[1]),
                        )

        filtered_preview = _apply_row_filters(
            analysis_base,
            cat_filters=cat_filters,
            numeric_ranges=numeric_ranges,
        )
        _render_filter_impact_summary(analysis_base, filtered_preview)
        n_rows_model = len(filtered_preview)
        max_k = max(1, n_rows_model - 1) if n_rows_model > 1 else 1
        default_k = min(int(model["DEFAULT_K_MEANS_CAP"]), max_k)
        elbow_cap = max(
            2, min(int(model["ELBOW_K_MAX_UPPER"]), max(1, n_rows_model - 1))
        )

        with st.expander("⚡ Model Parametreleri", expanded=True):
            col_options = list(analysis_base.columns.astype(str))
            target_variable = st.selectbox(
                "Açıklanacak hedef değişken",
                options=col_options,
                index=_default_target_column_index(col_options),
                help=(
                    "Random Forest özellik önemi, bu değişkeni açıklamada en etkili "
                    "sütunları gösterir."
                ),
                key=sess["TARGET_VARIABLE"],
            )
            n_clusters = st.number_input(
                "Küme sayısı (K-Means)",
                min_value=1,
                max_value=max_k,
                value=default_k,
                step=1,
                help=(
                    "Stabil kümeleme ve Silhouette için **k < satır sayısı** olmalıdır "
                    "(üst sınır buna göre ayarlanır). Dirsek eğrisinde turuncu çizgi bu k ile hizalanır."
                ),
            )
            contamination = st.slider(
                "Beklenen aykırı oranı (Isolation Forest)",
                min_value=float(model["CONTAMINATION_MIN"]),
                max_value=float(model["CONTAMINATION_MAX"]),
                value=float(model["CONTAMINATION_DEFAULT"]),
                step=float(model["CONTAMINATION_STEP"]),
                help=(
                    "Bu değer, veri setindeki kirlilik oranını (yaklaşık yüzde kaç "
                    "gözlemin anomali / aykırı olabileceğini) model öncesinde tahmin "
                    "eder. scikit-learn aralığı: (0, 0.5]."
                ),
            )
            elbow_k_max = st.slider(
                "Dirsek eğrisi: denenecek maks. k",
                min_value=2,
                max_value=elbow_cap,
                value=min(int(model["ELBOW_K_DEFAULT_CAP"]), elbow_cap),
                help="Her k için K-Means inertia (WCSS) hesaplanır.",
            )
            st.caption(
                f'PCA: **{model["PCA_COMPONENTS_UI"]}B + '
                f'{model["PCA_COMPONENTS_3D"]}B (3D Explorer)** görselleştirme.'
            )

        with st.expander("📈 Zaman Serisi Analiz Modu", expanded=time_series_mode):
            if datetime_candidates:
                st.success(
                    f"Tarih/zaman sütunu tespit edildi — **{len(datetime_candidates)}** aday."
                )
                auto_time = datetime_candidates[0]
                time_opts = list(dict.fromkeys(datetime_candidates))
                time_column = st.selectbox(
                    "Zaman ekseni sütunu",
                    options=time_opts,
                    index=time_opts.index(auto_time) if auto_time in time_opts else 0,
                    key=sess["TIME_COLUMN"],
                )
                ts_numeric_opts = [
                    c for c in numeric_for_model if c != time_column
                ]
                if ts_numeric_opts:
                    ts_value_column = st.selectbox(
                        "İzlenecek metrik (çizgi değeri)",
                        options=ts_numeric_opts,
                        index=0,
                        key=sess["TS_VALUE_COLUMN"],
                    )
                else:
                    st.caption("Zaman serisi için sayısal metrik sütunu seçin.")
            else:
                st.caption(
                    "Otomatik tarih/zaman sütunu bulunamadı. İsimde date/time/timestamp "
                    "geçen veya %85+ parse edilen sütunlar modu açar."
                )
                manual_opts = ["—"] + list(analysis_base.columns.astype(str))
                manual_time = st.selectbox(
                    "Manuel zaman sütunu (isteğe bağlı)",
                    options=manual_opts,
                    index=0,
                    key="wv_manual_time_col",
                )
                if manual_time != "—":
                    time_column = manual_time
                    time_series_mode = True
                    ts_numeric_opts = [
                        c for c in numeric_for_model if c != time_column
                    ]
                    if ts_numeric_opts:
                        ts_value_column = st.selectbox(
                            "İzlenecek metrik",
                            options=ts_numeric_opts,
                            index=0,
                            key="wv_manual_ts_value",
                        )

        with st.expander("🔬 Gelişmiş Ayarlar", expanded=False):
            use_log_transform = st.checkbox(
                "Veriye logaritmik dönüşüm uygula",
                value=False,
                help=(
                    "Seçili sayısal sütunlarda log(1+x) (yalnızca değerler ≥ 0); "
                    "aşırı çarpık dağılımlarda kümeleme ve aykırı tespiti genelde "
                    "daha kararlı olur. Eksenleri yorumlarken manuel grafikte "
                    "orijinal ölçek kullanılır."
                ),
            )

        with st.expander("🧑‍💻 Geliştirici Hakkında", expanded=False):
            st.markdown(f"**{dev['NAME']}**")
            st.markdown(dev["SCHOOL"])
            st.markdown(dev["TECH_STACK"])

        show_academic_tips = st.checkbox(
            ui["CHECKBOX_ACADEMIC_TIPS"],
            value=False,
            key="wv_academic_presentation_tips",
            help=ui.get("CHECKBOX_ACADEMIC_TIPS_HELP", ""),
        )

        jury_mode = st.toggle(
            ui.get("JURY_MODE_LABEL", "🎬 Jüri Sunum Modu"),
            value=False,
            key="wv_jury_presentation_mode",
            help=ui.get("JURY_MODE_HELP", ""),
        )

        st.divider()
        st.caption(CONFIG["VISION_NOTE"])
        if st.button(
            "Analizi sıfırla",
            help="Yüklenen dosyayı ve tüm analiz oturumunu temizler; başa dönersiniz.",
            use_container_width=True,
        ):
            _reset_app_session()

    filtered_base = _apply_row_filters(
        analysis_base,
        cat_filters=cat_filters,
        numeric_ranges=numeric_ranges,
    )
    if len(filtered_base) == 0:
        st.warning(
            "Seçilen filtreler hiç satır bırakmadı; filtreleri genişletin."
        )
        return
    if len(filtered_base) < len(analysis_base):
        st.caption(
            f"Veri filtresi aktif: **{len(filtered_base)}** / "
            f"{len(analysis_base)} satır gösteriliyor."
        )

    if st.session_state.get("_whatif_fp") != upload_fingerprint:
        st.session_state[whatif_key] = filtered_base.copy()
        st.session_state[f"{whatif_key}_snap"] = _dataframe_edit_fingerprint(
            filtered_base
        )
        st.session_state["_whatif_fp"] = upload_fingerprint
        st.session_state.pop("analiz_result", None)

    analysis_base = st.session_state.get(whatif_key, filtered_base)
    if not isinstance(analysis_base, pd.DataFrame) or analysis_base.empty:
        analysis_base = filtered_base.copy()
        st.session_state[whatif_key] = analysis_base.copy()

    min_num_feats = int(model["MIN_NUMERIC_FEATURES"])
    cols_for_log = (
        selected_numeric
        if len(selected_numeric) >= min_num_feats
        else list(numeric_for_model)
    )
    analysis_df = analysis_base.copy()
    log_skipped_cols: list[str] = []
    if use_log_transform and cols_for_log:
        analysis_df, log_skipped_cols = DataLoader.apply_log1p_to_numeric_columns(
            analysis_base, columns=cols_for_log
        )

    whatif_fp = st.session_state.get(f"{whatif_key}_snap", "")
    feature_selection_key = (
        upload_fingerprint,
        tuple(sorted(selected_numeric)),
        int(elbow_k_max),
        int(n_clusters),
        round(float(contamination), 4),
        bool(use_log_transform),
        _data_mode_id(dm_sess),
        str(target_variable),
        _filter_state_fingerprint(cat_filters, numeric_ranges),
        whatif_fp,
        str(time_column or ""),
        str(ts_value_column or ""),
    )
    if st.session_state.get("_feature_selection_key") != feature_selection_key:
        st.session_state.pop("analiz_result", None)
    st.session_state["_feature_selection_key"] = feature_selection_key

    jury_mode = bool(st.session_state.get("wv_jury_presentation_mode", False))
    result_early: dict[str, Any] | None = st.session_state.get("analiz_result")
    qm_dashboard: dict[str, Any] | None = None
    try:
        qm_dashboard = DataLoader.compute_fill_quality_metrics(df, cleaned)
    except (PreprocessingError, ValueError, KeyError, TypeError):
        qm_dashboard = None

    _render_data_qa_bar(
        result=result_early,
        analysis_df=analysis_df,
        analysis_base=analysis_base,
        target_col=str(target_variable),
        qm=qm_dashboard,
        n_clusters=int(n_clusters),
        ui=ui,
    )

    if jury_mode and result_early is not None:
        _inject_zen_mode_css(full=True)
        feat_for_jury = list(
            result_early.get("feature_columns") or selected_numeric or numeric_for_model
        )
        _render_jury_presentation_view(
            result_early,
            analysis_df=analysis_df,
            analysis_base=analysis_base,
            feat_cols=feat_for_jury,
            target_col=str(
                result_early.get("target_column") or target_variable or ""
            ),
            n_clusters=int(n_clusters),
            dm_sess=dm_sess,
        )
        st.divider()
        footer_j = ui.get("FOOTER_TEXT")
        if footer_j:
            st.caption(footer_j)
        return

    if jury_mode:
        st.info(
            "🎬 Jüri Sunum Modu açık — sunum görünümü için **Analiz Paneli** sekmesinde "
            "analizi çalıştırın; ardından odaklı slayt düzeni otomatik açılır."
        )

    tab_icons = ui.get("TAB_ICONS") or []
    tab_labels_base = list(ui["TABS"])
    if len(tab_icons) == len(tab_labels_base):
        tab_labels_ui = [f"{ic} {lbl}" for ic, lbl in zip(tab_icons, tab_labels_base)]
    else:
        tab_labels_ui = tab_labels_base
    tab_raw, tab_stats, tab_clean, tab_analiz, tab_exec = st.tabs(tab_labels_ui)

    with tab_raw:
        st.markdown("#### Kalite özeti (radar)")
        st.caption(
            "Yönetsel bakış: doluluk, benzersizlik, anomali azlığı (analiz sonrası güncellenir), "
            "varyans dengesi, sayısal özellik zenginliği ve imputasyon yükü."
        )
        try:
            qm_preview = DataLoader.compute_fill_quality_metrics(df, cleaned)
            res_prev = st.session_state.get("analiz_result")
            axes_prev = _data_quality_radar_axes(
                qm_preview,
                df,
                res_prev,
                len(numeric_for_model),
            )
            fig_prev = DataVisualizer(theme=_streamlit_theme()).plot_data_quality_radar(
                axes_prev,
                title="Veri kalitesi radarı",
            )
            _render_plotly_static(fig_prev, key="wv_plot_quality_radar_raw")
        except (PreprocessingError, ValueError, KeyError, TypeError):
            st.caption("Kalite radarı bu veri için hesaplanamadı.")
        st.markdown("#### Veri Sağlığı Haritası")
        st.caption(
            "Kırmızı hücreler eksik (NaN) değerleri gösterir; temizleme öncesi ham veri "
            "yapısının röntgenidir."
        )
        try:
            imput_lbl = _primary_imputation_method_label(df, cleaned)
            fig_miss = DataVisualizer(theme=_streamlit_theme()).plot_missing_data_matrix(
                df,
                max_rows=120,
                title="Kayıp veri matrisi (NaN ısı haritası)",
            )
            _render_plotly_static(fig_miss, key="wv_plot_missing_matrix")
            _render_ai_comment(
                StatisticalCommentator.missing_data_repair(
                    df,
                    cleaned,
                    method_label=imput_lbl,
                )
            )
        except (ValueError, PreprocessingError, KeyError, TypeError) as exc:
            st.caption(f"Veri sağlığı haritası oluşturulamadı: {exc}")
        st.dataframe(
            df.head(int(prev["RAW_CLEAN_HEAD"])),
            use_container_width=True,
        )

    with tab_stats:
        try:
            stats = loader.get_summary_stats(df)
        except PreprocessingError as exc:
            st.error(str(exc))
        else:
            st.subheader("Çıkarılan sütunlar")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Sayısal**")
                st.write(stats["numeric_columns"] or "(yok)")
            with c2:
                st.write("**Kategorik**")
                st.write(stats["categorical_columns"] or "(yok)")

            st.subheader("Sütun başına eksik değer")
            st.json(stats["missing_per_column"])

            with st.expander("Sayısal describe"):
                st.json(stats["numeric_describe"])

            with st.expander("Kategorik üst değerler"):
                st.json(stats["categorical_top_values"])

    with tab_clean:
        st.markdown("#### Canlı What-If veri düzenleyicisi")
        st.caption(
            "Hücreleri düzenleyin veya satır silin; değişiklik sonrası K-Means, PCA ve "
            "Random Forest modelleri **otomatik** yeniden çalışır."
        )
        editor_df = analysis_base.copy()
        edited_df = st.data_editor(
            editor_df,
            num_rows="dynamic",
            use_container_width=True,
            key="wv_whatif_data_editor",
            hide_index=False,
        )
        new_snap = _dataframe_edit_fingerprint(edited_df)
        old_snap = st.session_state.get(f"{whatif_key}_snap")
        if old_snap != new_snap:
            st.session_state[whatif_key] = edited_df.copy()
            st.session_state[f"{whatif_key}_snap"] = new_snap
            st.session_state.pop("analiz_result", None)
            st.session_state["whatif_auto_run"] = True
            st.rerun()

        if cleaned is not None:
            missing_after = int(analysis_base.isna().sum().sum())
            st.metric("Düzenlenmiş tablo — toplam NaN", missing_after)
            st.caption(f"**{len(analysis_base)}** satır analize hazır.")
        else:
            st.warning(
                clean_error or "Temizleme uygulanamadı; What-If ham veri üzerinde."
            )

    dec = int(fmt["TABLE_FLOAT_DECIMALS"])
    head_n = int(prev["ANALYSIS_TABLE_HEAD"])
    min_num = int(model["MIN_NUMERIC_FEATURES"])

    with tab_analiz:
        st.subheader("Analiz Paneli")
        _dm_tab = _data_mode_id(dm_sess)
        if _dm_tab != "tabular":
            st.info(msg["MSG_VECTORIZED_PIPELINE"])
        if cleaned is None and clean_error is not None:
            st.warning(
                "Temizleme uygulanamadı; analiz **ham veri** üzerinde çalışacak. "
                f"Sebep: {clean_error}"
            )

        st.markdown("#### Veri kalitesi")
        try:
            qm = DataLoader.compute_fill_quality_metrics(df, cleaned)
        except PreprocessingError as exc:
            st.caption(f"Veri kalitesi özeti alınamadı: {exc}")
        else:
            mq1, mq2, mq3 = st.columns(3)
            with mq1:
                st.metric(
                    "Ham doluluk oranı",
                    f"{qm['pct_filled']:.2f}%",
                    help="Ham veri tablosunda dolu hücrelerin oranı.",
                )
            with mq2:
                imputed_val = int(qm["imputed_cells"]) if cleaned is not None else 0
                st.metric(
                    "Doldurulan hücre sayısı",
                    f"{imputed_val:,}",
                    help="Temizleme adımında eksikten doluya çevrilen hücre (yalnızca temizleme uygulandıysa).",
                )
            with mq3:
                st.metric(
                    "Eksiklik oranı",
                    f"{qm['pct_missing']:.2f}%",
                    help="Ham veride boş (NaN) hücrelerin oranı.",
                )

        can_run = len(numeric_for_model) >= min_num and len(selected_numeric) >= min_num
        if use_log_transform and log_skipped_cols:
            st.caption(
                "**Log1p** atlanan sütunlar (negatif minimum vb.): "
                + ", ".join(log_skipped_cols)
            )
        if not can_run:
            st.warning(msg["WARN_MIN_NUMERIC"])
        whatif_auto = bool(st.session_state.pop("whatif_auto_run", False))
        run = st.button(
            "Analizi Çalıştır",
            type="primary",
            disabled=not can_run,
        )

        if (run or whatif_auto) and can_run:
            with st.spinner(ui["SPINNER_ANALYSIS"]):
                pipeline_out = _run_analysis_pipeline(
                    analysis_df,
                    n_clusters=int(n_clusters),
                    contamination=float(contamination),
                    elbow_k_max=int(elbow_k_max),
                    selected_numeric=list(selected_numeric),
                    target_variable=str(target_variable),
                    pca_components_2d=int(model["PCA_COMPONENTS_UI"]),
                    pca_components_3d=int(model["PCA_COMPONENTS_3D"]),
                    data_mode_id=_data_mode_id(dm_sess),
                )
            st.session_state["analiz_result"] = {
                **pipeline_out,
                "use_log_transform": bool(use_log_transform),
                "log_skipped_columns": list(log_skipped_cols),
                "time_column": time_column,
                "ts_value_column": ts_value_column,
                "time_series_mode": bool(
                    time_series_mode and time_column and ts_value_column
                ),
            }
            if whatif_auto:
                st.toast("What-If düzenlemesi uygulandı — modeller yenilendi.", icon="✨")

        result: dict[str, Any] | None = st.session_state.get("analiz_result")
        if result is None:
            st.caption(msg["CAPTION_ANALYSIS_EMPTY"])
            if _data_mode_id(dm_sess) == "text_corpus":
                _b0 = st.session_state.get(sess["TEXT_CORPUS_BLOB"])
                if _b0:
                    st.markdown("### Kelime bulutu (metin derlemi özeti)")
                    st.caption(
                        "Analizi çalıştırmadan önce ham metin önizlemesi; "
                        "model girdisi aşağıdaki sayısal özelliklerdir."
                    )
                    _render_wordcloud_section(str(_b0))
        else:
            viz = DataVisualizer(theme=_streamlit_theme())
            tips_cfg: dict[str, str] = CONFIG.get("PRESENTATION_TIPS", {})
            commentator = StatisticalCommentator()

            qm_kpi = qm_dashboard
            if qm_kpi is None:
                try:
                    qm_kpi = DataLoader.compute_fill_quality_metrics(df, cleaned)
                except (PreprocessingError, ValueError, KeyError, TypeError):
                    qm_kpi = {"pct_filled": 100.0, "pct_missing": 0.0}
            labels_kpi = result.get("labels")
            dom_cluster, dom_delta = _dominant_cluster_info(
                labels_kpi if isinstance(labels_kpi, pd.Series) else None
            )
            top_feat, feat_delta = _top_target_correlation_info(
                result.get("target_importance_df")
                if isinstance(result.get("target_importance_df"), pd.DataFrame)
                else None
            )
            _render_executive_kpi_row(
                n_obs=len(analysis_df),
                n_base=len(analysis_base),
                health_score=_compute_system_health_score(qm_kpi, result),
                dominant_cluster=dom_cluster,
                cluster_delta=dom_delta,
                top_feature=top_feat,
                feature_delta=feat_delta,
            )
            st.divider()

            feat_cols = result.get("feature_columns") or selected_numeric
            target_col = str(
                result.get("target_column") or target_variable or ""
            )
            if feat_cols:
                st.caption(
                    "Modele giren sayısal sütunlar: **"
                    + "**, **".join(feat_cols)
                    + "**"
                )
            if target_col:
                st.caption(f"Hedef değişken (RF odak): **{target_col}**")

            target_imp = result.get("target_importance_df")
            if isinstance(target_imp, pd.DataFrame) and not target_imp.empty:
                st.markdown("### Hedef değişken özellik önemi (Random Forest)")
                st.caption(
                    f"**{target_col}** değişkenini açıklamada en etkili sütunlar "
                    "(normalize önem, toplam = 1)."
                )
                try:
                    fig_tgt = viz.plot_target_feature_importance(
                        target_imp,
                        target_column=target_col,
                    )
                    _render_plotly_static(fig_tgt, key="wv_plot_target_imp")
                    _render_ai_comment(
                        commentator.feature_importance(
                            target_imp, target_name=target_col
                        )
                    )
                    _academic_tip_if(
                        show_academic_tips,
                        tips_cfg.get("rf_importance", ""),
                    )
                except ValueError as exc:
                    st.warning(str(exc))
            elif result.get("err_target_importance"):
                st.caption(
                    f"Hedef önemi atlandı: {result['err_target_importance']}"
                )

            st.markdown("### Akıllı iki değişken analizi")
            st.caption(
                "Sütun tiplerine göre otomatik scatter, kutu veya çubuk grafiği seçilir."
            )
            smart_cols = list(analysis_base.columns.astype(str))
            if len(smart_cols) >= 2:
                sc1, sc2 = st.columns(2)
                with sc1:
                    smart_a = st.selectbox(
                        "Sütun A",
                        options=smart_cols,
                        index=0,
                        key="wv_smart_col_a",
                    )
                with sc2:
                    smart_b_default = 1 if len(smart_cols) > 1 else 0
                    smart_b = st.selectbox(
                        "Sütun B",
                        options=smart_cols,
                        index=smart_b_default,
                        key="wv_smart_col_b",
                    )
                if smart_a != smart_b:
                    try:
                        fig_smart, smart_kind, smart_comment = (
                            viz.plot_smart_bivariate(
                                analysis_base,
                                smart_a,
                                smart_b,
                            )
                        )
                        _render_decision_banner(
                            build_bivariate_decision_banner(
                                analysis_base,
                                smart_a,
                                smart_b,
                                smart_kind,
                            )
                        )
                        _render_plotly_static(
                            fig_smart, key="wv_plot_smart_bivariate"
                        )
                        _render_ai_comment(smart_comment)
                        _academic_tip_if(
                            show_academic_tips,
                            tips_cfg.get("smart_chart", ""),
                        )
                    except ValueError as exc:
                        st.warning(str(exc))
                else:
                    st.caption("İki farklı sütun seçin.")
            else:
                st.info("Akıllı grafik için en az iki sütun gerekir.")

            st.markdown("### Korelasyon (ısı haritası)")
            if len(feat_cols) >= min_num:
                try:
                    fig_hm = viz.plot_correlation_heatmap(
                        analysis_df,
                        numeric_columns=feat_cols,
                        title=(
                            f"Korelasyon matrisi (Pearson) — {len(feat_cols)} sayısal sütun"
                        ),
                    )
                    _render_plotly_static(fig_hm, key="wv_plot_corr")
                    _render_ai_comment(
                        commentator.correlation_heatmap(analysis_df, feat_cols)
                    )
                    _academic_tip_if(
                        show_academic_tips,
                        tips_cfg.get("correlation", ""),
                    )
                except ValueError as exc:
                    st.warning(str(exc))
            else:
                st.info(
                    "Korelasyon haritası için en az iki sayısal sütun seçin."
                )

            st.markdown("### Dirsek yöntemi (inertia eğrisi)")
            if result.get("err_elbow"):
                st.warning(f"Dirsek taraması: {result['err_elbow']}")
            elif result.get("elbow_df") is not None and not result["elbow_df"].empty:
                try:
                    fig_e = viz.plot_elbow_curve(
                        result["elbow_df"],
                        selected_k=int(n_clusters),
                        title=(
                            "Dirsek yöntemi: K vs inertia — "
                            f"seçilen k = {int(n_clusters)} (turuncu çizgi)"
                        ),
                    )
                    _render_plotly_static(fig_e, key="wv_plot_elbow")
                    _render_ai_comment(
                        commentator.elbow(
                            result["elbow_df"],
                            int(n_clusters),
                        )
                    )
                    _academic_tip_if(
                        show_academic_tips,
                        tips_cfg.get("elbow", ""),
                    )
                except ValueError as exc:
                    st.warning(str(exc))

            st.markdown("### Kümeleme (K-Means)")
            if result["err_cluster"]:
                st.error(f"Kümeleme hatası: {result['err_cluster']}")
            elif result["labels"] is not None:
                labels = result["labels"]
                cluster_view = analysis_base.copy()
                cluster_view.insert(0, "cluster", labels.values)
                c_met1, c_met2 = st.columns(2)
                with c_met1:
                    if result.get("inertia") is not None:
                        st.metric(
                            "Inertia (WCSS, ölçeklenmiş özellik uzayında)",
                            f"{result['inertia']:.4f}",
                            help="Küme içi kare uzaklıklarının toplamı; düşük olması "
                            "daha sıkı kümeler demek değildir — k arttıkça genelde düşer.",
                        )
                with c_met2:
                    sil = result.get("silhouette")
                    if sil is not None:
                        st.metric(
                            "Kümeleme başarı notu (Silhouette)",
                            f"{sil:.3f}",
                            help="Ölçeklenmiş özellik uzayında, -1 ile 1 arası; 1 değerine "
                            "yakın kümeler daha iyi ayrışmış demektir. Tek kümede veya k=1 "
                            "iken hesaplanmaz.",
                        )
                    elif result.get("n_clusters", 0) < 2:
                        st.caption("Silhouette: yalnızca **k ≥ 2** için tanımlıdır.")
                    else:
                        st.caption("Silhouette bu veri/küme yapısı için hesaplanamadı.")
                _academic_tip_if(
                    show_academic_tips,
                    tips_cfg.get("clustering", ""),
                )
                _render_ai_comment(
                    commentator.silhouette(
                        result.get("silhouette"),
                        int(result.get("n_clusters", n_clusters)),
                    )
                )
                st.write(
                    f"**{result['n_clusters']}** küme, **{len(labels)}** satır."
                )
                st.dataframe(
                    _style_analysis_dataframe(cluster_view.head(head_n), float_decimals=dec),
                    use_container_width=True,
                )
                counts_df = (
                    pd.DataFrame({"cluster": labels.values})
                    .groupby("cluster", sort=True)
                    .size()
                    .reset_index(name="count")
                )
                st.text("Küme başına satır sayısı:")
                st.dataframe(
                    _style_analysis_dataframe(counts_df, float_decimals=dec),
                    use_container_width=True,
                    hide_index=True,
                )
                if feat_cols:
                    try:
                        cmeans = _cluster_profile_means(
                            analysis_base, labels, feat_cols
                        )
                        st.markdown("#### Küme profilleri (ortalama)")
                        st.caption(
                            "Seçilen sayısal sütunlarda küme başına ortalama "
                            "(mean); kümelerin profilini karşılaştırmak için."
                        )
                        st.dataframe(
                            _style_analysis_dataframe(
                                cmeans, float_decimals=dec
                            ),
                            use_container_width=True,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        st.caption(f"Küme ortalamaları hesaplanamadı: {exc}")

                imp_df = result.get("cluster_importance_df")
                if isinstance(imp_df, pd.DataFrame) and not imp_df.empty:
                    st.markdown("#### Küme ayırma önemi (Random Forest — ek)")
                    st.caption(
                        "K-Means etiketlerini sınıf kabul eden ek keşifsel önem analizi."
                    )
                    try:
                        fig_imp = viz.plot_cluster_feature_importance(imp_df)
                        _render_plotly_static(fig_imp, key="wv_plot_cluster_imp")
                        _render_ai_comment(
                            commentator.feature_importance(
                                imp_df, target_name="K-Means kümeleri"
                            )
                        )
                    except ValueError as exc:
                        st.warning(str(exc))
                elif result.get("err_importance"):
                    st.caption(f"Küme önemi atlandı: {result['err_importance']}")

                box_opts = [c for c in feat_cols if c in analysis_base.columns]
                if box_opts:
                    st.markdown("#### Kümelere göre kutu grafikleri")
                    st.caption(
                        "Seçilen özelliğin her kümedeki medyan ve yayılımı; "
                        "kümeler arası kutu konumu farkı, segmentasyonu destekler."
                    )
                    box_col = st.selectbox(
                        "Kutu grafiği sütunu",
                        options=box_opts,
                        key="wv_cluster_box_col",
                    )
                    try:
                        fig_box = viz.plot_cluster_boxplots(
                            analysis_base,
                            box_col,
                            labels.values,
                            title=f"{box_col} — küme bazında dağılım",
                        )
                        _render_plotly_static(fig_box, key="wv_plot_cluster_box")
                        box_comment_df = analysis_base.copy()
                        box_comment_df["Küme"] = labels.astype(int).astype(str)
                        _render_ai_comment(
                            commentator.boxplot_by_group(
                                box_comment_df,
                                box_col,
                                "Küme",
                                group_label="kümede",
                            )
                        )
                    except ValueError as exc:
                        st.warning(str(exc))

                if result["pca_coords"] is not None:
                    pca_enriched = result["pca_coords"].join(
                        analysis_df,
                        how="left",
                    )
                    try:
                        pv_c = result.get("pca_variance_pct")
                        fig_c = viz.plot_clustering(
                            pca_enriched,
                            labels.values,
                            variance_explained_pct=float(pv_c)
                            if pv_c is not None
                            else None,
                        )
                        _render_plotly_interactive(
                            fig_c,
                            key="wv_plot_cluster",
                            selection_base=pca_enriched,
                            float_decimals=dec,
                            show_selection_hint=True,
                        )
                        if result.get("pca_variance_pct") is not None:
                            pv = float(result["pca_variance_pct"])
                            st.info(
                                "Bu 2 boyutlu görselleştirme, orijinal verideki bilginin "
                                f"**%{pv:.2f}** kadarını temsil etmektedir "
                                "(PCA açıklanan varyans oranı; ölçeklenmiş sayısal sütunlar)."
                            )
                        _render_ai_comment(
                            commentator.silhouette(
                                result.get("silhouette"),
                                int(result.get("n_clusters", n_clusters)),
                            )
                        )
                    except ValueError as exc:
                        st.warning(f"Küme grafiği oluşturulamadı: {exc}")

                pca_3d = result.get("pca_coords_3d")
                if pca_3d is not None and labels is not None:
                    st.markdown("#### 3D PCA Explorer (PC1 · PC2 · PC3)")
                    st.caption(
                        "Fareyle döndürün; küme sınırlarını derinlik algısıyla inceleyin."
                    )
                    if result.get("err_pca_3d"):
                        st.warning(f"3D PCA: {result['err_pca_3d']}")
                    try:
                        pca3_enriched = pca_3d.join(analysis_df, how="left")
                        pv3 = result.get("pca_variance_pct_3d")
                        fig_3d = viz.plot_3d_pca_clusters(
                            pca3_enriched,
                            labels.values,
                            variance_explained_pct=float(pv3)
                            if pv3 is not None
                            else None,
                            chart_height=680,
                        )
                        _render_plotly_static(fig_3d, key="wv_plot_pca_3d")
                        if pv3 is not None:
                            _render_ai_comment(
                                f"PC1+PC2+PC3 ile açıklanan varyans **%{float(pv3):.1f}** "
                                "(ölçeklenmiş uzay); 3B görünüm bilgi kaybını 2B'ye göre "
                                "azaltır."
                            )
                    except ValueError as exc:
                        st.warning(f"3D PCA grafiği oluşturulamadı: {exc}")

                st.markdown("#### Manuel sütun kıyaslaması (K-Means renkli)")
                st.caption(
                    "İki gerçek özelliği seçin; noktalar K-Means küme rengiyle boyanır "
                    "(PCA’daki PC1/PC2’den farklı olarak eksenler doğrudan yorumlanır)."
                )
                if result.get("use_log_transform"):
                    st.caption(
                        "Kümeleme log-dönüşümlü özellikler üzerinde yapıldı; bu grafikte "
                        "eksiler **orijinal (temizlenmiş) ölçek**tedir."
                    )
                num_manual, _ = DataLoader.infer_column_types(analysis_base)
                if len(num_manual) >= 2:
                    m1, m2 = st.columns(2)
                    with m1:
                        mx = st.selectbox(
                            "X ekseni",
                            options=num_manual,
                            index=0,
                            key="wv_manual_scatter_x",
                        )
                    with m2:
                        default_y = 1 if len(num_manual) > 1 else 0
                        my = st.selectbox(
                            "Y ekseni",
                            options=num_manual,
                            index=default_y,
                            key="wv_manual_scatter_y",
                        )
                    if mx != my:
                        try:
                            plot_align = analysis_base.loc[labels.index]
                            fig_m, man_kind, man_comment = (
                                viz.plot_smart_bivariate(
                                    plot_align,
                                    mx,
                                    my,
                                    title=f"{mx} vs {my} (akıllı grafik + K-Means)",
                                    labels=labels.values,
                                )
                            )
                            if man_kind == "scatter":
                                fig_m = viz.plot_manual_cluster_scatter(
                                    plot_align,
                                    mx,
                                    my,
                                    labels.values,
                                    title=f"{mx} vs {my} ilişkisi (K-Means renkleri)",
                                )
                                _render_plotly_interactive(
                                    fig_m,
                                    key="wv_plot_manual_scatter",
                                    selection_base=plot_align,
                                    float_decimals=dec,
                                    show_selection_hint=True,
                                )
                            else:
                                _render_plotly_static(
                                    fig_m, key="wv_plot_manual_smart"
                                )
                            _render_ai_comment(man_comment)
                            _academic_tip_if(
                                show_academic_tips,
                                tips_cfg.get("manual_scatter", ""),
                            )
                        except ValueError as exc:
                            st.warning(f"Manuel grafik oluşturulamadı: {exc}")
                    else:
                        st.caption("X ve Y için farklı sütun seçin.")
                else:
                    st.info(
                        "Manuel kıyaslama için en az iki sayısal sütun gerekir."
                    )
            else:
                st.warning("Küme etiketleri yok.")

            st.markdown(
                f'### PCA ({model["PCA_COMPONENTS_UI"]} bileşen)'
            )
            if result["err_pca"]:
                st.error(f"PCA hatası: {result['err_pca']}")
            elif result["pca_coords"] is not None:
                st.write(
                    f"İlk {head_n} satır — "
                    f'PC1…PC{model["PCA_COMPONENTS_UI"]}:'
                )
                st.dataframe(
                    _style_analysis_dataframe(
                        result["pca_coords"].head(head_n),
                        float_decimals=dec,
                    ),
                    use_container_width=True,
                )
                if result.get("pca_variance_pct") is not None:
                    pv = float(result["pca_variance_pct"])
                    st.metric(
                        "PC1 + PC2 ile açıklanan varyans",
                        f"{pv:.2f}%",
                        help="Ölçeklenmiş sayısal özellikler üzerinde PCA "
                        "`explained_variance_ratio_` toplamı.",
                    )
                    st.caption(
                        "Bu 2 boyutlu görselleştirme, orijinal verideki bilginin "
                        f"%{pv:.2f} kadarını temsil etmektedir."
                    )
                _academic_tip_if(
                    show_academic_tips,
                    tips_cfg.get("pca", ""),
                )
            else:
                st.warning("PCA koordinatları yok.")

            _render_model_interpreter(result)
            _academic_tip_if(
                show_academic_tips,
                tips_cfg.get("interpreter", ""),
            )

            st.markdown("### Anomali tespiti (Isolation Forest)")
            if result["err_anomaly"]:
                st.error(f"Anomali hatası: {result['err_anomaly']}")
            elif result["pred"] is not None:
                pred = result["pred"]
                anomaly_view = analysis_base.copy()
                anomaly_view.insert(0, "anomaly_score_label", pred.values)
                n_out = int((pred.values == -1).sum())
                n_in = int((pred.values == 1).sum())
                st.write(
                    f"Tahmin özeti: **{n_out}** anomali (-1), **{n_in}** normal (1)."
                )
                if (
                    result.get("time_series_mode")
                    and result.get("time_column")
                    and result.get("ts_value_column")
                ):
                    st.markdown("#### Zaman serisi anomali motoru")
                    st.caption(
                        "Sistem logları / sensör verisi: çizgi üzerinde kırmızı sıçrama "
                        "noktaları Isolation Forest aykırılarını işaretler."
                    )
                    tcol = str(result["time_column"])
                    vcol = str(result["ts_value_column"])
                    try:
                        fig_ts = viz.plot_timeseries_anomalies(
                            analysis_base,
                            tcol,
                            vcol,
                            pred.values,
                            title=f"Zaman serisi — {vcol} (anomali sıçramaları)",
                        )
                        _render_plotly_static(fig_ts, key="wv_plot_timeseries_anom")
                        _render_ai_comment(
                            f"**{tcol}** ekseninde **{vcol}** izlendi; **{n_out}** anomali "
                            f"noktası kırmızı sıçrama olarak işaretlendi (~%"
                            f"{100.0 * n_out / max(1, len(pred)):.1f} oran)."
                        )
                    except ValueError as exc:
                        st.warning(f"Zaman serisi grafiği: {exc}")
                st.dataframe(
                    _style_analysis_dataframe(
                        anomaly_view.head(head_n),
                        float_decimals=dec,
                    ),
                    use_container_width=True,
                )
                if n_out > 0:
                    st.markdown("#### Şüpheli kayıtlar ve ‘neden şüpheli?’ özeti")
                    suspicious = analysis_base.loc[pred == -1]
                    cmp_tbl, cmp_summary = _anomaly_vs_normal_mean_table(
                        analysis_base,
                        pred,
                        list(feat_cols),
                    )
                    col_susp, col_why = st.columns(2)
                    with col_susp:
                        st.caption(
                            "Şüpheli kayıtlar listesi — Isolation Forest **-1**; "
                            "orijinal (log öncesi) ölçek."
                        )
                        st.dataframe(
                            _style_analysis_dataframe(
                                suspicious, float_decimals=dec
                            ),
                            use_container_width=True,
                        )
                    with col_why:
                        st.caption(
                            "Normal (1) vs anomali (-1) grup ortalamaları — keşifsel karşılaştırma."
                        )
                        if not cmp_tbl.empty:
                            st.dataframe(
                                _style_analysis_dataframe(
                                    cmp_tbl, float_decimals=dec
                                ),
                                use_container_width=True,
                            )
                        st.info(cmp_summary)
                        _academic_tip_if(
                            show_academic_tips,
                            tips_cfg.get("anomaly_character", ""),
                        )
                else:
                    st.caption("Bu çalıştırmada **-1** etiketli satır yok.")
                if result["pca_coords"] is not None:
                    try:
                        st.caption(CONFIG["PLOTLY"]["CAPTION_SELECTION"])
                        pv_a = result.get("pca_variance_pct")
                        fig_a = viz.plot_anomalies(
                            result["pca_coords"],
                            pred.values,
                            variance_explained_pct=float(pv_a)
                            if pv_a is not None
                            else None,
                        )
                        _render_plotly_interactive(
                            fig_a,
                            key="wv_plot_anomaly",
                            selection_base=analysis_base,
                            float_decimals=dec,
                            show_selection_hint=False,
                        )
                        _render_ai_comment(
                            f"Isolation Forest **{n_out}** aykırı (toplam "
                            f"{len(pred)} gözlemin ~%{100.0 * n_out / max(1, len(pred)):.1f}'i) "
                            "işaretlemiştir; kırmızı noktalar düşük yoğunluk adaylarıdır."
                        )
                        _academic_tip_if(
                            show_academic_tips,
                            tips_cfg.get("anomaly", ""),
                        )
                    except ValueError as exc:
                        st.warning(f"Anomali grafiği oluşturulamadı: {exc}")
            else:
                st.warning("Anomali tahminleri yok.")

            st.markdown("### Özellik dağılımı")
            cols_list = list(analysis_base.columns)
            col_pick = st.selectbox(
                "Grafik sütunu",
                options=cols_list,
                index=0,
                help="Sayısal: histogram. Kategorik/metin: frekans çubuğu.",
                key=sess["FEATURE_DIST_COLUMN"],
            )
            try:
                fig_d = viz.plot_feature_distribution(
                    analysis_base,
                    col_pick,
                    title=f"Özellik dağılımı: {col_pick}",
                )
                _render_plotly_static(fig_d, key="wv_plot_feature")
                _render_ai_comment(
                    commentator.distribution(analysis_base, col_pick)
                )
                _academic_tip_if(
                    show_academic_tips,
                    tips_cfg.get("feature_dist", ""),
                )
            except ValueError as exc:
                st.warning(str(exc))

            if _data_mode_id(dm_sess) == "text_corpus":
                _wc_blob = st.session_state.get(sess["TEXT_CORPUS_BLOB"])
                if _wc_blob:
                    st.markdown("### Kelime bulutu (metin derlemi özeti)")
                    st.caption(
                        "Ham metinden türetilen önizleme; kümeleme ve PCA sayısal "
                        "özellik sütunları üzerindedir."
                    )
                    _render_wordcloud_section(str(_wc_blob))

            st.divider()
            report_md = _build_academic_report_markdown(
                result=result,
                analysis_df=analysis_df,
                feat_cols=list(feat_cols),
                n_clusters=int(n_clusters),
                contamination=float(contamination),
                app_version=str(CONFIG["VERSION"]),
                float_decimals=dec,
                cluster_profile_df=analysis_base,
                use_log_transform=bool(result.get("use_log_transform")),
                log_skipped_columns=list(result.get("log_skipped_columns") or []),
                data_mode_id=_data_mode_id(dm_sess),
            )
            with st.expander(
                ui.get(
                    "REPORT_PREVIEW_EXPANDER",
                    "📜 İndirilecek Rapor İçeriğini Önizle",
                ),
                expanded=False,
            ):
                st.caption(
                    "Aşağıdaki metin indirilecek `.txt` dosyasıyla aynıdır (Markdown)."
                )
                st.markdown("**Biçimli önizleme**")
                st.markdown(report_md)
                st.markdown("**Ham Markdown kaynağı** (`language=\"markdown\"`)")
                st.code(report_md, language="markdown")
            code_py = _build_reproducible_analysis_script(
                feature_columns=list(feat_cols),
                target_column=str(target_col),
                n_clusters=int(n_clusters),
                contamination=float(contamination),
                elbow_k_max=int(elbow_k_max),
                use_log_transform=bool(result.get("use_log_transform")),
                app_version=str(CONFIG["VERSION"]),
                time_column=result.get("time_column"),
                ts_value_column=result.get("ts_value_column"),
            )
            try:
                excel_bytes = _build_multi_sheet_excel_bytes(analysis_base, result)
            except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
                excel_bytes = None
                st.caption(f"Excel dışa aktarımı hazırlanamadı: {exc}")

            strategy_md = _build_ai_strategy_advice_md(
                result,
                len(analysis_df),
                data_mode_id=_data_mode_id(dm_sess),
            )
            try:
                pdf_bytes = _build_executive_summary_pdf_bytes(
                    result,
                    app_version=str(CONFIG["VERSION"]),
                    n_obs=len(analysis_df),
                    n_clusters=int(n_clusters),
                    strategy_md=strategy_md,
                    feat_cols=list(feat_cols),
                )
            except (ImportError, OSError, ValueError, TypeError) as exc:
                pdf_bytes = None
                st.caption(f"PDF özeti hazırlanamadı: {exc}")

            models_zip = result.get("models_zip_bytes")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    label=ui["BTN_DOWNLOAD_REPORT"],
                    data=report_md.encode("utf-8"),
                    file_name=(
                        f'{ui["REPORT_FILENAME_PREFIX"]}_v{CONFIG["VERSION"]}.txt'
                    ),
                    mime="text/plain",
                    help=(
                        "Silhouette, PCA, sütunlar, küme ortalamaları ve model "
                        "yorumları (Markdown, .txt)."
                    ),
                    use_container_width=True,
                    key="wv_download_academic_report",
                )
            with dl2:
                st.download_button(
                    label=ui["BTN_DOWNLOAD_CODE"],
                    data=code_py.encode("utf-8"),
                    file_name=(
                        f'{ui["CODE_FILENAME_PREFIX"]}_v{CONFIG["VERSION"]}.py'
                    ),
                    mime="text/x-python",
                    help=(
                        "Panel ayarlarınızla aynı sklearn/pandas analizini yerelde "
                        "tekrarlayan şeffaf Python betiği."
                    ),
                    use_container_width=True,
                    key="wv_download_analysis_code",
                )
            dl3, dl4 = st.columns(2)
            with dl3:
                if excel_bytes:
                    st.download_button(
                        label=ui["BTN_DOWNLOAD_EXCEL"],
                        data=excel_bytes,
                        file_name=(
                            f'{ui["EXCEL_FILENAME_PREFIX"]}_v{CONFIG["VERSION"]}.xlsx'
                        ),
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        help=(
                            "3 sekme: Cleaned Data (Cluster_ID), Anomalies, Summary."
                        ),
                        use_container_width=True,
                        key="wv_download_excel_workbook",
                    )
                else:
                    st.caption("Excel indirme bu oturumda kullanılamıyor.")
            with dl4:
                if isinstance(models_zip, bytes) and len(models_zip) > 0:
                    st.download_button(
                        label=ui["BTN_DOWNLOAD_MODELS"],
                        data=models_zip,
                        file_name=(
                            f'{ui["MODELS_ZIP_PREFIX"]}_v{CONFIG["VERSION"]}.zip'
                        ),
                        mime="application/zip",
                        help=(
                            "K-Means, Isolation Forest, Random Forest + scaler "
                            "dosyaları (joblib .pkl)."
                        ),
                        use_container_width=True,
                        key="wv_download_models_zip",
                    )
                elif result.get("err_models_zip"):
                    st.caption(f"Model ZIP: {result['err_models_zip']}")
                else:
                    st.caption("Model ZIP henüz üretilmedi; analizi yeniden çalıştırın.")
            if isinstance(pdf_bytes, bytes) and len(pdf_bytes) > 0:
                st.download_button(
                    label=ui["BTN_DOWNLOAD_PDF"],
                    data=pdf_bytes,
                    file_name=(
                        f'{ui["PDF_FILENAME_PREFIX"]}_v{CONFIG["VERSION"]}.pdf'
                    ),
                    mime="application/pdf",
                    help=(
                        "Tek sayfalık yönetici özeti: küme dağılımı, metrikler ve "
                        "AI stratejik tavsiyeler."
                    ),
                    use_container_width=True,
                    key="wv_download_executive_pdf",
                )
            st.caption(
                "Kurumsal dışa aktarım: Excel (3 sekme), `.py` betiği, model `.zip` ve "
                "yönetici PDF — multimodal modlarda güvenli dışa aktarım."
            )
            st.markdown(
                CONFIG.get("UI", {}).get(
                    "REPORT_STRATEGY_SECTION_MD",
                    "## 🤖 AI Stratejik Tavsiyeler",
                )
            )
            st.markdown(strategy_md)

    with tab_exec:
        st.markdown("# Yönetici özeti")
        st.caption(
            "Sunum modu: Silhouette, PCA varyansı, anomali sayısı, kümeleme ve "
            "ayırt edici faktörler tek ekranda."
        )
        rex = st.session_state.get("analiz_result")
        ex_h = int(CONFIG.get("EXEC_SUMMARY_CHART_HEIGHT", 620))
        if rex is None:
            st.info(msg["CAPTION_ANALYSIS_EMPTY"])
        else:
            m1, m2, m3 = st.columns(3)
            sil_ex = rex.get("silhouette")
            with m1:
                st.metric(
                    "Silhouette (küme ayrışması)",
                    f"{float(sil_ex):.3f}" if sil_ex is not None else "—",
                    help="−1…1; yüksek değer daha ayrık kümeleri destekler.",
                )
            pv_ex = rex.get("pca_variance_pct")
            with m2:
                st.metric(
                    "PCA PC1+PC2 açıklanan varyans",
                    f"{float(pv_ex):.1f}%"
                    if pv_ex is not None
                    else "—",
                    help="Ölçeklenmiş özellikler üzerinde ilk iki bileşen.",
                )
            pred_ex = rex.get("pred")
            n_anom = (
                int((pred_ex == -1).sum())
                if pred_ex is not None
                else 0
            )
            with m3:
                st.metric(
                    "Tespit edilen anomali",
                    str(n_anom),
                    help="Isolation Forest −1 etiketli satır sayısı.",
                )

            st.markdown("### Kümeleme (PCA düzlemi)")
            viz_ex = DataVisualizer(theme=_streamlit_theme())
            if rex.get("err_cluster") or rex.get("labels") is None:
                st.warning(
                    "Kümeleme grafiği için geçerli sonuç yok; Analiz sekmesinde "
                    "hatayı kontrol edin veya analizi yeniden çalıştırın."
                )
            elif rex.get("pca_coords") is None:
                st.warning("PCA koordinatları yok; özet grafiği gösterilemiyor.")
            else:
                try:
                    pca_ex = rex["pca_coords"].join(analysis_df, how="left")
                    pv_c = rex.get("pca_variance_pct")
                    fig_ec = viz_ex.plot_clustering(
                        pca_ex,
                        rex["labels"].values,
                        variance_explained_pct=float(pv_c)
                        if pv_c is not None
                        else None,
                        title="K-Means (PCA) — yönetici özeti",
                    )
                    fig_ec.update_layout(height=ex_h)
                    _render_plotly_static(fig_ec, key="wv_exec_clustering")
                except ValueError as exc:
                    st.warning(str(exc))

            st.markdown("### Hedef değişkeni en çok etkileyen faktörler")
            tgt_ex = rex.get("target_importance_df")
            tgt_name_ex = str(rex.get("target_column") or target_variable or "")
            if isinstance(tgt_ex, pd.DataFrame) and not tgt_ex.empty:
                try:
                    fig_ei = viz_ex.plot_target_feature_importance(
                        tgt_ex,
                        target_column=tgt_name_ex,
                        title=(
                            f"Hedef: {tgt_name_ex} — Random Forest özellik önemi "
                            "(yönetici özeti)"
                        ),
                        chart_height=ex_h,
                    )
                    _render_plotly_static(fig_ei, key="wv_exec_importance")
                    _render_ai_comment(
                        StatisticalCommentator.feature_importance(
                            tgt_ex, target_name=tgt_name_ex
                        )
                    )
                except ValueError as exc:
                    st.warning(str(exc))
            elif rex.get("err_target_importance"):
                st.info(f"Hedef önemi: {rex['err_target_importance']}")
            else:
                st.info(
                    "Hedef değişken özellik önemi için Analiz sekmesinde analizi "
                    "çalıştırın ve hedef sütun seçin."
                )

    with st.expander("Geliştirici notu"):
        st.code(
            "DataLoadError, PreprocessingError, AIModelError — UI’da kullanıcıya "
            "sade mesaj; ayrıntı için log ekleyebilirsiniz."
        )

    st.divider()
    footer = ui.get("FOOTER_TEXT")
    if footer:
        st.caption(footer)


if __name__ == "__main__":
    main()
