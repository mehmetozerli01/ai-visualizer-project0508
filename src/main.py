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

from typing import Any, Literal

import numpy as np
import pandas as pd
import streamlit as st

from ai_engine import AIEngine
from exceptions import AIModelError, DataLoadError, PreprocessingError
from processor import DataLoader
from visualizer import DataVisualizer

# --- Merkezi ayarlar (bakım / tema / varsayılanlar) -----------------------------
CONFIG: dict[str, Any] = {
    "VERSION": "0.5.0",
    "UI": {
        "PAGE_TITLE": "AI Visualizer",
        "PAGE_LAYOUT": "wide",
        "APP_TITLE": "AI Visualizer — geliştirme paneli",
        "APP_CAPTION": (
            "CSV / Excel yükleme, özet, temizleme, AI analizleri ve Plotly grafikleri."
        ),
        "FILE_UPLOADER_LABEL": "CSV / Excel dosyası",
        "FILE_EXTENSIONS": ["csv", "xlsx", "xlsm"],
        "FILE_UPLOADER_HELP": "CSV veya Excel (.xlsx). Excel için openpyxl gerekir.",
        "INFO_NO_FILE": "Başlamak için sol menüden bir veri dosyası seçin veya örnek veriyi yükleyin.",
        "BTN_DEMO_DATA": "Örnek veriyi yükle",
        "BTN_DEMO_HELP": (
            "Iris / Wine (sklearn) veya sentetik anomali demosu; dosya seçmeden yüklenir."
        ),
        "DEMO_LABEL": "Örnek veri seti",
        "SUCCESS_DEMO": "Örnek veri yüklendi: **{dataset}**.",
        "BTN_DOWNLOAD_REPORT": "Akademik analiz raporunu indir",
        "REPORT_FILENAME_PREFIX": "akademik_analiz_raporu",
        "APP_TITLE_ICON": "🎯",
        "TAB_ICONS": ["📄", "📈", "🧹", "📊", "📋"],
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
        "REPORT_PREVIEW_EXPANDER": "📄 İndirilecek Rapor İçeriğini Önizle",
        "FOOTER_TEXT": (
            "v0.5.0 Final Release | Mehmet Özerli - Bitirme Projesi"
        ),
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
            "Bu grafik, gözetimsiz (unsupervised) kümelerin hangi gözetimli (supervised) "
            "özelliklerle en iyi açıklandığını gösterir."
        ),
        "anomaly_character": (
            "Bu tablo, aykırı değerlerin normal veriden hangi istatistiksel sapmalarla "
            "ayrıldığını kanıtlar."
        ),
    },
    "SESSION_KEYS": {
        "FILE_UPLOADER": "wv_dataset_upload",
        "FEATURE_DIST_COLUMN": "feature_dist_column",
        "DEMO_SAMPLE": "wv_demo_sample",
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
        "Bu platform; Finansal Tahminleme, Müşteri Segmentasyonu ve Endüstriyel "
        "Anomali Tespiti gibi çok modlu veri setlerinde derinlemesine analiz "
        "yapabilmek için tasarlanmıştır."
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
    "anomaly_synthetic": "Anomali Test Seti (Sentetik)",
}


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
    )
    for k in keys_to_drop:
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
    parts.extend(
        [
            "",
            "---",
            "*Bu rapor, analiz sekmesindeki o anki oturum çıktılarından üretilmiştir.*",
        ]
    )
    return "\n".join(parts)


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
    icon = ui.get("APP_TITLE_ICON", "")
    st.title(f"{icon} {ui['APP_TITLE']}".strip() if icon else ui["APP_TITLE"])
    st.caption(ui["APP_CAPTION"])

    demo_key = sess["DEMO_SAMPLE"]
    loader = DataLoader()
    uploaded = None

    with st.sidebar:
        st.markdown("**Hızlı başlangıç**")
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
            st.session_state.pop(sess["FILE_UPLOADER"], None)
            st.rerun()

        uploaded = st.file_uploader(
            ui["FILE_UPLOADER_LABEL"],
            type=list(ui["FILE_EXTENSIONS"]),
            help=ui["FILE_UPLOADER_HELP"],
            key=sess["FILE_UPLOADER"],
        )

    if uploaded is not None:
        st.session_state.pop(demo_key, None)
        try:
            df = loader.load_file(uploaded)
        except DataLoadError as exc:
            st.error(str(exc))
            return
        upload_fingerprint = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
        st.success(
            f"Yüklendi: **{len(df)}** satır, **{len(df.columns)}** sütun."
        )
    elif st.session_state.get(demo_key):
        try:
            dkey = str(st.session_state[demo_key])
            df = _load_demo_dataset(dkey)
        except (ImportError, OSError, ValueError) as exc:
            st.error(f"Örnek veri yüklenemedi: {exc}")
            return
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

    numeric_for_model, _ = DataLoader.infer_column_types(analysis_base)
    n_rows_model = len(analysis_base)
    max_k = max(1, n_rows_model - 1) if n_rows_model > 1 else 1
    default_k = min(int(model["DEFAULT_K_MEANS_CAP"]), max_k)
    elbow_cap = max(2, min(int(model["ELBOW_K_MAX_UPPER"]), max(1, n_rows_model - 1)))

    with st.sidebar:
        with st.expander("📂 Veri Kaynağı & Filtreleme", expanded=True):
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

        with st.expander("⚙️ Model Parametreleri", expanded=True):
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
                f'PCA: **{model["PCA_COMPONENTS_UI"]} boyut** (PC1, PC2) — sabit.'
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

        with st.expander("👨‍💻 Geliştirici Hakkında", expanded=False):
            st.markdown(f"**{dev['NAME']}**")
            st.markdown(dev["SCHOOL"])
            st.markdown(dev["TECH_STACK"])

        show_academic_tips = st.checkbox(
            ui["CHECKBOX_ACADEMIC_TIPS"],
            value=False,
            key="wv_academic_presentation_tips",
            help=ui.get("CHECKBOX_ACADEMIC_TIPS_HELP", ""),
        )

        st.divider()
        st.caption(CONFIG["VISION_NOTE"])
        if st.button(
            "Analizi sıfırla",
            help="Yüklenen dosyayı ve tüm analiz oturumunu temizler; başa dönersiniz.",
            use_container_width=True,
        ):
            _reset_app_session()

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

    feature_selection_key = (
        upload_fingerprint,
        tuple(sorted(selected_numeric)),
        int(elbow_k_max),
        int(n_clusters),
        round(float(contamination), 4),
        bool(use_log_transform),
    )
    if st.session_state.get("_feature_selection_key") != feature_selection_key:
        st.session_state.pop("analiz_result", None)
    st.session_state["_feature_selection_key"] = feature_selection_key

    tab_icons = ui.get("TAB_ICONS") or []
    tab_labels_base = list(ui["TABS"])
    if len(tab_icons) == len(tab_labels_base):
        tab_labels_ui = [f"{ic} {lbl}" for ic, lbl in zip(tab_icons, tab_labels_base)]
    else:
        tab_labels_ui = tab_labels_base
    tab_raw, tab_stats, tab_clean, tab_analiz, tab_exec = st.tabs(tab_labels_ui)

    with tab_raw:
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
        if cleaned is not None:
            st.dataframe(
                cleaned.head(int(prev["RAW_CLEAN_HEAD"])),
                use_container_width=True,
            )
            missing_after = int(cleaned.isna().sum().sum())
            st.metric("Temizlik sonrası toplam NaN", missing_after)
        else:
            st.error(clean_error or "Temizleme başarısız.")

    dec = int(fmt["TABLE_FLOAT_DECIMALS"])
    head_n = int(prev["ANALYSIS_TABLE_HEAD"])
    min_num = int(model["MIN_NUMERIC_FEATURES"])

    with tab_analiz:
        st.subheader("Analiz Paneli")
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
        run = st.button(
            "Analizi Çalıştır",
            type="primary",
            disabled=not can_run,
        )

        if run and can_run:
            labels: pd.Series | None = None
            pca_coords: pd.DataFrame | None = None
            pred: pd.Series | None = None
            err_cluster: str | None = None
            err_pca: str | None = None
            err_anomaly: str | None = None
            inertia_val: float | None = None
            silhouette_val: float | None = None
            pca_variance_pct: float | None = None
            elbow_df: pd.DataFrame | None = None
            err_elbow: str | None = None
            cluster_imp_df: pd.DataFrame | None = None
            err_importance: str | None = None

            with st.spinner(ui["SPINNER_ANALYSIS"]):
                engine = AIEngine()
                cols_arg = selected_numeric
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
                        n_components=int(model["PCA_COMPONENTS_UI"]),
                        numeric_columns=cols_arg,
                    )
                    pca_variance_pct = float(pca_var_info["variance_explained_pct"])
                except AIModelError as exc:
                    err_pca = str(exc)
                    pca_variance_pct = None

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

            st.session_state["analiz_result"] = {
                "labels": labels,
                "pca_coords": pca_coords,
                "pred": pred,
                "err_cluster": err_cluster,
                "err_pca": err_pca,
                "err_anomaly": err_anomaly,
                "n_clusters": n_clusters,
                "inertia": inertia_val,
                "silhouette": silhouette_val,
                "pca_variance_pct": pca_variance_pct,
                "elbow_df": elbow_df,
                "err_elbow": err_elbow,
                "feature_columns": list(selected_numeric),
                "use_log_transform": bool(use_log_transform),
                "log_skipped_columns": list(log_skipped_cols),
                "cluster_importance_df": cluster_imp_df,
                "err_importance": err_importance,
            }

        result: dict[str, Any] | None = st.session_state.get("analiz_result")
        if result is None:
            st.caption(msg["CAPTION_ANALYSIS_EMPTY"])
        else:
            viz = DataVisualizer(theme=_streamlit_theme())
            tips_cfg: dict[str, str] = CONFIG.get("PRESENTATION_TIPS", {})

            feat_cols = result.get("feature_columns") or selected_numeric
            if feat_cols:
                st.caption(
                    "Modele giren sayısal sütunlar: **"
                    + "**, **".join(feat_cols)
                    + "**"
                )

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
                    st.markdown("#### Küme önem derecesi (Random Forest)")
                    st.caption(
                        "K-Means etiketleri hedef alınarak ölçeklenmiş özelliklerde "
                        "Random Forest ile yaklaşık ayırma önemleri (normalize, toplam 1)."
                    )
                    try:
                        fig_imp = viz.plot_cluster_feature_importance(imp_df)
                        _render_plotly_static(fig_imp, key="wv_plot_cluster_imp")
                        _academic_tip_if(
                            show_academic_tips,
                            tips_cfg.get("rf_importance", ""),
                        )
                    except ValueError as exc:
                        st.warning(str(exc))
                elif result.get("err_importance"):
                    st.caption(f"Önem analizi atlandı: {result['err_importance']}")

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
                    except ValueError as exc:
                        st.warning(f"Küme grafiği oluşturulamadı: {exc}")

                st.markdown("#### Manuel sütun kıyaslaması")
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
                            _academic_tip_if(
                                show_academic_tips,
                                tips_cfg.get("manual_scatter", ""),
                            )
                        except ValueError as exc:
                            st.warning(f"Manuel saçılım oluşturulamadı: {exc}")
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
                _academic_tip_if(
                    show_academic_tips,
                    tips_cfg.get("feature_dist", ""),
                )
            except ValueError as exc:
                st.warning(str(exc))

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
            )
            with st.expander(
                ui.get(
                    "REPORT_PREVIEW_EXPANDER",
                    "📄 İndirilecek Rapor İçeriğini Önizle",
                ),
                expanded=False,
            ):
                st.caption("Aşağıdaki metin indirilecek `.txt` dosyasıyla aynıdır (Markdown).")
                st.markdown(report_md)
            st.download_button(
                label=ui["BTN_DOWNLOAD_REPORT"],
                data=report_md.encode("utf-8"),
                file_name=f'{ui["REPORT_FILENAME_PREFIX"]}_v{CONFIG["VERSION"]}.txt',
                mime="text/plain",
                help="Silhouette, PCA, sütunlar, küme ortalamaları ve model yorumları (Markdown, .txt).",
                use_container_width=True,
                key="wv_download_academic_report",
            )

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

            st.markdown("### Kümelenmeyi en çok etkileyen faktörler")
            imp_ex = rex.get("cluster_importance_df")
            if isinstance(imp_ex, pd.DataFrame) and not imp_ex.empty:
                try:
                    fig_ei = viz_ex.plot_cluster_feature_importance(
                        imp_ex,
                        title="Random Forest özellik önemi — yönetici özeti",
                        chart_height=ex_h,
                    )
                    _render_plotly_static(fig_ei, key="wv_exec_importance")
                except ValueError as exc:
                    st.warning(str(exc))
            elif rex.get("err_importance"):
                st.info(f"Önem analizi: {rex['err_importance']}")
            else:
                st.info(
                    "Özellik önemi için en az iki küme ve başarılı kümeleme gerekir "
                    "(Analiz sekmesinde RF bölümüne bakın)."
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
