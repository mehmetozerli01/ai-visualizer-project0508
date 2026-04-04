"""
Streamlit entrypoint for data loading, cleaning, ML analysis, and Plotly charts.

Run from the project root::

    streamlit run src/main.py
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import streamlit as st

from ai_engine import AIEngine
from exceptions import AIModelError, DataLoadError, PreprocessingError
from processor import DataLoader
from visualizer import DataVisualizer

# --- Merkezi ayarlar (bakım / tema / varsayılanlar) -----------------------------
CONFIG: dict[str, Any] = {
    "VERSION": "0.2.0",
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
        "INFO_NO_FILE": "Başlamak için sol menüden bir veri dosyası seçin.",
        "TABS": [
            "Raw preview",
            "Summary stats",
            "Cleaned preview",
            "Analiz Paneli",
        ],
        "SPINNER_ANALYSIS": "AI Analiz Yapıyor...",
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
    "SESSION_KEYS": {
        "FILE_UPLOADER": "wv_dataset_upload",
        "FEATURE_DIST_COLUMN": "feature_dist_column",
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
    )
    for k in keys_to_drop:
        st.session_state.pop(k, None)
    st.rerun()


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
    st.title(ui["APP_TITLE"])
    st.caption(ui["APP_CAPTION"])

    with st.sidebar:
        uploaded = st.file_uploader(
            ui["FILE_UPLOADER_LABEL"],
            type=list(ui["FILE_EXTENSIONS"]),
            help=ui["FILE_UPLOADER_HELP"],
            key=sess["FILE_UPLOADER"],
        )

    if uploaded is None:
        st.info(ui["INFO_NO_FILE"])
        return

    loader = DataLoader()

    try:
        df = loader.load_file(uploaded)
    except DataLoadError as exc:
        st.error(str(exc))
        return

    upload_fingerprint = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    if st.session_state.get("_upload_fingerprint") != upload_fingerprint:
        st.session_state["_upload_fingerprint"] = upload_fingerprint
        st.session_state.pop("analiz_result", None)
        st.session_state.pop("_feature_selection_key", None)

    st.success(f"Yüklendi: **{len(df)}** satır, **{len(df.columns)}** sütun.")

    cleaned: pd.DataFrame | None = None
    clean_error: str | None = None
    try:
        cleaned = loader.clean_data(df)
    except PreprocessingError as exc:
        clean_error = str(exc)

    analysis_df = cleaned if cleaned is not None else df

    numeric_for_model, _ = DataLoader.infer_column_types(analysis_df)
    n_rows_model = len(analysis_df)
    max_k = max(1, n_rows_model)
    default_k = min(int(model["DEFAULT_K_MEANS_CAP"]), max_k)
    elbow_cap = max(2, min(int(model["ELBOW_K_MAX_UPPER"]), n_rows_model))

    with st.sidebar:
        with st.expander("📂 Veri Kaynağı & Filtreleme", expanded=True):
            st.caption(f"Aktif: **{uploaded.name}** · {len(df)} satır")
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
                help="Satır sayısını aşamaz. Dirsek eğrisinde turuncu çizgi bu k ile gösterilir.",
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

        with st.expander("👨‍💻 Geliştirici Hakkında", expanded=False):
            st.markdown(f"**{dev['NAME']}**")
            st.markdown(dev["SCHOOL"])
            st.markdown(dev["TECH_STACK"])

        st.divider()
        st.caption(CONFIG["VISION_NOTE"])
        if st.button(
            "Analizi sıfırla",
            help="Yüklenen dosyayı ve tüm analiz oturumunu temizler; başa dönersiniz.",
            use_container_width=True,
        ):
            _reset_app_session()

    feature_selection_key = (
        upload_fingerprint,
        tuple(sorted(selected_numeric)),
        int(elbow_k_max),
        int(n_clusters),
        round(float(contamination), 4),
    )
    if st.session_state.get("_feature_selection_key") != feature_selection_key:
        st.session_state.pop("analiz_result", None)
    st.session_state["_feature_selection_key"] = feature_selection_key

    tab_raw, tab_stats, tab_clean, tab_analiz = st.tabs(list(ui["TABS"]))

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
            }

        result: dict[str, Any] | None = st.session_state.get("analiz_result")
        if result is None:
            st.caption(msg["CAPTION_ANALYSIS_EMPTY"])
        else:
            viz = DataVisualizer(theme=_streamlit_theme())

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
                        analysis_df, numeric_columns=feat_cols
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
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
                    )
                    st.plotly_chart(fig_e, use_container_width=True)
                except ValueError as exc:
                    st.warning(str(exc))

            st.markdown("### Kümeleme (K-Means)")
            if result["err_cluster"]:
                st.error(f"Kümeleme hatası: {result['err_cluster']}")
            elif result["labels"] is not None:
                labels = result["labels"]
                cluster_view = analysis_df.copy()
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
                if result["pca_coords"] is not None:
                    pca_enriched = result["pca_coords"].join(
                        analysis_df,
                        how="left",
                    )
                    try:
                        fig_c = viz.plot_clustering(pca_enriched, labels.values)
                        st.plotly_chart(fig_c, use_container_width=True)
                        if result.get("pca_variance_pct") is not None:
                            pv = float(result["pca_variance_pct"])
                            st.info(
                                "Bu 2 boyutlu görselleştirme, orijinal verideki bilginin "
                                f"**%{pv:.2f}** kadarını temsil etmektedir "
                                "(PCA açıklanan varyans oranı; ölçeklenmiş sayısal sütunlar)."
                            )
                    except ValueError as exc:
                        st.warning(f"Küme grafiği oluşturulamadı: {exc}")
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
            else:
                st.warning("PCA koordinatları yok.")

            st.markdown("### Anomali tespiti (Isolation Forest)")
            if result["err_anomaly"]:
                st.error(f"Anomali hatası: {result['err_anomaly']}")
            elif result["pred"] is not None:
                pred = result["pred"]
                anomaly_view = analysis_df.copy()
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
                if result["pca_coords"] is not None:
                    try:
                        fig_a = viz.plot_anomalies(
                            result["pca_coords"], pred.values
                        )
                        st.plotly_chart(fig_a, use_container_width=True)
                    except ValueError as exc:
                        st.warning(f"Anomali grafiği oluşturulamadı: {exc}")
            else:
                st.warning("Anomali tahminleri yok.")

            st.markdown("### Özellik dağılımı")
            cols_list = list(analysis_df.columns)
            col_pick = st.selectbox(
                "Grafik sütunu",
                options=cols_list,
                index=0,
                help="Sayısal: histogram. Kategorik/metin: frekans çubuğu.",
                key=sess["FEATURE_DIST_COLUMN"],
            )
            try:
                fig_d = viz.plot_feature_distribution(analysis_df, col_pick)
                st.plotly_chart(fig_d, use_container_width=True)
            except ValueError as exc:
                st.warning(str(exc))

    with st.expander("Geliştirici notu"):
        st.code(
            "DataLoadError, PreprocessingError, AIModelError — UI’da kullanıcıya "
            "sade mesaj; ayrıntı için log ekleyebilirsiniz."
        )


if __name__ == "__main__":
    main()
