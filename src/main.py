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


def main() -> None:
    st.set_page_config(page_title="AI Visualizer (dev)", layout="wide")
    st.title("AI Visualizer — geliştirme paneli")
    st.caption("CSV / Excel yükleme, özet, temizleme, AI analizleri ve Plotly grafikleri.")

    uploaded = st.file_uploader(
        "Dataset",
        type=["csv", "xlsx", "xlsm"],
        help="CSV veya Excel (.xlsx). Excel için openpyxl gerekir.",
    )

    if uploaded is None:
        st.info("Başlamak için bir dosya seçin.")
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

    st.success(f"Yüklendi: **{len(df)}** satır, **{len(df.columns)}** sütun.")

    cleaned: pd.DataFrame | None = None
    clean_error: str | None = None
    try:
        cleaned = loader.clean_data(df)
    except PreprocessingError as exc:
        clean_error = str(exc)

    analysis_df = cleaned if cleaned is not None else df

    tab_raw, tab_stats, tab_clean, tab_analiz = st.tabs(
        ["Raw preview", "Summary stats", "Cleaned preview", "Analiz Paneli"]
    )

    with tab_raw:
        st.dataframe(df.head(50), use_container_width=True)

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
            st.dataframe(cleaned.head(50), use_container_width=True)
            missing_after = int(cleaned.isna().sum().sum())
            st.metric("Temizlik sonrası toplam NaN", missing_after)
        else:
            st.error(clean_error or "Temizleme başarısız.")

    with tab_analiz:
        st.subheader("Analiz Paneli")
        if cleaned is None and clean_error is not None:
            st.warning(
                "Temizleme uygulanamadı; analiz **ham veri** üzerinde çalışacak. "
                f"Sebep: {clean_error}"
            )

        n_rows = len(analysis_df)
        max_k = max(1, n_rows)
        default_k = min(3, max_k)

        col_a, col_b = st.columns(2)
        with col_a:
            n_clusters = st.number_input(
                "Küme sayısı (K-Means)",
                min_value=1,
                max_value=max_k,
                value=default_k,
                step=1,
                help="Satır sayısını aşamaz.",
            )
        with col_b:
            contamination = st.slider(
                "Beklenen aykırı oranı (Isolation Forest)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="scikit-learn aralığı: (0, 0.5].",
            )

        st.caption("PCA: **2 boyut** (PC1, PC2) — sabit.")

        run = st.button("Analizi Çalıştır", type="primary")

        if run:
            labels: pd.Series | None = None
            pca_coords: pd.DataFrame | None = None
            pred: pd.Series | None = None
            err_cluster: str | None = None
            err_pca: str | None = None
            err_anomaly: str | None = None

            with st.spinner("AI Analiz Yapıyor..."):
                engine = AIEngine()
                try:
                    raw_labels = engine.perform_clustering(analysis_df, n_clusters)
                    labels = pd.Series(raw_labels, index=analysis_df.index)
                except AIModelError as exc:
                    err_cluster = str(exc)

                try:
                    pca_coords = engine.perform_pca(analysis_df, n_components=2)
                except AIModelError as exc:
                    err_pca = str(exc)

                try:
                    raw_pred = engine.detect_anomalies(
                        analysis_df, contamination=contamination
                    )
                    pred = pd.Series(raw_pred, index=analysis_df.index)
                except AIModelError as exc:
                    err_anomaly = str(exc)

            st.session_state["analiz_result"] = {
                "labels": labels,
                "pca_coords": pca_coords,
                "pred": pred,
                "err_cluster": err_cluster,
                "err_pca": err_pca,
                "err_anomaly": err_anomaly,
                "n_clusters": n_clusters,
            }

        result: dict[str, Any] | None = st.session_state.get("analiz_result")
        if result is None:
            st.caption("Sonuçları görmek için **Analizi Çalıştır** düğmesine basın.")
        else:
            viz = DataVisualizer(theme=_streamlit_theme())

            st.markdown("### Kümeleme (K-Means)")
            if result["err_cluster"]:
                st.error(f"Kümeleme hatası: {result['err_cluster']}")
            elif result["labels"] is not None:
                labels = result["labels"]
                cluster_view = analysis_df.copy()
                cluster_view.insert(0, "cluster", labels.values)
                st.write(
                    f"**{result['n_clusters']}** küme, **{len(labels)}** satır."
                )
                st.dataframe(cluster_view.head(100), use_container_width=True)
                counts_df = (
                    pd.DataFrame({"cluster": labels.values})
                    .groupby("cluster", sort=True)
                    .size()
                    .reset_index(name="count")
                )
                st.text("Küme başına satır sayısı:")
                st.dataframe(
                    counts_df,
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
                    except ValueError as exc:
                        st.warning(f"Küme grafiği oluşturulamadı: {exc}")
            else:
                st.warning("Küme etiketleri yok.")

            st.markdown("### PCA (2 bileşen)")
            if result["err_pca"]:
                st.error(f"PCA hatası: {result['err_pca']}")
            elif result["pca_coords"] is not None:
                st.write("İlk 100 satır — PC1, PC2:")
                st.dataframe(
                    result["pca_coords"].head(100), use_container_width=True
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
                st.dataframe(anomaly_view.head(100), use_container_width=True)
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
                key="feature_dist_column",
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
