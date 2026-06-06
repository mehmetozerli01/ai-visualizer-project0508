"""
Plotly ile kümeleme, anomali, dağılım ve korelasyon görselleştirmesi.

Streamlit’ten bağımsızdır; ``theme`` ile açık/koyu şablon seçilir. Korelasyon
ısı haritasında sayısal sütunlar :class:`processor.DataLoader` ile uyumlu tutulur.

**Görselleştirme notu:** PCA düzlemi üzerindeki saçılım, yüksek boyutlu öklidyen
yapının *doğrusal iki boyutlu* projeksiyonudur; bilgi kaybı PCA açıklanan
varyans oranı ile raporlanmalıdır. Küme renkleri tüm ilgili grafiklerde sabittir
(0=mavi, 1=turuncu, 2=yeşil, …).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from processor import DataLoader

# Küme indeksine göre sabit renkler (0=mavi, 1=turuncu, 2=yeşil); tüm küme grafiklerinde ortak.
_CLUSTER_COLORS_BY_INDEX: tuple[str, ...] = (
    "#2563EB",
    "#EA580C",
    "#16A34A",
    "#7C3AED",
    "#DB2777",
    "#0891B2",
    "#CA8A04",
    "#4F46E5",
)


def _cluster_id_str(v: Any) -> str:
    """Küme etiketini Plotly ``color_discrete_map`` anahtarı için normalize eder."""
    try:
        if pd.isna(v):
            return "?"
        return str(int(v))
    except (ValueError, TypeError):
        return str(v)


def cluster_color_discrete_map(labels: np.ndarray | pd.Series) -> dict[str, str]:
    """Görünen her küme kimliği için sabit hex renk (indeks mod uzunluk)."""
    lab = np.asarray(labels).ravel()
    uniq = sorted({_cluster_id_str(x) for x in lab}, key=lambda s: int(s) if s.lstrip("-").isdigit() else 10**9)
    out: dict[str, str] = {}
    for s in uniq:
        try:
            k = int(s)
        except ValueError:
            k = 0
        out[s] = _CLUSTER_COLORS_BY_INDEX[k % len(_CLUSTER_COLORS_BY_INDEX)]
    return out


# Random Forest önem çubukları: küme paletiyle uyumlu tek renk (Küme 0 mavisi).
_FEATURE_IMPORTANCE_BAR_COLOR: str = _CLUSTER_COLORS_BY_INDEX[0]

# Düşük varyanslı / kategorik sütun eşiği (benzersiz oran veya mutlak adet).
_LOW_VARIANCE_UNIQUE_RATIO: float = 0.08
_LOW_VARIANCE_MAX_UNIQUE: int = 12

SmartChartKind = Literal["scatter", "box", "bar"]


def is_continuous_numeric(df: pd.DataFrame, column: str) -> bool:
    """Sütunun sürekli sayısal (yüksek kardinalite) olduğunu heuristik olarak döner."""
    if column not in df.columns:
        return False
    inferred_numeric, _ = DataLoader.infer_column_types(df)
    if column not in inferred_numeric:
        return False
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(s) < 3:
        return False
    n_unique = int(s.nunique())
    n = int(len(s))
    if n_unique <= _LOW_VARIANCE_MAX_UNIQUE:
        ratio = n_unique / max(1, n)
        if ratio <= _LOW_VARIANCE_UNIQUE_RATIO or n_unique <= 5:
            return False
    return True


def is_categorical_or_low_variance(df: pd.DataFrame, column: str) -> bool:
    """Kategorik veya düşük varyanslı (ayrık) sütun mu?"""
    if column not in df.columns:
        return False
    inferred_numeric, categorical = DataLoader.infer_column_types(df)
    if column in categorical:
        return True
    if column in inferred_numeric:
        return not is_continuous_numeric(df, column)
    return True


def column_decision_type_label(df: pd.DataFrame, column: str) -> str:
    """Akıllı karar matrisi için sütun rol etiketi (Sürekli / Sayısal / Kategorik)."""
    if column not in df.columns:
        return "Kategorik"
    if is_continuous_numeric(df, column):
        return "Sürekli"
    inferred_numeric, categorical = DataLoader.infer_column_types(df)
    if column in categorical:
        return "Kategorik"
    if column in inferred_numeric:
        return "Sayısal"
    return "Kategorik"


def build_bivariate_decision_banner(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    chart_kind: SmartChartKind,
) -> str:
    """İki değişkenli grafik seçiminin gerekçesini kullanıcı dilinde özetler."""
    label_a = column_decision_type_label(df, col_a)
    label_b = column_decision_type_label(df, col_b)
    if chart_kind == "scatter":
        return (
            f"💡 <strong>Sistem Kararı:</strong> <strong>{col_a}</strong> ({label_a}) × "
            f"<strong>{col_b}</strong> ({label_b}) ilişkisi tespit edildi. İki sayısal "
            "değişkenin korelasyonunu ve dağılımını göstermek için en uygun görsel olan "
            "<strong>Saçılım Grafiği</strong> otomatik olarak seçildi."
        )
    if chart_kind == "box":
        return (
            f"💡 <strong>Sistem Kararı:</strong> <strong>{col_a}</strong> ({label_a}) × "
            f"<strong>{col_b}</strong> ({label_b}) ilişkisi tespit edildi. Grupların "
            "merkeze eğilimini ve varyans dağılımını kıyaslamak için en uygun görsel olan "
            "<strong>Kutu Grafiği</strong> otomatik olarak seçildi."
        )
    return (
        f"💡 <strong>Sistem Kararı:</strong> <strong>{col_a}</strong> ({label_a}) × "
        f"<strong>{col_b}</strong> ({label_b}) ilişkisi tespit edildi. Kategorik "
        "grupların ortalama düzeyini karşılaştırmak için en uygun görsel olan "
        "<strong>Çubuk Grafiği</strong> otomatik olarak seçildi."
    )


def count_structure_variables(df: pd.DataFrame) -> tuple[int, int]:
    """Filtre sonrası sürekli ve kategorik (ayrık dahil) sütun sayılarını döner."""
    if df.empty or len(df.columns) == 0:
        return 0, 0
    inferred_numeric, categorical = DataLoader.infer_column_types(df)
    n_continuous = sum(
        1 for col in df.columns if is_continuous_numeric(df, str(col))
    )
    n_categorical = len(categorical) + sum(
        1
        for col in inferred_numeric
        if not is_continuous_numeric(df, str(col))
    )
    return n_continuous, n_categorical


def recommend_bivariate_chart(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> tuple[SmartChartKind, str, str]:
    """İki sütun için en uygun grafik türünü ve (grup, değer) eksen eşlemesini önerir.

    Returns:
        ``(chart_kind, x_or_group_col, y_or_value_col)``
    """
    if col_a == col_b:
        raise ValueError("col_a and col_b must be different.")
    cont_a = is_continuous_numeric(df, col_a)
    cont_b = is_continuous_numeric(df, col_b)
    if cont_a and cont_b:
        return "scatter", col_a, col_b
    if cont_a and not cont_b:
        return "box", col_b, col_a
    if not cont_a and cont_b:
        return "box", col_a, col_b
    return "bar", col_a, col_b


def _pearson_strength_label(r: float) -> str:
    """|r| büyüklüğüne göre Türkçe güç etiketi."""
    a = abs(r)
    if a >= 0.7:
        strength = "güçlü"
    elif a >= 0.4:
        strength = "orta"
    elif a >= 0.2:
        strength = "zayıf"
    else:
        strength = "ihmal edilebilir"
    direction = "pozitif" if r >= 0 else "negatif"
    return f"{strength}/{direction}"


class StatisticalCommentator:
    """Grafik altı istatistiksel yorumları insan diline çevirir."""

    @staticmethod
    def scatter_pearson(df: pd.DataFrame, x_col: str, y_col: str) -> str:
        sub = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3:
            return "Yeterli sayısal gözlem yok; korelasyon yorumu üretilemedi."
        r = float(sub[x_col].corr(sub[y_col], method="pearson"))
        if not np.isfinite(r):
            return "Korelasyon hesaplanamadı (sabit veya eksik veri)."
        pct = abs(r) * 100.0
        label = _pearson_strength_label(r)
        return (
            f"**{x_col}** ile **{y_col}** arasında Pearson r = {r:.3f}; "
            f"yaklaşık **%{pct:.0f}** oranında **{label}** bir ilişki tespit edildi."
        )

    @staticmethod
    def boxplot_by_group(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        *,
        group_label: str = "grup",
    ) -> str:
        yvals = pd.to_numeric(df[value_col], errors="coerce")
        groups = df[group_col].astype(str)
        work = pd.DataFrame({"v": yvals, "g": groups}).dropna(subset=["v"])
        if work.empty:
            return "Kutu grafiği için yeterli sayısal veri yok."
        med = work.groupby("g", sort=True)["v"].median()
        means = work.groupby("g", sort=True)["v"].mean()
        top_med = str(med.idxmax())
        top_mean = str(means.idxmax())
        spread = float(work["v"].max() - work["v"].min())
        return (
            f"Medyan değerlerine göre en yüksek dağılım **{top_med}** "
            f"{group_label}inde toplanmıştır; ortalama en yüksek **{top_mean}** "
            f"{group_label}indedir. **{value_col}** genel yayılımı ≈ {spread:.4g}."
        )

    @staticmethod
    def bar_category_means(
        df: pd.DataFrame,
        category_col: str,
        value_col: str,
    ) -> str:
        yvals = pd.to_numeric(df[value_col], errors="coerce")
        cats = df[category_col].astype(str)
        work = pd.DataFrame({"c": cats, "v": yvals}).dropna(subset=["v"])
        if work.empty:
            return "Çubuk grafiği için yeterli veri yok."
        means = work.groupby("c", sort=True)["v"].mean()
        top = str(means.idxmax())
        bot = str(means.idxmin())
        return (
            f"**{category_col}** kırılımında **{value_col}** ortalaması en yüksek "
            f"**{top}** kategorisinde, en düşük **{bot}** kategorisindedir."
        )

    @staticmethod
    def missing_data_repair(
        raw_df: pd.DataFrame,
        cleaned_df: pd.DataFrame | None,
        *,
        method_label: str,
    ) -> str:
        """Eksik veri ısı haritası altı imputation özeti."""
        n_miss = int(raw_df.isna().sum().sum())
        if n_miss == 0:
            return (
                "Ham veride eksik hücre tespit edilmedi; temizleme adımı "
                "yapısal bütünlük kontrolü olarak geçildi."
            )
        if cleaned_df is None:
            return (
                f"Sistem verideki **{n_miss:,}** eksik hücreyi tespit etti; "
                "otomatik onarım bu oturumda uygulanmadı (ham veri modu)."
            )
        return (
            f"Sistem verideki eksikleri tespit etti ve **{method_label}** yöntemiyle "
            f"otomatik olarak onardı (**{n_miss:,}** hücre etkilendi)."
        )

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str]) -> str:
        sub = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        corr = sub.corr(numeric_only=True, method="pearson")
        if corr.shape[0] < 2:
            return "Korelasyon matrisi yorumu için yeterli sütun yok."
        pairs: list[tuple[float, str, str]] = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                r = float(corr.loc[a, b])
                if np.isfinite(r):
                    pairs.append((abs(r), a, b))
        if not pairs:
            return "Anlamlı korelasyon çifti bulunamadı."
        pairs.sort(reverse=True)
        _, a_max, b_max = pairs[0]
        r_max = float(corr.loc[a_max, b_max])
        return (
            f"En güçlü doğrusal ilişki **{a_max}** – **{b_max}** arasında "
            f"(Pearson r = {r_max:.3f}, {_pearson_strength_label(r_max)})."
        )

    @staticmethod
    def feature_importance(
        importance_df: pd.DataFrame,
        *,
        target_name: str | None = None,
    ) -> str:
        if importance_df.empty or importance_df.shape[1] < 2:
            return "Özellik önemi tablosu boş."
        feat_col = str(importance_df.columns[0])
        imp_col = str(importance_df.columns[1])
        top = importance_df.sort_values(imp_col, ascending=False).iloc[0]
        feat = str(top[feat_col])
        imp = float(top[imp_col])
        ctx = (
            f"**{target_name}** hedefini açıklamada"
            if target_name
            else "Ayırma / tahmin görevinde"
        )
        return (
            f"{ctx} en etkili değişken **{feat}** "
            f"(normalize önem ≈ {imp:.3f})."
        )

    @staticmethod
    def elbow(elbow_df: pd.DataFrame, selected_k: int | None) -> str:
        if elbow_df.empty or "inertia" not in elbow_df.columns:
            return "Dirsek eğrisi yorumu için veri yok."
        inert = elbow_df["inertia"].to_numpy(dtype=float)
        ks = elbow_df["k"].to_numpy(dtype=int)
        if len(inert) < 2:
            return "Dirsek tespiti için en az iki k değeri gerekir."
        drops = np.diff(inert)
        rel = np.abs(drops[1:] / drops[:-1]) if len(drops) > 1 else np.array([1.0])
        elbow_k = int(ks[1 + int(np.argmin(rel))]) if len(rel) else int(ks[0])
        sk_txt = f" Seçilen k = {selected_k}." if selected_k else ""
        return (
            f"Inertia düşüşü k ≈ **{elbow_k}** civarında belirgin yavaşlıyor (dirsek "
            f"sezgisel öneri).{sk_txt}"
        )

    @staticmethod
    def silhouette(silhouette: float | None, n_clusters: int) -> str:
        if silhouette is None:
            return f"Silhouette k = {n_clusters} için hesaplanamadı veya tanımsız."
        if silhouette > 0.5:
            return (
                f"Silhouette = {silhouette:.3f}: kümeler belirgin ve iyi ayrışmış "
                "görünüyor."
            )
        if silhouette < 0.2:
            return (
                f"Silhouette = {silhouette:.3f}: kümeler iç içe; k veya özellik "
                "seçimini gözden geçirin."
            )
        return f"Silhouette = {silhouette:.3f}: orta düzeyde küme ayrışması."

    @staticmethod
    def distribution(df: pd.DataFrame, column: str) -> str:
        if column not in df.columns:
            return "Dağılım yorumu için sütun bulunamadı."
        s = df[column]
        numeric_cols, _ = DataLoader.infer_column_types(df)
        if column in numeric_cols or pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce").dropna()
            if num.empty:
                return "Sayısal dağılım için veri yok."
            skew = float(stats.skew(num, nan_policy="omit"))
            skew_txt = (
                "sağa çarpık" if skew > 0.75 else "sola çarpık" if skew < -0.75 else "yaklaşık simetrik"
            )
            return (
                f"**{column}** medyan = {float(num.median()):.4g}, "
                f"IQR = {float(num.quantile(0.75) - num.quantile(0.25)):.4g}; "
                f"dağılım {skew_txt} (çarpıklık ≈ {skew:.2f})."
            )
        vc = s.astype(str).value_counts().head(3)
        top = ", ".join(f"{k} ({v})" for k, v in vc.items())
        return f"**{column}** en sık kategoriler: {top}."


class DataVisualizer:
    """Tutarlı başlık, eksen ve tema ile Plotly ``Figure`` üretir.

    **Rol:** Model çıktılarını (etiketler, PCA koordinatları, korelasyon matrisi)
    jüriye uygun, etkileşimli grafiklere çevirir; matematiksel tanım modülde değil,
    ``ai_engine`` içindedir.

    Args:
        theme: ``"light"`` → ``plotly_white``; ``"dark"`` → ``plotly_dark``.
    """

    def __init__(self, theme: Literal["light", "dark"] = "light") -> None:
        """Şablon ve font ailesini tema bayrağına göre sabitler."""
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
        """Tüm figürlerde ortak başlık, eksen, yükseklik ve yazı tipi düzenini üretir."""
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
        """İki boyutlu düzlem için x/y sütun adlarını çözer (``PC1``/``PC2`` öncelikli)."""
        if df_pca.shape[1] < 2:
            raise ValueError("df_pca must contain at least two coordinate columns.")
        cols = list(df_pca.columns)
        if "PC1" in df_pca.columns and "PC2" in df_pca.columns:
            return "PC1", "PC2"
        return str(cols[0]), str(cols[1])

    def plot_clustering(
        self,
        df_pca: pd.DataFrame,
        labels: np.ndarray | pd.Series,
        *,
        title: str | None = None,
        variance_explained_pct: float | None = None,
    ) -> Figure:
        """K-Means etiketlerini PCA düzleminde renk kodlu saçılım olarak gösterir.

        **Görselleştirdiği yapı:** İki ana bileşen ekseninde örnekler; renk küme
        ayrımını temsil eder (öklidyen uzaklık K-Means ile ölçeklenmiş uzayda
        minimize edilmiştir). Ek sütunlar hover ile satır detayı sağlar;
        ``custom_data`` satır seçimi için indeks taşır.

        Args:
            df_pca: En az iki koordinat sütunu (tercihen ``PC1``, ``PC2``).
            labels: Satır ile hizalı küme etiketleri.
            title: Özel başlık; ``None`` ise ``variance_explained_pct`` veya varsayılan.
            variance_explained_pct: İlk iki PC için açıklanan toplam varyans yüzdesi
                (başlığa eklenir; ``title`` verilmişse yok sayılır).

        Returns:
            ``plotly.graph_objects.Figure``.

        Raises:
            ValueError: Uzunluk uyumsuzluğu veya eksen çözülememesi.
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
        work["_cluster"] = [_cluster_id_str(x) for x in lab]
        work["_row_key"] = work.index.map(lambda i: str(i))

        hover_cols = [
            c
            for c in work.columns
            if c not in (pc_x, pc_y, "_cluster", "_row_key")
        ]
        hover_data = ["_cluster"] + hover_cols if hover_cols else ["_cluster"]

        cmap = cluster_color_discrete_map(lab)
        fig = px.scatter(
            work,
            x=pc_x,
            y=pc_y,
            color="_cluster",
            color_discrete_map=cmap,
            category_orders={"_cluster": sorted(cmap.keys(), key=lambda s: int(s) if s.lstrip("-").isdigit() else 0)},
            labels={
                pc_x: pc_x,
                pc_y: pc_y,
                "_cluster": "Küme",
            },
            hover_data=hover_data,
            custom_data=["_row_key"],
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")))
        if title is None:
            if variance_explained_pct is not None:
                title = (
                    "K-Means kümeleme (PCA) — PC1+PC2 açıklanan varyans: "
                    f"%{float(variance_explained_pct):.1f}"
                )
            else:
                title = "K-Means kümeleme (PCA düzlemi)"
        fig.update_layout(
            **self._base_layout(
                title,
                xaxis_title=pc_x,
                yaxis_title=pc_y,
            )
        )
        return fig

    @staticmethod
    def _resolve_pc3_columns(df_pca: pd.DataFrame) -> tuple[str, str, str]:
        """Üç boyutlu PCA eksen adlarını çözer (``PC1``, ``PC2``, ``PC3`` öncelikli)."""
        if df_pca.shape[1] < 3:
            raise ValueError("df_pca must contain at least three coordinate columns.")
        cols = list(df_pca.columns)
        if all(f"PC{i}" in df_pca.columns for i in (1, 2, 3)):
            return "PC1", "PC2", "PC3"
        return str(cols[0]), str(cols[1]), str(cols[2])

    def plot_3d_pca_clusters(
        self,
        df_pca: pd.DataFrame,
        labels: np.ndarray | pd.Series,
        *,
        title: str | None = None,
        variance_explained_pct: float | None = None,
        chart_height: int = 640,
    ) -> Figure:
        """K-Means kümelerini PC1–PC3 uzayında ``px.scatter_3d`` ile gösterir.

        Fareyle döndürülebilir 3B projeksiyon; küme sınırlarının derinlik algısı
        ile incelenmesi için tasarlanmıştır.

        Args:
            df_pca: En az üç koordinat sütunu (tercihen ``PC1``…``PC3``).
            labels: Satır ile hizalı küme etiketleri.
            title: Özel başlık.
            variance_explained_pct: İlk üç PC için açıklanan toplam varyans yüzdesi.
            chart_height: Grafik yüksekliği (px).

        Returns:
            ``plotly.graph_objects.Figure`` (3D scatter).

        Raises:
            ValueError: Uzunluk uyumsuzluğu veya eksen çözülememesi.
        """
        if not isinstance(df_pca, pd.DataFrame):
            raise ValueError("df_pca must be a pandas DataFrame.")
        lab = np.asarray(labels)
        if lab.shape[0] != len(df_pca):
            raise ValueError(
                f"labels length ({lab.shape[0]}) must match df_pca rows ({len(df_pca)})."
            )

        pc_x, pc_y, pc_z = self._resolve_pc3_columns(df_pca)
        work = df_pca.copy()
        work["_cluster"] = [_cluster_id_str(x) for x in lab]

        cmap = cluster_color_discrete_map(lab)
        fig = px.scatter_3d(
            work,
            x=pc_x,
            y=pc_y,
            z=pc_z,
            color="_cluster",
            color_discrete_map=cmap,
            category_orders={
                "_cluster": sorted(
                    cmap.keys(),
                    key=lambda s: int(s) if s.lstrip("-").isdigit() else 0,
                )
            },
            labels={
                pc_x: pc_x,
                pc_y: pc_y,
                pc_z: pc_z,
                "_cluster": "Küme",
            },
            opacity=0.88,
        )
        fig.update_traces(marker=dict(size=4, line=dict(width=0.3, color="white")))
        if title is None:
            if variance_explained_pct is not None:
                title = (
                    "3D PCA Explorer — PC1+PC2+PC3 açıklanan varyans: "
                    f"%{float(variance_explained_pct):.1f}"
                )
            else:
                title = "3D PCA Explorer (K-Means kümeleri)"
        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            template=self._template,
            height=int(chart_height),
            margin=dict(l=0, r=0, t=80, b=0),
            font=dict(family="system-ui, Segoe UI, sans-serif", size=13),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            scene=dict(
                xaxis_title=pc_x,
                yaxis_title=pc_y,
                zaxis_title=pc_z,
            ),
        )
        return fig

    def plot_timeseries_anomalies(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        anomaly_labels: np.ndarray | pd.Series,
        *,
        title: str | None = None,
        chart_height: int = 520,
    ) -> Figure:
        """Zaman serisi çizgisi üzerinde Isolation Forest anomalilerini kırmızı sıçrama noktalarıyla işaretler.

        Args:
            df: Zaman ve değer sütunlarını içeren tablo.
            time_col: Tarih/zaman ekseni sütunu.
            value_col: İzlenecek sayısal metrik sütunu.
            anomaly_labels: ``-1`` anomali, ``1`` normal (scikit-learn sözleşmesi).
            title: Grafik başlığı.
            chart_height: Piksel yüksekliği.

        Returns:
            Çizgi + anomali scatter ``Figure``.

        Raises:
            ValueError: Eksik sütun, uzunluk uyumsuzluğu veya boş seri.
        """
        if time_col not in df.columns or value_col not in df.columns:
            raise ValueError("time_col and value_col must exist in DataFrame.")
        preds = np.asarray(anomaly_labels).ravel()
        if preds.shape[0] != len(df):
            raise ValueError("anomaly_labels length must match df rows.")

        times = pd.to_datetime(df[time_col], errors="coerce")
        values = pd.to_numeric(df[value_col], errors="coerce")
        work = pd.DataFrame(
            {"_time": times, "_value": values, "_pred": preds},
            index=df.index,
        ).dropna(subset=["_time", "_value"])
        if work.empty:
            raise ValueError("No valid time-series points to plot.")
        work = work.sort_values("_time")

        normal = work["_pred"] != -1
        anomaly = work["_pred"] == -1

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=work.loc[normal, "_time"],
                y=work.loc[normal, "_value"],
                mode="lines+markers",
                name="Normal",
                line=dict(color="#2563EB", width=1.5),
                marker=dict(size=5, color="#2563EB", opacity=0.75),
                hovertemplate="%{x}<br>%{y:.4f}<extra>Normal</extra>",
            )
        )
        if anomaly.any():
            fig.add_trace(
                go.Scatter(
                    x=work.loc[anomaly, "_time"],
                    y=work.loc[anomaly, "_value"],
                    mode="markers",
                    name="Anomali (sıçrama)",
                    marker=dict(
                        color="#FF2D2D",
                        size=14,
                        symbol="diamond",
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                    hovertemplate="%{x}<br>%{y:.4f}<extra>Anomali (-1)</extra>",
                )
            )
        ts_title = (
            title
            if title is not None
            else f"Zaman serisi anomali motoru — {value_col}"
        )
        fig.update_layout(
            **self._base_layout(
                ts_title,
                xaxis_title=str(time_col),
                yaxis_title=str(value_col),
                height=int(chart_height),
            )
        )
        fig.update_xaxes(rangeslider_visible=True)
        return fig

    def plot_manual_cluster_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        labels: np.ndarray | pd.Series,
        *,
        title: str | None = None,
    ) -> Figure:
        """İki seçili özellik ekseninde gözlemleri K-Means küme rengiyle gösterir (keşif).

        **Amaç:** PCA bileşenleri soyutken, doğrudan anlaşılır değişken çiftlerinde
        (ör. gelir–yaş) segmentasyonu jüriye somutlaştırmak. Eksenler ``df`` içindeki
        ölçektedir; küme etiketleri ayrı bir uzayda üretilmiş olabilir (ör. log
        dönüşümü sonrası model).

        Args:
            df: ``x_col`` ve ``y_col`` içeren tablo; indeks ``labels`` ile hizalı.
            x_col, y_col: Sayısal eksen adları (farklı olmalı).
            labels: Satır ile eşleşen küme etiketleri.
            title: Özel başlık; ``None`` ise ``"{x} vs {y} ilişkisi"`` üretilir.

        Returns:
            ``Figure`` (Plotly Express scatter).

        Raises:
            ValueError: Eksik sütun, aynı eksen veya uzunluk uyumsuzluğu.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if x_col == y_col:
            raise ValueError("x_col and y_col must be different.")
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("x_col and y_col must exist in df.")
        lab = np.asarray(labels).ravel()
        if lab.shape[0] != len(df):
            raise ValueError(
                f"labels length ({lab.shape[0]}) must match df rows ({len(df)})."
            )

        work = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").copy()
        work["_cluster"] = [_cluster_id_str(x) for x in lab]
        work["_row_key"] = df.index.map(lambda i: str(i))

        cmap = cluster_color_discrete_map(lab)
        fig = px.scatter(
            work,
            x=x_col,
            y=y_col,
            color="_cluster",
            color_discrete_map=cmap,
            category_orders={"_cluster": sorted(cmap.keys(), key=lambda s: int(s) if s.lstrip("-").isdigit() else 0)},
            labels={
                x_col: str(x_col),
                y_col: str(y_col),
                "_cluster": "Küme",
            },
            hover_data=["_cluster"],
            custom_data=["_row_key"],
        )
        fig.update_traces(
            marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white"))
        )
        if title is None:
            title = f"{x_col} vs {y_col} ilişkisi (K-Means renkleri)"
        fig.update_layout(
            **self._base_layout(
                title,
                xaxis_title=str(x_col),
                yaxis_title=str(y_col),
            )
        )
        return fig

    def plot_anomalies(
        self,
        df_pca: pd.DataFrame,
        anomaly_labels: np.ndarray | pd.Series,
        *,
        title: str | None = None,
        variance_explained_pct: float | None = None,
    ) -> Figure:
        """Isolation Forest çıktısını PCA düzleminde normal (1) / anomali (-1) olarak çizer.

        **Okuma:** ``-1`` aykırı, ``1`` normal (scikit-learn sözleşmesi); kırmızı
        büyük işaretçiler modelin düşük yoğunluk / kısa yol skoruna dayalı kararını
        vurgular (nedensel açıklama değildir).

        Args:
            df_pca: İki boyutlu PCA koordinatları.
            anomaly_labels: Satır bazlı tahmin etiketleri.
            title: Özel başlık; ``None`` ise varyans veya varsayılan metin.
            variance_explained_pct: PCA düzleminde açıklanan varyans yüzdesi (başlıkta).

        Returns:
            İki izli (normal / anomali) ``Figure``.

        Raises:
            ValueError: Uzunluk veya eksen çözümü hatası.
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

        # Hover: index + coords + label; customdata = ham satır anahtarı (seçim eşlemesi)
        idx = df_pca.index.astype(str)
        idx_raw = df_pca.index.to_numpy()

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
                customdata=np.asarray(idx_raw[normal], dtype=object).reshape(-1, 1),
                hovertemplate="%{text}<extra></extra>",
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
                customdata=np.asarray(idx_raw[is_anomaly], dtype=object).reshape(-1, 1),
                hovertemplate="%{text}<extra></extra>",
            )
        )
        if title is None:
            if variance_explained_pct is not None:
                title = (
                    "Anomali tespiti (PCA) — PC1+PC2 açıklanan varyans: "
                    f"%{float(variance_explained_pct):.1f}"
                )
            else:
                title = "Anomali tespiti (PCA düzlemi)"
        fig.update_layout(**self._base_layout(title, pc_x, pc_y))
        return fig

    def plot_feature_distribution(
        self,
        df: pd.DataFrame,
        column: str,
        *,
        title: str | None = None,
    ) -> Figure:
        """Tek değişkenin dağılımını histogram (sayısal) veya çubuk (kategorik) ile gösterir.

        **Amaç:** Özellik uzayındaki marjinal dağılımı keşifsel olarak incelemek;
        küme veya anomali modeli ile doğrudan aynı optimizasyonu çözmez.

        Args:
            df: Kaynak tablo.
            column: Var olan sütun adı.
            title: Özel başlık; ``None`` ise sütun adından türetilir.

        Returns:
            Temalı ``Figure``.

        Raises:
            ValueError: Sütun yoksa.
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
            dist_title = title if title is not None else f"Dağılım: {column}"
            fig.update_layout(
                **self._base_layout(
                    dist_title,
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
            cat_title = title if title is not None else f"Kategori frekansı: {column}"
            fig.update_layout(
                **self._base_layout(
                    cat_title,
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
        title: str | None = None,
    ) -> Figure:
        """Dirsek yöntemi: *k* ile K-Means inertia (WCSS) eğrisini çizer.

        **Yorum:** Eğride belirgin dirsek, küme sayısı artışına karşı marjinal
        WCSS düşüşünün yavaşladığı bölgeyi işaret etmeye yardım eder (sezgisel
        seçim; istatistiksel test değildir). ``selected_k`` dikey çizgi ile UI
        seçimiyle hizalanır.

        Args:
            elbow_df: ``k`` ve ``inertia`` sütunları (:meth:`ai_engine.AIEngine.elbow_inertia_scan`).
            selected_k: İşaretlenecek kullanıcı *k* değeri.
            title: Özel grafik başlığı.

        Returns:
            Temalı ``Figure``.

        Raises:
            ValueError: Eksik sütun veya boş çerçeve.
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
        elbow_title = (
            title
            if title is not None
            else "Dirsek yöntemi: K vs inertia (WCSS, ölçeklenmiş uzayda)"
        )
        fig.update_layout(
            **self._base_layout(
                elbow_title,
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
        *,
        title: str | None = None,
    ) -> Figure:
        """Sayısal sütunlar için Pearson korelasyon matrisinin ısı haritası.

        **Matematiksel nesne:** Çiftler arası doğrusal ilişki katsayısı
        :math:`r \\in [-1,1]`; ölçekten bağımsız kıyas için değişkenler analiz
        öncesi tipik olarak standartlaştırılmış olmalıdır (bu grafik ham sütun
        üzerinden korelasyonu hesaplar — model uzayı ile uyum için yalnızca
        çıkarılmış sayısal sütunlar kullanılır).

        Args:
            df: Kaynak tablo.
            numeric_columns: Alt küme; ``None`` ise ``DataLoader`` sayısal listesi.
            title: Özel grafik başlığı.

        Returns:
            ``RdBu_r`` ile ``zmin=-1``, ``zmax=1`` sınırlı ``Figure``.

        Raises:
            ValueError: İki sayısal sütundan az kaldıysa.
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
        hm_title = title if title is not None else "Korelasyon matrisi (Pearson)"
        fig.update_layout(
            **self._base_layout(
                hm_title,
                xaxis_title="",
                yaxis_title="",
                height=560,
            )
        )
        fig.update_xaxes(side="bottom")
        return fig

    def plot_missing_data_matrix(
        self,
        df: pd.DataFrame,
        *,
        max_rows: int = 120,
        title: str | None = None,
    ) -> Figure:
        """Ham verideki eksik (NaN) hücreleri satır×sütun ısı haritası olarak gösterir."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty pandas DataFrame.")
        n = max(1, min(int(max_rows), len(df)))
        sub = df.iloc[:n].copy()
        missing = sub.isna().astype(int)
        row_labels = [f"Satır {i + 1}" for i in range(n)]
        fig = px.imshow(
            missing.values,
            x=[str(c) for c in sub.columns],
            y=row_labels,
            color_continuous_scale=[
                [0.0, "#22C55E"],
                [0.5, "#FDE68A"],
                [1.0, "#EF4444"],
            ],
            zmin=0,
            zmax=1,
            labels=dict(color="Eksik (1=NaN)"),
        )
        fig.update_traces(
            hovertemplate="Satır=%{y}<br>Sütun=%{x}<br>Eksik=%{z}<extra></extra>",
        )
        md_title = (
            title
            if title is not None
            else f"Kayıp veri matrisi (ilk {n} satır)"
        )
        h = min(560, 80 + n * 4)
        fig.update_layout(
            **self._base_layout(
                md_title,
                xaxis_title="Sütun",
                yaxis_title="Gözlem",
                height=h,
            )
        )
        fig.update_xaxes(tickangle=-35)
        return fig

    def plot_cluster_feature_importance(
        self,
        importance_df: pd.DataFrame,
        *,
        title: str | None = None,
        chart_height: int | None = None,
    ) -> Figure:
        """Küme ayrımında Random Forest ``feature_importances_`` yatay çubuk grafiği.

        Args:
            importance_df: En az iki sütun; birinci özellik adı, ikinci önem skoru.
            title: Grafik başlığı.
            chart_height: Piksel yüksekliği (``None`` → varsayılan).

        Returns:
            Yatay ``bar`` ``Figure``.

        Raises:
            ValueError: Boş veya eksik sütun.
        """
        if not isinstance(importance_df, pd.DataFrame) or importance_df.empty:
            raise ValueError("importance_df must be a non-empty DataFrame.")
        if importance_df.shape[1] < 2:
            raise ValueError("importance_df needs at least two columns.")
        feat_col = str(importance_df.columns[0])
        imp_col = str(importance_df.columns[1])
        plot_df = importance_df.sort_values(imp_col, ascending=True)
        fig = px.bar(
            plot_df,
            x=imp_col,
            y=feat_col,
            orientation="h",
            color_discrete_sequence=[_FEATURE_IMPORTANCE_BAR_COLOR],
        )
        fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
        h = int(chart_height) if chart_height is not None else 520
        bar_title = (
            title
            if title is not None
            else "Kümeleri en çok ayıran faktörler (Random Forest önemi)"
        )
        fig.update_layout(
            **self._base_layout(
                bar_title,
                xaxis_title="Önem (normalize, toplam = 1)",
                yaxis_title="Özellik",
                height=h,
            )
        )
        fig.update_layout(showlegend=False)
        return fig

    def plot_cluster_boxplots(
        self,
        df: pd.DataFrame,
        column: str,
        labels: np.ndarray | pd.Series,
        *,
        title: str | None = None,
        chart_height: int | None = None,
    ) -> Figure:
        """Tek sayısal sütunun küme etiketine göre kutu grafikleri (yan yana).

        Medyan ve çeyrekler arası aralık, küme içi yayılımı PCA’dan bağımsız
        olarak gösterir; ayrık kümelerde kutu konumları farklılaşır.

        Args:
            df: Kaynak tablo.
            column: Sayısal sütun adı.
            labels: Satır ile hizalı küme etiketleri.
            title: Başlık.
            chart_height: Piksel yüksekliği.

        Returns:
            ``px.box`` ``Figure``.

        Raises:
            ValueError: Sütun yok, sayısal veri yok veya boş sonuç.
        """
        if column not in df.columns:
            raise ValueError(f"Column {column!r} not found in DataFrame.")
        lab = np.asarray(labels).ravel()
        if lab.shape[0] != len(df):
            raise ValueError("labels length must match df rows.")

        yvals = pd.to_numeric(df[column], errors="coerce")
        work = pd.DataFrame(
            {
                str(column): yvals,
                "Küme": pd.Series(lab, index=df.index).astype(int).astype(
                    "string"
                ),
            },
            index=df.index,
        ).dropna(subset=[str(column)])
        if work.empty:
            raise ValueError("No valid numeric values for box plot.")

        cat_order = sorted(work["Küme"].unique(), key=lambda x: int(x))
        work["Küme"] = pd.Categorical(
            work["Küme"], categories=cat_order, ordered=True
        )

        cmap = cluster_color_discrete_map(lab)
        fig = px.box(
            work,
            x="Küme",
            y=str(column),
            color="Küme",
            color_discrete_map=cmap,
            category_orders={"Küme": cat_order},
        )
        h = int(chart_height) if chart_height is not None else 480
        box_title = (
            title
            if title is not None
            else f"{column} — küme bazında dağılım (kutu grafik)"
        )
        fig.update_layout(
            **self._base_layout(
                box_title,
                xaxis_title="Küme",
                yaxis_title=str(column),
                height=h,
            )
        )
        fig.update_layout(legend_title_text="Küme")
        return fig

    def plot_smart_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        *,
        title: str | None = None,
        color_col: str | None = None,
    ) -> Figure:
        """İki sürekli sayısal sütun için saçılım grafiği."""
        work = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if work.empty:
            raise ValueError("No valid numeric pairs for scatter plot.")
        fig = px.scatter(
            work,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col in df.columns else None,
            color_discrete_sequence=["#6366F1"],
        )
        fig.update_traces(marker=dict(size=9, opacity=0.82))
        sc_title = title if title is not None else f"{x_col} vs {y_col} (saçılım)"
        fig.update_layout(
            **self._base_layout(sc_title, str(x_col), str(y_col))
        )
        return fig

    def plot_smart_box(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        *,
        title: str | None = None,
    ) -> Figure:
        """Kategorik / düşük varyanslı gruplara göre kutu grafiği."""
        yvals = pd.to_numeric(df[value_col], errors="coerce")
        groups = df[group_col].astype(str)
        work = pd.DataFrame(
            {str(value_col): yvals, "Grup": groups},
            index=df.index,
        ).dropna(subset=[str(value_col)])
        if work.empty:
            raise ValueError("No valid data for box plot.")
        cat_order = sorted(work["Grup"].unique())
        work["Grup"] = pd.Categorical(work["Grup"], categories=cat_order, ordered=True)
        fig = px.box(
            work,
            x="Grup",
            y=str(value_col),
            color="Grup",
            color_discrete_sequence=list(_CLUSTER_COLORS_BY_INDEX),
            category_orders={"Grup": cat_order},
        )
        bx_title = (
            title
            if title is not None
            else f"{value_col} — {group_col} kırılımında kutu grafik"
        )
        fig.update_layout(
            **self._base_layout(
                bx_title,
                str(group_col),
                str(value_col),
            )
        )
        fig.update_layout(showlegend=False)
        return fig

    def plot_smart_bar(
        self,
        df: pd.DataFrame,
        category_col: str,
        value_col: str,
        *,
        title: str | None = None,
    ) -> Figure:
        """İki ayrık sütun için kategori başına ortalama çubuk grafiği."""
        yvals = pd.to_numeric(df[value_col], errors="coerce")
        cats = df[category_col].astype(str)
        work = pd.DataFrame({"category": cats, "value": yvals}).dropna()
        if work.empty:
            raise ValueError("No valid data for bar chart.")
        agg = work.groupby("category", sort=True)["value"].mean().reset_index()
        fig = px.bar(
            agg,
            x="category",
            y="value",
            color_discrete_sequence=["#0EA5E9"],
        )
        br_title = (
            title
            if title is not None
            else f"{value_col} ortalaması — {category_col} kırılımı"
        )
        fig.update_layout(
            **self._base_layout(
                br_title,
                str(category_col),
                f"Ort. {value_col}",
            )
        )
        fig.update_xaxes(tickangle=-35)
        return fig

    def plot_smart_bivariate(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        *,
        title: str | None = None,
        labels: np.ndarray | pd.Series | None = None,
    ) -> tuple[Figure, SmartChartKind, str]:
        """Akıllı grafik seçici: sütun tiplerine göre scatter, box veya bar üretir.

        Returns:
            ``(figure, chart_kind, ai_comment)``
        """
        kind, group_col, value_col = recommend_bivariate_chart(df, col_a, col_b)
        commentator = StatisticalCommentator()
        if kind == "scatter":
            fig = self.plot_smart_scatter(df, group_col, value_col, title=title)
            comment = commentator.scatter_pearson(df, group_col, value_col)
        elif kind == "box":
            fig = self.plot_smart_box(df, group_col, value_col, title=title)
            comment = commentator.boxplot_by_group(
                df, value_col, group_col, group_label="kategoride"
            )
        else:
            fig = self.plot_smart_bar(df, group_col, value_col, title=title)
            comment = commentator.bar_category_means(df, group_col, value_col)
        return fig, kind, comment

    def plot_target_feature_importance(
        self,
        importance_df: pd.DataFrame,
        *,
        target_column: str,
        title: str | None = None,
        chart_height: int | None = None,
    ) -> Figure:
        """Hedef değişken odaklı Random Forest özellik önemi grafiği."""
        if not isinstance(importance_df, pd.DataFrame) or importance_df.empty:
            raise ValueError("importance_df must be a non-empty DataFrame.")
        feat_col = str(importance_df.columns[0])
        imp_col = str(importance_df.columns[1])
        plot_df = importance_df.sort_values(imp_col, ascending=True)
        fig = px.bar(
            plot_df,
            x=imp_col,
            y=feat_col,
            orientation="h",
            color_discrete_sequence=[_FEATURE_IMPORTANCE_BAR_COLOR],
        )
        fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
        h = int(chart_height) if chart_height is not None else 520
        tgt_title = (
            title
            if title is not None
            else f"Hedef: {target_column} — en etkili özellikler (Random Forest)"
        )
        fig.update_layout(
            **self._base_layout(
                tgt_title,
                xaxis_title="Önem (normalize, toplam = 1)",
                yaxis_title="Özellik",
                height=h,
            )
        )
        fig.update_layout(showlegend=False)
        return fig

    def plot_data_quality_radar(
        self,
        axis_values: list[tuple[str, float]],
        *,
        title: str | None = None,
        chart_height: int | None = None,
    ) -> Figure:
        """Veri kalitesi göstergelerini 0–100 ölçeğinde radar (örümcek ağı) grafiği.

        Args:
            axis_values: ``(eksen etiketi, skor 0…100)`` çiftleri; en az üç, tipik olarak beş–altı eksen.
            title: Grafik başlığı.
            chart_height: Piksel yüksekliği.

        Returns:
            ``plotly.graph_objects.Figure`` (``Scatterpolar``).

        Raises:
            ValueError: Boş veya çok kısa eksen listesi.
        """
        if not axis_values or len(axis_values) < 3:
            raise ValueError("axis_values must contain at least three (label, score) pairs.")
        labels = [str(t[0]) for t in axis_values]
        vals = [float(max(0.0, min(100.0, t[1]))) for t in axis_values]
        labels_closed = labels + [labels[0]]
        vals_closed = vals + [vals[0]]

        line_color = "#818CF8" if self._theme == "dark" else "#4F46E5"
        fill_rgba = "rgba(129,140,248,0.35)" if self._theme == "dark" else "rgba(79,70,229,0.28)"

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=labels_closed,
                fill="toself",
                fillcolor=fill_rgba,
                line=dict(color=line_color, width=2),
                name="Kalite skoru",
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            )
        )
        rad_title = title if title is not None else "Veri kalitesi radarı (0–100)"
        h = int(chart_height) if chart_height is not None else 480
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                ),
                angularaxis=dict(rotation=90, direction="counterclockwise"),
            ),
            showlegend=False,
            **self._base_layout(rad_title, "", "", height=h),
        )
        return fig
