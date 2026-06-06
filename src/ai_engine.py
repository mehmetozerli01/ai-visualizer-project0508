"""
Makine öğrenmesi çekirdeği: ölçekleme, kümeleme, boyut indirgeme ve anomali tespiti.

Bu modül Streamlit veya çizim kütüphanelerine bağlı değildir. Sayısal sütun seçimi
:class:`processor.DataLoader` ile uyumludur; böylece özet istatistik, temizleme ve
modelleme aynı özellik uzayını paylaşır.

Matematiksel özeti
-------------------
* **StandardScaler** — Özellikleri sıfır ortalama ve birim varyansa getirir (z-skoru);
  öklidyen uzaklığın ölçeğe duyarlılığını giderir.
* **K-Means** — Öklidyen uzaklıkta küme içi kare hatalar toplamını (inertia / WCSS)
  yerel olarak minimize eden ayrım problemidir.
* **PCA** — Kovaryans yapısına dayalı doğrusal projeksiyon; varyansın üst üste
  binmeyen doğrusal bileşenlere ayrıştırılması (SVD tabanlı çözüm).
* **Isolation Forest** — Rastgele ayrımlarla aykırıların daha kısa yoldan izole
  edildiği ansamble yöntem; ``contamination`` ön bilgisine bağlıdır.
* **Random Forest (küme / hedef önemi)** — K-Means etiketleri veya kullanıcı
  hedef değişkeni üzerinde ölçeklenmiş özelliklerde ``feature_importances_``;
  keşifsel yorum içindir, nedensel kanıt değildir.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from exceptions import AIModelError
from processor import DataLoader


class AIEngine:
    """Tabular veride öklidyen-uzaklık tabanlı modelleri tek tip ön işlemle çalıştırır.

    Tüm fit işlemleri önce **StandardScaler** ile aynı ölçeklenmiş uzayda yapılır;
    bu, K-Means (WCSS minimizasyonu), PCA (varyans yönleri) ve Isolation Forest
    (rastgele altuzay ayrımları) için numerik kararlılık sağlar.

    Attributes:
        _random_state: Stokastik tahminleyicilere iletilen tekrarlanabilirlik tohumu.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Tahminleyiciler için ortak RNG tohumunu ayarlar.

        Args:
            random_state: K-Means, PCA ve Isolation Forest için sabit tohum
                (aynı veri-tekrar üretilebilir sonuç).
        """
        self._random_state: Final[int] = random_state

    def _prepare_data(
        self,
        df: pd.DataFrame,
        numeric_columns: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Sayısal matris hazırlar: seçim, zorunlu sayısallaştırma, eksik doldurma, z-skoru.

        **Matematiksel rol:** :math:`X \\in \\mathbb{R}^{n \\times d}` üzerinde sütun
        bazlı ortalama ile eksik imputation, ardından
        :math:`z_{ij} = (x_{ij} - \\mu_j) / \\sigma_j` (``StandardScaler``).
        Böylece sonraki adımlarda öklidyen uzaklık anlamlı hale gelir.

        ``DataLoader.infer_column_types`` ile yükleme katmanıyla aynı sayısal sütun
        tanımı kullanılır. ``numeric_columns`` verilirse yalnızca bu alt küme.

        Returns:
            ``(X_scaled, use_cols)`` — ölçeklenmiş ``float64`` matris ve sütun adları.

        Raises:
            AIModelError: Boş çerçeve, sayısal sütun yokluğu veya ölçekleme hatası.
        """
        if not isinstance(df, pd.DataFrame):
            raise AIModelError("Input must be a pandas DataFrame.")
        if df.empty:
            raise AIModelError("Input DataFrame is empty.")

        try:
            inferred_numeric, _ = DataLoader.infer_column_types(df)
            if not inferred_numeric:
                raise AIModelError(
                    "No numeric columns found. Ensure the dataset has numeric "
                    "features or numeric-like text columns after cleaning."
                )

            if numeric_columns is not None:
                if not numeric_columns:
                    raise AIModelError(
                        "At least one numeric column must be selected for analysis."
                    )
                missing_names = set(numeric_columns) - set(df.columns)
                if missing_names:
                    raise AIModelError(
                        f"Unknown columns (not in DataFrame): {sorted(missing_names)}"
                    )
                bad = [c for c in numeric_columns if c not in inferred_numeric]
                if bad:
                    raise AIModelError(
                        "These columns are not treated as numeric for modeling: "
                        + ", ".join(bad)
                    )
                use_cols = list(numeric_columns)
            else:
                use_cols = inferred_numeric

            X_frame = df[use_cols].copy()
            for col in use_cols:
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

        return X_scaled, use_cols

    def elbow_inertia_scan(
        self,
        df: pd.DataFrame,
        k_max: int = 10,
        numeric_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Dirsek yöntemi için her *k* değerinde K-Means **inertia** (WCSS) tarar.

        **Matematiksel problem:** Her sabit *k* için K-Means, ölçeklenmiş uzayda
        :math:`\\sum_{i} \\min_{c} \\|x_i - \\mu_c\\|^2` (küme içi kare uzaklık
        toplamı) üzerinde yerel minimum arar; burada raporlanan ``inertia`` bu
        değerdir. *k* arttıkça WCSS genelde azalır; dirsek, marjinal kazancın
        kırıldığı bölgeyi seçmeye yardım eder.

        Args:
            df: Girdi tablosu.
            k_max: Denenecek üst *k* (satır sayısına kısıtlanır).
            numeric_columns: İsteğe bağlı sayısal sütun alt kümesi.

        Returns:
            ``k`` ve ``inertia`` sütunlu ``DataFrame``.

        Raises:
            AIModelError: Hazırlık veya geçerli *k* aralığı yoksa.
        """
        if k_max < 1:
            raise AIModelError("k_max must be at least 1.")

        X, _ = self._prepare_data(df, numeric_columns=numeric_columns)
        n_samples = int(X.shape[0])
        # k = n_samples anlamsız kümeleme ve Silhouette ile uyumsuz; tarama üst sınırı n-1.
        upper = min(int(k_max), max(1, n_samples - 1))
        if upper < 1:
            raise AIModelError("Not enough rows for elbow scan.")

        rows: list[tuple[int, float]] = []
        for k in range(1, upper + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self._random_state,
                n_init=10,
            )
            model.fit(X)
            rows.append((k, float(model.inertia_)))

        return pd.DataFrame(rows, columns=["k", "inertia"])

    def perform_clustering(
        self,
        df: pd.DataFrame,
        n_clusters: int = 3,
        numeric_columns: list[str] | None = None,
    ) -> tuple[np.ndarray, float, float | None]:
        """Ölçeklenmiş uzayda **K-Means** ile ayrım; inertia ve isteğe bağlı Silhouette.

        **Matematiksel problem:** *k* sabit küme merkezi :math:`\\{\\mu_c\\}_{c=1}^k`
        ile örnekleri en yakın merkeze atayarak öklidyen WCSS’i minimize etmeye
        çalışır (NP-zor global optimum; Lloyd tipi yinelemeli yaklaşım).

        **Silhouette:** Aynı ölçeklenmiş uzayda, gözlemlerin kendi kümesine göre
        komşu kümeye göre ne kadar daha iyi oturduğunun ortalaması
        (:math:`s \\in [-1,1]`); yüksek değer daha ayrık kümeleri destekler.

        Args:
            df: Girdi tablosu.
            n_clusters: Küme sayısı (1 … satır sayısı).
            numeric_columns: İsteğe bağlı sayısal sütun listesi.

        Returns:
            ``(labels, inertia, silhouette)`` — etiketler, WCSS, ve *k* ≥ 2 ile en az
            iki farklı küme varsa ortalama Silhouette; aksi halde ``silhouette=None``.

        Raises:
            AIModelError: Geçersiz argüman, ``n_clusters >= n_samples`` (stabilite) veya
                eğitim hatası.
        """
        if n_clusters < 1:
            raise AIModelError("n_clusters must be at least 1.")

        try:
            X, _ = self._prepare_data(df, numeric_columns=numeric_columns)
            n_samples = int(X.shape[0])
            if n_clusters >= n_samples:
                raise AIModelError(
                    "Kümeleme ve Silhouette için **satır sayısı, küme sayısından büyük** "
                    f"olmalıdır (şu an {n_samples} satır, k = {n_clusters}). "
                    "k değerini düşürün veya daha fazla gözlem ekleyin."
                )

            model = KMeans(
                n_clusters=n_clusters,
                random_state=self._random_state,
                n_init=10,
            )
            labels = model.fit_predict(X)
            inertia = float(model.inertia_)

            silhouette: float | None = None
            if n_clusters >= 2 and n_samples > n_clusters:
                uniq = int(len(np.unique(labels)))
                if uniq >= 2:
                    try:
                        silhouette = float(
                            silhouette_score(
                                X,
                                labels,
                                metric="euclidean",
                                random_state=self._random_state,
                            )
                        )
                    except ValueError:
                        silhouette = None
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(f"Clustering failed: {exc}") from exc

        return np.asarray(labels, dtype=np.int32), inertia, silhouette

    def cluster_feature_importance_rf(
        self,
        df: pd.DataFrame,
        labels: np.ndarray | pd.Series,
        numeric_columns: list[str] | None = None,
        *,
        n_estimators: int = 200,
    ) -> pd.DataFrame:
        """K-Means etiketlerini hedef alarak **Random Forest** özellik önemleri.

        Özellikler :meth:`_prepare_data` ile ölçeklenir; orman, küme kimliğini
        sınıf etiketi gibi öğrenir ve ``feature_importances_`` ile marjinal
        katkıyı yaklaşıklar. **Yorum:** K-Means ile tutarlı ama farklı optimizasyon;
        hangi değişkenlerin ayırıcı olduğuna dair keşifsel ipucu verir.

        Args:
            df: Model girdi çerçevesi (``analysis_df`` ile uyumlu).
            labels: Satır ile hizalı küme etiketleri.
            numeric_columns: Modele giren sayısal sütunlar.
            n_estimators: Ağaç sayısı (daha fazla → daha kararlı tahmin, daha yavaş).

        Returns:
            ``feature``, ``importance`` sütunları; ``importance`` satır içi normalize
            (toplam 1), azalan sıra.

        Raises:
            AIModelError: Tek küme, yetersiz satır veya eğitim hatası.
        """
        y = np.asarray(labels).ravel()
        if y.shape[0] == 0:
            raise AIModelError("labels is empty.")

        try:
            X, use_cols = self._prepare_data(df, numeric_columns=numeric_columns)
        except AIModelError:
            raise

        if int(X.shape[0]) != int(y.shape[0]):
            raise AIModelError("labels length must match DataFrame rows.")

        n_unique = int(len(np.unique(y)))
        if n_unique < 2:
            raise AIModelError(
                "Özellik önemi için en az **iki farklı küme** gerekir (k ≥ 2 ve "
                "benzersiz etiket)."
            )

        if int(X.shape[0]) < 4:
            raise AIModelError(
                "Random Forest önemi için birkaç gözlemden fazla satır gerekir."
            )

        try:
            n_feat = int(X.shape[1])
            max_depth = min(16, max(3, n_feat * 2))
            rf = RandomForestClassifier(
                n_estimators=int(n_estimators),
                random_state=self._random_state,
                max_depth=max_depth,
                class_weight="balanced_subsample",
            )
            rf.fit(X, y)
            imp = np.asarray(rf.feature_importances_, dtype=np.float64)
            s = float(np.sum(imp))
            if s > 0:
                imp = imp / s
            out = pd.DataFrame(
                {"feature": use_cols, "importance": imp}
            ).sort_values("importance", ascending=False, ignore_index=True)
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(
                f"Random Forest önem analizi başarısız: {exc}"
            ) from exc

        return out

    def target_feature_importance_rf(
        self,
        df: pd.DataFrame,
        target_column: str,
        numeric_columns: list[str] | None = None,
        *,
        n_estimators: int = 200,
    ) -> pd.DataFrame:
        """Seçilen hedef değişkeni tahmin eden **Random Forest** özellik önemleri.

        Hedef sürekli sayısal ise ``RandomForestRegressor``; aksi halde sınıflandırıcı
        kullanılır. Hedef sütun özellik matrisinden çıkarılır; kalan sayısal sütunlar
        açıklayıcı değişken kabul edilir.

        Args:
            df: Model girdi çerçevesi.
            target_column: Açıklanacak hedef sütun adı.
            numeric_columns: Modele girebilecek sayısal sütunlar (hedef hariç tutulur).
            n_estimators: Orman ağacı sayısı.

        Returns:
            ``feature``, ``importance`` sütunları; normalize, azalan sıra.

        Raises:
            AIModelError: Hedef yok, yeterli özellik yok veya eğitim hatası.
        """
        if not target_column or target_column not in df.columns:
            raise AIModelError(
                f"Hedef sütun bulunamadı: {target_column!r}"
            )

        inferred_numeric, _ = DataLoader.infer_column_types(df)
        if numeric_columns is not None:
            feat_pool = [c for c in numeric_columns if c != target_column]
        else:
            feat_pool = [c for c in inferred_numeric if c != target_column]

        if len(feat_pool) < 1:
            raise AIModelError(
                "Hedef dışında en az bir sayısal açıklayıcı sütun gerekir."
            )

        try:
            y_raw = df[target_column]
            if DataLoader._is_numeric_feature_column(y_raw):
                y = pd.to_numeric(y_raw, errors="coerce")
                is_regression = True
                if y.nunique(dropna=True) <= 10:
                    is_regression = False
            else:
                y = y_raw.astype(str)
                is_regression = False

            work = df[feat_pool].apply(pd.to_numeric, errors="coerce")
            col_means = work.mean()
            work = work.fillna(col_means).fillna(0.0)
            y = y.reindex(work.index)
            mask = y.notna() & work.notna().all(axis=1)
            X = np.asarray(work.loc[mask], dtype=np.float64)
            y_fit = y.loc[mask]
            if int(X.shape[0]) < 4:
                raise AIModelError(
                    "Hedef önemi için yeterli tam gözlem yok (en az 4 satır)."
                )

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            n_feat = int(X_scaled.shape[1])
            max_depth = min(16, max(3, n_feat * 2))

            if is_regression:
                model: RandomForestRegressor | RandomForestClassifier = (
                    RandomForestRegressor(
                        n_estimators=int(n_estimators),
                        random_state=self._random_state,
                        max_depth=max_depth,
                    )
                )
                y_train = np.asarray(y_fit, dtype=np.float64)
            else:
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    random_state=self._random_state,
                    max_depth=max_depth,
                    class_weight="balanced_subsample",
                )
                y_train = np.asarray(y_fit)

            if len(np.unique(y_train)) < 2:
                raise AIModelError(
                    "Hedef değişkende en az iki farklı değer gerekir."
                )

            model.fit(X_scaled, y_train)
            imp = np.asarray(model.feature_importances_, dtype=np.float64)
            s = float(np.sum(imp))
            if s > 0:
                imp = imp / s
            out = pd.DataFrame(
                {"feature": feat_pool, "importance": imp}
            ).sort_values("importance", ascending=False, ignore_index=True)
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(
                f"Hedef değişken önem analizi başarısız: {exc}"
            ) from exc

        return out

    def perform_pca(
        self,
        df: pd.DataFrame,
        n_components: int = 2,
        numeric_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Ölçeklenmiş veriyi doğrusal **PCA** alt uzayına yansıtır.

        **Matematiksel problem:** Özellik kovaryansının özdeğer ayrışımına eşdeğer
        olarak, birincil bileşenler varyansı sırayla maksimize eden ortogonal
        yönlerdir; projeksiyon bilgi kaybını (yaklaşık) kayıp varyans üzerinden
        ölçülebilir kılar. ``explained_variance_ratio_`` her bileşenin toplam
        varyansa (ölçeklenmiş uzayda) oransal katkısıdır.

        Args:
            df: Girdi tablosu.
            n_components: Tutulacak bileşen sayısı (en fazla ``min(n, d)``).
            numeric_columns: İsteğe bağlı sayısal sütun alt kümesi.

        Returns:
            ``(coordinates_df, variance_info)`` — ``PC1…PCn`` koordinatları ve
            ``explained_variance_ratio``, ``cumulative_variance_ratio``,
            ``variance_explained_pct`` anahtarları.

        Raises:
            AIModelError: Geçersiz ``n_components`` veya SVD/PCA hatası.
        """
        if n_components < 1:
            raise AIModelError("n_components must be at least 1.")

        try:
            X, _ = self._prepare_data(df, numeric_columns=numeric_columns)
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
            evr = np.asarray(model.explained_variance_ratio_, dtype=np.float64)
            cumulative = float(np.sum(evr))
            variance_info: dict[str, Any] = {
                "explained_variance_ratio": evr,
                "cumulative_variance_ratio": cumulative,
                "variance_explained_pct": float(100.0 * cumulative),
            }
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(f"PCA failed: {exc}") from exc

        return result, variance_info

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        contamination: float = 0.1,
        numeric_columns: list[str] | None = None,
    ) -> np.ndarray:
        """Ölçeklenmiş özelliklerde **Isolation Forest** ile aykırı etiketleri üretir.

        **Matematiksel fikir:** Rastgele özellik ve eşik seçimleriyle ağaçlar inşa
        edilir; nadir ve kısa yoldan izole edilen yapraklara düşen gözlemler aykırı
        adaylarıdır. ``contamination`` beklenen aykırı oranına dair ön bilgidir
        (hiperparametre, fiziksel ölçüm değildir).

        Args:
            df: Girdi tablosu.
            contamination: scikit-learn kısıtı ``(0, 0.5]``.
            numeric_columns: İsteğe bağlı sayısal sütun alt kümesi.

        Returns:
            Satır bazında ``-1`` (anomali) veya ``1`` (normal), ``df`` sırasıyla uyumlu.

        Raises:
            AIModelError: ``contamination`` aralık dışı veya eğitim hatası.
        """
        if contamination <= 0.0 or contamination > 0.5:
            raise AIModelError(
                "contamination must be in the interval (0.0, 0.5] for "
                "IsolationForest."
            )

        try:
            X, _ = self._prepare_data(df, numeric_columns=numeric_columns)
            n_samples = int(X.shape[0])
            if n_samples < 2:
                raise AIModelError(
                    "Anomali tespiti için en az **2 satır** gerekir; "
                    "çok küçük örneklemde model kararlı çalışmaz."
                )

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

    def fit_deployable_models(
        self,
        df: pd.DataFrame,
        *,
        n_clusters: int,
        contamination: float,
        numeric_columns: list[str],
        target_column: str,
        n_estimators: int = 200,
    ) -> dict[str, Any]:
        """Canlı ortama aktarım için K-Means, Isolation Forest ve Random Forest eğitir.

        Tüm modeller aynı özellik uzayında tutarlı ön işlemle eğitilir; dönen
        sözlükte ``scaler``, ``target_scaler`` ve metadata bulunur.

        Args:
            df: Model girdi çerçevesi.
            n_clusters: K-Means küme sayısı.
            contamination: Isolation Forest contamination.
            numeric_columns: Sayısal özellik listesi.
            target_column: Hedef değişken (Random Forest).
            n_estimators: Random Forest ağaç sayısı.

        Returns:
            ``kmeans``, ``isolation_forest``, ``random_forest``, ``scaler``,
            ``target_scaler``, ``feature_columns``, ``target_feature_columns``,
            ``target_column``, ``rf_task`` anahtarları.

        Raises:
            AIModelError: Hazırlık veya eğitim hatası.
        """
        if n_clusters < 1:
            raise AIModelError("n_clusters must be at least 1.")
        if contamination <= 0.0 or contamination > 0.5:
            raise AIModelError("contamination must be in (0.0, 0.5].")

        try:
            X, use_cols = self._prepare_data(df, numeric_columns=numeric_columns)
            n_samples = int(X.shape[0])
            if n_clusters >= n_samples:
                raise AIModelError(
                    f"Satır sayısı ({n_samples}) küme sayısından ({n_clusters}) büyük olmalı."
                )
            if n_samples < 2:
                raise AIModelError("En az 2 satır gerekir.")

            frame = df[use_cols].apply(pd.to_numeric, errors="coerce")
            col_means = frame.mean()
            frame = frame.fillna(col_means).fillna(0.0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(np.asarray(frame, dtype=np.float64))

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self._random_state,
                n_init=10,
            )
            kmeans.fit(X_scaled)

            iso = IsolationForest(
                contamination=contamination,
                random_state=self._random_state,
            )
            iso.fit(X_scaled)

            rf_model: RandomForestRegressor | RandomForestClassifier | None = None
            target_scaler: StandardScaler | None = None
            target_feat_cols: list[str] = []
            rf_task = "none"

            if target_column and target_column in df.columns:
                feat_pool = [c for c in use_cols if c != target_column]
                if feat_pool:
                    y_raw = df[target_column]
                    if DataLoader._is_numeric_feature_column(y_raw):
                        y = pd.to_numeric(y_raw, errors="coerce")
                        is_regression = y.nunique(dropna=True) > 10
                    else:
                        y = y_raw.astype(str)
                        is_regression = False

                    t_frame = df[feat_pool].apply(pd.to_numeric, errors="coerce")
                    t_means = t_frame.mean()
                    t_frame = t_frame.fillna(t_means).fillna(0.0)
                    y = y.reindex(t_frame.index)
                    mask = y.notna() & t_frame.notna().all(axis=1)
                    X_t = np.asarray(t_frame.loc[mask], dtype=np.float64)
                    y_fit = y.loc[mask]
                    if int(X_t.shape[0]) >= 4 and len(np.unique(y_fit)) >= 2:
                        target_scaler = StandardScaler()
                        X_t_scaled = target_scaler.fit_transform(X_t)
                        n_feat = int(X_t_scaled.shape[1])
                        max_depth = min(16, max(3, n_feat * 2))
                        if is_regression:
                            rf_model = RandomForestRegressor(
                                n_estimators=int(n_estimators),
                                random_state=self._random_state,
                                max_depth=max_depth,
                            )
                            rf_model.fit(
                                X_t_scaled,
                                np.asarray(y_fit, dtype=np.float64),
                            )
                            rf_task = "regression"
                        else:
                            rf_model = RandomForestClassifier(
                                n_estimators=int(n_estimators),
                                random_state=self._random_state,
                                max_depth=max_depth,
                                class_weight="balanced_subsample",
                            )
                            rf_model.fit(X_t_scaled, np.asarray(y_fit))
                            rf_task = "classification"
                        target_feat_cols = list(feat_pool)

            return {
                "kmeans": kmeans,
                "isolation_forest": iso,
                "random_forest": rf_model,
                "scaler": scaler,
                "target_scaler": target_scaler,
                "feature_columns": list(use_cols),
                "target_feature_columns": target_feat_cols,
                "target_column": str(target_column),
                "rf_task": rf_task,
                "n_clusters": int(n_clusters),
                "contamination": float(contamination),
                "random_state": int(self._random_state),
            }
        except AIModelError:
            raise
        except Exception as exc:
            raise AIModelError(f"Deployable model training failed: {exc}") from exc
