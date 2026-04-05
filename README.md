# AI Visualizer

Streamlit tabanlı tabular veri yükleme, temizleme, K-Means kümeleme, PCA görselleştirme, Isolation Forest anomali tespiti ve Plotly grafikleri sunan bitirme / araştırma paneli.

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

Proje kökünden:

```bash
streamlit run src/main.py
```

## Özellikler (kısa)

- CSV / Excel yükleme veya **Iris / Wine** örnek verisi (sklearn)
- Sayısal sütun seçimi, veri kalitesi metrikleri, dirsek eğrisi ve inertia
- Silhouette ve PCA açıklanan varyans ile şeffaflık
- Küme bazlı ortalama profil tablosu ve akademik rapor indirme (`.txt`, Markdown)

---

## Jüri savunma rehberi — soru & cevap

Aşağıdaki terimler, kurulda tek cümlelik netlik için özetlenmiştir.

### Inertia nedir?

**Inertia**, küme içi benzerlik (sıkılık) ölçüsüdür: K-Means’in minimize ettiği *küme içi kare uzaklıklar toplamı (WCSS)* olup, düşük değer merkezlere yakın kümeleri gösterir; *k* arttıkça genelde düştüğü için dirsek ve Silhouette ile birlikte yorumlanmalıdır.

### PCA nedir?

**PCA**, bilgi kaybını (varyans açısından) kontrol ederek boyut indirgeme yapar: veriyi doğrusal bileşenlere yansıtır ve ilk bileşenler ölçeklenmiş uzaydaki toplam varyansın mümkün olduğunca büyük payını sırayla açıklar.

### StandardScaler neden kullanılıyor?

**StandardScaler**, veriyi ölçek olarak *karşılaştırılabilir* hale getirir (her özelliği sıfır ortalama / birim varyansa getirir); böylece “elma ile armut” farklı birim ve aralıklarda olsa da uzaklık tabanlı modellerde adil kıyas yapılır — bu işlem normalizasyon / standartlaştırma olarak bilinir.

### Silhouette skoru neyi ölçer?

**Silhouette**, kümelerin birbirinden ne kadar *ayrık* olduğunun kanıtıdır: noktanın kendi kümesine yakınlığı ile en yakın komşu kümeye uzaklığını kıyaslar; 1’e yakın değerler daha net ayrışmış segmentleri, düşük veya negatif değerler iç içe geçmeyi düşündürür.

### Ek: Isolation Forest’ta contamination ne?

**Contamination:** Modelin veride yaklaşık ne oranda aykırı gözlem beklediğine dair ön bilgidir; fiziksel bir ölçüm değil, algoritma hiperparametresidir ve (0, 0.5] aralığında tanımlıdır.

---

## Mimari (modüller)

| Modül | Rol |
|--------|-----|
| `src/main.py` | Streamlit UI, CONFIG, rapor indirme |
| `src/processor.py` | Dosya okuma, tip çıkarımı, temizleme |
| `src/ai_engine.py` | Ölçekleme, K-Means, PCA, Isolation Forest |
| `src/visualizer.py` | Plotly figürleri |
| `src/exceptions.py` | Ayrık hata türleri |

---

## Lisans / atıf

Bitirme projesi kapsamında kullanım için uygundur; kendi kurumunuzun teslim koşullarına göre atıf veya ek belge ekleyin.
