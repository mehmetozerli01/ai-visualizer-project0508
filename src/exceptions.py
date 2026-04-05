"""ML ve ön işlem boru hattı için ayrıştırılmış istisna türleri.

Her sınıf, hatanın **hangi aşamada** (dosya okuma, doğrulama, model) oluştuğunu
belirlemeye yarar; arayüzde kullanıcıya kısa mesaj, geliştiriciye log ayrımı
yapılabilir.
"""


class ProcessorError(Exception):
    """Yükleme, temizleme veya tablo doğrulama aşamasında oluşan temel hata."""


class DataLoadError(ProcessorError):
    """Dosya biçimi, boş içerik veya ayrıştırma başarısız olduğunda fırlatılır."""


class PreprocessingError(ProcessorError):
    """``DataFrame`` şeması / boşluk / temizleme kuralları ihlal edildiğinde."""


class AIModelError(Exception):
    """Ölçekleme, uyum (fit) veya çıkarım sırasında ML çekirdeğinde oluşan hata."""
