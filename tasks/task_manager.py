"""
DS-47 — İş İlanı Çıkarım Motoru

Senaryo: İş piyasası analizi için bir veri bilimcisin. Elinde dağınık, serbest
metin iş ilanları var. Bunları LLM ile YAPISAL VERİYE çeviriyorsun (structured
extraction): pozisyon, beceriler, deneyim yılı, uzaktan çalışma, maaş bilgisi.
Sonra toplu işleyip "en çok aranan beceriler" gibi içgörüler çıkarıyorsun:
ham ilan → structured output ile çıkarım → toplu apply → İÇGÖRÜ.

⚠️ LLM'i dışarıdan PARAMETRE olarak alıyoruz (`llm`). Böylece:
  - Gerçek kullanımda kendi OpenAI/Anthropic çağrını verirsin (README'ye bak).
  - Testlerde deterministik bir sahte LLM verilir (API key gerekmez).
`llm` bir fonksiyondur: prompt (str) alır, JSON string döndürür.

Her fonksiyonun `pass` kısmını doldur. Test: `python watch.py` veya
`pytest tests/test_question.py -v`
"""
import json
from collections import Counter

from pydantic import BaseModel, Field


# 0. Veri seti
def load_data():
    """
    Koda gömülü küçük bir veri seti döndür:
      - ilanlar: list[str] — 10-12 DAĞINIK, serbest metin Türkçe iş ilanı.
    İlanlar çeşitli olsun: bazısı uzaktan, bazısı değil; bazısı maaş belirtir,
    bazısı belirtmez; farklı beceriler ve deneyim yılları.

    Returns:
        list[str]
    """
    pass


# 1. Structured output şeması (ML-04)
class IlanKaydi(BaseModel):
    """
    Bir iş ilanının yapılandırılmış kaydı. Şu alanları TANIMLA:
      - pozisyon: str
      - beceriler: list[str]
      - deneyim_yili: int, 0-40 arası (Field(ge=0, le=40))
      - uzaktan_mi: bool
      - maas_belirtilmis: bool
    """
    pass


# 2. Prompt kurulumu (ML-02)
def build_prompt(ilan):
    """
    Modele verilecek prompt'u kur. Prompt MUTLAKA:
      - `ilan` metnini içermeli,
      - modelden pozisyon + beceriler + deneyim_yili + uzaktan_mi +
        maas_belirtilmis alanlarını JSON olarak istemeli.

    Args:
        ilan: str
    Returns:
        str: prompt
    """
    pass


# 3. Tek ilanı çıkar
def cikar(ilan, llm):
    """
    1. build_prompt ile prompt kur.
    2. `cevap = llm(prompt)` ile modeli çağır (JSON string döner).
    3. Cevabı IlanKaydi şemasıyla doğrula ve dict olarak döndür.
       İpucu: IlanKaydi.model_validate_json(cevap).model_dump()

    Args:
        ilan: str
        llm: callable — prompt(str) alır, JSON string döndürür
    Returns:
        dict: {"pozisyon", "beceriler", "deneyim_yili", "uzaktan_mi",
               "maas_belirtilmis"}
    """
    pass


# 4. Toplu çıkarım (ML-06 — batch apply)
def toplu_cikar(ilanlar, llm):
    """
    Her ilanı `cikar` ile yapısal kayda çevir, dict listesi döndür.

    Args:
        ilanlar: list[str]
        llm: callable
    Returns:
        list[dict]
    """
    pass


# 5. İÇGÖRÜ — en çok aranan beceriler (ML-06)
def en_cok_beceri(kayitlar, n=5):
    """
    Tüm kayıtların `beceriler` listelerini topla, en çok geçen n beceriyi
    (string) liste olarak döndür. İpucu: collections.Counter.

    Args:
        kayitlar: list[dict]
        n: int
    Returns:
        list[str]
    """
    pass


# 6. İÇGÖRÜ — uzaktan çalışma oranı
def uzaktan_orani(kayitlar):
    """
    `uzaktan_mi == True` olan kayıtların oranını (0.0-1.0) döndür.
    Kayıt yoksa 0.0 döndür.

    Args:
        kayitlar: list[dict]
    Returns:
        float
    """
    pass


# 7. İÇGÖRÜ — ortalama deneyim
def ortalama_deneyim(kayitlar):
    """
    Tüm kayıtların `deneyim_yili` ortalamasını döndür.
    Kayıt yoksa 0.0 döndür.

    Args:
        kayitlar: list[dict]
    Returns:
        float
    """
    pass


# 8. Uçtan uca pipeline
def pipeline(ilanlar, llm):
    """
    Hepsini birleştir:
      1. toplu_cikar ile tüm ilanları yapısal kayda çevir (kayitlar),
      2. en_cok_beceri ile en çok aranan becerileri bul,
      3. uzaktan_orani ile uzaktan çalışma oranını bul,
      4. ortalama_deneyim ile ortalama deneyimi bul.

    Returns:
        dict: {"kayitlar": [...], "en_cok_beceri": [...],
               "uzaktan_orani": float, "ortalama_deneyim": float}
    """
    pass
