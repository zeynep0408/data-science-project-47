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
    ilanlar = [
        "Senior Python Developer aranıyor. 5 yıl deneyim, SQL ve Docker bilgisi şart. Uzaktan çalışma mevcut. Maaş: 90.000 TL.",
        "Junior Data Analyst pozisyonu. Excel ve SQL bilen, 1 yıl deneyimli adaylar. Ofisten çalışma. İstanbul.",
        "Backend geliştirici (Java). 3 yıl tecrübe, Docker ve AWS deneyimi tercih edilir. Hibrit model.",
        "Frontend Developer - React uzmanı. 2 yıl deneyim yeterli. Tamamen remote ekip. Rekabetçi ücret sunuyoruz.",
        "Veri Bilimci aranıyor. Python, Pandas ve SQL ileri seviye. 4 yıl deneyim. Uzaktan. Maaş görüşülecektir.",
        "Stajyer yazılım geliştirici. Deneyim aranmıyor. Ofis İzmir. Java veya Python bilen öğrenciler.",
        "DevOps Mühendisi. AWS ve Docker zorunlu. 6 yıl deneyim. Remote pozisyon. 120.000 TL üzeri.",
        "Full Stack Developer. React ve Python bilgisi gerekli. 3 yıl deneyim. Hibrit çalışma, maaş 75.000 TL.",
        "Pazarlama uzmanı. Excel raporlama. 2 yıl tecrübe. Ofisten, Ankara.",
        "Machine Learning Engineer. Python, Pandas, AWS deneyimi. 5 yıl. Uzaktan çalışma imkanı var.",
        "Database Administrator. SQL ve Docker. 7 yıl deneyim şart. Ofis İstanbul. Maaş yüksek.",
        "Mobil geliştirici. Java bilgisi. 1 yıl deneyim yeterli. Remote. Ücret belirtilmemiştir.",
    ]
    return ilanlar
    pass


# 1. Structured output şeması (ML-04)
class IlanKaydi(BaseModel):
    pozisyon: str
    beceriler: list[str]
    deneyim_yili: int = Field(ge=0, le=40)
    uzaktan_mi: bool
    maas_belirtilmis: bool
    pass


# 2. Prompt kurulumu (ML-02)
def build_prompt(ilan):
    return (
        "Şu iş ilanını analiz et ve SADECE JSON döndür.\n"
        "Alanlar: pozisyon (str), beceriler (str listesi), "
        "deneyim_yili (0-40 tam sayı, belirtilmemişse 0), "
        "uzaktan_mi (true/false), maas_belirtilmis (true/false).\n\n"
        f"İlan: {ilan}"
    )
    pass


# 3. Tek ilanı çıkar
def cikar(ilan, llm):
    prompt = build_prompt(ilan)
    cevap = llm(prompt)  # JSON string döner
    return IlanKaydi.model_validate_json(cevap).model_dump()
    pass


# 4. Toplu çıkarım (ML-06 — batch apply)
def toplu_cikar(ilanlar, llm):
    return [cikar(i, llm) for i in ilanlar]
    pass


# 5. İÇGÖRÜ — en çok aranan beceriler (ML-06)
def en_cok_beceri(kayitlar, n=5):
    sayac = Counter()
    for k in kayitlar:
        sayac.update(k["beceriler"])
    return [beceri for beceri, _ in sayac.most_common(n)]

    pass


# 6. İÇGÖRÜ — uzaktan çalışma oranı
def uzaktan_orani(kayitlar):
    if not kayitlar:
        return 0.0
    uzaktan = sum(1 for k in kayitlar if k["uzaktan_mi"])
    return uzaktan / len(kayitlar)
    pass


# 7. İÇGÖRÜ — ortalama deneyim
def ortalama_deneyim(kayitlar):
    if not kayitlar:
        return 0.0
    return sum(k["deneyim_yili"] for k in kayitlar) / len(kayitlar)
    pass


# 8. Uçtan uca pipeline
def pipeline(ilanlar, llm):
    kayitlar = toplu_cikar(ilanlar, llm)
    return {
        "kayitlar": kayitlar,
        "en_cok_beceri": en_cok_beceri(kayitlar),
        "uzaktan_orani": uzaktan_orani(kayitlar),
        "ortalama_deneyim": ortalama_deneyim(kayitlar),
    }
    pass
