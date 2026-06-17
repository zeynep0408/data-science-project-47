# DS-47 — İş İlanı Çıkarım Motoru

**Modül**: Generative AI ve LLM (ML-07) • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

İş piyasası analizi yapan bir ekipte **veri bilimcisin**. Elinde yüzlerce **dağınık, serbest metin iş ilanı** var — her biri farklı yazılmış, standart yok. Yönetici soruyor: *"Bu ilanlardan ne çıkarabiliriz? En çok hangi beceriler aranıyor, kaç ilan uzaktan?"*

LLM ile **uçtan uca bir yapısal çıkarım (structured extraction) pipeline'ı** kuruyorsun:

1. Dağınık metni **structured output** ile yapısal kayda çevir (pozisyon + beceriler + deneyim + uzaktan + maaş),
2. Tüm veriye **toplu** uygula,
3. Yapısal kayıtlardan **İÇGÖRÜ** çıkar (en çok aranan beceriler, uzaktan çalışma oranı, ortalama deneyim).

Bir veri bilimcinin LLM ile gerçekte yaptığı iş budur: **yapılandırılmamış metni ölçekte yapısal veriye çevirmek** ve analiz etmek.

## 🧩 Kullandığın Araçlar

- ✅ **Pydantic** structured output şeması (`Field` ile kısıtlı alanlar) — ML-04
- ✅ **Prompt kurulumu** — ML-02
- ✅ **Toplu (batch) çıkarım** deseni — ML-06
- ✅ **collections.Counter** ile içgörü (en çok aranan beceriler)
- ✅ Oran / ortalama gibi temel **agregasyon** metrikleri

## 🔌 LLM Nasıl Çalışıyor? (önemli)

Fonksiyonların LLM'i **parametre olarak alır** (`llm`). Bu bir fonksiyondur: prompt (str) alır, JSON string döndürür.

- **Gerçek kullanımda** kendi OpenAI/Anthropic çağrını verirsin (aşağıya bak) → gerçek çıkarım.
- **Testlerde** deterministik bir sahte LLM verilir → **API key gerekmez**, puanlama her zaman aynı.

Aynı kod, sadece verdiğin `llm` değişir. (Bu, profesyonel **dependency injection** desenidir — test edilebilir LLM kodu yazmanın yolu.)

## 📦 Kurulum

```bash
git clone <your-fork-url>
cd ds-47-is-ilani-cikarim

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python watch.py                 # otomatik test (dosya değişince)
# veya
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 727 # ← Bu projeye ait, DOKUNMA
```

## 📝 Görevler (`tasks/task_manager.py`)

| # | Fonksiyon | Ne yapar |
|---|-----------|----------|
| 0 | `load_data()` | Gömülü dağınık iş ilanı metinleri |
| 1 | `IlanKaydi` | Pydantic şema: pozisyon / beceriler / deneyim_yili / uzaktan_mi / maas_belirtilmis |
| 2 | `build_prompt(ilan)` | Modele verilecek prompt'u kur |
| 3 | `cikar(ilan, llm)` | Prompt kur → llm çağır → şemayla parse |
| 4 | `toplu_cikar(ilanlar, llm)` | Hepsini yapısal kayda çevir |
| 5 | `en_cok_beceri(kayitlar, n=5)` | En çok aranan n beceri |
| 6 | `uzaktan_orani(kayitlar)` | Uzaktan çalışma oranı (0-1) |
| 7 | `ortalama_deneyim(kayitlar)` | Ortalama deneyim yılı |
| 8 | `pipeline(ilanlar, llm)` | Hepsini uçtan uca birleştir |

## 🚀 Gerçek LLM ile Çalıştır (deneyim — puanlanmaz ama YAP!)

### 🔑 Önce API key (sadece gerçek-LLM denemesi için — testlerde GEREKMEZ)

1. **Anahtar al:** [platform.openai.com](https://platform.openai.com) → Billing'e küçük bakiye yükle → **API keys → Create**. (Anthropic için console.anthropic.com.)
2. **`.env` oluştur** ve anahtarını oraya koy — bu dosya `.gitignore`'da, **commit edilmez**:
   ```bash
   cp .env.example .env
   # .env içeriği:  OPENAI_API_KEY=sk-...
   ```
3. **Kur:** `pip install openai python-dotenv`
4. ⚠️ Anahtarı **ASLA** koda, `kaizu_config.py`'ye ya da git'e yazma — yalnızca `.env`'e. (Spending limit koymayı da unutma.)


Testler sahte LLM ile geçer; ama asıl öğrenme, kodu **gerçek** bir modelle çalıştırmaktır:

```python
from dotenv import load_dotenv
load_dotenv()                       # .env'den OPENAI_API_KEY okunur
from openai import OpenAI
from tasks.task_manager import load_data, pipeline

client = OpenAI()
def gercek_llm(prompt):
    r = client.chat.completions.create(
        model="gpt-5.4-mini",                       # ucuz, toplu iş için ideal
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return r.choices[0].message.content

ilanlar = load_data()
print(pipeline(ilanlar, gercek_llm))   # gerçek çıkarım + içgörü
```

> 💡 Aynı `pipeline` fonksiyonu hem testte (sahte llm) hem burada (gerçek llm) çalışır — işte dependency injection'ın gücü.
