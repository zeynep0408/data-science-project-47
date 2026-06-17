import pytest
import sys
import os
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_data, IlanKaydi, build_prompt, cikar,
    toplu_cikar, en_cok_beceri, uzaktan_orani, ortalama_deneyim, pipeline,
)


# ──────────────────────────────────────────────────────
# MOCK LLM — testler için deterministik sahte LLM
# Gerçek projede burası senin OpenAI/Anthropic çağrın olur.
# Mock, prompt içindeki anahtar kelimelere bakıp sabit JSON döndürür.
# ──────────────────────────────────────────────────────
def fake_llm(prompt):
    p = prompt.lower()

    # beceriler — bilinen becerileri prompt içinde ara
    bilinen = ["python", "sql", "docker", "excel", "java", "react", "aws", "pandas"]
    beceriler = [b for b in bilinen if b in p]

    # deneyim_yili — "<sayı> yıl" kalıbını yakala, yoksa 0
    m = re.search(r"(\d+)\s*yıl", p)
    deneyim_yili = int(m.group(1)) if m else 0

    # uzaktan_mi — "uzaktan" veya "remote" geçiyorsa True
    uzaktan_mi = ("uzaktan" in p) or ("remote" in p)

    # maas_belirtilmis — "maaş" / "tl" / "₺" geçiyorsa True
    maas_belirtilmis = ("maaş" in p) or ("tl" in p) or ("₺" in p)

    # pozisyon — mock için sabit (testte içerik ASSERT edilmez)
    return json.dumps({
        "pozisyon": "Geliştirici",
        "beceriler": beceriler,
        "deneyim_yili": deneyim_yili,
        "uzaktan_mi": uzaktan_mi,
        "maas_belirtilmis": maas_belirtilmis,
    })


@pytest.fixture(scope="module")
def data():
    return load_data()


# 0. Veri yapısı
def test_load_data_structure(data):
    ilanlar = data
    assert isinstance(ilanlar, list)
    assert 10 <= len(ilanlar) <= 12
    assert all(isinstance(t, str) and t for t in ilanlar)


# 1. Şema — geçerli veriyi kabul eder
def test_schema_valid():
    o = IlanKaydi(
        pozisyon="Senior Python Developer",
        beceriler=["python", "sql"],
        deneyim_yili=5,
        uzaktan_mi=True,
        maas_belirtilmis=True,
    )
    assert o.pozisyon == "Senior Python Developer"
    assert o.beceriler == ["python", "sql"]
    assert o.deneyim_yili == 5
    assert o.uzaktan_mi is True
    assert o.maas_belirtilmis is True


# 1b. Şema — geçersiz veriyi reddeder (aralık + tip)
def test_schema_rejects_invalid():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        IlanKaydi(
            pozisyon="X", beceriler=[], deneyim_yili=-1,
            uzaktan_mi=False, maas_belirtilmis=False,
        )
    with pytest.raises(ValidationError):
        IlanKaydi(
            pozisyon="X", beceriler=[], deneyim_yili="çok",
            uzaktan_mi=False, maas_belirtilmis=False,
        )


# 2. Prompt — ilanı ve istenen alanları içerir
def test_build_prompt():
    ilan = "Python ve SQL bilen 5 yıl deneyimli developer"
    p = build_prompt(ilan)
    assert ilan in p                       # prompt ilanı içermeli
    assert "pozisyon" in p.lower()
    assert "beceriler" in p.lower()
    assert "deneyim_yili" in p.lower()
    assert "uzaktan_mi" in p.lower()
    assert "maas_belirtilmis" in p.lower()


# 3. cikar — tek ilan (mock llm ile)
def test_cikar_bilinen_ilan():
    ilan = "Python ve SQL bilen, 5 yıl deneyim. Uzaktan çalışma. Maaş yüksek."
    r = cikar(ilan, fake_llm)
    assert "python" in r["beceriler"]
    assert "sql" in r["beceriler"]
    assert r["deneyim_yili"] == 5
    assert r["uzaktan_mi"] is True
    assert r["maas_belirtilmis"] is True
    assert isinstance(r["pozisyon"], str) and r["pozisyon"]
    assert set(r.keys()) == {
        "pozisyon", "beceriler", "deneyim_yili", "uzaktan_mi", "maas_belirtilmis"
    }


# 4. toplu_cikar
def test_toplu_cikar(data):
    ilanlar = data
    kayitlar = toplu_cikar(ilanlar, fake_llm)
    assert isinstance(kayitlar, list) and len(kayitlar) == len(ilanlar)
    assert all(
        set(k.keys()) == {
            "pozisyon", "beceriler", "deneyim_yili", "uzaktan_mi", "maas_belirtilmis"
        }
        for k in kayitlar
    )


# 5. en_cok_beceri — elle verilen kayıtlarla deterministik
def test_en_cok_beceri():
    kayitlar = [
        {"pozisyon": "A", "beceriler": ["python", "sql"], "deneyim_yili": 3,
         "uzaktan_mi": True, "maas_belirtilmis": True},
        {"pozisyon": "B", "beceriler": ["python", "docker"], "deneyim_yili": 2,
         "uzaktan_mi": False, "maas_belirtilmis": False},
        {"pozisyon": "C", "beceriler": ["python", "sql", "aws"], "deneyim_yili": 5,
         "uzaktan_mi": True, "maas_belirtilmis": True},
    ]
    top = en_cok_beceri(kayitlar, n=2)
    assert isinstance(top, list)
    assert top[0] == "python"          # 3 kez geçiyor → en çok
    assert "sql" in top                # 2 kez geçiyor → ikinci
    assert len(top) == 2


# 6. uzaktan_orani — elle verilen kayıtlarla deterministik
def test_uzaktan_orani():
    kayitlar = [
        {"uzaktan_mi": True}, {"uzaktan_mi": False},
        {"uzaktan_mi": True}, {"uzaktan_mi": True},
    ]
    assert abs(uzaktan_orani(kayitlar) - 0.75) < 1e-9
    assert uzaktan_orani([]) == 0.0


# 7. ortalama_deneyim — elle verilen kayıtlarla deterministik
def test_ortalama_deneyim():
    kayitlar = [
        {"deneyim_yili": 2}, {"deneyim_yili": 4}, {"deneyim_yili": 6},
    ]
    assert abs(ortalama_deneyim(kayitlar) - 4.0) < 1e-9
    assert ortalama_deneyim([]) == 0.0


# 8. pipeline — uçtan uca (mock llm)
def test_pipeline(data):
    ilanlar = data
    sonuc = pipeline(ilanlar, fake_llm)
    assert set(sonuc.keys()) >= {
        "kayitlar", "en_cok_beceri", "uzaktan_orani", "ortalama_deneyim"
    }
    assert len(sonuc["kayitlar"]) == len(ilanlar)
    assert isinstance(sonuc["en_cok_beceri"], list)
    assert 0.0 <= sonuc["uzaktan_orani"] <= 1.0
    assert sonuc["ortalama_deneyim"] >= 0.0


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────
import requests


def _send_score(user_score):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return
    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return
    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {"user_id": USER_ID, "project_id": PROJECT_ID, "user_score": user_score, "is_auto": True}
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
