# HANDOFF (Geliştirici → İnceleyici)

## Görev: T12 — README + demo GIF

**Değişen/eklenen dosyalar**
- `README.md` — Tam proje dokümantasyonu: tek cümlelik özet, demo GIF + altında zorunlu CC BY 4.0 atfı, özellikler, iki geçişli mimari diyagramı ve modül haritası, Windows `uv` kurulumu (Python 3.11), opsiyonel CUDA PyTorch kurulumu, repo kökü (CWD) notu, TLS hatasında `curl.exe -fL` ile elle ağırlık indirme komutu, kullanım (`configs/default.yaml`, `configs/fast.yaml`, `--max-frames`, `make_demo_gif.py`, `pytest`, `ruff`), çıktı formatı, tekrarlanabilir CPU hız tablosu (`GPU: Ölçülmedi`), 2 hafif örnek kare, dürüst kısıtlar (ID değişiminde takım etiketi gecikmesi + Kare 500 hakem örneği, COCO top kısıtı, saha dışı tespitler, sentetik testle doğrulanan sahne kesmesi, CPU hızı), sonraki adımlar (Roboflow `trackers` dahil) ve MIT lisans + CC BY 4.0 atıf bölümü. Kişisel bilgi (yerel kullanıcı yolu / e-posta) içermez.
- `configs/fast.yaml` — CPU'da hızlı önizleme için `detector.imgsz: 640` kullanan (diğer alanlarda kalibre edilmiş varsayılanları miras alan) profil eklendi.
- `scripts/make_demo_gif.py` — İşaretlenmiş videonun belirli bir zaman aralığından Pillow + OpenCV ile optimize edilmiş, `<= max_mb` (varsayılan `10.0 MiB`) boyutlu animasyonlu GIF üreten CLI ve `make_demo_gif()` fonksiyonu eklendi.
- `docs/assets/demo.gif` (`7.89 MiB`, `960×540`, `10 fps`, `72` kare / `7.2 sn`, penaltı koşusu + vuruş + gol + sevinç akışı) — `outputs/t11_final/annotated.mp4` dosyasından `--start 8.8 --duration 7.2 --width 960 --fps 10 --colors 128` ile üretildi.
- `docs/assets/t11_frame_200.jpg` (`70.2 KB`) & `docs/assets/t11_frame_500.jpg` (`80.8 KB`) — T11'deki ~1.7 MB tutan 3 PNG (`t11_frame_200.png`, `t11_frame_350.png`, `t11_frame_500.png`) silindi; repo boyutunu küçük tutmak için README'de kullanılan 2 örnek kare `<= 200 KB` JPG'ye çevrildi.
- `tests/test_config.py` — `configs/fast.yaml` yükleme ve `imgsz == 640` doğrulama testi eklendi.
- `tests/test_make_demo_gif.py` — Sentetik videodan GIF üretimi ve eksik dosyada temiz exit `1` hata mesajı birim testleri eklendi.
- `docs/PLAN.md` — T12 durumu `DOING` → `REVIEW` olarak güncellendi.

---

### Tekrarlanabilir CPU Performans Ölçümleri (`--max-frames 300`, `1920×1080`, `stride=1`)
- **CPU:** `11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz` (4 fiziksel / 8 mantıksal çekirdek, 15.7 GB RAM, PyTorch CPU, CUDA yok)
- **Ölçüm yöntemi:** Başka iş yokken ikişer ardışık koşu:
  - `configs/fast.yaml` (`imgsz=640`):
    - Koşu 1: `pass1_fps=6.35`, `pass1_sec=47.22`, `pass2_sec=6.69`, `elapsed_sec=54.13`
    - Koşu 2: `pass1_fps=6.20`, `pass1_sec=48.35`, `pass2_sec=6.54`, `elapsed_sec=55.10`
    - **Ortalama:** **`6.28 pass1_fps`** (`pass1: 47.8 sn`, `pass2: 6.6 sn`, `toplam: 54.6 sn`)
  - `configs/default.yaml` (`imgsz=1280`):
    - Koşu 1: `pass1_fps=2.41`, `pass1_sec=124.55`, `pass2_sec=6.35`, `elapsed_sec=131.13`
    - Koşu 2: `pass1_fps=2.34`, `pass1_sec=128.43`, `pass2_sec=6.29`, `elapsed_sec=134.93`
    - **Ortalama:** **`2.38 pass1_fps`** (`pass1: 126.5 sn`, `pass2: 6.3 sn`, `toplam: 133.0 sn`)
  - **GPU (CUDA):** Ölçülmedi.

*(İlk `640` denemesinde arka plan sistem yükü nedeniyle geçici olarak `2.07–2.56 fps` ve yüksek `pass2_sec` (`11.26 sn`) gözlenince saf çıkarım gecikmesi doğrulandı (`640`: `9.82 fps`, `1280`: `2.31 fps`) ve `640` ikili koşusu temiz sistemde tekrarlanarak tutarlı `6.35 / 6.20 fps` ölçüldü.)*

---

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe scripts\make_demo_gif.py --input outputs\t11_final\annotated.mp4 --output docs\assets\demo.gif --start 8.8 --duration 7.2 --width 960 --fps 10 --colors 128
-> Saved docs\assets\demo.gif (7.89 MiB)

.venv\Scripts\python.exe -m futbolcv --help
-> Exit 0 (tüm CLI seçenekleri README ile birebir uyumlu)

.venv\Scripts\python.exe scripts\make_demo_gif.py --help
-> Exit 0

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 27 files already formatted

.venv\Scripts\pytest.exe -q
-> 104 passed, 1 deselected in 13.29s

README PII kontrolü (Users / Projeler / e-posta):
-> PII check: []

docs/assets/ boyutları:
-> demo.gif: 8075.5 KB (7.89 MiB), t08_annotator_preview.png: 91.8 KB, t11_frame_200.jpg: 70.2 KB, t11_frame_500.jpg: 80.8 KB
```

**Kabul kriterleri**
- [x] `README.md` tüm PLAN T12 ve REVIEW.md bölümlerini içeriyor (tek cümlelik özet, demo GIF, zorunlu CC BY 4.0 atfı + "Değiştirildi: tespit/takip katmanları eklendi", özellikler, mimari, Windows `uv` + opsiyonel CUDA + CWD + TLS `curl.exe -fL` notu, `configs/fast.yaml`, kullanım, çıktı formatı, 2×300 kare tekrarlanabilir CPU performans tablosu + "GPU: Ölçülmedi", dürüst kısıtlar, sonraki adımlar, MIT lisans & atıflar; kişisel bilgi yok).
- [x] `scripts/make_demo_gif.py` çalışıyor ve `docs/assets/demo.gif` (`7.89 MiB <= 10 MiB`) üretildi.
- [x] T11 PNG'leri temizlendi; README'de kullanılan 2 kare `<= 200 KB` JPG (`70.2 KB` ve `80.8 KB`) olarak tutuldu.
- [x] `pytest -q` (104 passed, 1 slow deselected), `ruff check .`, `ruff format --check .` temiz.
- [x] Repoda video (`.mp4`/`.webm`) veya ağırlık (`.pt`) dosyası yok; T12 commit'i atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `configs/fast.yaml` yalnızca `detector.imgsz: 640` override'ını içerir (diğer tüm alanları `AppConfig` varsayılanlarından miras alır). Gerekçe: `default.yaml` ile gelecekte parametre kayması (config drift) riskini sıfırlar ve `test_load_fast_config` testiyle birebir eşitlik doğrulanır.
- `scripts/make_demo_gif.py` ek paket (`imageio`/`ffmpeg`) gerektirmemek için halihazırda bağımlılık grafiğinde bulunan `cv2` + `PIL` (Pillow) kullanır.

**Açık sorular / riskler**
- `T13 — (Opsiyonel) Futbol veri setiyle fine-tune` görevine geçilip geçilmeyeceği inceleyici/kullanıcı kararına bırakılmıştır.
