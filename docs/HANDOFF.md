# HANDOFF (Geliştirici → İnceleyici)

## Görev: T10 — Pipeline + CLI (REVIEW.md FIX sonrası)

**Değişen/eklenen dosyalar**
- `src/futbolcv/cli.py` — `ConnectionError` (ve genel `OSError`) yakalandı; ağırlık indirme hatasında traceback basmadan `models/` altına elle indirme talimatı ve GitHub release URL'si (`Error: failed to download model weights (<weights>). Download manually and place under models/: <url>`) yazdırılıp çıkış kodu `1` dönülüyor.
- `src/futbolcv/export.py` — `build_summary` ve `export_summary_json` imzalarına `timings: Mapping[str, float] | None = None` eklendi; `summary.json` → `processing` altında `elapsed_sec`, `model_load_sec`, `pass1_sec`, `pass1_fps` ve `pass2_sec` alanları üretiliyor (geçersiz anahtar/negatif/NaN değerler `ValueError` verir).
- `src/futbolcv/pipeline.py` — `model_load_sec`, `pass1_sec` (ısınma tespitleri + Pass 1 takip/takım + top enterpolasyonu; model yükleme hariç, kare başına tek tespit) ve `pass2_sec` (çizim + MP4 yazma) süreleri ölçülüp `export_summary_json(..., timings=...)` içine aktarılıyor.
- `tests/test_cli.py` — `Detector` `ConnectionError` fırlattığında exit `1`, traceback yokluğu ve `stderr`'de `models/` + URL mesajını doğrulayan test eklendi.
- `tests/test_export.py` & `tests/test_pipeline.py` — yeni `processing` şeması (`elapsed_sec`, `model_load_sec`, `pass1_sec`, `pass1_fps`, `pass2_sec`) ve geçersiz `timings` doğrulamaları güncellendi.
- `docs/PLAN.md` — T10 durumu `FIX` → `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\pytest.exe -q
-> ........................................................................ [ 73%]
-> ..........................                                               [100%]
-> 98 passed, 1 deselected in 30.57s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 26 files already formatted

git commit -m "fix: handle download errors and stage timings in summary (T10)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `Detector` `ConnectionError` fırlattığında CLI traceback basmadan anlaşılır hata mesajı (`models/` + elle indirme URL'si) verip çıkış kodu `1` döner (birim testiyle doğrulandı).
- [x] `summary.json` → `processing` altında `elapsed_sec`, `model_load_sec`, `pass1_sec`, `pass1_fps` ve `pass2_sec` bulunur; `pass1_fps` model yükleme süresinden arındırılmış gerçek tespit+takip hızını raporlar.
- [x] `pytest -q` (98 passed, 1 slow deselected), `ruff check .`, `ruff format --check .` temiz.
- [x] Düzeltme commit'i (`fix: handle download errors and stage timings in summary (T10)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `pass1_sec`, ısınma ön geçişindeki YOLO tespitleri + Pass 1 takip/takım döngüsü + top enterpolasyonunun toplam süresini ölçer. Sebep: ısınma ön geçişinde önbelleklenen tespitler kısa kliplerde tüm kareleri kapsayabildiğinden yalnızca ikinci döngüyü ölçmek `pass1_fps`'i yapay olarak aşırı yükseltirdi; birlikte ölçmek her kare için tam olarak bir kez yapılan tespit+takip hızını dürüstçe yansıtır.

**Açık sorular / riskler**
- Yok. (REVIEW.md'deki T11 notları — kare kare Bhattacharyya mesafe dağılımı ve geniş planla başlayan klip seçimi — T11 için not alındı.)
