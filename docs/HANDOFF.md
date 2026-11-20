# HANDOFF (Geliştirici → İnceleyici)

## Görev: T10 — Pipeline + CLI

**Değişen/eklenen dosyalar**
- `configs/default.yaml` & `src/futbolcv/config.py` — sahne kesmesi eşiği için `tracker.scene_cut_threshold: 0.6` (`float | None`, `(0, 1]`) eklendi (R3).
- `src/futbolcv/pipeline.py` — `run(input_path, output_dir, cfg, max_frames=None, *, detector=None, show_progress=True) -> Path` yazıldı:
  - Etkin FPS (`source_fps / stride`), geçersiz/0 FPS'te `UserWarning` + `25.0` fallback, `frame_count` yalnızca `tqdm` tahmini.
  - İlk `n_init_frames` karede takım ısınma ön geçişi (`add_init_frame` → `fit()`), ana geçiş 0. kareden başlar (tespit kutuları önbelleklenerek çift YOLO çıkarımı önlenir, kareler RAM'de tutulmaz — R7).
  - HSV H-S histogramı Bhattacharyya mesafesi ile sahne kesmesi tespiti; kesmede `PlayerTracker.reset()`, `BallTracker.reset()`, `TeamClassifier.reset_history()` birlikte çağrılır ve top enterpolasyonu kesme sınırlarında bölünerek her parça ayrı enterpole edilir (`max(1, max_gap_frames // stride)`).
  - Pass 2'de video yeniden okunup çizilir, MP4 + `tracks.csv` + `summary.json` (`VideoInfo.source = src_path.name`) yazılır.
- `src/futbolcv/cli.py` & `src/futbolcv/__main__.py` — `python -m futbolcv --input ... --output ... --config ... [--max-frames N] [--device DEV]` arayüzü; `--help` ve yol/config/bayrak hatalarında PyTorch yüklemeden hızlı ve traceback'siz `Error: ...` çıktısı.
- `tests/test_config.py` — `tracker.scene_cut_threshold` sınır doğrulama testleri eklendi.
- `tests/test_pipeline.py` — mock Detector ile 10 karelik uçtan uca MP4+CSV+JSON + 0. kare takım ısınması + top enterpolasyonu, `stride=2` etkin FPS ve boşluk ölçekleme, sahne kesmesinde 3 reset + kesme aşmayan enterpolasyon, `fps=0` fallback uyarısı ve `max_frames` testleri (5 test) eklendi.
- `tests/test_cli.py` — `python -m futbolcv --help` alt süreç testi, olmayan video/config/hatalı bayraklarda traceback'siz hata mesajı testleri ve mock Detector ile uçtan uca CLI testi (4 test) eklendi.
- `docs/PLAN.md` — T10 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe -m futbolcv --help
-> usage: python -m futbolcv [-h] --input INPUT [--output OUTPUT] [--config CONFIG] [--max-frames MAX_FRAMES] [--device DEVICE] (exit code 0)

.venv\Scripts\pytest.exe -q
-> ........................................................................ [ 74%]
-> .........................                                                [100%]
-> 97 passed, 1 deselected in 24.21s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 26 files already formatted

git commit -m "feat: end-to-end analysis pipeline (T10)"   # 402dfe3
git commit -m "feat: cli entrypoint for futbolcv (T10)"    # (bu teslim)
-> Push edilmedi.
```

**Kabul kriterleri**
- [x] `run(...) -> Path` iki geçişli akış + takım ısınma ön geçişi + `tqdm` ilerleme çubuğu ile çalışıyor; bellekte tam kare tutulmuyor.
- [x] Etkin FPS (`source_fps / stride`) hem çıktı videosuna hem ByteTrack'e veriliyor; `fps=0/NaN` durumunda uyarı + `25.0` fallback; `frame_count` yalnızca `tqdm` için.
- [x] Top için `BallTracker.update(..., frame_idx=<kaynak kare>)` ve enterpolasyon için `max(1, max_gap_frames // stride)` kullanılıyor.
- [x] Sahne kesmesinde 3 reset (`PlayerTracker.reset`, `BallTracker.reset`, `TeamClassifier.reset_history`) birlikte çağrılıyor ve enterpolasyon kesme sınırını aşmıyor.
- [x] `VideoInfo.source` yalnızca dosya adı (`Path(input_path).name`), tam yol sızdırılmıyor.
- [x] Detector mock'lu testler 10 karelik sentetik videoda MP4 + CSV + JSON üretiyor; `--help` çalışıyor; hatalı yol/config traceback basmadan anlaşılır hata mesajı verip çıkış kodu `1` dönüyor.
- [x] `pytest -q`, `ruff check .`, `ruff format --check .` temiz; GPU/internet/ağırlık gerektirmiyor.
- [x] İki odaklı commit atıldı (`feat: end-to-end analysis pipeline (T10)` ve `feat: cli entrypoint for futbolcv (T10)`), push edilmedi.

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- Python yerleşik `input()` fonksiyonunu gölgelememek için `pipeline.run` ilk parametre adı `input_path` yapıldı (konumsal çağrı `run(path, out_dir, cfg, max_frames=...)` aynı kalır); test enjeksiyonu için opsiyonel `detector` ve `show_progress` keyword-only parametreleri eklendi.
- Isınma ön geçişinde ilk `n_init_frames` karede elde edilen hafif `sv.Detections` kutuları (kareler değil) önbelleklenip Pass 1'de yeniden kullanıldı; böylece ilk 30 kare için çift YOLO çıkarımı önlendi (R8 CPU hızı + R7 bellek güvenliği).
- Sahne kesmesi için `tracker.scene_cut_threshold: 0.6` config alanı eklendi (kodda sihirli sayı bırakmamak için §2).
- `cli.py` içinde `from futbolcv.pipeline import run` çağrısı argüman/yol/config doğrulamasından sonra (lazy) yapılır; böylece `--help` ve hatalı yol kontrolleri PyTorch yüklemeden anında yanıt verir.

**Açık sorular / riskler**
- `tracker.scene_cut_threshold = 0.6` (HSV H-S histogram Bhattacharyya mesafesi) sentetik sahne kesmesinde doğrulandı; T11'de gerçek yayın klibinde kamera kesmeleri / hızlı pan hareketlerine göre gerçek değer gözlemlenecek.
