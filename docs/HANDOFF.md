# HANDOFF (Geliştirici → İnceleyici)

## Görev: T04 — Detector

**Değişen/eklenen dosyalar**
- `src/futbolcv/detector.py` — `Detector(cfg)` YOLO11 sarmalayıcısı (`detect(frame) -> tuple[sv.Detections, sv.Detections]`), `resolve_device("auto" -> "0"/"cpu")` ve yalın ağırlık adlarını `models/` altına yönlendiren `resolve_weights_path` eklendi.
- `tests/test_detector.py` — YOLO'yu mock'layarak modelin tek kez yüklenmesini, sınıf/güven ayrımını (`player_class_ids`, `ball_class_ids`, `conf`, `ball_conf`, `min_player_box_height_px`), boş sonuçları, cihaz ve ağırlık yolu çözümlemesini doğrulayan 5 hızlı birim testi + gerçek `yolo11s.pt` ağırlığıyla `ultralytics/assets/zidane.jpg` üzerinde ≥1 `person` bulan 1 `@pytest.mark.slow` testi eklendi.
- `docs/PLAN.md` — T04 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\pytest.exe -q
-> ..............................................                           [100%]
-> 46 passed, 1 deselected in 12.19s

.venv\Scripts\pytest.exe -m slow -v
-> tests/test_detector.py::test_detector_real_weights_finds_person PASSED   [100%]
-> 1 passed, 46 deselected in 22.94s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 19 files already formatted

git commit -m "feat: yolo11 detector wrapper (T04)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `Detector(cfg)` modeli `__init__` içinde yalnızca bir kez yükler; `detect(frame)` `(players, ball_candidates)` olarak `tuple[sv.Detections, sv.Detections]` döner; sınıf filtreleri config'den alınır.
- [x] `device="auto"` → CUDA varsa `"0"`, yoksa `"cpu"` (monkeypatch ile her iki dal test edildi).
- [x] Yalın `weights` adı (`yolo11s.pt`) `models/yolo11s.pt` olarak çözümlenir; proje kökünde `.pt` oluşmadığı ve `models/yolo11s.pt`'nin `git status`'ta görünmediği doğrulandı.
- [x] Mock'lu birim testleri sınıf/güven ayrımını doğrular; `@pytest.mark.slow` testi gerçek `yolo11s.pt` ağırlığıyla `zidane.jpg` üzerinde ≥1 `person` tespit etti ve varsayılan `pytest -q` çalıştırmasında atlandı.
- [x] `ruff check .`, `ruff format --check .` ve `pytest -q` temiz.
- [x] Görev sonu commit'i (`feat: yolo11 detector wrapper (T04)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `detect()` içinde YOLO'ya `conf=min(cfg.conf, cfg.ball_conf)` ve `classes=sorted(set(player_class_ids) | set(ball_class_ids))` geçilip sonuçlar sonrasında oyuncular için `cfg.conf` (+ opsiyonel `min_player_box_height_px`), top adayları için `cfg.ball_conf` ile ayrıştırıldı (R1 düşük top eşiği ve R4 kutu boyutu alt sınırı tek inference geçişinde desteklenir).
- `Detector.__init__` test edilebilirlik için opsiyonel `models_dir: Path = Path("models")` parametresi alır.
- `@pytest.mark.slow` testi repo içine telifli görsel eklememek için (R10) `ultralytics` paketiyle hazır gelen `assets/zidane.jpg` görselini kullanır.

**Açık sorular / riskler**
- **Windows sistem sertifikası (TLS):** Bu Windows ortamında `requests`/`certifi` GitHub release indirmesinde `SSLCertVerificationError` verdi (T01'deki `uv --system-certs` ile aynı kök neden). `yolo11s.pt`, Windows'un yerel TLS deposunu kullanan `curl.exe -fL` ile `models/yolo11s.pt` altına indirilerek `@pytest.mark.slow` doğrulandı. T12 README'de Windows/kurumsal ağ kullanıcıları için `models/yolo11s.pt` manuel indirme notu düşülebilir.
