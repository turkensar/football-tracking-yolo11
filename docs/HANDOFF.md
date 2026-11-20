# HANDOFF (Geliştirici → İnceleyici)

## Görev: T02 — Config

**Değişen/eklenen dosyalar**
- `configs/default.yaml` — 6 bölüm (`detector`, `tracker`, `ball`, `team`, `annotate`, `output`) için varsayılan YAML ayarları eklendi.
- `src/futbolcv/config.py` — `DetectorConfig`, `TrackerConfig`, `BallConfig`, `TeamConfig`, `AnnotateConfig`, `OutputConfig`, `AppConfig` iç içe dataclass'ları, tip/aralık doğrulamaları ve `load_config(path) -> AppConfig` fonksiyonu yazıldı.
- `tests/test_config.py` — varsayılan dosya yükleme, eksik anahtar/boş YAML varsayılan dönüşü, bilinmeyen bölüm/anahtar `ValueError`, dosya yokluğu, hatalı tip/aralık doğrulama testleri (20 test) eklendi.
- `docs/PLAN.md` — T02 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe -m pytest -q
-> .....................                                                    [100%]
-> 21 passed in 1.28s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 17 files already formatted

git commit -m "feat: config loader (T02)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `configs/default.yaml` + `src/futbolcv/config.py` içinde 6 iç içe dataclass (`detector`, `tracker`, `ball`, `team`, `annotate`, `output`) ve `load_config(path) -> AppConfig` tanımlandı.
- [x] Asgari alanlar (`detector.weights`, `detector.conf`, `detector.imgsz`, `detector.device="auto"`, `detector.player_class_ids=[0]`, `detector.ball_class_ids=[32]`, `ball.max_gap_frames=15`, `team.n_init_frames=30`) mevcut ve varsayılan YAML ile dataclass varsayılanlarının birebir eşleştiği testle doğrulandı (`assert cfg == AppConfig()`).
- [x] Varsayılan dosya yükleme, bilinmeyen bölüm/anahtar için açıklayıcı `ValueError`, eksik anahtar/bölüm için varsayılan değer dönüşü birim testleriyle doğrulandı.
- [x] `ruff check .`, `ruff format --check .` ve `pytest -q` (GPU/internet/ağırlık gerektirmeden) temiz.
- [x] Görev sonu commit'i (`feat: config loader (T02)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `configs/default.yaml` ve dataclass'lara sonraki görevlerde (T03–T11) sihirli sayı kalmaması için ek makul alanlar eklendi: `detector.ball_conf` (R1), `detector.min_player_box_height_px` (R4, varsayılan `0`), `tracker.*` (`sv.ByteTrack.__init__` parametreleriyle birebir eşleşen alanlar), `ball.max_jump_px` (T06), `team.{color_space,vote_window,outlier_distance_threshold,jersey_top_ratio,jersey_center_crop_ratio,random_state}` (T07), `annotate.*` (T08 BGR renkleri + HUD), `output.*` (T03/T09 dosya adları, `codec`, `stride`).
- YAML'da `device: 0` gibi sayısal GPU indeksi yazıldığında hata vermemesi için `int` → `str` dönüşümü eklendi (`bool` hariç).

**Açık sorular / riskler**
- `detector.imgsz` varsayılanı CPU hızını koruyarak `640` bırakıldı (R1/T11'de top tespiti için `1280` denenmesi planlandığından). Varsayılanın baştan `1280` olmasını isterseniz tek satırla güncelleyebiliriz.
