# HANDOFF (Geliştirici → İnceleyici)

## Görev: T13b — Futbol modeli için config + hakem sınıfı (+ T13a not defteri `test` split eklemesi)

> **Kullanıcıya not:** `notebooks/finetune_colab.ipynb` içindeki `test` split değerlendirmesi eklendi; **not defteri artık çalıştırılabilir.** Colab'da eğitimi tamamlayıp `models/football_yolo11s.pt` dosyasını ve `val_metrics.json` çıktısını aldığınızda T13c'ye geçebiliriz. (Roboflow sayfasını açtığınızda lisansın **CC BY 4.0** olduğunu gözle teyit etmeyi unutmayın.)

**Değişen/eklenen dosyalar**
- `notebooks/finetune_colab.ipynb` — REVIEW.md isteği doğrultusunda 5. hücre güncellendi: `best.pt` hem `split="val"` hem de `data.yaml`'da tanımlıysa `split="test"` üzerinde değerlendirilir; `val_metrics.json` `{"val": {"overall": ..., "per_class": ...}, "test": {"overall": ..., "per_class": ...}}` biçiminde kaydedilir (çıktı hücreleri boş, ruff temiz).
- `configs/football.yaml` — Fine-tune futbol modeli için profil eklendi: `weights: "football_yolo11s.pt"`, `ball_class_ids: [0]`, `player_class_ids: [1, 2]` (goalkeeper + player), `referee_class_ids: [3]`; diğer parametreler `AppConfig` varsayılanlarından miras alınır.
- `configs/default.yaml` & `src/futbolcv/config.py` — `DetectorConfig.referee_class_ids: list[int] = field(default_factory=list)` eklendi ve `_validate_int_list` ile doğrulandı; `default.yaml` / `fast.yaml` için `[]` olarak COCO davranışını korur.
- `src/futbolcv/detector.py` — `Detector.detect`, `player_class_ids | referee_class_ids` sınıflarını birlikte ilk `sv.Detections` içinde (orijinal `class_id` korunarak) döner; böylece hakemler oyuncularla birlikte takip edilir.
- `src/futbolcv/team.py` — `TeamClassifier.__init__` opsiyonel `referee_class_ids: Sequence[int] = ()` parametresi eklendi:
  - `add_init_frame`: `class_id in referee_class_ids` tespitlerini atlar (hakem formaları KMeans eğitimine girmez).
  - `predict`: `class_id in referee_class_ids` tespitlerine forma renginden ve önceki oy geçmişinden bağımsız olarak doğrudan `-1` atar.
- `src/futbolcv/pipeline.py` — `TeamClassifier(cfg.team, referee_class_ids=cfg.detector.referee_class_ids)` bağlantısı eklendi.
- `tests/test_config.py`, `tests/test_detector.py`, `tests/test_tracker.py`, `tests/test_team.py`, `tests/test_pipeline.py`, `tests/test_train.py` — 6 yeni test eklendi:
  - `test_load_football_config` + geçersiz `referee_class_ids` doğrulaması
  - `test_detector_includes_referee_classes_with_players`
  - `test_player_tracker_preserves_class_ids_for_players_and_referees` (`sv.ByteTrack`'in `class_id` koruduğunun doğrulaması)
  - `test_referee_class_excluded_from_kmeans_and_always_unknown` (aynı takım renginde ve eski takım oy geçmişli ID'de bile hakem → `-1`, KMeans merkezleri etkilenmez)
  - `test_pipeline_football_config_keeps_referee_unknown` (uçtan uca mocked futbol pipeline testi)
  - `test_colab_notebook_valid_clean_and_secret_safe` içine `evaluate_split("val")` + `evaluate_split("test")` kontrolü eklendi.
- `docs/PLAN.md` — T13b durumu `DOING` → `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 30 files already formatted

.venv\Scripts\pytest.exe -q
-> 119 passed, 1 deselected in 13.94s
```

**Kabul kriterleri**
- [x] `notebooks/finetune_colab.ipynb` 5. hücrede `test` split'i varsa `split="test"` ile de değerlendiriyor ve `val_metrics.json`'ı `{"val": ..., "test": ...}` olarak kaydediyor; çıktılar boş; **not defteri artık çalıştırılabilir.**
- [x] `configs/football.yaml` yükleniyor (`weights: football_yolo11s.pt`, `ball_class_ids: [0]`, `player_class_ids: [1, 2]`, `referee_class_ids: [3]`).
- [x] `DetectorConfig.referee_class_ids` eklendi; hakem tespitleri oyuncularla birlikte takip ediliyor, `sv.ByteTrack` `class_id` değerini koruyor (testle doğrulandı).
- [x] Hakem tespitleri KMeans eğitimine girmiyor ve `TeamClassifier.predict` içinde renkten ve önceki oy geçmişinden bağımsız olarak `-1` alıyor (birim ve uçtan uca mocked pipeline testiyle doğrulandı).
- [x] COCO profilleri (`default.yaml`, `fast.yaml`) `referee_class_ids=[]` ile değişmeden çalışıyor.
- [x] `pytest -q` (119 passed, 1 deselected), `ruff check .`, `ruff format --check .` temiz; T13b commit'i atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `TeamClassifier.predict` içinde hakem sınıfı (`class_id in referee_class_ids`) doğrudan `-1` döner ve o `tracker_id`'nin oy geçmişine (`_vote_history`) dokunmaz. Gerekçe: (a) bir hakem örtüşmede oyuncunun eski ID'sini devraldığında eski takımın oy penceresinden etkilenmeden anında `-1` olur, (b) tek karelik yanlış hakem sınıfı tespiti bir oyuncunun geçmişini kirletmez.

**Açık sorular / riskler**
- `T13c` görevine kullanıcı Colab eğitimini tamamlayıp `models/football_yolo11s.pt` dosyasını yerleştirene kadar **başlanmayacaktır**.
