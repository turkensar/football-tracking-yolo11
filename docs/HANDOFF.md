# HANDOFF (Geliştirici → İnceleyici)

## Görev: T05 — Oyuncu takibi

**Değişen/eklenen dosyalar**
- `configs/default.yaml` — `tracker.frame_rate` değeri `null` olarak güncellendi.
- `src/futbolcv/config.py` — `TrackerConfig.frame_rate` tipi `float | None = None` yapıldı ve yalnızca `None` değilken pozitif/sonlu sayı doğrulaması uygulandı.
- `src/futbolcv/tracker.py` — `PlayerTracker(cfg, fps)` (`update(dets) -> sv.Detections`, `reset()`) yazıldı; `supervision==0.30.x` sabitleme yorumu ve yalnızca `sv.ByteTrack(...)` kurulumuna daraltılmış `warnings.catch_warnings()` eklendi.
- `tests/test_config.py` — `tracker.frame_rate` varsayılan `None`, geçerli override (`50.0`) ve geçersiz değer (`0`, `true`) testleri eklendi.
- `tests/test_tracker.py` — 20 karelik 5 px/kare kayan 2 kutu ID sürekliliği, `reset()` sonrası ID sıfırlama, `fps` vs `cfg.frame_rate` override, boş tespit (`sv.Detections.empty()`), yerel `FutureWarning` susturma ve geçersiz `fps` testleri (10 test) eklendi.
- `docs/PLAN.md` — T05 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\pytest.exe -q
-> ...........................................................              [100%]
-> 59 passed, 1 deselected in 24.43s

.venv\Scripts\pytest.exe -W error tests/test_tracker.py -v
-> 10 passed in 4.93s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 20 files already formatted

git commit -m "feat: bytetrack player tracker (T05)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `PlayerTracker(cfg, fps)`: `update(dets) -> sv.Detections` (`tracker_id` dolu) ve `reset()` çalışıyor.
- [x] `tracker.frame_rate: float | None = None` (`configs/default.yaml`'da `null`); `None` ise `fps` parametresi, dolu ise config override'ı `sv.ByteTrack(..., frame_rate=...)` içine aktarılıyor.
- [x] `FutureWarning` yalnızca `sv.ByteTrack` oluşturma çağrısında `warnings.catch_warnings()` ile susturuldu (`pytest -W error` ve birim testiyle global filtreye dokunulmadığı doğrulandı); dosya başında 0.30.x sabitleme yorumu mevcut.
- [x] Sağa 5 px/kare kayan 2 sentetik kutu 20 kare boyunca aynı 2 `tracker_id`'yi koruyor; `reset()` sonrası ID'ler baştan başlıyor.
- [x] `pytest -q`, `ruff check .`, `ruff format --check .` temiz; GPU/internet/ağırlık gerektirmiyor.
- [x] Görev sonu commit'i (`feat: bytetrack player tracker (T05)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `update()` içinde `sv.Detections.empty()` (`confidence=None`) geldiğinde `sv.ByteTrack.update_with_detections()` `ValueError` fırlattığı için, boş tespitler boş `float32` `confidence` dizisiyle normalize edilip ByteTrack'e iletildi; böylece boş karelerde hata alınmaz ve ByteTrack'in iç `frame_id` sayacı / kayıp iz yaşlandırması doğru ilerler.

**Açık sorular / riskler**
- Yok.
