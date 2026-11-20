# HANDOFF (Geliştirici → İnceleyici)

## Görev: T06 — Top takibi + enterpolasyon

**Değişen/eklenen dosyalar**
- `src/futbolcv/types.py` — PLAN §2 veri sözleşmesine uygun `BallState(xyxy, conf, interpolated)` ve `FrameResult(frame_idx, players, ball)` dataclass'ları eklendi.
- `src/futbolcv/ball.py` — `BallTracker(cfg)` (`select(candidates, prev_xyxy, *, frames_since_last=1)`, durumlu `update(candidates, *, frame_idx=None)`, `reset()`) ve doğrusal boşluk doldurma fonksiyonu `interpolate(states, max_gap) -> list[BallState]` eklendi.
- `tests/test_ball.py` — (a) 3 karelik boşluk doğrusal koordinat kontrolü, (b) `max_gap` vs `max_gap+1` sınır kontrolü, (c) baştaki/sondaki boşlukların doldurulmaması, (d) 2 adaydan öncekine yakın olanın seçilmesi, önceki yokken en yüksek güven seçimi, kare başına sıçrama eşiği (`max_jump_px * frames_since_last`), durumlu `update`/`reset` ve `FrameResult` sözleşme testleri (10 test) eklendi.
- `docs/PLAN.md` — T06 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\pytest.exe -q
-> .....................................................................    [100%]
-> 69 passed, 1 deselected in 14.31s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 21 files already formatted

git commit -m "feat: ball tracker and interpolation (T06)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `src/futbolcv/types.py` içine §2'deki `BallState` ve `FrameResult` veri sözleşmesi yazıldı.
- [x] `BallTracker.select(candidates, prev_xyxy)`: birden çok adayda öncekine en yakını seçer (eşit mesafede yüksek güvenle kırar), `prev_xyxy=None` ise en yüksek güvenli adayı seçer; `max_jump_px * frames_since_last` üzerindeki sıçramaları reddeder.
- [x] `interpolate(states, max_gap)`: ≤ `max_gap` iç boşlukları doğrusal doldurur (`conf=0.0`, `interpolated=True`); `max_gap+1` ve baş/son boşlukları `None` kalır; girdi listesini değiştirmez.
- [x] Zorunlu 4 senaryo (a, b, c, d) ve ek kenar durumlar birim testleriyle doğrulandı.
- [x] `pytest -q`, `ruff check .`, `ruff format --check .` temiz; GPU/internet/ağırlık gerektirmiyor.
- [x] Görev sonu commit'i (`feat: ball tracker and interpolation (T06)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `BallTracker.select` imzasına varsayılanı `1` olan keyword-only `frames_since_last: int = 1` parametresi eklendi; ayrıca T10 pipeline'ın son görülmeden beri geçen kare sayısını temiz yönetmesi için `select()` üzerine kurulu durumlu `update(candidates, *, frame_idx=None)` ve `reset()` metotları eklendi.

**Açık sorular / riskler**
- Yok.
