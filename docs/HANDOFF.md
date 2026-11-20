# HANDOFF (Geliştirici → İnceleyici)

## Görev: T09 — Exporter

**Değişen/eklenen dosyalar**
- `src/futbolcv/export.py` — `CSV_COLUMNS`, `VideoInfo`, `export_tracks_csv`, `build_summary` ve `export_summary_json` eklendi; `tracks.csv` içinde top satırlarında `track_id` ve `team_id` boş hücre (`""`) olarak yazılır, `summary.json` video bilgisi, toplam kare, benzersiz oyuncu ID sayısı, takım başına baskın oyuncu ID sayıları, top görülme/enterpolasyon oranları ve işlem süresi/FPS alanlarını içerir.
- `tests/test_export.py` — CSV şeması/satır sayısı/boş top ID hücreleri, `summary.json` tüm alan/oran doğrulamaları, takım değişen bir `track_id`'nin baskın takımda tek sayılması, boş sonuç / sıfır süre koruması, geçersiz `elapsed_sec` hatası ve eksik opsiyonel oyuncu alanları testleri (5 test) eklendi.
- `docs/PLAN.md` — T09 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\pytest.exe -q
-> ........................................................................ [ 83%]
-> ..............                                                           [100%]
-> 86 passed, 1 deselected in 21.17s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 24 files already formatted

git commit -m "feat: csv tracks and json summary exporters (T09)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `tracks.csv` sütunları `frame, track_id, team_id, x1, y1, x2, y2, conf, kind, interpolated` sırasıyla üretilir; top satırlarında `track_id` ve `team_id` boş hücredir (`""`, `-1` değil).
- [x] `summary.json` video bilgisi (`source`, `width`, `height`, `source_fps`, `effective_fps`, `stride`), toplam kare, benzersiz oyuncu ID sayısı, takım başına ID sayısı, topun görüldüğü/enterpole edildiği kare sayısı & oranları, işlem süresi ve işlem FPS'ini içerir.
- [x] Sahte `FrameResult` listesinden üretilen CSV satır sayısı ve JSON alanları birim testleriyle doğrulandı.
- [x] `pytest -q`, `ruff check .`, `ruff format --check .` temiz; GPU/internet/ağırlık gerektirmiyor.
- [x] Görev sonu commit'i (`feat: csv tracks and json summary exporters (T09)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `player_ids_by_team` hesaplanırken bir oyuncunun takımı nadiren `-1` veya diğer takıma kaymışsa çift sayım olmaması için her benzersiz `track_id` kareler boyunca en sık aldığı **baskın takım** altında tek kez sayılır (`sum(player_ids_by_team.values()) == unique_player_ids` korunur).
- `ball` bölümünde hem ham tespit oranı (`detected_ratio`), hem enterpolasyon oranı (`interpolated_ratio`), hem de toplam görünürlük oranı (`visible_ratio`) ile kare sayıları ayrı ayrı verildi (T11 ve T13 karşılaştırmalarını netleştirir).

**Açık sorular / riskler**
- Yok.
