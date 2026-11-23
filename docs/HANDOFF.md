# HANDOFF (Geliştirici → İnceleyici)

## Görev: T13d — Top takipçisinin sabit yanlış pozitife kilitlenmesini önleme

**Değişen/eklenen dosyalar**
- `src/futbolcv/config.py` & `configs/default.yaml` — `BallConfig` içine 4 yeni doğrulanmış alan eklendi (kodda sabit sayı yok; `default.yaml` ile birebir senkron):
  - `distance_weight: 0.3` (`>= 0.0`)
  - `static_radius_px: 5.0` (`> 0.0`)
  - `static_frames: 25` (`>= 1`)
  - `reacquire_conf: 0.5` (`0.0..1.0`)
- `src/futbolcv/ball.py` — `BallTracker` güncellendi:
  - `_StaticSpot` sabit nokta hafızası: ardışık karelerde `static_radius_px` içinde kalan adaylar aynı noktada birikir; `hits >= static_frames` olunca sabit nokta işaretlenir; `static_frames` kareden uzun süre görünmeyen noktalar unutulur; `reset()` hem takip geçmişini hem sabit nokta hafızasını temizler.
  - Durumlu `update()` seçim sırası:
    - **(a)** Sıçrama kapısı içindeki hareketli (sabit olmayan) adaylardan en yüksek `conf - distance_weight * (mesafe / izin_verilen_sıçrama)` puanlı olan seçilir.
    - **(b)** Kapı içinde hareketli aday yoksa **ve** mevcut takip sabit bir nokta üzerindeyse **ve** kapı dışında `conf >= reacquire_conf` olan hareketli bir aday varsa en yüksek güvenli hareketli adaya **yeniden kilitlenilir**.
    - **(c)** Hareketli aday seçilmezse sabit adaylar normal kapı ve puan kuralıyla değerlendirilir (penaltı hazırlığı / duran toplar kesintisiz takip edilir).
  - `select()` mevcut T06 sözleşmesini ve testlerini bozmadan (`static_mask=None, use_score=False`) korundu; ek opsiyonel parametreler (`static_mask`, `on_static_spot`, `use_score`) desteklendi.
- `tests/test_ball.py` & `tests/test_config.py` — 8 yeni test eklendi (`119 → 127` test):
  - Sentetik penaltı noktası testi (0–49 sabit nokta, 50+ 20 px/kare uzaklaşan gerçek top → 50. karede, yani `<= 3` kare içinde gerçek topa geçiş ve sonuna kadar koruma).
  - Kapı dışı yeniden kilitlenme testi (`conf=0.35 < 0.50` kilidi kıramaz, `conf=0.74 >= 0.50` 500+ px uzaktan anında kilidi kırar).
  - Tek sabit top (duran top / korner / penaltı hazırlığı) 60 kare kesintisiz takip testi.
  - Sabit noktanın `> static_frames` yoklukta unutulması ve `reset()` ile temizlenmesi testi.
  - Yeni `BallConfig` alanlarının varsayılan ve geçersiz değer testleri.
- `README.md` — Karşılaştırma tablosu T13d takipçi gerçek-top sayılarıyla güncellendi; **Bilinen Kısıtlar #2** kısmen çözüldü olarak dürüstçe güncellendi; **Sonraki Adımlar** yenilendi. Demo GIF (`f=220–400`, penaltı+gol penceresi) etkilenmediği için gereksiz binary şişkinliği yaratmamak adına değiştirilmedi.
- `docs/PLAN.md` — T13d durumu `DOING` → `REVIEW` olarak güncellendi.

---

### Gerçek Klip Doğrulaması (`data/ecuador_qatar_wc2022.webm`, `543` kare, tam klip)

| Koşu | `setup (0–275)` | `on_spot (300–542)` (≤15 px) | `on_spot (430–542)` (top orta sahada) | `midfield (430–542)` (gerçek top) | Gerçek top (nokta dışı — ham / görünür) | `summary.json` `detected / visible` |
|---|---:|---:|---:|---:|---:|---:|
| **Football ÖNCE (T13c)** | `276/276` | `201` | `90` | `20/113` (`17.7%`) | `315` (`58.0%`) / `342` (`63.0%`) | `516 (95.0%)` / `543 (100.0%)` |
| **Football SONRA (T13d)** | **`276/276`** | **`115`** *(-43%)* | **`10`** ***(-89%)*** | **`103/113`** ***(`91.2%`)*** | **`409` (`75.3%`)** / **`428` (`78.8%`)** | `524 (96.5%)` / `543 (100.0%)` |
| **COCO ÖNCE (T11)** | `150/276` | `19` | `0` | `104/113` | `215` (`39.6%`) / `323` (`59.5%`) | `222 (40.9%)` / `342 (63.0%)` |
| **COCO SONRA (T13d)** | **`150/276`** | **`19`** | **`0`** | **`104/113`** | **`215` (`39.6%`)** / **`323` (`59.5%`)** | `222 (40.9%)` / `342 (63.0%)` |

- **Görsel doğrulama (Kare 500, `outputs/t13d_football/annotated.mp4`):** T13c'de kaleci `46`'nın yanındaki boyalı penaltı noktasında kilitli kalan top üçgeni, T13d'de tam olarak orta sahada `3` numaralı oyuncunun önündeki gerçek topun üzerine taşındı.
- **Kalan kısım (dürüst rapor):** `f=300–429` arasında top ağların içinde / kalecinin kucağında tamamen görünmezken sahada başka hareketli aday olmadığından, duran top takibini (`276/276`) koruyan `(c)` kuralı bu ~105 karede boyalı penaltı noktasını seçmeye devam eder.
- **COCO regresyon kontrolü:** `outputs/t13d_coco` ile `outputs/t11_final` kare kare birebir aynı (`0` regresyon).

---

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe -m futbolcv --input data\ecuador_qatar_wc2022.webm --output outputs\t13d_football --config configs\football.yaml
-> Exit 0 | 543 kare | pass1_fps=2.59 | detected=524 (96.5%) | visible=543 (100.0%) | non_spot_raw=409 (75.3%) | non_spot_vis=428 (78.8%)

.venv\Scripts\python.exe -m futbolcv --input data\ecuador_qatar_wc2022.webm --output outputs\t13d_coco --config configs\default.yaml
-> Exit 0 | 543 kare | pass1_fps=1.92 | detected=222 (40.9%) | visible=342 (63.0%) (T11 ile birebir aynı)

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 30 files already formatted

.venv\Scripts\pytest.exe -q
-> 127 passed, 1 deselected
```

**Kabul kriterleri**
- [x] Sentetik penaltı noktası testi geçiyor (50. karede, `<= 3` kare içinde gerçek topa geçiş ve sonuna kadar takip).
- [x] Kapı dışı yeniden kilitlenme `(b)` kuralı test edildi ve gerçek klipte orta saha top kilidini kırdı (`f=430–542`: `90 → 10` penaltı noktası karesi, `20 → 103/113` gerçek orta saha topu).
- [x] Tek sabit top (penaltı hazırlığı / korner) seçilmeye devam ediyor (sentetik 60 kare testi + gerçek klip `276/276`).
- [x] Mevcut T06 testleri kırılmadı; COCO `default.yaml` tam klip regresyon kontrolü `0` regresyon verdi.
- [x] Yeni `BallConfig` alanları (`distance_weight`, `static_radius_px`, `static_frames`, `reacquire_conf`) doğrulamalı ve `default.yaml` ile senkron; kodda sabit sayı yok.
- [x] `README.md` karşılaştırma tablosu ve Bilinen Kısıtlar #2 dürüstçe güncellendi.
- [x] `pytest -q` (127 passed, 1 deselected), `ruff check .`, `ruff format --check .` temiz; T13d commit'i atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- **`static_radius_px: 5.0` (planda örnek verilen `4.0` yerine) ve nokta merkezinin eşleşen adayla güncellenmesi:** Gerçek yayın klibinde gol sonrası kamera yavaş pan yaptığı için boyalı penaltı noktası kare başına medyan `1.21 px`, p95 `4.41 px` kayıyor; buna karşılık orta sahadaki gerçek top medyan `4.77 px/kare` (p90 `14.48 px/kare`) hareket ediyor. `static_radius_px=4.0` ile pan sırasındaki 4.1–4.8 px'lik adımlar sabit nokta sayacını sıfırlayıp orta saha takibini `56/113`'te bırakırken, `5.0` ile ek sabit/karmaşıklık eklemeden orta saha takibi `103/113` (`91.2%`) seviyesine çıktı ve COCO'da sıfır regresyon verdi.

**Açık sorular / riskler**
- Yok (tüm plan görevleri T01–T13d tamamlandı; inceleme onayı sonrası kullanıcı repoyu GitHub'a push edip `<GITHUB_USER>/<REPO_NAME>` yer tutucularını güncelleyebilir).
