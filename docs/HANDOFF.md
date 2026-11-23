# HANDOFF (Geliştirici → İnceleyici)

## Görev: T13d — Sabit yanlış pozitif kilidini önleme (REVIEW.md FIX sonrası)

**Değişen/eklenen dosyalar**
- `src/futbolcv/ball.py` — **"Terk edilmiş sabit nokta" (`abandoned`) kuralı** eklendi:
  - `_StaticSpot(eq=False)` içine `abandoned: bool = False` eklendi (yan etki notu docstring'de belgelendi: aynı noktaya yeniden konan top, unutma süresi dolana kadar takip edilmez).
  - Takipçi bir sabit noktadan hareketli bir adaya (`> 2 * static_radius_px` ayrılma, küçük kamera titreşimleri hariç) geçtiğinde o nokta `abandoned=True` olarak işaretlenir.
  - Terk edilmiş noktalar hem hareketli adaylardan hem `(c)` geri-düşüş kuralından hariç tutulur; başka hareketli aday yoksa `None` döner.
  - Terk edilmiş noktalar kısa örtüşmelerdeki yavaş kamera pan/tilt kaymasını yakalamak için `min(static_radius_px * dt, max_jump_px * 0.5)` eşleşme yarıçapı kullanır; `static_frames` kareden uzun süre görünmezlerse unutulurlar; `reset()` tüm sabit ve terk edilmiş noktaları temizler.
- `tests/test_ball.py` — `test_abandoned_static_spot_returns_none_when_only_spot_remains` eklendi: 0–49 sabit nokta seçilir → 50–59 gerçek top uzaklaşır (takip edilir, nokta terk edilir) → 60–99 yalnızca sabit nokta kalır ve tüm karelerde `None` döner → `reset()` sonrası hafıza temizlenir. Mevcut 60 karelik duran top testi dahil tüm testler geçer.
- `docs/assets/demo.gif` (`7.91 MiB <= 10 MiB`) — `outputs/t13d_fix_football/annotated.mp4` dosyasından yeniden üretildi; gol sonrası (`f=300–400`) boş penaltı noktasındaki sahte top üçgeni artık GIF'te görünmüyor (Kare 350 görsel olarak doğrulandı).
- `README.md` —
  - Tablo etiketi **"Gerçek top (takipçi çıktısı, penaltı noktası yanlış pozitifi hariç — ham / görünür)"** olarak düzeltildi; değerler **`78.1% / 84.3%`** (`424 / 458`) ve yoruma *"T13d öncesi: `58.0% / 63.0%`"* eklendi.
  - `ball.detected_ratio` (`79.0%`) ve `ball.visible_ratio` (`85.3%`) yeni çıktıya göre güncellendi.
  - **Bilinen Kısıtlar #2** yeni sonuca (`201 → 5`, `-97.5%`) ve terk edilmiş noktaya yeniden konan top (penaltı tekrarı) yan etkisine göre güncellendi.
- `docs/PLAN.md` — T13d durumu `FIX` → `REVIEW` olarak güncellendi.

---

### Gerçek Klip Karşılaştırması (`data/ecuador_qatar_wc2022.webm`, `543` kare, tam klip)

| Koşu | `setup (0–275)` | `spot (300–429)` | `spot (430–542)` | Toplam `spot (300–542)` | `midfield (430–542)` | Gerçek top (ham / görünür) | `summary.json` `det / vis` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Football T13c (ÖNCE)** | `276/276` | `111` | `90` | `201` | `20/113` | `315 (58.0%)` / `342 (63.0%)` | `516 (95.0%)` / `543 (100.0%)` |
| **Football T13d** | `276/276` | `105` | `10` | `115` | `103/113` | `409 (75.3%)` / `428 (78.8%)` | `524 (96.5%)` / `543 (100.0%)` |
| **Football T13d-fix** | **`276/276`** | **`5`** | **`0`** | **`5`** ***(-97.5%)*** | **`113/113`** ***(`100%`)*** | **`424 (78.1%)`** / **`458 (84.3%)`** | `429 (79.0%)` / `463 (85.3%)` |
| **COCO T11 (ÖNCE)** | `150/276` | `19` | `0` | `19` | `104/113` | `215 (39.6%)` / `323 (59.5%)` | `222 (40.9%)` / `342 (63.0%)` |
| **COCO T13d** | `150/276` | `19` | `0` | `19` | `104/113` | `215 (39.6%)` / `323 (59.5%)` | `222 (40.9%)` / `342 (63.0%)` |
| **COCO T13d-fix** | **`150/276`** | **`19`** | **`0`** | **`19`** | **`104/113`** | **`215 (39.6%)`** / **`323 (59.5%)`** | `222 (40.9%)` / `342 (63.0%)` |

- Gol sonrası penaltı noktası yanlış pozitifleri **`201 → 5` (`-97.5%`)**, `300–429` aralığı **`111 → 5`**, `430–542` aralığı **`90 → 0`** oldu; `visible_ratio` **`85.3%` (`< 100%`)** değerine indi.
- COCO `default.yaml` tam klip regresyon kontrolü kare kare birebir aynı (`0` regresyon).

---

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe -m futbolcv --input data\ecuador_qatar_wc2022.webm --output outputs\t13d_fix_football --config configs\football.yaml
-> Exit 0 | 543 kare | pass1_fps=3.01 | detected=429 (79.0%) | visible=463 (85.3%) | spot(300-542)=5 | midfield=113/113

.venv\Scripts\python.exe -m futbolcv --input data\ecuador_qatar_wc2022.webm --output outputs\t13d_fix_coco --config configs\default.yaml
-> Exit 0 | 543 kare | pass1_fps=2.80 | detected=222 (40.9%) | visible=342 (63.0%) (T11 ile birebir aynı)

.venv\Scripts\python.exe scripts\make_demo_gif.py --input outputs\t13d_fix_football\annotated.mp4 --output docs\assets\demo.gif --start 8.8 --duration 7.2 --width 960 --fps 10 --colors 128
-> Saved docs\assets\demo.gif (7.91 MiB)

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 30 files already formatted

.venv\Scripts\pytest.exe -q
-> 128 passed, 1 deselected in 10.22s
```

**Kabul kriterleri**
- [x] Terk edilmiş sabit nokta kuralı eklendi; unutma süresi (`static_frames`) ve `reset()` ile temizleniyor; penaltı tekrarı yan etkisi kod yorumu ve README kısıtlarında yazıldı.
- [x] Yeni sentetik test (0–49 sabit → 50–59 hareketli → 60–99 yalnızca sabit → `None`) ve 60 karelik duran top testi geçiyor (`128 passed`).
- [x] Gerçek klip `football.yaml` tam ölçümü: `spot(300–542)` `201 → 5` (`-97.5%`), `spot(300–429)` `111 → 5`, `visible_ratio` `85.3%` (`< 100%`), `midfield` `113/113`; `COCO` tam klip `0` regresyon.
- [x] README etiketi "Gerçek top (takipçi çıktısı, penaltı noktası yanlış pozitifi hariç)" olarak düzeltildi, T13d öncesi (`58.0%`) eklendi ve Kısıt #2 güncellendi.
- [x] `demo.gif` (`7.91 MiB`) temiz çıktıyla yenilendi; `ruff` ve `pytest` temiz; commit atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- Terk edilme eşiği olarak `departure > 2.0 * static_radius_px` (`10 px`) ve terk edilmiş noktanın kısa örtüşme sonrası yeniden eşleşmesi için `min(static_radius_px * dt, max_jump_px * 0.5)` yarıçapı kullanıldı. Gerekçe: (a) penaltı hazırlığında `f=64`'teki `5.39 px`'lik kamera titreşiminin gerçek duran topu yanlışlıkla terk edilmiş işaretlemesini önler (`setup=276/276` korunur), (b) şut sırasındaki 18 karelik (`f=279 → 297`) kamera tiltinde `16.4 px` kayan boyalı penaltı noktasının terk edilmiş olarak tanınmasını sağlar (`spot(300–429): 99 → 5`).

**Açık sorular / riskler**
- Yok.
