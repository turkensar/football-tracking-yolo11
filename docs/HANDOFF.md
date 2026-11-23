# HANDOFF (Geliştirici → İnceleyici)

## Görev: T13c — Karşılaştırma + README

**Değişen/eklenen dosyalar**
- `configs/football.yaml` — Fine-tune futbol modelinin top güven dağılımı (`0.54–0.84`) doğrultusunda `detector.ball_conf: 0.25` olarak ayarlandı (düşük güvenli 119 gürültülü aday elendi, gerçek top yakalaması korundu). Sınıf sırası ağırlıktan doğrudan doğrulandı: `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}`.
- `tests/test_config.py` — `test_load_football_config` içine `ball_conf == 0.25` kontrolü eklendi.
- `docs/assets/demo.gif` (`7.93 MiB`, `960×540`, `10 fps`, `7.2 sn`) — `outputs/t13c_football_final/annotated.mp4` dosyasından yeniden üretildi (penaltı şutunu havada takip ediyor, uzak oyuncuları yakalıyor, saha dışı fotoğrafçılar yok, hakemler gri `-1`).
- `docs/assets/t13c_frame_285.jpg` (`70.0 KB`) — Fine-tune modelin penaltı şutunu havada yakaladığı, orta (`16`) ve yan (`13`) hakemi doğrudan `-1` (gri) etiketlediği örnek kare.
- `README.md` —
  - Yeni **Futbol Veri Setiyle Fine-Tune (Colab GPU)** bölümü: Open in Colab rozeti (yer tutucu), veri seti + CC BY 4.0 lisansı, eğitim parametreleri, `test` split mAP tablosu (`val` genel özeti dipnotta), `scripts/train.py` komutu ve GitHub Release yer tutuculu indirme talimatı (`models/football_yolo11s.pt`).
  - **Kullanım** ve **Performans** bölümlerine `configs/football.yaml` (300 kare CPU kıyaslama satırı dahil) ve tam 543 karelik **COCO (`default.yaml`) vs. Fine-Tune (`football.yaml`)** karşılaştırma tablosu eklendi.
  - **Bilinen Kısıtlar** bölümüne boyalı penaltı noktası yanlış pozitifi ve `BallTracker.select()` kilitlenmesi dürüstçe eklendi (ham `%95.0` vs. gerçek top `%75.7+` ayrımıyla).
  - **Lisans ve Atıflar** bölümüne Roboflow Universe `football-players-detection-3zvbc` (CC BY 4.0) veri seti atfı eklendi; PII yok.
- `docs/PLAN.md` — T13c durumu `DOING` → `REVIEW` olarak güncellendi.

---

### Colab Eğitim Sonuçları (`outputs/val_metrics.json`, gitignored)
- **Ağırlık (gitignored):** `models/football_yolo11s.pt` (`19.24 MB`), sınıf haritası doğrulandı: `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}`.
- **`test` split (README'de kullanılan birincil tablo):**
  - Genel: `mAP50 = 0.8627`, `mAP50-95 = 0.6513`, `Precision = 0.8559`, `Recall = 0.8494`
  - `ball (0)`: `AP50 = 0.5319`, `AP50-95 = 0.2667`
  - `goalkeeper (1)`: `AP50 = 0.9575`, `AP50-95 = 0.8067`
  - `player (2)`: `AP50 = 0.9906`, `AP50-95 = 0.8168`
  - `referee (3)`: `AP50 = 0.9709`, `AP50-95 = 0.7149`
- **`val` split:** Genel `mAP50 = 0.8781`, `mAP50-95 = 0.6721`, `ball AP50 = 0.5616`.

---

### T11 Klibinde Karşılaştırma (`data/ecuador_qatar_wc2022.webm`, `543` kare / `21.72 sn`)

| Metrik | COCO (`configs/default.yaml`, `outputs/t11_final`) | Fine-Tune (`configs/football.yaml`, `outputs/t13c_football_final`) |
|---|---:|---:|
| `ball.detected_ratio` | `0.4088` (`222/543`) | **`0.9503`** (`516/543`)* |
| `ball.interpolated_ratio` | `0.2210` (`120/543`) | `0.0497` (`27/543`) |
| `ball.visible_ratio` | `0.6298` (`342/543`) | **`1.0000`** (`543/543`)* |
| `unique_player_ids` | `48` | `64` |
| `player_ids_by_team` | `{"0": 23, "1": 16, "-1": 9}` | `{"0": 26, "1": 22, "-1": 16}` |
| `pass1_fps` (tam 543 kare) | `1.22` | **`3.18`** |
| `pass1_fps` (`--max-frames 300`, adil CPU) | `2.38` *(2 koşu ort.)* | **`2.45`** (`outputs/t13c_bench_football_300`) |

*\*Dürüst analiz — gerçek top vs. boyalı penaltı noktası FP'si:*
- Fine-tune model penaltı hazırlığında (`f=0..275`) gerçek topu **`276/276`** karede (`c=0.73–0.84`), penaltı şutu havadayken (`f=285, 290`, COCO tamamen kaçırıyordu) ve gol sonrası orta saha yuvarlanmasında (`f=300..542`) **`135/243`** karede yakalıyor — yani **gerçek top ham oranı `%75.7+`** (COCO'nun `%40.9` değerinin neredeyse 2× üstünde!).
- Ancak gol sonrası penaltı noktası boş kaldığında fine-tune model beyaz penaltı noktasını `ball` (`c=0.50–0.76`) olarak tespit ediyor; `BallTracker.select()` yalnızca önceki konuma en yakın adayı seçtiği için `f=440..520` arasında gerçek top daha yüksek güvenle (`c=0.70–0.77`) yakalansa bile takipçi bir süre penaltı noktasında kilitli kalıyor. Bu durum README'de **Bilinen Kısıtlar** ve **Sonraki Adımlar** bölümlerinde açıkça belgelendi.

#### `ball_conf` Taraması (543 kare, `football_yolo11s.pt`)
- `0.12`: `raw_cands=819`, `detected=521 (95.9%)`, `visible=543 (100.0%)`
- **`0.25` (seçilen):** `raw_cands=700`, `detected=516 (95.0%)`, `visible=543 (100.0%)` — 119 düşük güvenli gürültü elendi, gerçek top tespitleri (`0.54–0.84`) eksiksiz korundu.
- `0.40`: `raw_cands=634`, `detected=508 (93.6%)`, `visible=543 (100.0%)`

---

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe -m futbolcv --input data\ecuador_qatar_wc2022.webm --output outputs\t13c_football_final --config configs\football.yaml
-> Exit 0 | 543 kare | pass1_fps=3.18 | ball.detected_ratio=0.9503 | ball.visible_ratio=1.0

.venv\Scripts\python.exe -m futbolcv --input data\ecuador_qatar_wc2022.webm --output outputs\t13c_bench_football_300 --config configs\football.yaml --max-frames 300
-> Exit 0 | 300 kare | pass1_fps=2.45 | pass1=122.59s | pass2=6.04s | elapsed=128.86s

.venv\Scripts\python.exe scripts\make_demo_gif.py --input outputs\t13c_football_final\annotated.mp4 --output docs\assets\demo.gif --start 8.8 --duration 7.2 --width 960 --fps 10 --colors 128
-> Saved docs\assets\demo.gif (7.93 MiB)

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 30 files already formatted

.venv\Scripts\pytest.exe -q
-> 119 passed, 1 deselected in 17.74s

README PII kontrolü: []
docs/assets/ boyutları: demo.gif 8116.2 KB (7.93 MiB), t08_annotator_preview.png 91.8 KB, t11_frame_200.jpg 70.2 KB, t11_frame_500.jpg 80.8 KB, t13c_frame_285.jpg 70.0 KB
```

**Kabul kriterleri**
- [x] T11 klibinde `default.yaml` ve `football.yaml` tam 543 karede (ve 300 karelik adil CPU hız testinde) karşılaştırıldı; `ball_conf: 0.25` ayarlandı.
- [x] Top görünürlüğü COCO modeline göre aynı klipte belirgin arttı (`%40.9 → %95.0` ham / `%75.7+` gerçek top; `%63.0 → %100.0` görünürlük) ve boyalı penaltı noktası FP kilitlenmesi dürüstçe analiz edildi.
- [x] `README.md` güncellendi: fine-tune eğitim bölümü (CC BY 4.0 veri seti atfı, Colab rozeti, parametreler, `test` split mAP tablosu, GitHub Release indirme yer tutucusu), COCO vs. futbol karşılaştırma tablosu, kısıtlar ve atıflar.
- [x] `docs/assets/demo.gif` (`7.93 MiB <= 10 MiB`) fine-tune çıktıdan yeniden üretildi.
- [x] Ağırlık dosyası (`models/football_yolo11s.pt`) ve videolar git'te yok; `pytest -q` (119 passed), `ruff check .`, `ruff format --check .` temiz; T13c commit'i atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `BallTracker.select()` bu görevde değiştirilmedi; boyalı penaltı noktası FP kilitlenmesi verisiyle tespit edilip hem README "Bilinen Kısıtlar" hem "Sonraki Adımlar" bölümüne şeffaf biçimde yazıldı (T13c kapsamı ölçüm + config + README olduğundan takipçi algoritmasını değiştirmekten kaçınıldı).

**Açık sorular / riskler**
- GitHub push ve Release oluşturma kullanıcıda: push sonrasında `README.md` ve `notebooks/finetune_colab.ipynb` içindeki `<GITHUB_USER>/<REPO_NAME>` yer tutucuları gerçek repo adıyla doldurulacak ve `models/football_yolo11s.pt` GitHub Release varlığı olarak yüklenecektir.
