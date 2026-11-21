# futbol-cv

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

Futbol maç videolarında oyuncu ve top tespiti, kalıcı ID takibi ve forma rengine göre takım sınıflandırması yapan iki geçişli Python komut satırı aracı.

![Demo — Ekvador vs. Katar penaltı ve gol takibi](docs/assets/demo.gif)

> **Demo görüntü atfı:** *[2022 FIFA World Cup's first goal by Enner Valencia of Ecuador against Qatar](https://commons.wikimedia.org/wiki/File:2022_FIFA_World_Cup%27s_first_goal_by_Enner_Valencia_of_Ecuador_against_Qatar.webm)* — Yazar: **Mehr News Agency** ([mehrnews.com](https://www.mehrnews.com)), Lisans: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Değiştirildi: tespit/takip katmanları eklendi.**

---

## Özellikler

- **Oyuncu ve top tespiti:** Ultralytics YOLO11 (`yolo11s.pt`) ile ayrı oyuncu (`conf`) ve top (`ball_conf`) güven eşikleri; `device: "auto"` (CUDA varsa GPU, yoksa CPU).
- **Kalıcı oyuncu takibi:** `supervision` ByteTrack sarmalayıcısı; `frame_rate` değeri otomatik olarak videonun etkin FPS'inden (`fps / stride`) alınır.
- **Top takibi ve enterpolasyon:** Çoklu top adayında önceki konuma yakınlık + kare başına sıçrama sınırı (`max_jump_px × geçen_kare`); kısa kopukluklarda (`<= max_gap_frames`) doğrusal enterpolasyon.
- **Forma rengine göre takım sınıflandırma:** Üst gövde kırpımından yeşil çim pikselleri (`exclude_grass`) elenerek CIE Lab ortalama renk çıkarılır; ilk `n_init_frames` karede KMeans (`k=2`) eğitilir, `tracker_id` başına kayan pencere çoğunluk oyu uygulanır, küme merkezlerinden uzak hakem/kaleciler `-1` olarak etiketlenir.
- **Sahne kesmesi algılama:** Ardışık karelerde HSV histogram Bhattacharyya mesafesi eşiği aştığında oyuncu takibi, top takibi ve takım oy geçmişi sıfırlanır; top enterpolasyonu kesme sınırını aşmaz.
- **Bellek dostu iki geçişli akış:** İlk geçişte yalnızca kutular ve takip sonuçları tutulur (kareler RAM'de biriktirilmez); ikinci geçişte video yeniden okunarak çizim yapılır.
- **Yapılandırılmış çıktılar:** İşaretlenmiş MP4 (`annotated.mp4`), kare-kare tespit tablosu (`tracks.csv`) ve koşu özeti (`summary.json`).

---

## Mimari

```text
Girdi Video (.mp4 / .webm)
   │
   ├─► Isınma Ön Geçişi (ilk n_init_frames kare)
   │      YOLO11 tespit (önbelleğe alınır) ─► TeamClassifier.add_init_frame ─► KMeans(k=2) fit
   │
   ├─► Geçiş 1 — Tespit & Takip (tüm kareler)
   │      VideoReader ─► Sahne kesmesi (HSV Bhattacharyya) ─► kesmede tracker/ball/team reset
   │                 ─► Detector (YOLO11) ─► PlayerTracker (ByteTrack) ─► TeamClassifier.predict
   │                                      ─► BallTracker.update
   │                 ─► Kesme segmentlerine göre BallTracker.interpolate
   │
   └─► Geçiş 2 — Çizim & Dışa Aktarım
          VideoReader ─► Annotator (elips + ID rozeti + top üçgeni + HUD) ─► annotated.mp4
                      ─► Exporter ─► tracks.csv + summary.json
```

Proje `src/` yerleşimini kullanır:

- `src/futbolcv/config.py` — YAML yükleme ve tip/aralık doğrulaması (`configs/default.yaml`, `configs/fast.yaml`)
- `src/futbolcv/video_io.py` — Yeniden yinelenebilir `VideoReader` ve bağlam yöneticili `VideoWriter`
- `src/futbolcv/detector.py` — YOLO11 sarmalayıcısı (`models/` çözümlemesi, cihaz seçimi)
- `src/futbolcv/tracker.py` — `PlayerTracker` (`sv.ByteTrack` sarmalayıcısı)
- `src/futbolcv/ball.py` — `BallTracker` (aday seçimi, sıçrama filtresi, doğrusal enterpolasyon)
- `src/futbolcv/team.py` — `TeamClassifier` (çim maskeli gövde rengi, KMeans, çoğunluk oyu)
- `src/futbolcv/annotate.py` — `Annotator` (forma tonu canlılaştırma, parlaklığa duyarlı metin, dolu/boş top üçgeni)
- `src/futbolcv/export.py` — `tracks.csv` ve `summary.json` dışa aktarıcı
- `src/futbolcv/pipeline.py` & `src/futbolcv/cli.py` — İki geçişli orkestrasyon ve CLI

---

## Kurulum (Windows)

Gereksinimler: **Python 3.11** (`.python-version`) ve [uv](https://docs.astral.sh/uv/).

```powershell
git clone <repo-url>
cd futbol-cv

uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1

# Geliştirme ve test araçları dahil kurulum (önerilen)
uv pip install -r requirements-dev.txt -e .

# Yalnızca çalışma zamanı bağımlılıkları için:
# uv pip install -r requirements.txt -e .
```

TLS/sertifika hatası alırsanız sistem sertifikalarıyla kurun:

```powershell
uv pip install --system-certs -r requirements-dev.txt -e .
```

> **Önemli — Çalışma Dizini (CWD):** Tüm komutları **repo kök dizininden** (`futbol-cv/`) çalıştırın. `configs/default.yaml` içindeki yalın ağırlık adı (`yolo11s.pt`) çalışma dizinine göre `models/yolo11s.pt` olarak çözülür.

### Opsiyonel: NVIDIA GPU (CUDA) Kurulumu

`requirements.txt` bilinçli olarak `torch` içermez (`ultralytics` varsayılan olarak CPU sürümünü kurar). NVIDIA GPU kullanmak için [PyTorch yerel kurulum seçicisinden](https://pytorch.org/get-started/locally/) işletim sisteminize ve CUDA sürümünüze uygun `--index-url` değerini alın ve şu biçimde çalıştırın:

```powershell
uv pip install --reinstall torch torchvision --index-url <pytorch-cuda-index-url>
```

`detector.device: "auto"` CUDA kullanılabilir olduğunda otomatik olarak GPU (`0`), aksi halde CPU seçer.

### TLS / Sertifika Hatasında Model Ağırlığını Elle İndirme

Kurumsal proxy veya özel kök sertifika bulunan Windows ortamlarında Ultralytics otomatik ağırlık indirmesi `SSLCertVerificationError` / bağlantı hatası verebilir. Bu durumda ağırlığı Windows yerel TLS yığınıyla (`curl.exe`) doğrudan `models/` altına indirip tekrar çalıştırın:

```powershell
curl.exe -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt -o models\yolo11s.pt
```

---

## Kullanım

Komut satırı seçeneklerini görmek için:

```powershell
python -m futbolcv --help
```

### 1. Varsayılan Ayar (`imgsz: 1280` — yüksek doğruluk)

Küçük top ve uzak oyuncular için `configs/default.yaml` `imgsz: 1280` kullanır (GPU için uygundur; CPU'da yavaştır):

```powershell
python -m futbolcv --input data\mac.mp4 --output outputs\run1 --config configs\default.yaml
```

### 2. Hızlı CPU Önizleme (`configs/fast.yaml`, `imgsz: 640`)

CPU üzerinde hızlı deneme yapmak için `imgsz: 640` kullanan `configs/fast.yaml` (~2.6× daha hızlı) ve/veya `--max-frames` seçeneğini kullanın:

```powershell
python -m futbolcv --input data\mac.mp4 --output outputs\preview --config configs\fast.yaml --max-frames 300
```

Cihazı geçersiz kılmak için `--device cpu` veya `--device 0` verilebilir. Daha uzun videolarda `output.stride: 2` ayarlanarak her 2. kare işlenebilir.

### 3. Demo GIF Üretme

İşaretlenmiş çıktı videosundan ≤10 MiB boyutunda bir GIF üretmek için:

```powershell
python scripts\make_demo_gif.py --input outputs\run1\annotated.mp4 --output docs\assets\demo.gif --start 8.8 --duration 7.2 --width 960 --fps 10
```

### 4. Testler ve Kod Kalitesi

```powershell
# Hızlı birim testleri (GPU, internet ve model ağırlığı gerektirmez)
pytest -q

# Gerçek yolo11s.pt ağırlığıyla entegrasyon testi (models/yolo11s.pt gerekir)
pytest -m slow -v

# Lint ve biçim kontrolü
ruff check .
ruff format --check .
```

---

## Çıktı Formatı

Her çalıştırma `--output` klasöründe üç dosya üretir:

1. **`annotated.mp4`** — Oyuncu ayak altı takım elipsleri, ID rozetleri (`-1` nötr gri), top üçgeni (algılanan dolu, enterpole içi boş) ve sol üstte kare/FPS bilgisi.
2. **`tracks.csv`** — Kare bazında tüm oyuncu ve top kayıtları:
   ```text
   frame,track_id,team_id,x1,y1,x2,y2,conf,kind,interpolated
   ```
   Top satırlarında `track_id` ve `team_id` boştur; `kind` alanı `player` veya `ball` değerini alır.
3. **`summary.json`** — Video metadatası, benzersiz oyuncu ID sayıları (`player_ids_by_team`), top tespit/enterpolasyon oranları ve aşama süreleri (`elapsed_sec`, `model_load_sec`, `pass1_sec`, `pass1_fps`, `pass2_sec`).

---

## Sonuçlar ve Performans

Ölçümler 1920×1080, 25 fps açık lisanslı yayın klibi üzerinde (`stride=1`) yapılmıştır.

### Hız Kıyaslaması (`--max-frames 300`, 2 koşu ortalaması)

| Donanım | Yapılandırma | `imgsz` | `pass1_fps` (ort.) | `pass1` (sn) | `pass2` (sn) | Toplam (sn) |
|---|---|---:|---:|---:|---:|---:|
| CPU — `11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz` (4C/8T, 15.7 GB RAM) | `configs/fast.yaml` | 640 | **6.28** *(6.35 / 6.20)* | 47.8 | 6.6 | 54.6 |
| CPU — `11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz` (4C/8T, 15.7 GB RAM) | `configs/default.yaml` | 1280 | **2.38** *(2.41 / 2.34)* | 126.5 | 6.3 | 133.0 |
| GPU (CUDA) | `configs/default.yaml` | 1280 | Ölçülmedi | Ölçülmedi | Ölçülmedi | Ölçülmedi |

*Not: `640` hızlı önizleme için uygundur ancak geniş yayın açısında uzak oyuncuları ve topu daha sık kaçırır, saha kenarı tripodunu yanlışlıkla oyuncu sanabilir.*

### Tam Klip Özeti (`543` kare / `21.72 sn`, `configs/default.yaml`)

- **Benzersiz oyuncu ID:** `48` (Takım 0: `23`, Takım 1: `16`, Uç değer `-1`: `9` — ByteTrack ayarı ile varsayılan `90`'dan `%47` düşürüldü)
- **Top görünürlüğü:** Ham tespit `%40.9` (`222/543`), enterpolasyon `+22.1%` (`120/543`), toplam görünürlük **`%63.0`** (`342/543`)

| Penaltı anı (Kare 200) | Orta saha akışı (Kare 500) |
|---|---|
| ![Kare 200 — penaltı](docs/assets/t11_frame_200.jpg) | ![Kare 500 — orta saha](docs/assets/t11_frame_500.jpg) |
| Penaltı noktasında top işaretli; Ekvador (sarı) ve Katar (bordo) ayrışıyor; siyah hakem `12` gri (`-1`). | Top `29` önünde işaretli; turkuaz kaleci `37` gri (`-1`); saha dışı görevliler (`44`, `45`) ve ID değişimi sonrası hakem (`21`) gecikmesi görülüyor. |

---

## Bilinen Kısıtlar

1. **Örtüşmede ID değişimi ve takım etiketi gecikmesi:** ByteTrack yalnızca hareket (Kalman) ve IoU kullandığından ceza sahası kalabalığında ID değişimleri oluşur (22 oyuncu yerine 543 karede 48 benzersiz ID). Bir oyuncu veya hakem başka birinin ID'sini devraldığında kayan çoğunluk oyu penceresi (`vote_window: 15`) eski takımı birkaç kare taşır — **ID değişiminde takım etiketi birkaç kare gecikmeli düzelir** (ör. Kare 500'de siyah formalı hakem `21` geçici olarak bordo rozet taşıyor).
2. **Hazır COCO modeliyle top tespiti:** `yolo11s.pt` futbol özelinde eğitilmediğinden hızlı şut ve hareket bulanıklığında topu kaçırabilir (`%40.9` ham tespit, enterpolasyonla `%63.0`); düşük `ball_conf` (0.12) nedeniyle nadiren çömelen oyuncu veya reklam panolarında geçici yanlış pozitifler görülebilir.
3. **Saha dışı `person` tespitleri:** Saha sınır maskesi olmadığı için reklam panosu arkasındaki fotoğrafçı/görevliler (Kare 500'de `44`, `45`) ve bazı açılarda saha tripodu `person` olarak yakalanabilir.
4. **Sahne kesmesi algılama:** Test klibi tek kesintisiz kamera çekimi olduğu için `scene_cut_threshold: 0.4` yalnızca normal kamera panında yanlış pozitif üretmediği (`max=0.069`) yönünde test edildi; **kesme algılama sentetik testle doğrulandı, gerçek kesme içeren klipte doğrulanmadı.**
5. **CPU hızı:** `imgsz: 1280` CPU'da ~2.38 fps'tir (uzun videolarda ısınma/arka plan yüküyle daha da düşebilir). CPU'da hızlı önizleme için `configs/fast.yaml`, `--max-frames` veya `output.stride: 2` kullanın.

---

## Sonraki Adımlar (Future Work)

- `supervision 0.30.x`'te deprecated olan `sv.ByteTrack` yerine Roboflow `trackers` paketine geçiş
- Açık lisanslı futbol veri setiyle (`player`, `goalkeeper`, `referee`, `ball`) YOLO11 fine-tune
- Saha anahtar noktalarıyla homografi ve 2D kuşbakışı taktik radar
- Saha maskesiyle seyirci/görevli filtreleme ve Re-ID öznitelikleriyle ID değişimi azaltma
- Oyuncu hızı, kat edilen mesafe, top sahipliği yüzdesi ve Streamlit etkileşimli arayüzü

---

## Lisans ve Atıflar

- **Kod lisansı:** [GNU AGPL-3.0](LICENSE) — Ultralytics YOLO AGPL-3.0 lisanslı olduğu için proje kodu da GNU AGPL-3.0 altında lisanslanmıştır.
- **Demo görselleri ve GIF kaynağı:**
  - Başlık: *[2022 FIFA World Cup's first goal by Enner Valencia of Ecuador against Qatar](https://commons.wikimedia.org/wiki/File:2022_FIFA_World_Cup%27s_first_goal_by_Enner_Valencia_of_Ecuador_against_Qatar.webm)*
  - Yazar: **Mehr News Agency** ([mehrnews.com](https://www.mehrnews.com))
  - Lisans: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
  - Değişiklik bildirimi: **Değiştirildi: tespit/takip katmanları eklendi** (kırpılmış/ölçeklenmiş kareler üzerine YOLO11, ByteTrack ve takım katmanları çizilmiştir).
- **Üçüncü parti kütüphaneler:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), [Roboflow Supervision](https://github.com/roboflow/supervision), OpenCV, scikit-learn, NumPy, Pillow.
