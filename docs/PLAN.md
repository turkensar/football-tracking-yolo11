# futbol-cv — Proje Planı

> Mimar/İnceleyici: Claude · Geliştirici: Gemini
> Haberleşme yalnızca `docs/` üzerinden: **PLAN.md** (bu dosya, görevler + durum), **HANDOFF.md** (geliştirici → inceleyici), **REVIEW.md** (inceleyici → geliştirici).

## 0. Hedef

Bir futbol maç videosunu girdi olarak alıp:
1. Oyuncuları (ve hakemi) tespit eden ve **kalıcı ID** ile takip eden,
2. Topu tespit edip kayıp karelerde **enterpolasyonla** dolduran,
3. Oyuncuları forma rengine göre **iki takıma** ayıran,
4. Açıklamalı bir çıktı videosu + kare bazlı **CSV/JSON** veri dosyası üreten

bir Python komut satırı aracı. Staj portföyü: temiz kod, testler, iyi bir README ve demo GIF.

**Kapsam dışı (şimdilik):** saha homografisi / kuşbakışı harita, hız-mesafe ölçümü, top sahipliği istatistiği, web arayüzü. (Bkz. §6 "Sonraki adımlar".)

## 1. Teknoloji ve sürümler

| Bileşen | Seçim | Not |
|---|---|---|
| Python | 3.11 | `uv venv --python 3.11 .venv` (sistemde 3.13 var, kullanılmaz) |
| Tespit | `ultralytics` (YOLO11, varsayılan `yolo11s.pt`) | COCO sınıfları: `person=0`, `sports ball=32` |
| Takip | `supervision` içindeki `sv.ByteTrack` | Detector'dan bağımsız, test edilebilir |
| Görüntü/video | `opencv-python`, `numpy` | |
| Takım ayrımı | `scikit-learn` KMeans | |
| Test | `pytest` | GPU ve internet **gerektirmeyen** testler |
| Kalite | `ruff` (lint + format) | |

`requirements.txt` sürümleri **sabitlenir** (`==`). PyTorch CUDA kurulumu README'de ayrı adım olarak anlatılır (requirements'a torch yazılmaz; ultralytics CPU torch çeker, GPU isteyen elle kurar).

## 2. Mimari

```
video ──► VideoReader ──► frame ──► Detector (YOLO11)
                                       │  sv.Detections (xyxy, conf, class_id)
                         ┌─────────────┴──────────────┐
                    oyuncular (person)            top (sports ball)
                         │                            │
                  PlayerTracker (ByteTrack)     BallTracker (en iyi aday seç
                         │  + tracker_id          + kısa boşlukları enterpole)
                  TeamClassifier (KMeans)             │
                         │  + team_id (0/1/-1)        │
                         └─────────────┬──────────────┘
                                  FrameResult
                         ┌─────────────┼──────────────┐
                    Annotator      Exporter       (istatistik)
                         │             │
                  VideoWriter      tracks.csv / summary.json
```

**İlkeler**
- Her modül tek iş yapar; `pipeline.py` sadece onları birbirine bağlar.
- Modeller yalnızca `detector.py` içinde yüklenir. Diğer modüller numpy / `sv.Detections` alır → model olmadan test edilebilir.
- Tüm ayarlar `configs/default.yaml` + `config.py` (dataclass) üzerinden; kodda sihirli sayı yok.
- Yollar her yerde `pathlib.Path`. Windows/Linux farkı yok.
- Top iki aşamalı işlenir: önce tüm video taranıp ham top konumları toplanır, sonra enterpolasyon yapılır, sonra çizim. (Bu yüzden pipeline **iki geçişli**: 1) tespit+takip → bellekte `list[FrameResult]`, 2) enterpolasyon + çizim + yazma.) Uzun videolarda bellek riski için bkz. R7.

**Veri sözleşmesi** (`src/futbolcv/types.py`):
```python
@dataclass
class BallState:
    xyxy: np.ndarray | None   # (4,) ya da None
    conf: float               # enterpole ise 0.0
    interpolated: bool

@dataclass
class FrameResult:
    frame_idx: int
    players: sv.Detections    # tracker_id dolu; data["team_id"] -> np.ndarray[int]
    ball: BallState
```

## 3. Klasör yapısı

```
futbol-cv/
├─ README.md
├─ LICENSE                      (MIT)
├─ requirements.txt
├─ requirements-dev.txt         (pytest, ruff)
├─ pyproject.toml               (paket + pytest + ruff ayarları)
├─ .gitignore                   (.venv, data/, models/, outputs/, *.pt, *.mp4)
├─ configs/
│  └─ default.yaml
├─ src/futbolcv/
│  ├─ __init__.py
│  ├─ config.py                 (yaml → dataclass)
│  ├─ types.py                  (BallState, FrameResult)
│  ├─ video_io.py               (VideoReader, VideoWriter)
│  ├─ detector.py               (YOLO sarmalayıcı)
│  ├─ tracker.py                (PlayerTracker: ByteTrack)
│  ├─ ball.py                   (BallTracker: seçim + enterpolasyon)
│  ├─ team.py                   (TeamClassifier)
│  ├─ annotate.py               (çizim)
│  ├─ export.py                 (CSV/JSON)
│  ├─ pipeline.py               (uçtan uca akış)
│  └─ cli.py                    (argparse; `python -m futbolcv ...`)
├─ src/futbolcv/__main__.py
├─ tests/
│  ├─ conftest.py               (sentetik kare/video fixture'ları)
│  └─ test_*.py
├─ scripts/
│  └─ make_demo_gif.py
├─ data/        (gitignore — girdi videoları)
├─ models/      (gitignore — .pt ağırlıkları)
├─ outputs/     (gitignore — çıktı video/CSV)
└─ docs/
   ├─ PLAN.md  HANDOFF.md  REVIEW.md
   └─ assets/   (README görselleri, demo.gif ≤ 10 MB)
```

## 4. Görevler

Durumlar: `TODO` · `DOING` · `REVIEW` (handoff yazıldı) · `FIX` (düzeltme istendi) · `DONE`

| # | Görev | Durum |
|---|---|---|
| T01 | İskelet + ortam | DONE |
| T02 | Config | DONE |
| T03 | Video I/O | DONE |
| T04 | Detector | DONE |
| T05 | Oyuncu takibi | DONE |
| T06 | Top takibi + enterpolasyon | DONE |
| T07 | Takım sınıflandırma | DONE |
| T08 | Annotator | DONE |
| T09 | Exporter | DONE |
| T10 | Pipeline + CLI | DONE |
| T11 | Uçtan uca deneme + ayar | REVIEW |
| T12 | README + demo GIF | TODO |
| T13 | (Opsiyonel) Futbol veri setiyle fine-tune | TODO |

Her görevde genel kabul kriterleri (aşağıda tekrar yazılmaz):
- `ruff check .` ve `ruff format --check .` temiz.
- `pytest -q` yeşil; yeni testler GPU/internet/model ağırlığı **istemez** (gerekenler `@pytest.mark.slow` ile işaretlenir ve varsayılan çalıştırmada atlanır).
- Public fonksiyon/sınıflarda tip ipucu ve kısa docstring.
- HANDOFF.md güncel.

### T01 — İskelet + ortam
- §3'teki klasörleri, `pyproject.toml` (src layout, `pip install -e .`), `requirements*.txt`, `.gitignore`, `LICENSE`, boş modüller.
- `pyproject.toml`'da pytest `markers = ["slow"]` ve `addopts = "-m 'not slow'"`.
- Ortam `uv` ile kurulur: `uv venv --python 3.11 .venv`; repoya `.python-version` (`3.11`) eklenir. `git init` yapılır, ilk commit `.gitignore` ile birlikte atılır.
- **Kabul:** Temiz venv'de `uv pip install -r requirements-dev.txt -e .` çalışır; `git status` içinde `.venv/`, `data/`, `models/` ve `outputs/` görünmez; `python -c "import futbolcv"` hata vermez; `pytest -q` (0 veya 1 dummy test) yeşil.

### T02 — Config
- `configs/default.yaml` + `config.py`: `load_config(path) -> AppConfig` (iç içe dataclass'lar: `detector`, `tracker`, `ball`, `team`, `annotate`, `output`).
- Asgari alanlar: `detector.weights`, `detector.conf`, `detector.imgsz`, `detector.device` (`"auto"`), `detector.player_class_ids=[0]`, `detector.ball_class_ids=[32]`, `ball.max_gap_frames=15`, `team.n_init_frames=30`.
- **Kabul:** Test: varsayılan dosya yüklenir; bilinmeyen anahtar → anlaşılır `ValueError`; eksik anahtar → varsayılan değer.

### T03 — Video I/O
- `VideoReader(path)`: iterator (`frame_idx, frame`), `fps`, `width`, `height`, `frame_count`; `max_frames` / `stride` parametreleri; olmayan dosyada `FileNotFoundError`.
- `VideoWriter(path, fps, size)`: context manager; klasörü oluşturur; codec `mp4v`.
- **Kabul:** Test, fixture'da OpenCV ile 10 karelik sentetik video yazar → okur; kare sayısı, boyut ve fps eşleşir.

### T04 — Detector
- `Detector(cfg)`: modeli bir kez yükler; `detect(frame) -> tuple[sv.Detections, sv.Detections]` (oyuncular, top adayları). Sınıf filtreleri config'den.
- `device="auto"` → CUDA varsa `0`, yoksa `cpu`.
- `weights` yalın bir dosya adıysa (`yolo11s.pt`) `models/` altında aranır, yoksa ultralytics oraya indirir. Böylece proje kökünde `.pt` dosyası birikmez.
- **Kabul:** Birim testi YOLO'yu **mock**layarak sınıf ayrımını doğrular. `@slow` testi gerçek ağırlıkla bir görüntüde ≥1 `person` bulur.

### T05 — Oyuncu takibi
- `PlayerTracker(cfg, fps)`: `update(dets) -> sv.Detections` (tracker_id dolu), `reset()`.
- ByteTrack'in `frame_rate` değeri **videodan** gelir (`fps / stride`); config'teki `tracker.frame_rate` `float | None = None` olur ve yalnızca elle zorlamak için kullanılır. (Yayınlar çoğunlukla 25/50 fps'tir; sabit 30 değeri `lost_track_buffer` süresini kaydırır.)
- **Kabul:** Test: sağa doğru 5 px/kare kayan 2 sentetik kutu, 20 kare boyunca **aynı** 2 tracker_id'yi korur; `reset()` sonrası ID'ler yeniden başlar.

### T06 — Top takibi + enterpolasyon
- `BallTracker.select(candidates, prev_xyxy) -> BallState`: birden çok aday varsa önceki konuma en yakını, yoksa en yüksek güvenli olanı seçer; mantıksız sıçrama reddedilir. Eşik **kare başına** tanımlanır (`max_jump_px × son görülmeden beri geçen kare`), aksi halde birkaç kare kaybolan top hiç geri yakalanamaz.
- `interpolate(states, max_gap) -> list[BallState]`: ≤ `max_gap` karelik boşlukları doğrusal doldurur (`interpolated=True`); daha uzun boşluklar `None` kalır.
- **Kabul:** Testler: (a) 3 karelik boşluk doğru koordinatlarla doldurulur, (b) `max_gap+1` boşluk doldurulmaz, (c) baştaki/sondaki boşluk doldurulmaz, (d) 2 adaydan öncekine yakın olan seçilir.

### T07 — Takım sınıflandırma
- `TeamClassifier`: kutunun **üst yarısının orta bölgesinden** (forma) renk özniteliği (HSV ya da Lab ortalaması); ilk `n_init_frames` karedeki örneklerle KMeans(k=2) `fit`; `predict(frame, dets) -> np.ndarray[int]`.
- Tutarlılık: bir `tracker_id`'nin takımı son N tahminin çoğunluk oyu ile belirlenir (titreme önlenir).
- Uç değer (hakem/kaleci): küme merkezine uzaklık eşiği aşılırsa `-1`.
- **Kabul:** Test: kırmızı ve mavi forma bölgeli sentetik kutular doğru 2 kümeye ayrılır; sarı kutu `-1` olur; çoğunluk oyu tek hatalı kareyi düzeltir.

### T08 — Annotator
- Oyuncu: ayak altına elips + ID etiketi, takım rengine göre; `-1` gri. Top: üçgen işaretçi; enterpole ise farklı stil (ör. içi boş). Sol üstte kare no/FPS.
- Girdi kareyi **değiştirmez** (kopya döner).
- **Kabul:** Test: çıktı aynı şekil/dtype, girdi değişmemiş, boş tespitlerde çökmez. Görsel kontrol HANDOFF'ta ekran görüntüsüyle (`docs/assets/`).

### T09 — Exporter
- `tracks.csv`: `frame, track_id, team_id, x1, y1, x2, y2, conf, kind(player|ball), interpolated`.
- `summary.json`: video bilgisi, toplam kare, benzersiz oyuncu ID sayısı, takım başına ID sayısı, topun görüldüğü/enterpole edildiği kare oranı, işlem süresi ve FPS.
- **Kabul:** Test: sahte `FrameResult` listesinden üretilen CSV satır sayısı ve JSON alanları doğru.

### T10 — Pipeline + CLI
- `run(input, output_dir, cfg, max_frames=None) -> Path` — §2'deki iki geçiş; ilerleme çubuğu (`tqdm`).
- Etkin fps = `reader.fps / stride`. Bu değer hem çıktı videosunun fps'i hem de ByteTrack'in `frame_rate` değeri olur. `reader.fps` 0 ya da geçersizse uyarı verilir ve 25 kullanılır. `frame_count` yalnızca ilerleme çubuğu için kullanılır, güvenilir kabul edilmez.
- Top: `BallTracker.update(..., frame_idx=<kaynak kare no>)` çağrılır. `interpolate` liste indeksiyle çalıştığı için ona `max(1, max_gap_frames // stride)` verilir.
- Sahne kesmesi (R3) algılandığında `PlayerTracker.reset()` ve `BallTracker.reset()` birlikte çağrılır. Enterpolasyon da kesme noktasını aşmamalı: listeyi kesme noktalarında parçalara bölüp her parçayı ayrı enterpole et. Sahne kesmesinde `TeamClassifier.reset_history()` de çağrılır.
- Takım ısınması: Ana geçişten önce ilk `n_init_frames` kare için kısa bir ön geçiş yapılır (tespit + `add_init_frame`), ardından `fit()` çağrılır. Ana geçiş 0. kareden başlar. Böylece ilk karelerdeki oyuncular `-1` (gri) görünmez.
- CLI: `python -m futbolcv --input data\mac.mp4 --output outputs\ --config configs\default.yaml [--max-frames 300] [--device cpu]`.
- **Kabul:** Detector mock'lu test, 10 karelik sentetik videoda çıktı mp4 + csv + json üretir. `--help` çalışır. Hatalı yol → anlaşılır hata mesajı, traceback değil.

### T11 — Uçtan uca deneme + ayar
- Gerçek bir klip (30–60 sn, yayın kamerası) üzerinde çalıştır. `conf`, `imgsz` (top için 1280 dene), ByteTrack eşikleri ayarla (`lost_track_buffer` için 60–90 dene).
- Takım: Gölge/güneş geçişinde aynı takım L (parlaklık) kanalı yüzünden iki kümeye bölünüyor ya da `-1` oluyorsa `team.lightness_weight` eklenir (Lab'de L kanalını örn. 0.3 ile çarp). `outlier_distance_threshold` gerçek görüntüye göre kalibre edilir.
- Top: Sabit bir yanlış pozitife kilitlenme görülürse `select()` mesafe ve güveni birlikte puanlayacak şekilde güncellenir.
- **Kabul:** HANDOFF'ta: kullanılan klip kaynağı + lisansı, donanım, FPS, `summary.json` içeriği, gözlenen sorunlar (ID değişimi, top kaybı) ve 3 örnek kare `docs/assets/` içinde.

### T12 — README + demo GIF
- README: tek cümlelik özet, demo GIF, özellikler, mimari şeması, Windows kurulum (venv, opsiyonel CUDA torch), kullanım, çıktı formatı, sonuçlar/kısıtlar, sonraki adımlar, lisans/atıflar. Komutların repo kökünden çalıştırılması gerektiği (`models/` CWD'ye göre çözülüyor) ve TLS/sertifika hatasında `models/yolo11s.pt` dosyasının elle nasıl indirileceği (`curl.exe -fL ...`) notu.
- `scripts/make_demo_gif.py`: çıktı videodan ≤10 MB GIF.
- **Kabul:** README'deki komutlar temiz klonda birebir çalışır; repoda ağırlık/video yok (`git status` temiz, büyük dosya yok).

### T13 — (Opsiyonel) Fine-tune
- Açık lisanslı bir futbol veri setiyle (sınıflar: player, goalkeeper, referee, ball) YOLO11 fine-tune; eğitim komutu ve sonuçlar (mAP) README'ye.
- **Kabul:** Top tespit oranı T11'deki COCO modeline göre aynı klipte ölçülebilir şekilde artar; ağırlık repoya değil, release/indirme linkine konur.

## 5. Riskler ve önlemler

| # | Risk | Önlem |
|---|---|---|
| R1 | **Top küçük ve hareket bulanıklığı** → COCO modeli topu sık kaçırır | `imgsz=1280`, düşük top `conf`, enterpolasyon (T06); gerekirse fine-tune (T13) |
| R2 | Kalabalık/örtüşmede **ID değişimi** | ByteTrack `lost_track_buffer` ayarı; T11'de ölç ve README'de dürüstçe yaz |
| R3 | **Kamera kesmeleri / tekrar görüntüleri** takibi bozar | Kareler arası histogram farkı büyükse `tracker.reset()` (T10'a küçük ek, T11'de gerekirse) |
| R4 | Seyirci/yedekler de `person` olarak algılanır | Kutu boyutu alt sınırı + takım sınıflandırmasında `-1`; ileride saha maskesi |
| R5 | Hakem/kaleci forması takım kümelemesini bozar | KMeans k=2 + uzaklık eşiği → `-1`; çoğunluk oyu |
| R6 | **Windows**: CUDA torch kurulumu, OpenCV codec (`avc1` yok) | README'de ayrı torch adımı; `mp4v` codec; `device="auto"` CPU'ya düşer |
| R7 | İki geçişli pipeline uzun videoda bellek | Bellekte sadece kutular tutulur, kareler değil; 2. geçişte video yeniden okunur |
| R8 | CPU'da yavaşlık | `--max-frames`, `stride`; README'de CPU/GPU FPS tablosu |
| R9 | Kütüphane API değişimi (supervision/ultralytics) | `==` ile sabit sürüm; tracker/detector sarmalayıcı arkasında |
| R10 | **Telif**: maç görüntüsü README/GIF'te | Açık lisanslı klip kullan, kaynağı yaz; reponun içine video koyma |
| R11 | Büyük dosyaların git'e girmesi | `.gitignore` T01'de; T12'de kontrol |

## 6. Sonraki adımlar (kapsam dışı, README'de "Future work")
`sv.ByteTrack` (supervision 0.30.x'te deprecated) → Roboflow `trackers` paketine geçiş, saha homografisi + kuşbakışı radar, oyuncu hızı/mesafesi, top sahipliği yüzdesi, Streamlit demo.

## 7. Çalışma protokolü (Gemini için)
1. Görevleri **sırayla** ve **tek tek** yap (T01 → …). Bir görevin bittiğini göstermeden sonrakine geçme.
2. Başlarken PLAN.md'de durumu `DOING`, bitince `REVIEW` yap. `DONE`'ı sadece inceleyici yazar.
3. Bitince `docs/HANDOFF.md`'yi şablona göre **üzerine yaz** (değişen dosyalar, çalıştırılan komutlar ve çıktıları, sapmalar, açık sorular).
4. `docs/REVIEW.md`'de `FIX` gelirse önce onları düzelt, aynı görev numarasıyla tekrar HANDOFF yaz.
5. Plana uymayan bir karar gerekiyorsa uygula ama HANDOFF'ta **"Sapmalar"** başlığında gerekçesiyle yaz.
