# HANDOFF (Geliştirici → İnceleyici)

## Görev: T13a — Colab eğitim not defteri

**Değişen/eklenen dosyalar**
- `notebooks/finetune_colab.ipynb` — Google Colab (T4 GPU) eğitim not defteri (nbformat 4.5, **tüm kod çıktıları ve `execution_count` alanları temizlenmiş**):
  - Başta yer tutuculu **Open in Colab** rozeti (`<GITHUB_USER>/<REPO_NAME>`).
  - Hücre 1: `nvidia-smi` + `torch.cuda.is_available()` GPU kontrolü (GPU yoksa açık hata verir).
  - Hücre 2: `requirements.txt` ile birebir aynı `ultralytics==8.4.216` + `roboflow` kurulumu.
  - Hücre 3: Yalnızca Colab Secrets (`userdata.get("ROBOFLOW_API_KEY")`) ile Roboflow'dan veri setini (`yolov11` formatı) indirme, `data.yaml` yollarını Colab için köke sabitleme ve sınıf haritasını ekrana yazdırma.
  - Hücre 4: `yolo11s.pt`'den ince ayar (`imgsz=1280`, `epochs=50`, `batch=8`, `patience=15`, `seed=42`).
  - Hücre 5: `best.pt` üzerinde doğrulama (`mAP50`, `mAP50-95`, sınıf bazlı `AP50` / `AP50-95` ve özellikle `ball` vurgusu) + `val_metrics.json` kaydı.
  - Hücre 6: `best.pt`'yi `football_yolo11s.pt` adıyla kopyalayıp `results.csv` ve `val_metrics.json` ile birlikte indirme (+ yorum satırında opsiyonel Google Drive yedeği).
- `scripts/train.py` — Not defterindeki eğitim çağrısının `argparse`'lı yerel/sunucu karşılığı (`--data`, `--weights`, `--epochs`, `--imgsz`, `--batch`, `--device`, `--project`, `--name`, `--patience`, `--seed`). `ultralytics` ve `torch` tembel (lazy) import edilir; `--help` ve hata yolları hızlı ve traceback'siz çalışır.
- `tests/test_train.py` — 9 yeni birim testi: `scripts/train.py --help`, mocked `ultralytics.YOLO.train` argüman doğrulaması, `auto` cihaz çözümlemesi, geçersiz argüman / eksik dosya hata kontrolü, ve not defteri doğrulaması (geçerli JSON, boş çıktı hücreleri, `Open in Colab` rozeti, `userdata.get("ROBOFLOW_API_KEY")`, gizli anahtar/yerel yol yokluğu, `requirements.txt` ile eşleşen `ultralytics==8.4.216` sabiti).
- `docs/PLAN.md` — T13a durumu `DOING` → `REVIEW` olarak güncellendi.

---

### Seçilen Veri Seti ve Lisans Doğrulaması
- **Veri seti:** Roboflow Universe — *Football Players Detection* (`roboflow-jvuqo/football-players-detection-3zvbc`, v1)
- **Kaynak URL:** https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
- **Lisans:** **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/) — T13c'de `README.md` atıf bölümüne eklenecektir.
- **Sınıflar (4):** `0: ball`, `1: goalkeeper`, `2: player`, `3: referee`
- **Veri sızıntısı kontrolü:** T11'de kullanılan 2022 Dünya Kupası Ekvador–Katar klibi bu eğitim setinde yer almaz (Avrupa/Bundesliga yayın karelerinden oluşur).
- **Doğrulama notu:** `universe.roboflow.com` HTTP istemcisinde Cloudflare doğrulama sayfası döndürdüğü için sayfa kazınmadı; lisans (`CC BY 4.0`) ve `data.yaml` şeması arama sonuçlarından ve Roboflow Sports dokümantasyonundan teyit edildi. Ayrıca not defterinin 3. hücresi indirilen `data.yaml` içeriğini eğitim başlamadan ekrana basar.

---

### Kullanıcı İçin: Colab'da Nasıl Çalıştırılır?
1. **Not defterini açın:** [Google Colab](https://colab.research.google.com/) adresine gidin → `Dosya -> Not defteri yükle` ile yerel repodaki `notebooks/finetune_colab.ipynb` dosyasını yükleyin (repo GitHub'a push edildikten sonra üstteki **Open in Colab** rozetiyle de açılabilir).
2. **GPU seçin:** Üst menüden `Çalışma zamanı -> Çalışma zamanı türünü değiştir -> Donanım hızlandırıcı: T4 GPU -> Kaydet` adımlarını izleyin.
3. **Roboflow API anahtarını Secrets'a ekleyin:**
   - Ücretsiz Roboflow hesabınızdan `Settings -> Roboflow API` bölümündeki Private API Key'i kopyalayın.
   - Colab sol kenar çubuğundaki **Anahtar (Secrets)** simgesine tıklayın → **Yeni gizli anahtar ekle** → Ad: `ROBOFLOW_API_KEY`, Değer: anahtarınız → **Not defteri erişimi** anahtarını açın. (Anahtarı kod hücrelerine yazmayın.)
4. **Hücreleri çalıştırın:** `Çalışma zamanı -> Tümünü çalıştır` seçin.
   - 1. hücrede `Tesla T4` göründüğünü, 3. hücrede `names: ['ball', 'goalkeeper', 'player', 'referee']` yazdırıldığını kontrol edin.
   - 4. hücrede `imgsz=1280, batch=8` ile bellek hatası (CUDA OOM) alırsanız `BATCH = 4` veya `BATCH = -1` yapıp o hücreyi tekrar çalıştırın.
   - 5. hücrede genel `mAP50` / `mAP50-95` ve `ball` sınıfı metriklerini inceleyin.
5. **Çıktıları indirip yerele koyun:**
   - 6. hücre `football_yolo11s.pt`, `results.csv` ve `val_metrics.json` dosyalarını tarayıcınıza indirir (kopma riskine karşı opsiyonel Google Drive yedek satırlarını da açabilirsiniz).
   - İndirilen `football_yolo11s.pt` dosyasını yerel repoda **`models/football_yolo11s.pt`** konumuna kaydedin (`models/` gitignored olduğu için repoya girmez).
   - `val_metrics.json` içeriğini T13c karşılaştırma ve README güncellemesi için paylaşın.

---

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe scripts\train.py --help
-> Exit 0 (--data, --weights, --epochs, --imgsz, --batch, --device, --project, --name, --patience, --seed)

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 30 files already formatted

.venv\Scripts\pytest.exe -q
-> 113 passed, 1 deselected in 13.35s
```

**Kabul kriterleri**
- [x] `python scripts/train.py --help` çalışır.
- [x] Birim testi `ultralytics.YOLO.train`'i mock'layarak tüm argümanların doğru geçtiğini doğrular.
- [x] `notebooks/finetune_colab.ipynb` geçerli JSON'dur, tüm kod hücrelerinin çıktıları ve `execution_count` alanları boştur (testle doğrulanır).
- [x] Not defterinde gizli anahtar veya yerel dosya yolu yoktur; API anahtarı yalnızca Colab Secrets (`userdata.get("ROBOFLOW_API_KEY")`) ile okunur; `ultralytics==8.4.216` sürümü `requirements.txt` ile eşleşir.
- [x] Veri seti kaynağı + CC BY 4.0 lisansı ve adım adım "Colab'da nasıl çalıştırılır" rehberi HANDOFF'a eklendi.
- [x] `pytest -q` (113 passed, 1 deselected), `ruff check .`, `ruff format --check .` temiz; T13a commit'i atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- Not defteri 3. hücrede Roboflow'un ürettiği `data.yaml` içindeki göreli yolları (`../train/images`) mutlak `path: <dataset_dir>` + `train/images` biçimine normalize eder. Gerekçe: Ultralytics'in Colab'da varsayılan `datasets_dir` ayarı yüzünden göreli `../` yollarında `FileNotFoundError` verme riskini önler.
- Not defteri 6. hücrede `best.pt` dosyasını doğrudan `football_yolo11s.pt` adıyla kopyalayıp `val_metrics.json` ile birlikte indirir; böylece kullanıcının elle dosya adı değiştirmesi gerekmez ve T13c için sınıf bazlı metrikler kaybolmaz.

**Açık sorular / riskler**
- T4 (16 GB VRAM) üzerinde `imgsz=1280` + `batch=8` genellikle sığar ancak Colab ortamındaki PyTorch sürümüne göre VRAM sınırına yaklaşabilir; bu durum için not defterine `BATCH = 4` veya `-1` (AutoBatch) notu eklendi.
- Sıradaki yerel görev **T13b** (T13a onayı sonrası); **T13c** ise kullanıcı `models/football_yolo11s.pt` dosyasını yerleştirdikten sonra başlayacak.
