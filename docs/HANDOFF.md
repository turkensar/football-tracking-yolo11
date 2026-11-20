# HANDOFF (Geliştirici → İnceleyici)

## Görev: T03 — Video I/O

**Değişen/eklenen dosyalar**
- `src/futbolcv/video_io.py` — OpenCV tabanlı `VideoReader` (metadata özellikleri, `max_frames`/`stride`, iki geçiş için yeniden yinelenebilir iterator, context manager, `FileNotFoundError`) ve `VideoWriter` (context manager, üst klasör oluşturma, `mp4v` codec, şekil/dtype/kapalı yazma koruması) eklendi.
- `tests/conftest.py` — OpenCV ile 10 karelik sentetik video üreten `synthetic_video_path`, `synthetic_video_meta` ve `synthetic_frame` fixture'ları eklendi.
- `tests/test_video_io.py` — `VideoReader` ve `VideoWriter` için 20 birim testi (metadata/kare eşleşmesi, `max_frames`, `stride`, çift geçişte tekrar okuma, olmayan/bozuk dosya, şekil/dtype/kapalı yazıcı hataları, iç içe klasör oluşturma) eklendi.
- `docs/PLAN.md` — T03 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\pytest.exe -q
-> .........................................                                [100%]
-> 41 passed in 2.58s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 18 files already formatted

git commit -m "feat: video io reader and writer (T03)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `VideoReader(path)`: `(frame_idx, frame)` iterator, `fps`, `width`, `height`, `frame_count` özellikleri, `max_frames` ve `stride` desteği, olmayan dosyada `FileNotFoundError`.
- [x] `VideoWriter(path, fps, size)`: context manager, üst klasörleri otomatik oluşturur, varsayılan codec `mp4v`.
- [x] Fixture'da OpenCV ile 10 karelik sentetik video yazılıp okundu; kare sayısı (`10`), boyut (`320x240`) ve FPS (`25.0`) eşleşmesi testle doğrulandı (`VideoWriter` round-trip testiyle de doğrulandı).
- [x] `pytest -q`, `ruff check .`, `ruff format --check .` temiz; GPU/internet/ağırlık gerektirmiyor.
- [x] Görev sonu commit'i (`feat: video io reader and writer (T03)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- `VideoReader` metadata'yı `__init__` içinde okuyup `VideoCapture`'ı hemen serbest bırakır; her `__iter__` çağrısında kendi `VideoCapture`'ını açıp `finally` bloğunda kapatır. Bu sayede Windows'ta dosya kilidi sızıntısı önlenir ve T10'daki iki geçişli (two-pass) pipeline'da aynı `VideoReader` nesnesi güvenle iki kez gezilebilir.
- `VideoReader` atlanan karelerde (`stride > 1`) tam decode yerine `cap.grab()` kullanır ve `frame_idx` olarak kaynak videodaki orijinal kare indeksini (`0, stride, 2*stride, ...`) üretir; `max_frames` döndürülen kare sayısını sınırlar.
- `VideoWriter.write()` OpenCV'nin sessizce kare düşürmesini engellemek için kare boyutu (`(height, width, 3)`) ve `uint8` doğrulaması yapar.

**Açık sorular / riskler**
- Yok.
