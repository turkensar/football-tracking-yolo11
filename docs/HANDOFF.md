# HANDOFF (Geliştirici → İnceleyici)

## Görev: T13c — Karşılaştırma + README (REVIEW.md FIX sonrası)

**Değişen/eklenen dosyalar**
- `README.md` — REVIEW.md'deki 3 dürüstlük düzeltmesi uygulandı:
  1. **Hız kıyaslaması:** Hız tablosu altındaki "4 sınıflı başlık ... biraz daha yüksektir" cümlesi kaldırıldı; karşılaştırma tablosundaki `pass1_fps` satırında yalnızca 300 karelik adil CPU ölçümleri (`2.38` / `2.45`) bırakılıp yorum *"Aynı mimari (`yolo11s`, `imgsz 1280`); hız eşdeğer (fark ölçüm gürültüsü içinde)."* olarak düzeltildi.
  2. **Top oranları vurgusu:** Tablonun en üstüne **Gerçek top tespiti (kare kare doğrulanmış)** satırı eklendi (`≈ 40.9%*` → **`≥ 75.7%`**, COCO için elle doğrulanmadığı dipnotuyla); `ball.detected_ratio` (`95.0%`) ve `ball.visible_ratio` (`100.0%`) satırlarındaki kalınlık kaldırıldı, yanlarına *(penaltı noktası yanlış pozitifi dahil)* eklendi ve "kesintisiz akış" ifadesi silindi.
  3. **`unique_player_ids` yorumu:** *"ID parçalanması da arttı — ByteTrack eşikleri COCO modeline göre ayarlandı, fine-tune model için yeniden ayarlanmadı."* notu eklendi.
- `docs/PLAN.md` — T13c durumu `FIX` → `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 30 files already formatted

$env:YOLO_CONFIG_DIR='...'; .venv\Scripts\pytest.exe -q
-> 119 passed, 1 deselected
```

**Kabul kriterleri**
- [x] README hız iddiası düzeltildi (`2.38` / `2.45`, eşdeğer hız notu).
- [x] README top tablosunda asıl vurgu **Gerçek top tespiti** (`≥ 75.7%`) satırına alındı; `detected_ratio` ve `visible_ratio` kalınlığı kaldırılıp penaltı noktası FP uyarısı eklendi.
- [x] `unique_player_ids` (`48 → 64`) yorumuna ByteTrack ID parçalanması notu eklendi.
- [x] `ruff check .`, `ruff format --check .` ve `pytest -q` temiz; commit atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- Yok.

**Açık sorular / riskler**
- REVIEW.md sırasına uyularak önce T13c FIX teslim edildi; onay/devam sonrası **T13d — Top takipçisi: sabit yanlış pozitif kilidini önleme** görevine geçilecektir.
