# HANDOFF (Geliştirici → İnceleyici)

## Görev: Yayına hazırlık (`turkensar/football-tracking-yolo11`)

**Değişen/eklenen dosyalar**
- `README.md` —
  - Başlık `# football-tracking-yolo11` olarak güncellendi (Python paket adı `futbolcv` ve CLI `python -m futbolcv` korundu).
  - Kurulum bölümündeki `git clone` URL'si `https://github.com/turkensar/football-tracking-yolo11.git`, dizin adı `cd football-tracking-yolo11` ve CWD notu `football-tracking-yolo11/` olarak güncellendi.
  - GitHub Release indirme komutu ve Open in Colab rozetindeki tüm `<GITHUB_USER>/<REPO_NAME>` yer tutucuları `turkensar/football-tracking-yolo11` ile değiştirildi (`blob/main` korundu).
  - **Bilinen Kısıtlar #2** içine *"ayrıca terk edilmiş noktanın hemen yanından çok yavaş geçen top kısa süre kaçırılabilir"* cümlesi eklendi.
- `notebooks/finetune_colab.ipynb` — Başlık, açıklama ve Open in Colab rozeti `turkensar/football-tracking-yolo11` (`blob/main`) ile güncellendi; tüm kod çıktıları ve `execution_count` alanları boş bırakıldı.

**Çalıştırılan komutlar ve sonuçları**
```
Select-String -Path README.md, notebooks\finetune_colab.ipynb -Pattern 'GITHUB_USER|REPO_NAME|<repo-url>'
-> 0 eşleşme (kalan yer tutucu yok)

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 30 files already formatted

$env:YOLO_CONFIG_DIR='...'; .venv\Scripts\pytest.exe -q
-> 128 passed, 1 deselected

git commit -m "docs: set repo name and links for publishing"
-> Commit atıldı (push edilmedi, yerel dal master olarak korundu).
```

**Kabul kriterleri**
- [x] `README.md` ve `notebooks/finetune_colab.ipynb` içindeki tüm `<GITHUB_USER>/<REPO_NAME>` ve `<repo-url>` yer tutucuları `turkensar/football-tracking-yolo11` ile değiştirildi (`Select-String` ile 0 kalan doğrulandı).
- [x] README başlığı `# football-tracking-yolo11`; Python paketi (`futbolcv`) ve CLI (`python -m futbolcv`) değişmedi.
- [x] Kurulumdaki `cd football-tracking-yolo11` ve CWD notu `football-tracking-yolo11/` olarak güncellendi.
- [x] Linkler `blob/main` olarak korundu; yerel dal `master` olarak bırakıldı.
- [x] Bilinen Kısıtlar #2'ye yavaş geçen top notu eklendi.
- [x] Not defterinin çıktıları boş, `pytest -q` (128 passed, not defteri testi dahil), `ruff check .` ve `ruff format --check .` temiz.
- [x] İstenen commit mesajıyla (`docs: set repo name and links for publishing`) commit atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- Yok. (Kurulumdaki `git clone <repo-url>` yer tutucusu da yayına hazırlık kapsamında gerçek `https://github.com/turkensar/football-tracking-yolo11.git` adresiyle dolduruldu.)

**Açık sorular / riskler**
- Yok — repo GitHub'da yayınlanmaya hazır (`git branch -M main`, remote ekleme, `git push -u origin main`, ardından `models/football_yolo11s.pt` için GitHub Release oluşturma).
