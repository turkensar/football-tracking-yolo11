# HANDOFF (Geliştirici → İnceleyici)

## Görev: T12 — README + demo GIF (REVIEW.md FIX sonrası)

**Değişen/eklenen dosyalar**
- `README.md` — REVIEW.md'deki 4 düzeltme uygulandı:
  1. **Kurulum TLS notu:** Kurulum bölümüne `uv pip install --system-certs -r requirements-dev.txt -e .` komutu eklendi.
  2. **CUDA PyTorch komutu:** Sabit `cu124` indeksi kaldırıldı; `https://pytorch.org/get-started/locally/` seçicisinden uygun `--index-url` alınarak `uv pip install --reinstall torch torchvision --index-url <pytorch-cuda-index-url>` biçiminde çalıştırılması yönlendirildi.
  3. **AGPL-3.0 lisans rozeti ve bölümü:** Başlığın altına `License: AGPL v3` rozeti eklendi; "Lisans ve Atıflar" bölümündeki "Kod lisansı" satırı `[GNU AGPL-3.0](LICENSE)` olarak güncellenip tek cümlelik gerekçe ("Ultralytics YOLO AGPL-3.0 lisanslı olduğu için...") eklendi. Demo görüntüsünün CC BY 4.0 atfı ve değişiklik bildirimi aynen korundu.
- `requirements.txt` — `pillow==13.0.0` açıkça sabitlendi.
- `pyproject.toml` — `license = "AGPL-3.0-or-later"` olarak güncellendi ve `dependencies` listesine `"pillow"` eklendi.
- `LICENSE` — Resmi GNU Affero General Public License v3.0 tam metni (`https://www.gnu.org/licenses/agpl-3.0.txt`, `34,523` bayt, 661 satır) doğrudan indirilerek değiştirildi (elle düzenleme yapılmadı).
- `docs/PLAN.md` — T12 durumu `FIX` → `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
curl.exe -fL https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE
-> 34,523 bytes (resmi GNU AGPL-3.0 tam metni, 661 satır)

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 27 files already formatted

.venv\Scripts\pytest.exe -q
-> 104 passed, 1 deselected in 17.39s

pyproject.toml metadata doğrulaması (tomllib):
-> license: AGPL-3.0-or-later
-> deps: ['numpy', 'opencv-python', 'pillow', 'pyyaml', 'scikit-learn', 'supervision<0.31', 'tqdm', 'ultralytics']
```

**Kabul kriterleri**
- [x] README kurulum bölümünde `--system-certs` TLS notu yer alıyor.
- [x] `requirements.txt` (`pillow==13.0.0`) ve `pyproject.toml` (`"pillow"`) içinde Pillow açıkça tanımlı.
- [x] CUDA kurulumu sabit `cu124` yerine `https://pytorch.org/get-started/locally/` seçicisine yönlendiriyor.
- [x] Karar A (AGPL-3.0) eksiksiz uygulandı: resmi `LICENSE` metni (`agpl-3.0.txt`), `pyproject.toml` `license = "AGPL-3.0-or-later"`, README AGPL-3.0 rozeti ve gerekçesi; demo görüntüsü CC BY 4.0 atfı korundu.
- [x] `pytest -q` (104 passed, 1 slow deselected), `ruff check .`, `ruff format --check .` temiz.
- [x] Düzeltme commit'i atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- Yok (REVIEW.md'deki 4 düzeltme ve kullanıcı kararı A birebir uygulandı).

**Açık sorular / riskler**
- `T13 — (Opsiyonel) Futbol veri setiyle fine-tune` için karar kullanıcı/inceleyici onayına bırakıldı; T12 onaylanana kadar başlanmayacak.
