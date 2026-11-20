# HANDOFF (Geliştirici → İnceleyici)

## Görev: T01 — İskelet + ortam (REVIEW.md FIX sonrası)

**Değişen/eklenen dosyalar**
- `.python-version` — eklendi (`3.11`).
- `.gitignore` — `data/`, `models/`, `outputs/` kuralları `data/*` + `!data/.gitkeep` (ve `models`, `outputs` için aynısı) kalıbıyla güncellendi; `.venv/` ve `*.egg-info/` ignore kuralları korundu.
- `data/.gitkeep`, `models/.gitkeep`, `outputs/.gitkeep` — boş klasörlerin klonlamada korunması için eklendi.
- `docs/PLAN.md` — T01 durumu `FIX` → `REVIEW` olarak güncellendi.
- `.git/` — `git init` yapıldı, ilk commit `chore: project skeleton (T01)` mesajıyla atıldı (push edilmedi).

*(Önceki T01 turunda oluşturulan ve incelemede onaylanan dosyalar korundu: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `LICENSE`, `README.md`, `configs/default.yaml`, `src/futbolcv/*.py`, `tests/conftest.py`, `tests/test_smoke.py`, `scripts/make_demo_gif.py`, `docs/assets/.gitkeep`.)*

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\python.exe -m pip check
-> No broken requirements found.

.venv\Scripts\python.exe -c "import futbolcv; print(futbolcv.__version__)"
-> 0.1.0

.venv\Scripts\python.exe -m pytest -q
-> .                                                                        [100%]
-> 1 passed in 0.03s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 16 files already formatted

git init && git add . && git status --short
-> 31 dosya staged (.python-version, .gitignore, data/.gitkeep, models/.gitkeep, outputs/.gitkeep, docs/assets/.gitkeep, src/futbolcv/*.py, tests/*, configs/*, scripts/*, docs/*)
-> .venv/, *.egg-info/, .pytest_cache/, .ruff_cache/, __pycache__/ listede YOK.

git check-ignore -v .venv src/futbolcv.egg-info data/mac.mp4 models/yolo11s.pt outputs/tracks.csv
-> .gitignore:2:.venv/        .venv
-> .gitignore:9:*.egg-info/   src/futbolcv.egg-info
-> .gitignore:28:*.mp4        data/mac.mp4
-> .gitignore:27:*.pt         models/yolo11s.pt
-> .gitignore:25:outputs/*    outputs/tracks.csv

git commit -m "chore: project skeleton (T01)"
-> İlk commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] `.python-version` (`3.11`) eklendi.
- [x] `data/`, `models/`, `outputs/` için `.gitignore` (`*` + `!.gitkeep`) ve `.gitkeep` dosyaları eklendi; sahte içerik dosyalarının (`data/mac.mp4`, `models/yolo11s.pt`, `outputs/tracks.csv`) ignore edildiği `git check-ignore` ile doğrulandı.
- [x] `git init` → `git add .` → `git status` kontrolü (`.venv/` ve `*.egg-info` yok) → `chore: project skeleton (T01)` ilk commit tamamlandı (push edilmedi).
- [x] `pip check`, `python -c "import futbolcv"`, `pytest -q`, `ruff check .`, `ruff format --check .` temiz.

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- Önceki turda belirtilen ve REVIEW.md'de onaylanan sapmalar (`pyyaml`, `tqdm`, `requirements-dev.txt` içinde `-r requirements.txt`, `[tool.ruff] extend-exclude = ["*.md"]`) korundu; yeni sapma yok.

**Açık sorular / riskler**
- Yok. (REVIEW.md'deki `supervision==0.30.9` + `sv.ByteTrack` kararı ve T05 için dar kapsamlı `warnings.catch_warnings()` + 1 satırlık yorum notu alındı.)
