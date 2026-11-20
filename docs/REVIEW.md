# REVIEW (İnceleyici → Geliştirici)

## T01 — İskelet + ortam → **FIX**

Doğrulandı: Python 3.11.17, `pytest` (1 passed), `ruff check`, `ruff format --check` ve `pip check` temiz. Yapı §3'e uygun ve sapmaların gerekçeleri yerinde. Sapmaları (pyyaml, tqdm, `-r requirements.txt`, `*.md` exclude) **onaylıyorum**.

### Düzeltilecekler
1. **`.python-version` yok.** İçinde `3.11` yazan dosyayı ekle. (Önceki REVIEW'da ve PLAN T01'de yazıyordu, lütfen her görevden önce REVIEW.md'yi oku.)
2. **`git init` yapılmadı.** Bu da PLAN T01'de yazıyor. `git init` → `git add .` → `git status` ile `.venv/` ve `*.egg-info` dosyalarının listede olmadığını kontrol et → ilk commit. Commit mesajı: `chore: project skeleton (T01)`. Push etme.
3. **Boş klasörler klonda kaybolur.** `data/`, `models/`, `outputs/` tümüyle ignore edildiği için repo klonlandığında bu klasörler oluşmaz. `.gitignore`'da şu kalıbı kullan: `data/*` + `!data/.gitkeep` (models ve outputs için de aynısı). Ardından üç klasöre `.gitkeep` ekle.

### Soruya cevap: ByteTrack
**`supervision==0.30.9` + `sv.ByteTrack` ile devam.** Gerekçe: kullanım `PlayerTracker` sarmalayıcısının arkasında kalıyor, yani ileride `trackers` paketine geçiş tek dosyalık bir iş olur. T05'te:
- `FutureWarning`'ı yalnızca ByteTrack oluşturulduğu satırda `warnings.catch_warnings()` ile sustur, global olarak susturma.
- `tracker.py` başına neden 0.30.x'e sabitlendiğini anlatan 1 satırlık yorum ekle.

`trackers` paketine geçiş, PLAN §6'ya "Future work" maddesi olarak eklendi.

Düzeltmeler bitince T01 HANDOFF'unu güncelle; onaydan sonra T02'ye geçilecek.
