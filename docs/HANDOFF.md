# HANDOFF (Geliştirici → İnceleyici)

## Görev: T07 — Takım sınıflandırma

**Değişen/eklenen dosyalar**
- `src/futbolcv/team.py` — `extract_jersey_color` (kutunun üst yarısının orta bölgesinden Lab/HSV ortalama renk özniteliği) ve `TeamClassifier(cfg)` (`add_init_frame`, `fit`, `predict(frame, dets) -> np.ndarray[int]`, `reset_history`) eklendi; tahminler `dets.data["team_id"]` içine `np.ndarray[int]` olarak yazılır, uzaklık eşiği aşan uç değerler `-1` atanır ve `tracker_id` başına son `vote_window` tahminin çoğunluk oyu uygulanır.
- `tests/test_team.py` — üst-orta kırpma doğrulaması, kırmızı/mavi 2 küme ayrımı + sarı hakemin `-1` atanması + `dets.data["team_id"]` sözleşmesi, çoğunluk oyunun tek hatalı kareyi düzeltmesi ve `reset_history()`, `n_init_frames` sınırı, eğitilmemiş/tek renkli/boş tespit kenar durumları ve HSV modu testleri (6 test) eklendi.
- `docs/PLAN.md` — T07 durumu `REVIEW` olarak güncellendi.

**Çalıştırılan komutlar ve sonuçları**
```
.venv\Scripts\pytest.exe -W error tests/test_team.py -v
-> 6 passed in 6.66s

.venv\Scripts\pytest.exe -q
-> ........................................................................ [ 96%]
-> ...                                                                      [100%]
-> 75 passed, 1 deselected in 23.59s

.venv\Scripts\ruff.exe check .
-> All checks passed!

.venv\Scripts\ruff.exe format --check .
-> 22 files already formatted

git commit -m "feat: kmeans team classifier with majority voting (T07)"
-> Commit atıldı (push edilmedi).
```

**Kabul kriterleri**
- [x] Kutunun üst yarısının orta bölgesinden (`jersey_top_ratio`, `jersey_center_crop_ratio`) Lab (varsayılan) veya HSV ortalama renk özniteliği çıkarılır; şort ve kenar çimi dışarıda bırakılır.
- [x] İlk `n_init_frames` karedeki örneklerle `KMeans(k=2)` `fit` edilir (hem doğrudan `fit(samples)` hem bellek dostu artımlı `add_init_frame` + `fit()` desteklenir).
- [x] `predict(frame, dets) -> np.ndarray[int]` sonucu hem döndürür hem de §2 / REVIEW.md sözleşmesine uygun şekilde `dets.data["team_id"]` içine `np.ndarray[int]` olarak yazar.
- [x] Kırmızı ve mavi forma kutuları doğru 2 kümeye (`0`, `1`) ayrılır; sarı kutu küme merkezine uzaklık eşiğini aşarak `-1` olur.
- [x] `tracker_id` başına son `vote_window` tahminin çoğunluk oyu tek hatalı kareyi düzeltir.
- [x] `pytest -q`, `pytest -W error tests/test_team.py`, `ruff check .`, `ruff format --check .` temiz; GPU/internet/ağırlık gerektirmiyor.
- [x] Görev sonu commit'i (`feat: kmeans team classifier with majority voting (T07)`) atıldı (push edilmedi).

**Sapmalar** (plandan farklı yapılanlar + gerekçe)
- İki geçişli pipeline'ın (T10, R7) kareleri bellekte tutmadan yalnızca 3 elemanlı renk vektörlerini biriktirebilmesi için `add_init_frame(frame, dets)` ve sahne kesmelerinde (R3) KMeans merkezlerini koruyup yalnızca `tracker_id` oy geçmişini sıfırlayan `reset_history()` metotları eklendi.
- Küme merkezleri fit sonrası deterministik sıraya dizildi (`team 0` / `team 1` etiketleri tekrar üretilebilir).
- Benzersiz renk örneği sayısı `< 2` olduğunda `KMeans` `ConvergenceWarning` üretmemesi için model `is_fitted=False` bırakılıp tüm kutulara `-1` dönülür.

**Açık sorular / riskler**
- `outlier_distance_threshold=45.0` sentetik ve tipik forma renkleri (kırmızı/mavi/sarı/siyah/beyaz) için OpenCV Lab uzayında doğrulandı; gerçek yayın ışığında (gölge/güneş geçişi) T11'de gözlemlenip gerekirse kalibre edilecek.
