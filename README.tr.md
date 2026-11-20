# LOWLAT — kayıpsız-öncelikli uzak-masaüstü codec'i

**[English](README.md) | Türkçe**

> ⚠️ **Durum: deneysel.** Bu depo bir kanıt-tanımı / deneme niteliğindedir
> (uçtan uca incelenen bir VNC-yerine codec: klasik C çekirdek, oran-kontrollü
> streaming modu, ölçüm harness'ları, yayınlanmış demolar). Geliştirme
> Kasım2026'da son halinde **burada durduruldu** ve proje **aktif olarak
> bakılmıyor** — ama incelemek, fork'layıp üzerine inşa etmek için yeterince
> tamamlanmış durumda. Nasıl isterseniz kullanın: **MIT lisanslı.**

**Varsayılan olarak kayıpsız, *yalnızca* zorunlu olduğunda bozulan bir VNC
yerine geçişi.** Masaüstünü30 fps'de son derece düşük bir bit hızında
(hedef **10 KB/s**) yayınlar, **metni piksel-birebir** tutar, tekrar eden
desenleri önbelleğe alır ve tam ekran kaydırmayı neredeyse bedava hâle
getirir.

```
kare ─► [kirli karolar] ─► [global-scroll / ME] ─► [karo önbelleği] ─► [hibrit kodlayıcı] ─► deflate ─► hat
                                                                                         (CBR 10 KB/s)
hat   ─► inflate ─► [RLE bitmap] ─► [karo başına: PRED/REF/varint/submask/down/skip] ─► imleç birleştir ─► ekran
```

---

## Öne çıkan sonuçlar

Tüm sayılar **gerçektir**, ölçülmüştür ve **bit-birebir doğrulanmıştır**
(kayıp-olmayan karolar piksel birebir). `enc/dec` tek-çekirdekli bir
dizüstünde; çekirdek C'dir.

### Masaüstü etkinliği (kayıpsız)

| çözünürlük | etkinlik | bit hızı (kayıpsız) | not |
|---|---|---|---|
| 480p  | boşta    | **2.1 KB/s** | yalnız imleç/yanıp-sönme |
| 720p  | yazma    | **0.9 KB/s** | metin asla bozulmaz |
| 1080p | boşta    | **1.0 KB/s** | |
| 1080p | yazma    | **0.7 KB/s** | |
| 1080p | yoğun    | 8.8 KB/s | hâlâ kayıpsız |
| herhangi | çok yoğun | 12–20 KB/s | bütçe aşılıyor → uyarlanır |

### Tekrarlayan-desen önbelleği (A↔B arayüzünü değiştir)

| | önbellek yok | **önbellek** |
|---|---|---|
| toplam bayt | 20,216 B | **5,592 B** (3.6×) |
| önbellek referansı | 0 | **373** |

### Sorunsuz makale kaydırması — kayıpsız (global-scroll *band* modu)

Saf dikey kaydırma yalnızca yeni-açılan şeridi gönderir (Paeth + deflate);
gerisi tek bir kaydırmadır. Kayıpsız, tam-kare bit-birebir:

| kaydırma hızı | px/sn | bit hızı (kayıpsız) |
|---|---|---|
| 1.0 px/kare | 30 | **6.4 KB/s** |
| 1.5 px/kare | 45 | **9.1 KB/s**  ← 10 KB/s altında |
| 2.0 px/kare | 60 | 11.7 KB/s |
| 4.0 px/kare | 120 | 22.1 KB/s |

Maliyet *yeni açılan metin* ile orantılıdır — bu gerçek bilgidir. Normal
okuma hızındaki kaydırmada kayıpsız akış10 KB/s altında kalır. (Daha önceki
sürümlerde burada ~40 KB/s idi; çünkü şerit karo-karo yeniden kodlanıyor ve
karo başına ek yük biniyordu; band yolu bunu kaldırıyor.)

Kaydırma gerçekten yeni metin açığa çıkarır; oran yeni içerikle orantılıdır.
Statik/yazma/etkileşimli masaüstleri10 KB/s'nin çok altında oturur — bir VNC
yerine geçişi tam olarak burada yaşar.

### Metin koruması (temel vaat)

Yoğun içerikte sert10 KB/s bütçesi altında **yalnızca metin-olmayan karolar
bozulur** (atla / 2× / 4× örnekleme). Her kare doğrulandı:

```
kayıp-olmayan-karo piksel uyuşmazlığı : 0
metin-olarak-kayıp-sınıflandırılan karo : 0
```

### Gerçek60 saniyelik masaüstü kaydı,15 KB/s (LOWLAT-S)

(terminal, duvar kâğıdı, pencere geçişleri, bildirimler, sahne kesimleri
dahil) gerçek bir1080p/30 masaüstünün tam dakikası, sert15 KB/s bütçesi
altında oran-kontrollü `stream/` codec'inden geçirildi:

| metrik | değer |
|---|---|
| ortalama oran | **14.97 KB/s** (898.3 / 900 KB) ✅ |
| en uzun gönderi boşluğu (sessiz hat) | **0.4 s** — hiç ölü ekran yok |
| etkileşim gecikmesi (yazma → ekran) | **p50 0.13 s** |
| PSNR orta / p5 | **24.28 / 13.44 dB** |
| karo kalitesi | **q82+ zemini** (kodlamaların %97'si q82–92; bantlar ≥ q75) |

Kodlayıcı iki şeritli çalışır: **recency-öncelikli etkileşim** (az önce
yazdığınız satır önce gider, oran-paylı olduğu için bulk'ı aç bırakamaz) ve
**bulk** (yoğun değişiklikler yaş-hedefli crop-bantlar olarak, seyrek
değişiklikler yaş-en-eski prefix-paketli koşular olarak) — hat hiçbir zaman
saniyeler süren snapshot banketmesinde donmaz. İletiş sayfası, oran/PSNR
zaman çizelgesi ve tam istatistikler için
[`experiments/streaming/`](experiments/streaming/)'e bakın veya
[yan-yana demo videosunu](https://github.com/kedybaykedy-bit/lowlat/releases/download/demo-15kbps/lowlat-s_15kbps_compare.mp4)
izleyin.

---

## Özellikler

- **Varsayılan olarak kayıpsız.** Kare-başına bütçe bozmayı zorlamadığı
  sürece her kare bayt-birebirdir.
- **Korumalı metin, uyarlanabilir kalite.** Karolar metin-benzeri olarak
  sınıflandırılır (güçlü-kenar yoğunluğu + iki-uçlu parlaklık histogramı).
  Metin karoları *asla* bozulmaz; yalnızca düz/fotoğraf/video karoları
  atla / 2× / 4× örneklenebilir — ve yalnızca kare aksi halde bayt
  bütçesini aşacaksa.
- **Karo önbelleği (FIFO halka).** Daha önce gönderilmiş bir karoya eşit
  bir karo ~7 bayta mal olur (halka-yuva referansı) tam piksel yerine.
  Tekrarlanan arayüz çerçeveleri, geri gelen pencereler, açılıp-kapanan
  paneller, geri-kaydırılan içerik neredeyse bedava hâle gelir.
- **Global-scroll modu.** Tüm kare sabit bir `dy` kadar kaydıysa tek bayt
  taşır; geri-çözümler kaydırmayı tahmin eder ve yalnızca yeni-açılan
  şerit kodlanır. Makale/kod kaydırmasını sorunsuz ve ucuz yapar.
- **Karo-başına hibrit kodlayıcı.** Her karo için kodlayıcı en küçüğünü
  seçer: `MV`-birebir `PRED` (0 yük), önbellek `REF`, zigzag-varint
  artık-değeri, ham, değişen-piksel alt-maskesi, 2×/4× örnekleme veya
  `SKIP`.
- **Tamsayı hareket tahmini** komşu-MV ipucu ve metin-güvenli global
  arama ile.
- **Katmanlı imleç.** Masaüstü imleçsiz kodlanır; imleç2–4 baytlık bir
  yan-kanaldır, istemcide birleştirilir. Görüntülenen kare bit-birebir
  kalır.
- **CBR ritimleme** yakalama-patlaması yok → düz, dürüst bant genişliği.
- **Tek-akış deflate** tüm karo yükleri için (karo-başına zlib başlığı yok).

---

##10 KB/s neden mümkün (ve nerede değil)

30 fps'de2K masaüstü ham olarak ~331 MB/s'dir. Onu10 KB/s'de kayıpsız
göndermenin **tek yolu**, çoğu zaman *neredeyse hiçbir şey* göndermektir —
masaüstü çalışması tam olarak budur: statik bölgeler, hareketli bir imleç,
küçük metin düzenlemeleri. LOWLAT şunları kullanır:

1. zamansal redundancy (kirli karolar + hareket + global scroll),
2. karolar içindeki uzamsal redundancy (artık-değer/alt-maske + deflate),
3. tekrar (karo önbelleği).

**Saf rastgele gürültü** veya **tam-hareketli doğal video** için kayıpsız
10 KB/s *bilgi-kurumsal olarak imkânsızdır* (Shannon). LOWLAT bunu dürüstçe
raporlar: `noise_demo.py --mode true` her genel-amaçlı codec'i (ve bizimkini)
gerçek rastgele karelerde ~1.00× gösterir (~178 MB/s gerekir). **Tohum-tanımlı**
sentetik gürültü için `noise_demo.py --mode seed`8 baytlık bir tohum gönderir
ve1080p kareyi bit-birebir yeniden üretir —0.23 KB/s'ye.

---

## Derleme ve çalıştırma

Gereksinimler: Python3.10+, `numpy`, `opencv-python-headless` ve `gcc` +
`zlib`. C çekirdeği ilk import'ta kendiliğinden derlenir.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# canlı pencere (ffplay), gerçek zamanlı
python live_gui.py --res 720
python live_gui.py --res 2k --activity busy

# yayın yaparken kaydet
python live_gui.py --res 1080 --record out/live.mp4 --seconds 12

# benchmark'lar
python bench_all.py          # tüm senaryolar × çözünürlükler
python final_bench.py        # VNC-stili zlib tabanına karşı
python noise_demo.py --mode seed    #0.23 KB/s'de kayıpsız gürültü
python noise_demo.py --mode true    # imkânsızlık ispatı

# birim testler
pip install pytest && pytest -q
```

`live_gui.py` şunları gösterir: `SOURCE | DECODED`, canlı bant-grafiği
(LINK vs CONTENT), gecikme, önbellek isabetleri ve `BIT-EXACT OK` /
`ADAPTIVE nT (text exact)`.

---

## Depo düzeni

```
lowlat/
  ccodec.c / ccodec.py     # C codec çekirdeği + ctypes sarmalayıcı  (motor)
  rate.py                  # sıkı CBR ritimleyici (yakalama yok)
  vnc_scenario.py          # masaüstü / makale / stres içerik üreteçleri
  noise.py                 # tohum-tanımlı deterministik gürültü
  content.py               # genel amaçlı animasyonlu içerik
  # --- daha önceki neural deneme (isteğe bağlı, torch gerektirir) ---
  model.py train.py codec.py ans.c ans.py freq.c freq.py sparse.py classical.py tiles.py
stream/                    # LOWLAT-S: oran-kontrollü streaming modu (15 KB/s demosu)
  codec.py                 #   karo ızgarası, prob'lar, kalite zeminleri, ritimli bütçe
  run.py                   #   bir kaydı bütçe altında kodla + istatistik
  compare.py               #   yan-yana / iletişim sayfası / oran+PSNR çizelgesi
  latency.py               #   değişiklik→ekran gecikmesi ölçümü
experiments/streaming/     # yayınlanmış15 KB/s demosu (iletişim sayfası, çizelge, istatistik)
live_gui.py                # canlı görsel demo (ffplay penceresi + HUD)
final_bench.py demo.py live_demo.py noise_demo.py bench_all.py
tests/test_codec.py
```

## Lisans

MIT — `LICENSE` dosyasına bakın. İstediğiniz gibi kullanın.
