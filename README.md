# PV Vision AI

PV Vision AI, fotovoltaik güneş hücrelerinin elektrolüminesans (EL) görüntülerinde bulunan üretim kusurlarını YOLO tabanlı nesne tespiti ile belirlemeyi hedefleyen bir bilgisayar mühendisliği staj projesidir.

V1 hedefi; kullanıcının bir EL görüntüsü yüklediği, sistemin eğitilmiş bir YOLO modeliyle kusurları tespit ettiği ve sonucu Türkçe, anlaşılır bir Streamlit arayüzünde gösterdiği çalışan bir uygulama geliştirmektir.

## Proje Kapsamı

- PVEL-AD veri setinin incelenmesi
- Annotation formatının YOLO nesne tespiti formatına dönüştürülmesi
- Ultralytics YOLO ile model eğitimi
- Model değerlendirme metriklerinin raporlanması
- Streamlit tabanlı Türkçe web arayüzü
- Kusur konumlarının bounding box ile görselleştirilmesi

## Güncel Proje Yapısı

```text
PV Vision AI/
├── .streamlit/
│   └── config.toml
├── app/
│   ├── app.py
│   ├── services/model_service.py
│   └── utils/image_utils.py
├── data/
│   ├── raw/
│   └── processed/dataset.yaml
├── models/
│   └── weights/
│       ├── best.pt
│       └── model_info.json
├── outputs/
│   ├── colab/
│   ├── predictions/
│   ├── reports/
│   └── training/
├── tests/
│   └── test_*.py
├── notebooks/
│   └── PV_Vision_AI_Colab_Resume.ipynb
├── training/
│   ├── analyze_dataset.py
│   ├── convert_annotations.py
│   ├── evaluate.py
│   ├── finalize.py
│   ├── predict.py
│   ├── prepare_colab.py
│   ├── prepare_dataset.py
│   ├── status.py
│   └── train.py
├── config.py
├── model_registry.py
├── requirements.txt
└── README.md
```

## Kullanılan Temel Teknolojiler

- Python
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Pillow
- Matplotlib
- PyYAML
- Streamlit

## Türkçe Kusur Sınıfları

Projede kullanıcıya gösterilecek kusur sınıfları `config.py` içinde merkezi olarak tanımlanmıştır:

- Parmak izi kusuru
- Çatlak
- Siyah çekirdek
- Kalın çizgi
- Yatay hizasızlık
- Kısa devre
- Dikey hizasızlık
- Yıldız çatlağı
- Baskı hatası
- Köşe kusuru
- Parçalanma
- Çizik

## Kurulum

Python 3.10 veya 3.11 kullanılması önerilir.
Apple Silicon üzerinde doğrulanan eğitim ortamı `torch==2.8.0`,
`torchvision==0.23.0` ve Ultralytics `8.3.186` sürümlerini kullanır.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows kullanılıyorsa sanal ortam aktivasyonu:

```bash
.venv\Scripts\activate
```

## Veri Seti Yerleşimi

PVEL-AD veri setinin orijinal dosyaları değiştirilmeden şu klasöre konulmalıdır:

```text
data/raw/
```

YOLO formatına dönüştürülmüş veri seti şu klasörde tutulur:

```text
data/processed/
```

## Geliştirme Aşamaları

1. Proje altyapısının oluşturulması
2. Python ortamının hazırlanması
3. PVEL-AD veri setinin alınması
4. Veri setinin incelenmesi
5. Veri setinin YOLO formatına hazırlanması
6. İlk YOLO model eğitimi
7. Model değerlendirmesi
8. Streamlit arayüzünün geliştirilmesi
9. Eğitilmiş modelin uygulamaya bağlanması
10. Test ve V1 teslim

## Veri Hazırlama

Ham PVEL-AD veri seti `data/raw/` altında durmalıdır. YOLO formatındaki veri setini üretmek için:

```bash
python training/prepare_dataset.py
```

Bu komut `data/processed/dataset.yaml`, `images/` ve `labels/` klasörlerini oluşturur.
Komut farklı bir seed veya validation oranıyla yeniden çalıştırılırsa önceki
split'ten kalan üretilmiş dosyalar temizlenir; train/validation/test arasında
aynı görüntü bulunursa veri sızıntısı olarak işlem durdurulur. Ham veri dosyaları
hiçbir durumda değiştirilmez.

## Eğitim

Güncel checkpoint, model aşaması ve sıradaki doğru komutu görmek için:

```bash
python training/status.py
```

Durum aracı `model_info.json` içindeki kaynak run kaydını izler; V1 sonrasında
quality fine-tuning başlatılmışsa eski V1 yerine güncel quality checkpoint'ini
ve ona uygun devam/finalizasyon komutunu gösterir.

Hızlı sistem testi için:

```bash
python training/train.py --preset smoke
```

CPU üzerinde kısa V1 eğitimi için:

```bash
python training/train.py --preset cpu-v1
```

Apple Silicon Mac üzerinde MPS hızlandırmalı gerçek V1 eğitimi için:

```bash
python training/train.py --preset mps-v1
```

Bu preset `10 epoch`, `640px`, `batch=8` ve `device=mps` kullanır. Eğitimi daha
önce kontrollü biçimde durdurduysan mevcut optimizer ve epoch durumuyla toplam
10 epoch'a devam etmek için:

```bash
python training/train.py --resume-from outputs/training/pv_vision_yolov8n_mps_v1/weights/last.pt --resume-epochs 10 --device mps
```

Resume komutu eğitimi sıfırdan başlatmaz. `Ctrl+C` ile durdurulduğunda tamamlanan
son epoch `last.pt` içinde, o ana kadarki en iyi ara model de
`models/weights/best.pt` içinde korunur.

### Google Colab'da 6/10'dan Devam Etme

Train/validation görüntülerini symlink yerine gerçek dosya olarak paketleyen,
ham veri ve resmi test setini dışarıda bırakan Colab arşivini üretmek için:

```bash
python training/prepare_colab.py \
  --output "$HOME/Desktop/pv_vision_colab.tar.gz" \
  --target-epochs 10 \
  --force
```

Arşiv Google Drive'daki `PV_Vision_AI_Colab` klasörüne yüklenir ve
`notebooks/PV_Vision_AI_Colab_Resume.ipynb` Colab ile açılır. Notebook GPU'yu,
3600/900 train-validation eşleşmesini, optimizer durumunu ve tamamlanan epoch'u
doğrular. Drive'da daha yeni bir `last.pt` varsa eski 6 epoch checkpoint'iyle
üzerine yazmaz. Eğitim CUDA `device=0` ile toplam 10 epoch'a ulaşır; test spliti
bu pakette bulunmadığı için eğitim sırasında kullanılamaz.

V1 değerlendirildikten sonra gerekirse daha uzun kalite eğitimi için:

```bash
python training/train.py --preset quality
```

Bu preset temel YOLO ağırlığından yeniden başlamaz; tamamlanmış V1'in merkezi
`models/weights/best.pt` modelini başlangıç ağırlığı olarak kullanarak yeni bir
fine-tuning çalışması açar. Ara veya hash kaydı uyuşmayan modelle kalite eğitimi
başlatılmaz. Kalite çalışması tamamlanırsa finalizasyon komutu şöyledir:

```bash
python training/finalize.py --checkpoint outputs/training/pv_vision_yolov8n_quality/weights/last.pt --target-epochs 60 --allow-early-stop --include-test --device mps --batch 8
```

Eğitim tamamlandığında en iyi model ağırlığı şu konuma kopyalanır:

```text
models/weights/best.pt
```

Modelin ara/deneme/final durumu, epoch bilgisi ve validation metrikleri
`models/weights/model_info.json` dosyasında tutulur.

## Veri Seti Sınıf Analizi

Train/validation/test sınıf dağılımını, az örnekli sınıfları ve validation
kapsama eksiklerini raporlamak için:

```bash
python training/analyze_dataset.py
```

CSV, JSON ve Türkçe Markdown raporları `outputs/reports/` altında üretilir.
Mevcut veri dağılımında Köşe kusuru, Parçalanma ve Çizik sınıfları çok az eğitim
örneğine sahiptir; Köşe kusuru ve Baskı hatası validation setinde hiç yoktur.
Uygulama bu sınırlılıkları “Model kapsamı ve veri sınırlılıkları” bölümünde
açıkça gösterir.

## Değerlendirme

Validation seti üzerinde değerlendirme:

```bash
python training/evaluate.py --split val --device mps --batch 8
```

Test seti üzerinde değerlendirme:

```bash
python training/evaluate.py --split test --device mps --batch 8
```

Özet ve sınıf bazlı metrik raporları `outputs/reports/` altında; Türkçe sınıf
adlarını kullanan Precision/Recall eğrileri ve karışıklık matrisi
`outputs/training/evaluation_<split>_<model-hash>/` altında üretilir. Her
değerlendirmenin hash içeren değişmez bir rapor kopyası da saklanır; güncel
sonuç aynı zamanda `evaluation_<split>_metrics.json` dosyasına yazılır.
Test değerlendirmesi yalnızca final model hazır olduğunda çalıştırılmalıdır.

## Finalizasyon

Eğitimin gerçekten 10 epoch tamamlandığını hiçbir dosyayı değiştirmeden kontrol etmek:

```bash
python training/finalize.py --check-only
```

Ultralytics normal tamamlanan eğitimde `last.pt` dosyasını küçültüp iç epoch
alanını temizlediği için finalizasyon ayrıca `results.csv` satırlarını doğrular.
Kesilmiş bir eğitimde optimizer ve checkpoint epoch bilgisi bulunduğundan yarım
çalışma yanlışlıkla tamamlanmış kabul edilmez.

Kontrol başarılı olduktan sonra validation ve resmi test değerlendirmesini
çalıştırıp kalite eşiğini geçen modeli final olarak yayımlamak:

```bash
python training/finalize.py --include-test --device mps --batch 8
```

Test seti 19.150 görüntü içerdiği için bu komut uzun sürebilir. Aynı modelin test
setinde yanlışlıkla tekrar tekrar değerlendirilmesi engellenir. Final kabul için
aggregate Precision, Recall ve mAP eşikleriyle birlikte, yeterli eğitim örneğine
sahip sınıfların sınıf bazlı mAP50 değerleri de kontrol edilir. Sonuç
`outputs/reports/final_delivery_summary.md` dosyasına yazılır.
Final test tamamlandıktan sonra süreç kesilirse aynı modelin hash ile doğrulanmış
test raporu yeniden kullanılır; kanonik rapor kaybolsa bile değişmez hash raporu
aynı modeli tanır ve 19.150 görüntülük test gereksiz yere tekrarlanmaz.
Validation kalite eşiği geçmezse resmi test seti hiç açılmaz; model önce
validation sonuçlarına göre iyileştirilir.

V1 teknik kabul eşikleri: Precision ≥ 0.40, Recall ≥ 0.30, mAP50 ≥ 0.35,
mAP50-95 ≥ 0.20. Eğitim setinde en az 100 kutusu bulunan sınıfların mAP50
değeri de en az 0.10 olmalıdır. Az örnekli sınıflar bu kapıdan gizlice
geçirilmez; veri seti sınırlılığı olarak ayrıca raporlanır.

## Tahmin Çıktısı Üretme

Tek bir görüntü için kutulu sonuç, CSV tespit tablosu ve Türkçe özet üretmek:

```bash
python training/predict.py --source data/processed/images/val/img000004.jpg
```

Bir klasördeki görüntüler için:

```bash
python training/predict.py --source data/processed/images/val --conf 0.25
```

Çıktılar `outputs/predictions/` altında saklanır.

## Web Uygulaması

Streamlit arayüzünü başlatmak için:

```bash
streamlit run app/app.py
```

Uygulamada EL görüntüsü yüklenir, model analizi başlatılır ve tespit edilen
kusurlar hem tabloda hem de görüntü üzerindeki kutularda Türkçe isimlerle
gösterilir. İşaretlenmiş PNG, CSV tespit tablosu ve Türkçe analiz özeti
indirilebilir.

### Kalite puanı ve fiyat önerisi

Web uygulaması her analizde kusur adedini, YOLO kutularının görüntüde kapladığı
çakışmasız yaklaşık alanı, 0-100 kalite puanını ve A/B/C kalite sınıfını
gösterir. İlk V1 puanı; kusur türünün merkezi önem ağırlığı, kutu alanı, model
güveni ve kusur adedinden açıklanabilir şekilde hesaplanır:

- A: 85-100, fiyat katsayısı 0.95
- B: 60-84.9, fiyat katsayısı 0.75
- C: 0-59.9, fiyat katsayısı 0.45

`Fiyat önerisini hesapla` seçeneği isteğe bağlıdır. Referans tutar TRY, USD veya
EUR olarak girilebilir; kur dönüşümü yapılmaz. Öneri, referans fiyatın kalite
sınıfı katsayısıyla çarpılmasıdır. Bu değer bir ekspertiz veya kesin piyasa
değeri değil, kalibre edilmeye açık tahmini karar desteğidir. Alan değeri de
fiziksel panel yüzdesi değil, yüklenen EL görüntüsündeki tespit kutularının
yaklaşık birleşimidir.

### Tahmini üretim performansı

Her analizde nominal panel gücü varsayılan `550 W` olmak üzere değiştirilebilir.
Sistem yeni bir model çalıştırmadan mevcut kalite sonucundan tahmini performansı
hesaplar: `performans = 100 - ((100 - kalite puanı) x 0.30)`. Tahmini panel gücü,
nominal güç ile bu oranın çarpımıdır. Nominal güç değiştiğinde mevcut YOLO sonucu
korunur ve yalnızca güç hesabı yenilenir.

Bu değer görüntüdeki kusurlara dayalı açıklanabilir bir V1 tahminidir; gerçek
elektriksel ölçüm veya enerji üretim garantisi değildir.

### Panel sağlığı, bakım ve kayıp analizi

Uygulama mevcut kalite sonucunu, birleşik kusurlu alanı ve güven ağırlıklı kusur
önemlerini kullanarak 0-100 panel sağlık skoru üretir. Sağlık sonucu `Çok İyi`,
`İyi`, `Orta` veya `Kritik`; risk seviyesi ise kritik kusurlar da dikkate
alınarak `Düşük`, `Orta` veya `Yüksek` olarak gösterilir. Puan düşüşü kalite,
alan ve kusur sınıfı etkileriyle açıklanır.

Bakım bölümü risk seviyesine ek olarak en etkili üç kusur sınıfına özel,
tekrarsız kontrol önerileri sunar. Performans kaybı tahmini üretim performansının
100'den farkıdır. Fiyat hesabı etkinleştirildiğinde mevcut A/B/C katsayılarıyla
değer kaybı yüzdesi, kayıp tutarı ve tahmini panel değeri gösterilir. Bu
sonuçlar gerçek elektriksel ölçüm, laboratuvar testi veya kesin finansal değer
yerine geçmez.

## Otomatik Testler

Veri seti bütünlüğü, VOC → YOLO dönüşümü, sınıf sırası, checkpoint resume,
model yayımlama, Türkçe tahmin çıktısı ve Streamlit açılış testi:

```bash
python -m unittest discover -s tests -v
```

## Mevcut Durum

Proje altyapısı, PVEL-AD veri dönüşümü, güvenli checkpoint resume, model
değerlendirme, tahmin çıktı aracı, otomatik testler ve Streamlit V1 arayüzü
hazırdır. Uygulama `model_info.json` içindeki duruma göre deneme, ara, final adayı
veya final model kullandığını açıkça gösterir. Gerçek V1 modelinin final kabul edilmesi için
10 epoch eğitimin tamamlanması ve ardından validation/test değerlendirmesinin
raporlanması gerekir.
