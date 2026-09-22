# Müşteri Kaybı (Churn) Tahmini

Bu proje, bir abonelik/telekom işletmesinin müşteri verisi üzerinde uçtan uca bir
makine öğrenmesi çalışmasıdır: keşifsel veri analizinden modele, model
değerlendirmesinden iş önerilerine kadar tüm süreci kapsar.

## Proje Özeti

6.000 müşterilik veri seti üzerinde hangi müşterilerin hizmeti bırakma
(churn) olasılığının yüksek olduğunu tahmin eden iki sınıflandırma modeli
geliştirdim: **Lojistik Regresyon** ve **Random Forest**. Modelleri
**scikit-learn gibi hazır bir kütüphane kullanmadan, NumPy ile sıfırdan
kodladım** — amaç, algoritmaların iç mantığını (gradyan inişi, Gini
impurity, bootstrap agregasyonu) gösterebilmekti.

## Veri Seti

Gerçekçi ilişkiler içerecek şekilde parametrik olarak üretilmiştir
(`src/generate_data.py`): aylık sözleşmeli, düşük abonelik süresi olan,
teknik destek/online güvenliği olmayan ve sık destek çağrısı yapan
müşterilerin churn olasılığı daha yüksek olacak şekilde modellenmiştir.

| Alan | Açıklama |
|---|---|
| Sözleşme Tipi | Aylık / 1 Yıllık / 2 Yıllık |
| Abonelik Süresi | Ay cinsinden |
| İnternet Hizmeti, Teknik Destek, Online Güvenlik | Hizmet detayları |
| Aylık Ücret, Toplam Ücret | Finansal alanlar |
| Destek Çağrısı Sayısı | Müşteri memnuniyetsizliği göstergesi |
| Churn | Hedef değişken (0/1) |

## Yöntem

1. **Keşifsel Veri Analizi** — sözleşme tipi, abonelik süresi ve destek
   çağrısı sayısının churn ile ilişkisi görselleştirildi
2. **Feature Engineering** — kategorik değişkenler one-hot encoding ile
   dönüştürüldü, sayısal değişkenler standardize edildi
3. **Model Eğitimi (sıfırdan NumPy implementasyonu)**
   - `LogisticRegressionScratch`: gradyan inişi + L2 regularizasyon
   - `DecisionTreeScratch` / `RandomForestScratch`: Gini impurity ile CART,
     bootstrap agregasyonu ve rastgele öznitelik alt kümesi seçimi
4. **Değerlendirme** — accuracy, precision, recall, F1, ROC eğrisi ve AUC,
   karmaşıklık matrisi (hepsi sıfırdan hesaplandı)
5. **Değişken Önem Analizi** — Random Forest'ın hangi değişkenlere en çok
   ağırlık verdiği görselleştirildi

## Sonuçlar

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Lojistik Regresyon | 0.737 | 0.662 | 0.575 | 0.616 | 0.785 |
| Random Forest | 0.713 | 0.638 | 0.500 | 0.561 | 0.771 |

En önemli değişkenler: **abonelik süresi, aylık ücret, toplam ücret, yaş,
destek çağrısı sayısı** ve **aylık sözleşme tipi**.

## Kullanılan Teknolojiler

- Python
- NumPy (sıfırdan model implementasyonu ve metrik hesaplama)
- Pandas (veri işleme, one-hot encoding)
- Matplotlib (görselleştirme)

## Proje Yapısı

```
customer-churn-prediction/
├── data/                    # Üretilen veri seti (CSV)
├── src/
│   ├── generate_data.py     # Sentetik veri üretimi
│   ├── ml_models.py         # Sıfırdan Lojistik Regresyon + Random Forest
│   └── analysis.py          # EDA + model eğitimi + görselleştirme
├── outputs/                 # Üretilen grafikler (PNG)
└── README.md
```

## Çalıştırma

```bash
python src/generate_data.py   # Veri setini üretir
python src/analysis.py        # EDA yapar, modelleri eğitir, grafikleri kaydeder
```

## İş Önerileri

- Aylık sözleşmeli müşteriler (%50 churn oranı) yıllık sözleşmeye
  geçirilmeye teşvik edilmeli (indirim/kampanya)
- İlk 12 ay içindeki müşteriler için proaktif memnuniyet takibi yapılmalı
- 2+ destek çağrısı yapan müşteriler risk sinyali olarak izlenmeli
- Teknik destek ve online güvenlik paketleri olmayan müşterilere bu
  hizmetler önerilmeli

## Yazar

Bu proje, veri bilimi portföyü amacıyla hazırlanmıştır.
