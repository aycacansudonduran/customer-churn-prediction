"""
Eğitilmiş Lojistik Regresyon modelini (ağırlıklar, bias, standardizasyon
parametreleri) JSON'a aktarır — tarayıcıda (JavaScript) çalıştırmak için.
Sunucu gerektirmeyen bir web uygulaması için kullanılır.
"""
import os
import json
import numpy as np

from analysis import load_data, build_features, standardize
from ml_models import LogisticRegressionScratch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(BASE_DIR, "..", "docs")
os.makedirs(WEBAPP_DIR, exist_ok=True)

CATEGORICAL_COLS = ["Cinsiyet", "Partner", "Bakmakla Yükümlü", "Sözleşme Tipi",
                     "İnternet Hizmeti", "Teknik Destek", "Online Güvenlik",
                     "Kağıtsız Fatura", "Ödeme Yöntemi"]
NUMERIC_COLS = ["Yas", "Abonelik Süresi (Ay)", "Destek Çağrısı Sayısı", "Aylık Ücret", "Toplam Ücret"]


def main():
    df = load_data()
    X, y, feature_names = build_features(df)

    # Tüm veri üzerinde standardizasyon parametreleri (deploy edilen model için)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_s = (X - mean) / std

    model = LogisticRegressionScratch(lr=0.3, n_iter=4000, l2=0.01)
    model.fit(X_s, y)

    proba = model.predict_proba(X_s)
    pred = (proba >= 0.5).astype(int)
    acc = (pred == y).mean()
    print(f"Tüm veri üzerinde accuracy: {acc:.3f}")

    # Her kategorik sütun için hangi değerin "referans" (dummy'si olmayan) olduğunu bul
    categories = {}
    for col in CATEGORICAL_COLS:
        all_values = sorted(df[col].unique().tolist())
        dummy_values = [v for v in all_values if f"{col}_{v}" in feature_names]
        baseline = [v for v in all_values if v not in dummy_values]
        categories[col] = {"all_values": all_values, "baseline": baseline[0] if baseline else all_values[0]}

    export = {
        "feature_order": feature_names,
        "weights": model.weights.tolist(),
        "bias": float(model.bias),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "categories": categories,
        "accuracy_on_full_data": round(float(acc), 4),
    }

    out_path = os.path.join(WEBAPP_DIR, "model.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"Model dışa aktarıldı -> {out_path}")
    print("Feature order:", feature_names)
    print("Categories:", json.dumps(categories, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
