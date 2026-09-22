"""
Sentetik ama gerçekçi bir abonelik/telekom müşteri kaybı (churn) veri seti üretir.
Telco Customer Churn tarzı bir yapı, gerçekçi değişken ilişkileriyle.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = 6000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n=N) -> pd.DataFrame:
    contract_type = RNG.choice(
        ["Aylık", "1 Yıllık", "2 Yıllık"], size=n, p=[0.58, 0.24, 0.18]
    )
    tenure_months = np.clip(RNG.gamma(shape=2.2, scale=14, size=n), 0, 72).round().astype(int)
    internet_service = RNG.choice(["Fiber", "DSL", "Yok"], size=n, p=[0.45, 0.35, 0.20])
    tech_support = RNG.choice(["Var", "Yok"], size=n, p=[0.35, 0.65])
    online_security = RNG.choice(["Var", "Yok"], size=n, p=[0.32, 0.68])
    payment_method = RNG.choice(
        ["Otomatik Ödeme", "Elektronik Çek", "Posta Çeki"], size=n, p=[0.45, 0.35, 0.20]
    )
    paperless_billing = RNG.choice(["Evet", "Hayır"], size=n, p=[0.6, 0.4])
    gender = RNG.choice(["Kadın", "Erkek"], size=n, p=[0.5, 0.5])
    partner = RNG.choice(["Evet", "Hayır"], size=n, p=[0.48, 0.52])
    dependents = RNG.choice(["Evet", "Hayır"], size=n, p=[0.3, 0.7])
    age = np.clip(RNG.normal(42, 14, size=n), 18, 85).round().astype(int)
    num_support_calls = RNG.poisson(lam=1.4, size=n)
    num_support_calls = np.clip(num_support_calls, 0, 10)

    base_charge = np.where(internet_service == "Fiber", 75, np.where(internet_service == "DSL", 50, 25))
    monthly_charge = base_charge + RNG.normal(10, 12, size=n)
    monthly_charge = np.clip(monthly_charge, 18, 140).round(2)
    total_charges = (monthly_charge * tenure_months * RNG.uniform(0.92, 1.08, size=n)).round(2)

    # ---- Churn olasılığını mantıklı ilişkilerle kur ----
    z = np.full(n, -1.6)
    z += np.where(contract_type == "Aylık", 1.55, np.where(contract_type == "1 Yıllık", 0.15, -0.9))
    z += -0.038 * tenure_months
    z += 0.014 * (monthly_charge - 60)
    z += np.where(tech_support == "Yok", 0.55, -0.25)
    z += np.where(online_security == "Yok", 0.40, -0.20)
    z += 0.22 * num_support_calls
    z += np.where(payment_method == "Elektronik Çek", 0.35, 0.0)
    z += np.where(internet_service == "Fiber", 0.25, 0.0)
    z += np.where(paperless_billing == "Evet", 0.10, -0.05)
    z += np.where(dependents == "Evet", -0.20, 0.0)
    z += RNG.normal(0, 0.55, size=n)  # gerçekçi gürültü

    churn_prob = sigmoid(z)
    churn = (RNG.uniform(size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "MüsteriID": [f"CUST-{10000+i}" for i in range(n)],
        "Cinsiyet": gender,
        "Yas": age,
        "Partner": partner,
        "Bakmakla Yükümlü": dependents,
        "Abonelik Süresi (Ay)": tenure_months,
        "Sözleşme Tipi": contract_type,
        "İnternet Hizmeti": internet_service,
        "Teknik Destek": tech_support,
        "Online Güvenlik": online_security,
        "Kağıtsız Fatura": paperless_billing,
        "Ödeme Yöntemi": payment_method,
        "Destek Çağrısı Sayısı": num_support_calls,
        "Aylık Ücret": monthly_charge,
        "Toplam Ücret": total_charges,
        "Churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "..", "data", "customer_churn.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"{len(df):,} satır üretildi -> {out_path}")
    print(f"Churn oranı: %{df['Churn'].mean()*100:.1f}")
