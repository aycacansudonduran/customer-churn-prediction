"""
Müşteri Kaybı (Churn) Tahmini — Uçtan Uca Analiz
EDA -> Feature Engineering -> Model Eğitimi (sıfırdan) -> Değerlendirme -> Vitrin Dashboard
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ml_models import (
    LogisticRegressionScratch, RandomForestScratch,
    classification_metrics, roc_curve_scratch,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "customer_churn.csv")
OUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Palet (dataviz skill referansı) ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CAT_1_BLUE = "#2a78d6"
CAT_2_ORANGE = "#eb6834"
CAT_3_AQUA = "#1baf7a"
CAT_8_RED = "#e34948"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 11,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def clean_axes(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0)
    if y_grid:
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
        ax.set_axisbelow(True)


# ------------------------------------------------------------------
# Veri yükleme ve feature engineering
# ------------------------------------------------------------------
def load_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def build_features(df):
    df = df.copy()
    numeric_cols = ["Yas", "Abonelik Süresi (Ay)", "Destek Çağrısı Sayısı", "Aylık Ücret", "Toplam Ücret"]
    categorical_cols = ["Cinsiyet", "Partner", "Bakmakla Yükümlü", "Sözleşme Tipi",
                         "İnternet Hizmeti", "Teknik Destek", "Online Güvenlik",
                         "Kağıtsız Fatura", "Ödeme Yöntemi"]

    X_num = df[numeric_cols].to_numpy(dtype=float)
    X_cat = pd.get_dummies(df[categorical_cols], drop_first=True)
    feature_names = numeric_cols + list(X_cat.columns)
    X = np.hstack([X_num, X_cat.to_numpy(dtype=float)])
    y = df["Churn"].to_numpy(dtype=float)
    return X, y, feature_names


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ------------------------------------------------------------------
# EDA grafikleri
# ------------------------------------------------------------------
def chart_churn_by_contract(df, ax=None, standalone=True):
    rate = df.groupby("Sözleşme Tipi")["Churn"].mean().reindex(["Aylık", "1 Yıllık", "2 Yıllık"]) * 100
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = [CAT_8_RED, SEQ_BLUE[3], STATUS_GOOD]
    bars = ax.bar(rate.index, rate.values, color=colors, width=0.55, zorder=3)
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"%{val:.0f}",
                ha="center", fontsize=10.5, color=INK_SECONDARY, fontweight="bold")
    clean_axes(ax)
    ax.set_title("Sözleşme Tipine Göre Churn Oranı", fontsize=14, fontweight="bold",
                 color=INK_PRIMARY, loc="left", pad=14)
    ax.set_ylabel("Churn Oranı (%)")
    if standalone and fig:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "01_sozlesme_churn.png"), dpi=200)
        plt.close(fig)


def chart_tenure_distribution(df, ax=None, standalone=True):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    churned = df[df["Churn"] == 1]["Abonelik Süresi (Ay)"]
    stayed = df[df["Churn"] == 0]["Abonelik Süresi (Ay)"]
    bins = np.linspace(0, 72, 19)
    ax.hist(stayed, bins=bins, color=CAT_1_BLUE, alpha=0.55, label="Kalan Müşteri", zorder=3)
    ax.hist(churned, bins=bins, color=CAT_8_RED, alpha=0.55, label="Kaybedilen Müşteri", zorder=3)
    clean_axes(ax)
    ax.set_title("Abonelik Süresine Göre Churn Dağılımı", fontsize=14, fontweight="bold",
                 color=INK_PRIMARY, loc="left", pad=14)
    ax.set_xlabel("Abonelik Süresi (Ay)")
    ax.legend(frameon=False, fontsize=9.5)
    if standalone and fig:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "02_abonelik_suresi_dagilimi.png"), dpi=200)
        plt.close(fig)


def chart_support_calls(df, ax=None, standalone=True):
    rate = df.groupby("Destek Çağrısı Sayısı")["Churn"].mean() * 100
    rate = rate[rate.index <= 7]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(rate.index, rate.values, color=CAT_2_ORANGE, linewidth=2.5, marker="o",
            markersize=6, zorder=3)
    clean_axes(ax)
    ax.set_title("Destek Çağrısı Sayısı ile Churn İlişkisi", fontsize=14, fontweight="bold",
                 color=INK_PRIMARY, loc="left", pad=14)
    ax.set_xlabel("Destek Çağrısı Sayısı")
    ax.set_ylabel("Churn Oranı (%)")
    if standalone and fig:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "03_destek_cagrisi_churn.png"), dpi=200)
        plt.close(fig)


def chart_roc_curve(results, ax=None, standalone=True):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = {"Lojistik Regresyon": CAT_1_BLUE, "Random Forest": CAT_2_ORANGE}
    for name, res in results.items():
        fpr, tpr, auc = res["fpr"], res["tpr"], res["auc"]
        ax.plot(fpr, tpr, linewidth=2.5, color=colors[name], label=f"{name} (AUC={auc:.3f})", zorder=3)
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1.2, zorder=2)
    clean_axes(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)")
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)")
    ax.set_title("ROC Eğrisi — Model Karşılaştırması", fontsize=14, fontweight="bold",
                 color=INK_PRIMARY, loc="left", pad=14)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    if standalone and fig:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "04_roc_egrisi.png"), dpi=200)
        plt.close(fig)


def chart_confusion_matrix(cm, title, ax=None, standalone=True, fname=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    labels = np.array([["Doğru Negatif", "Yanlış Pozitif"], ["Yanlış Negatif", "Doğru Pozitif"]])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() * 0.55 else INK_PRIMARY
            ax.text(j, i, f"{labels[i, j]}\n{cm[i, j]}", ha="center", va="center",
                    fontsize=10.5, color=text_color, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Tahmin: Kalır", "Tahmin: Kaybedilir"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Gerçek: Kalır", "Gerçek: Kaybedilir"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK_PRIMARY, loc="left", pad=12)
    if standalone and fig:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=200)
        plt.close(fig)


def _pretty_feature_name(name):
    mapping = {
        "Abonelik Süresi (Ay)": "Abonelik Süresi",
        "Destek Çağrısı Sayısı": "Destek Çağrısı",
        "Yas": "Yaş",
    }
    if name in mapping:
        return mapping[name]
    if "_" in name:
        base, val = name.split("_", 1)
        return f"{base}: {val}"
    return name


def chart_feature_importance(importances, feature_names, ax=None, standalone=True, top_n=8):
    order = np.argsort(importances)[::-1][:top_n]
    names = [_pretty_feature_name(feature_names[i]) for i in order][::-1]
    vals = importances[order][::-1] * 100
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [SEQ_BLUE[min(len(SEQ_BLUE) - 1, max(1, top_n - 1 - i))] for i in range(top_n)]
    bars = ax.barh(names, vals, color=colors, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(val + vals.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                f"%{val:.1f}", va="center", fontsize=9.5, color=INK_SECONDARY)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks([])
    ax.set_title("Random Forest — En Önemli 8 Değişken", fontsize=14, fontweight="bold",
                 color=INK_PRIMARY, loc="left", pad=14)
    if standalone and fig:
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "06_feature_importance.png"), dpi=200)
        plt.close(fig)


def build_dashboard(df, results, cm_rf, importances, feature_names):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Müşteri Kaybı (Churn) Tahmini — Model Özeti", fontsize=17,
                 fontweight="bold", color=INK_PRIMARY, x=0.02, ha="left", y=0.975)
    fig.text(0.02, 0.938, "Python · NumPy · Sıfırdan Lojistik Regresyon & Random Forest",
              fontsize=10, color=INK_MUTED, ha="left")

    gs = fig.add_gridspec(2, 2, left=0.09, right=0.97, top=0.86, bottom=0.07, hspace=0.6, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    chart_churn_by_contract(df, ax=ax1, standalone=False)
    chart_roc_curve(results, ax=ax2, standalone=False)
    chart_feature_importance(importances, feature_names, ax=ax3, standalone=False, top_n=6)
    chart_confusion_matrix(cm_rf, "Random Forest — Karmaşıklık Matrisi", ax=ax4, standalone=False)

    fig.savefig(os.path.join(OUT_DIR, "00_dashboard_vitrin.png"), dpi=200)
    plt.close(fig)


def main():
    df = load_data()
    print("Veri yüklendi:", df.shape)

    chart_churn_by_contract(df)
    chart_tenure_distribution(df)
    chart_support_calls(df)

    X, y, feature_names = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_s, X_test_s = standardize(X_train, X_test)

    print("\n--- Lojistik Regresyon eğitiliyor (sıfırdan, gradyan inişi) ---")
    log_reg = LogisticRegressionScratch(lr=0.3, n_iter=3000, l2=0.01)
    log_reg.fit(X_train_s, y_train)
    proba_lr = log_reg.predict_proba(X_test_s)
    pred_lr = (proba_lr >= 0.5).astype(int)
    metrics_lr = classification_metrics(y_test, pred_lr)
    fpr_lr, tpr_lr, auc_lr = roc_curve_scratch(y_test, proba_lr)
    print(f"Lojistik Regresyon -> Accuracy: {metrics_lr['accuracy']:.3f}, "
          f"Precision: {metrics_lr['precision']:.3f}, Recall: {metrics_lr['recall']:.3f}, "
          f"F1: {metrics_lr['f1']:.3f}, AUC: {auc_lr:.3f}")

    print("\n--- Random Forest eğitiliyor (sıfırdan, 25 ağaç) ---")
    rf = RandomForestScratch(n_trees=25, max_depth=6, min_samples_split=25, seed=42)
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)
    pred_rf = (proba_rf >= 0.5).astype(int)
    metrics_rf = classification_metrics(y_test, pred_rf)
    fpr_rf, tpr_rf, auc_rf = roc_curve_scratch(y_test, proba_rf)
    print(f"Random Forest -> Accuracy: {metrics_rf['accuracy']:.3f}, "
          f"Precision: {metrics_rf['precision']:.3f}, Recall: {metrics_rf['recall']:.3f}, "
          f"F1: {metrics_rf['f1']:.3f}, AUC: {auc_rf:.3f}")

    results = {
        "Lojistik Regresyon": {"fpr": fpr_lr, "tpr": tpr_lr, "auc": auc_lr},
        "Random Forest": {"fpr": fpr_rf, "tpr": tpr_rf, "auc": auc_rf},
    }
    chart_roc_curve(results)
    chart_confusion_matrix(metrics_lr["confusion_matrix"], "Lojistik Regresyon — Karmaşıklık Matrisi",
                            fname="05a_confusion_logreg.png")
    chart_confusion_matrix(metrics_rf["confusion_matrix"], "Random Forest — Karmaşıklık Matrisi",
                            fname="05b_confusion_rf.png")

    importances = rf.feature_importances(len(feature_names))
    chart_feature_importance(importances, feature_names)

    build_dashboard(df, results, metrics_rf["confusion_matrix"], importances, feature_names)

    print("\nTüm grafikler outputs/ klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
