"""Bionluk vitrin/portfolyo kapağı için 16:9 küçük resim (1400x788px)."""
import os
import numpy as np
import matplotlib.pyplot as plt

from analysis import (
    load_data, build_features, train_test_split, standardize, clean_axes,
    SURFACE, INK_PRIMARY, INK_SECONDARY, INK_MUTED, CAT_1_BLUE, CAT_2_ORANGE,
    CAT_8_RED, STATUS_GOOD, SEQ_BLUE, OUT_DIR,
)
from ml_models import LogisticRegressionScratch, RandomForestScratch, roc_curve_scratch

plt.rcParams.update({
    "font.family": "Segoe UI",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def build_thumbnail(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.88))  # 1400x788px @100dpi
    fig.suptitle("Müşteri Kaybı (Churn) Tahmini", fontsize=20,
                 fontweight="bold", color=INK_PRIMARY, x=0.03, ha="left", y=0.97)
    fig.text(0.03, 0.905, "Sıfırdan NumPy ile Lojistik Regresyon & Random Forest",
              fontsize=12.5, color=INK_MUTED, ha="left")

    # Sol panel: sözleşme tipine göre churn oranı
    ax = axes[0]
    rate = df.groupby("Sözleşme Tipi")["Churn"].mean().reindex(["Aylık", "1 Yıllık", "2 Yıllık"]) * 100
    colors = [CAT_8_RED, SEQ_BLUE[3], STATUS_GOOD]
    bars = ax.bar(rate.index, rate.values, color=colors, width=0.55, zorder=3)
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"%{val:.0f}",
                ha="center", fontsize=13, color=INK_SECONDARY, fontweight="bold")
    clean_axes(ax)
    ax.tick_params(labelsize=12)
    ax.set_title("Sözleşme Tipine Göre Churn Oranı", fontsize=15, fontweight="bold",
                 color=INK_PRIMARY, loc="left", pad=12)

    # Sağ panel: ROC eğrisi
    ax = axes[1]
    X, y, _ = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_s, X_test_s = standardize(X_train, X_test)

    log_reg = LogisticRegressionScratch(lr=0.3, n_iter=1500, l2=0.01).fit(X_train_s, y_train)
    fpr_lr, tpr_lr, auc_lr = roc_curve_scratch(y_test, log_reg.predict_proba(X_test_s))

    rf = RandomForestScratch(n_trees=15, max_depth=6, min_samples_split=25, seed=42).fit(X_train, y_train)
    fpr_rf, tpr_rf, auc_rf = roc_curve_scratch(y_test, rf.predict_proba(X_test))

    ax.plot(fpr_lr, tpr_lr, linewidth=2.8, color=CAT_1_BLUE, label=f"Lojistik Reg. (AUC={auc_lr:.2f})")
    ax.plot(fpr_rf, tpr_rf, linewidth=2.8, color=CAT_2_ORANGE, label=f"Random Forest (AUC={auc_rf:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1.2)
    clean_axes(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=11)
    ax.set_title("ROC Eğrisi — Model Performansı", fontsize=15, fontweight="bold",
                 color=INK_PRIMARY, loc="left", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out_path = os.path.join(OUT_DIR, "vitrin_kapak_thumbnail_1400x788.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Kaydedildi: {out_path}")


if __name__ == "__main__":
    df = load_data()
    build_thumbnail(df)
