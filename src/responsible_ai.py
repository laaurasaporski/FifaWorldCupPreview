"""Gera as figuras da análise de IA responsável (Etapa 3).

Salva PNGs em reports/figures/. Foco: por que o empate é difícil, o que o
modelo aprendeu, e quão confiáveis são suas probabilidades.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, recall_score, accuracy_score

from .features import FEATURE_COLUMNS
from .modeling import (
    build_models, prepare, temporal_split, train_calibrated,
    naive_baseline_accuracy,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
LABELS = ["home_win", "draw", "away_win"]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare()
    train, test = temporal_split(df)
    Xte, yte = test[FEATURE_COLUMNS], test["result"]

    gb = build_models()["grad_boosting"]
    gb.fit(train[FEATURE_COLUMNS], train["result"])
    gb_bal = build_models(class_weight="balanced")["grad_boosting"]
    gb_bal.fit(train[FEATURE_COLUMNS], train["result"])
    gb_cal = train_calibrated(train, build_models()["grad_boosting"])

    # 1. Matriz de confusao (gradient boosting)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        yte, gb.predict(Xte), labels=LABELS, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Matriz de confusão — Gradient Boosting")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_confusion_matrix.png", dpi=130); plt.close(fig)

    # 2. Importancia das features (permutacao)
    perm = permutation_importance(gb, Xte, yte, n_repeats=5, random_state=42, n_jobs=-1)
    order = perm.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(FEATURE_COLUMNS)[order], perm.importances_mean[order], color="#2b6cb0")
    ax.set_xlabel("Queda de acurácia ao embaralhar a feature")
    ax.set_title("Importância das features (permutação)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_permutation_importance.png", dpi=130); plt.close(fig)

    # 3. O trade-off do empate
    variants = {
        "Gradient\nBoosting": gb,
        "GB +\nclass_weight": gb_bal,
        "GB\ncalibrado": gb_cal,
    }
    draw_recalls, accs = [], []
    for m in variants.values():
        p = m.predict(Xte)
        draw_recalls.append(recall_score(yte, p, labels=["draw"], average="macro"))
        accs.append(accuracy_score(yte, p))
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, accs, w, label="Acurácia geral", color="#2b6cb0")
    ax.bar(x + w/2, draw_recalls, w, label="Recall do empate", color="#dd6b20")
    ax.axhline(naive_baseline_accuracy(test), ls="--", c="gray", label="Baseline ingênuo")
    ax.set_xticks(x); ax.set_xticklabels(variants.keys())
    ax.set_ylim(0, 1); ax.set_title("Trade-off: acurácia geral × recall do empate")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_draw_tradeoff.png", dpi=130); plt.close(fig)

    # 4. Curva de calibracao para a classe "empate"
    fig, ax = plt.subplots(figsize=(6, 5))
    y_draw = (yte == "draw").astype(int)
    for name, m in [("Sem calibrar", gb), ("Calibrado", gb_cal)]:
        idx = list(m.classes_).index("draw")
        prob = m.predict_proba(Xte)[:, idx]
        frac, mean_pred = calibration_curve(y_draw, prob, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="Calibração perfeita")
    ax.set_xlabel("Probabilidade prevista de empate")
    ax.set_ylabel("Frequência real de empate")
    ax.set_title("Confiabilidade das probabilidades (empate)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_calibration_draw.png", dpi=130); plt.close(fig)

    print("Figuras salvas em reports/figures/:")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
