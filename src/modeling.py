"""Treino, calibração e avaliação com split TEMPORAL.

Modelos: Random Forest (baseline) e HistGradientBoosting (gradient boosting
nativo do scikit-learn — equivalente ao XGBoost, sem dependência extra).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, classification_report, log_loss

from .data import add_target, load_results, PROCESSED_DIR
from .features import FEATURE_COLUMNS, build_features

MODELS_DIR = PROCESSED_DIR.parent.parent / "models"


def temporal_split(df: pd.DataFrame, cutoff: str = "2018-01-01"):
    """Treino = jogos antes do corte; teste = jogos a partir do corte.

    Split temporal (nunca aleatório): simula prever o futuro com o passado.
    """
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


def prepare() -> pd.DataFrame:
    """Carrega, cria target, gera features e salva a base processada."""
    df = build_features(add_target(load_results()))
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "matches_features.csv", index=False)
    return df


def naive_baseline_accuracy(test: pd.DataFrame) -> float:
    """Acurácia de chutar sempre a classe mais comum (mandante vence)."""
    return (test["result"] == "home_win").mean()


def build_models(class_weight=None):
    """Retorna os modelos a comparar."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=18, random_state=42,
            n_jobs=-1, class_weight=class_weight,
        ),
        "grad_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=6,
            random_state=42, class_weight=class_weight,
        ),
    }


def train_calibrated(train: pd.DataFrame, model, calib_cutoff: str = "2014-01-01"):
    """Treina e calibra probabilidades sem leakage.

    Divide o treino em núcleo (mais antigo) e calibração (mais recente),
    ajusta o modelo no núcleo e calibra no pedaço seguinte (método isotônico,
    cv='prefit'). Assim a calibração nunca vê dados de teste.
    """
    core = train[train["date"] < calib_cutoff]
    calib = train[train["date"] >= calib_cutoff]
    model.fit(core[FEATURE_COLUMNS], core["result"])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(calib[FEATURE_COLUMNS], calib["result"])
    return calibrated


def evaluate(name, model, test, *, draws_focus=False):
    preds = model.predict(test[FEATURE_COLUMNS])
    acc = accuracy_score(test["result"], preds)
    print(f"\n===== {name} =====")
    print(f"Acuracia: {acc:.1%}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        ll = log_loss(test["result"], proba, labels=sorted(test['result'].unique()))
        print(f"Log loss: {ll:.3f}  (quanto menor, melhor calibrado)")
    print(classification_report(test["result"], preds, digits=2))
    return acc


def run_comparison(cutoff: str = "2018-01-01"):
    df = prepare()
    train, test = temporal_split(df, cutoff)
    print(f"Treino: {len(train):,} | Teste: {len(test):,}")
    print(f"Baseline ingenuo: {naive_baseline_accuracy(test):.1%}")

    results = {}
    # 1) modelos crus
    for name, mdl in build_models().items():
        mdl.fit(train[FEATURE_COLUMNS], train["result"])
        results[name] = evaluate(name, mdl, test)

    # 2) gradient boosting com peso de classe (foco no empate)
    gb_w = build_models(class_weight="balanced")["grad_boosting"]
    gb_w.fit(train[FEATURE_COLUMNS], train["result"])
    results["grad_boosting_balanced"] = evaluate("grad_boosting (class_weight=balanced)", gb_w, test)

    # 3) gradient boosting calibrado
    gb_cal = train_calibrated(train, build_models()["grad_boosting"])
    results["grad_boosting_calibrated"] = evaluate("grad_boosting (calibrado)", gb_cal, test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gb_cal, MODELS_DIR / "grad_boosting_calibrated.joblib", compress=3)
    return results


if __name__ == "__main__":
    run_comparison()
