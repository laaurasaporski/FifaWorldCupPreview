"""Lógica de previsão de um confronto — independente da camada web.

Monta o vetor de features a partir do estado atual das duas seleções e
devolve a probabilidade de cada resultado. Isso é o 'cérebro' que a API expõe.
"""
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS
from .team_state import load_team_states

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "grad_boosting_calibrated.joblib"

_model = None
_states = None


def _load():
    global _model, _states
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _states = load_team_states()
    return _model, _states


def available_teams() -> list[str]:
    _, states = _load()
    return sorted(states.keys())


def _features(home: dict, away: dict, neutral: bool) -> pd.DataFrame:
    row = {
        "home_elo_pre": home["elo"],
        "away_elo_pre": away["elo"],
        "elo_diff": home["elo"] - away["elo"],
        "home_form_points": home["form_points"],
        "away_form_points": away["form_points"],
        "form_points_diff": home["form_points"] - away["form_points"],
        "home_form_gf": home["form_gf"],
        "away_form_gf": away["form_gf"],
        "home_form_ga": home["form_ga"],
        "away_form_ga": away["form_ga"],
        "home_pen_share": home["pen_share"],
        "away_pen_share": away["pen_share"],
        "home_scorer_diversity": home["scorer_diversity"],
        "away_scorer_diversity": away["scorer_diversity"],
        "is_neutral": int(neutral),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def predict_match(home_team: str, away_team: str, neutral: bool = False) -> dict:
    model, states = _load()
    if home_team not in states:
        raise KeyError(f"Seleção desconhecida: {home_team}")
    if away_team not in states:
        raise KeyError(f"Seleção desconhecida: {away_team}")

    X = _features(states[home_team], states[away_team], neutral)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    p = {c: float(proba[i]) for i, c in enumerate(classes)}

    label_map = {"home_win": home_team, "draw": "Empate", "away_win": away_team}
    pred = max(p, key=p.get)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": neutral,
        "probabilities": {
            home_team: round(p.get("home_win", 0), 3),
            "draw": round(p.get("draw", 0), 3),
            away_team: round(p.get("away_win", 0), 3),
        },
        "most_likely": label_map[pred],
        "home_elo": states[home_team]["elo"],
        "away_elo": states[away_team]["elo"],
    }


if __name__ == "__main__":
    import json
    for h, a in [("Brazil", "Argentina"), ("Germany", "France"), ("Brazil", "Tahiti")]:
        try:
            print(json.dumps(predict_match(h, a, neutral=True), ensure_ascii=False, indent=2))
        except KeyError as e:
            print(e)
