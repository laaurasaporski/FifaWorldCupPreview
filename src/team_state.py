"""Gera o 'estado atual' de cada seleção para uso na API.

Replaya todo o histórico e guarda, por seleção, o último Elo e a forma recente.
A API usa esse retrato para montar as features de um confronto novo, sem
precisar reprocessar 49 mil jogos a cada requisição.
"""
import json
from pathlib import Path

import pandas as pd

from .data import add_target, load_goalscorers, load_results
from .features import add_elo_ratings

STATE_PATH = Path(__file__).resolve().parents[1] / "models" / "team_state.json"
FORM_WINDOW = 5
GOAL_WINDOW = 10


def build_team_states() -> dict:
    df = add_target(load_results())

    # Elo final por seleção: replaya o histórico e guarda o rating pós-último jogo
    df = add_elo_ratings(df)
    ratings = {}
    K, HOME_ADV, INIT = 30, 65, 1500
    for r in df.itertuples(index=False):
        rh = ratings.get(r.home_team, INIT)
        ra = ratings.get(r.away_team, INIT)
        adv = 0 if bool(r.neutral) else HOME_ADV
        eh = 1 / (1 + 10 ** (-((rh + adv) - ra) / 400))
        sh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        ratings[r.home_team] = rh + K * (sh - eh)
        ratings[r.away_team] = ra + K * ((1 - sh) - (1 - eh))

    # forma recente (ultimos FORM_WINDOW jogos de cada time)
    home = df[["date", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "gf", "ga"]
    away = df[["date", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "gf", "ga"]
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = (long.gf > long.ga) * 3 + (long.gf == long.ga) * 1
    long = long.sort_values(["team", "date"])
    form = long.groupby("team").tail(FORM_WINDOW).groupby("team").agg(
        form_points=("points", "mean"), form_gf=("gf", "mean"), form_ga=("ga", "mean")
    )

    # features de artilheiro (ultimos GOAL_WINDOW jogos com gols)
    g = load_goalscorers()
    agg = g.groupby(["date", "team"]).agg(
        goals=("scorer", "size"), pen=("penalty", "sum"), div=("scorer", "nunique")
    ).reset_index().sort_values(["team", "date"])
    tail = agg.groupby("team").tail(GOAL_WINDOW)
    gfeat = tail.groupby("team").agg(
        pen_sum=("pen", "sum"), goal_sum=("goals", "sum"), scorer_diversity=("div", "mean")
    )
    gfeat_pen_share = (gfeat.pen_sum / gfeat.goal_sum).fillna(0.0)

    states = {}
    for team, e in ratings.items():
        f = form.loc[team] if team in form.index else None
        states[team] = {
            "elo": round(float(e), 1),
            "form_points": round(float(f.form_points), 3) if f is not None else 1.0,
            "form_gf": round(float(f.form_gf), 3) if f is not None else 1.0,
            "form_ga": round(float(f.form_ga), 3) if f is not None else 1.0,
            "pen_share": round(float(gfeat_pen_share.get(team, 0.0)), 3),
            "scorer_diversity": round(float(gfeat.scorer_diversity.get(team, 1.0)), 3)
                if team in gfeat.index else 1.0,
        }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(states, fh, ensure_ascii=False, indent=1)
    return states


def load_team_states() -> dict:
    with open(STATE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    s = build_team_states()
    print(f"Estados salvos: {len(s)} selecoes")
    for t in ["Brazil", "Argentina", "France", "Germany"]:
        if t in s:
            print(t, s[t])
