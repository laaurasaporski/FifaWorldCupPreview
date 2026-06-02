"""Engenharia de features SEM data leakage.

Princípio central: para cada partida, só podemos usar informação que já
existia ANTES daquela partida acontecer. Por isso tudo é calculado de forma
cronológica, e a estatística de cada jogo nunca inclui o próprio jogo.
"""
import numpy as np
import pandas as pd

from .data import load_goalscorers


def assign_match_id(df: pd.DataFrame) -> pd.DataFrame:
    """Garante uma chave única por partida, usada para juntar features."""
    df = df.reset_index(drop=True).copy()
    if "match_id" not in df.columns:
        df["match_id"] = df.index
    return df


def add_elo_ratings(
    df: pd.DataFrame,
    k: int = 30,
    home_advantage: int = 65,
    init_rating: int = 1500,
) -> pd.DataFrame:
    """Adiciona o rating Elo de cada seleção ANTES de cada partida.

    O Elo pré-jogo é uma feature legítima (estado conhecido antes da bola rolar).
    Após registrar o pré-jogo, atualizamos os ratings com o resultado — mas essa
    atualização só afeta jogos FUTUROS daquela seleção, nunca o jogo atual.

    Espera as colunas: home_team, away_team, home_score, away_score, neutral.
    """
    df = df.copy()
    ratings: dict[str, float] = {}
    home_elo_pre, away_elo_pre = [], []

    for row in df.itertuples(index=False):
        rh = ratings.get(row.home_team, init_rating)
        ra = ratings.get(row.away_team, init_rating)

        # registra o estado ANTES do jogo (isso vira feature)
        home_elo_pre.append(rh)
        away_elo_pre.append(ra)

        # vantagem de mando só quando não é campo neutro
        adv = 0 if bool(row.neutral) else home_advantage
        exp_home = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400.0))

        if row.home_score > row.away_score:
            score_home = 1.0
        elif row.home_score < row.away_score:
            score_home = 0.0
        else:
            score_home = 0.5

        # atualização (afeta apenas partidas futuras)
        ratings[row.home_team] = rh + k * (score_home - exp_home)
        ratings[row.away_team] = ra + k * ((1.0 - score_home) - (1.0 - exp_home))

    df["home_elo_pre"] = home_elo_pre
    df["away_elo_pre"] = away_elo_pre
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    return df


def add_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Adiciona forma recente (pontos e gols) das últimas `window` partidas.

    Usa `.shift()` ANTES do rolling para garantir que o jogo atual nunca entra
    no cálculo da sua própria média — esse é o detalhe que evita o vazamento.
    """
    df = assign_match_id(df)

    # formato longo: uma linha por (time, partida)
    home = df[["match_id", "date", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["match_id", "date", "team", "gf", "ga"]
    home["side"] = "home"

    away = df[["match_id", "date", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["match_id", "date", "team", "gf", "ga"]
    away["side"] = "away"

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select(
        [long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], default=0
    )
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    grp = long.groupby("team", group_keys=False)
    long["form_points"] = grp["points"].apply(
        lambda s: s.shift().rolling(window, min_periods=1).mean()
    )
    long["form_gf"] = grp["gf"].apply(
        lambda s: s.shift().rolling(window, min_periods=1).mean()
    )
    long["form_ga"] = grp["ga"].apply(
        lambda s: s.shift().rolling(window, min_periods=1).mean()
    )

    feats = ["form_points", "form_gf", "form_ga"]
    wide = long.pivot_table(index="match_id", columns="side", values=feats)
    wide.columns = [f"{side}_{feat}" for feat, side in wide.columns]
    wide = wide.reset_index()

    return df.merge(wide, on="match_id", how="left")


def add_goalscorer_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Features de ataque derivadas de goalscorers.csv, sem data leakage.

    Para cada seleção, em janela móvel das últimas `window` partidas COM gols
    registrados:
      - pen_share: proporção de gols de pênalti (dependência da bola parada)
      - scorer_diversity: média de artilheiros distintos por jogo (quanto o time
        depende de um único goleador)

    Tudo com `.shift()` para nunca usar a própria partida.
    """
    df = assign_match_id(df)
    g = load_goalscorers()

    # agrega por (data, time): gols, gols de pênalti, artilheiros distintos
    agg = (
        g.groupby(["date", "team"])
        .agg(
            goals=("scorer", "size"),
            pen_goals=("penalty", "sum"),
            distinct_scorers=("scorer", "nunique"),
        )
        .reset_index()
    )

    # tabela longa de partidas (mandante e visitante) com match_id
    home = df[["match_id", "date", "home_team"]].rename(columns={"home_team": "team"})
    home["side"] = "home"
    away = df[["match_id", "date", "away_team"]].rename(columns={"away_team": "team"})
    away["side"] = "away"
    long = pd.concat([home, away], ignore_index=True)

    long = long.merge(agg, on=["date", "team"], how="left")
    long[["goals", "pen_goals", "distinct_scorers"]] = long[
        ["goals", "pen_goals", "distinct_scorers"]
    ].fillna(0)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    grp = long.groupby("team", group_keys=False)
    # razao de gols de penalti acumulada na janela (soma pen / soma gols)
    sum_pen = grp["pen_goals"].apply(lambda s: s.shift().rolling(window, min_periods=1).sum())
    sum_gls = grp["goals"].apply(lambda s: s.shift().rolling(window, min_periods=1).sum())
    long["pen_share"] = (sum_pen / sum_gls).where(sum_gls > 0, np.nan)
    long["scorer_diversity"] = grp["distinct_scorers"].apply(
        lambda s: s.shift().rolling(window, min_periods=1).mean()
    )

    feats = ["pen_share", "scorer_diversity"]
    wide = long.pivot_table(index="match_id", columns="side", values=feats)
    wide.columns = [f"{side}_{feat}" for feat, side in wide.columns]
    wide = wide.reset_index()

    out = df.merge(wide, on="match_id", how="left")
    # neutralidade: sem historico -> share 0 e diversidade 1 (depende de 1 jogador)
    for col in ["home_pen_share", "away_pen_share"]:
        out[col] = out[col].fillna(0.0)
    for col in ["home_scorer_diversity", "away_scorer_diversity"]:
        out[col] = out[col].fillna(1.0)
    return out


def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Pipeline completo de features na ordem correta (cronológica)."""
    df = assign_match_id(df)
    df = add_elo_ratings(df)
    df = add_rolling_form(df, window=window)
    df = add_goalscorer_features(df, window=10)
    df["form_points_diff"] = df["home_form_points"] - df["away_form_points"]
    df["is_neutral"] = df["neutral"].astype(int)
    return df


FEATURE_COLUMNS = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff",
    "home_form_points",
    "away_form_points",
    "form_points_diff",
    "home_form_gf",
    "away_form_gf",
    "home_form_ga",
    "away_form_ga",
    "home_pen_share",
    "away_pen_share",
    "home_scorer_diversity",
    "away_scorer_diversity",
    "is_neutral",
]
