"""Carregamento dos dados brutos e criação da variável alvo (target)."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_results(filename: str = "results.csv") -> pd.DataFrame:
    """Carrega o results.csv e ordena cronologicamente.

    A ordenação por data é OBRIGATÓRIA: todas as features de histórico
    dependem de processar os jogos na ordem em que aconteceram.
    """
    df = pd.read_csv(RAW_DIR / filename, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_goalscorers(filename: str = "goalscorers.csv") -> pd.DataFrame:
    """Carrega os artilheiros (um gol por linha)."""
    g = pd.read_csv(RAW_DIR / filename, parse_dates=["date"])
    return g


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Cria o target multiclasse do ponto de vista do mandante.

    home_win  -> mandante venceu
    draw      -> empate
    away_win  -> visitante venceu
    """
    df = df.copy()
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        ["home_win", "draw"],
        default="away_win",
    )
    return df


if __name__ == "__main__":
    data = add_target(load_results())
    print(f"Partidas carregadas: {len(data):,}")
    print("\nDistribuicao do target (%):")
    print((data["result"].value_counts(normalize=True) * 100).round(1))
