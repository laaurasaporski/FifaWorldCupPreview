# 📖 Dicionário de Dados

## Base bruta — `results.csv`

| Coluna       | Tipo    | Descrição                                                        |
|--------------|---------|------------------------------------------------------------------|
| `date`       | data    | Data da partida                                                  |
| `home_team`  | texto   | Seleção mandante                                                 |
| `away_team`  | texto   | Seleção visitante                                                |
| `home_score` | inteiro | Gols do mandante (tempo normal + prorrogação, sem pênaltis)      |
| `away_score` | inteiro | Gols do visitante                                                |
| `tournament` | texto   | Competição (ex: FIFA World Cup, Friendly, Copa América)          |
| `city`       | texto   | Cidade onde ocorreu a partida                                    |
| `country`    | texto   | País-sede da partida                                             |
| `neutral`    | booleano| Se a partida foi em campo neutro (sem mando real)                |

## Variável alvo (target) — criada em `src/data.py`

| Coluna   | Tipo  | Valores                          | Descrição                              |
|----------|-------|----------------------------------|----------------------------------------|
| `result` | texto | `home_win`, `draw`, `away_win`   | Resultado do ponto de vista do mandante|

## Features de engenharia — criadas em `src/features.py`

Todas calculadas **cronologicamente**, usando apenas dados anteriores a cada partida.

| Coluna              | Tipo  | Descrição                                                       |
|---------------------|-------|-----------------------------------------------------------------|
| `home_elo_pre`      | float | Rating Elo do mandante imediatamente antes da partida          |
| `away_elo_pre`      | float | Rating Elo do visitante antes da partida                        |
| `elo_diff`          | float | Diferença de Elo (mandante − visitante)                         |
| `home_form_points`  | float | Média de pontos do mandante nas últimas 5 partidas              |
| `away_form_points`  | float | Média de pontos do visitante nas últimas 5 partidas             |
| `form_points_diff`  | float | Diferença de forma em pontos (mandante − visitante)             |
| `home_form_gf`      | float | Média de gols marcados pelo mandante (últimas 5)                |
| `away_form_gf`      | float | Média de gols marcados pelo visitante (últimas 5)               |
| `home_form_ga`      | float | Média de gols sofridos pelo mandante (últimas 5)                |
| `away_form_ga`      | float | Média de gols sofridos pelo visitante (últimas 5)               |
| `is_neutral`        | int   | 1 se campo neutro, 0 caso contrário                             |

### Features de artilheiros (`goalscorers.csv`) — janela de 10 jogos

| Coluna                  | Tipo  | Descrição                                                   |
|-------------------------|-------|-------------------------------------------------------------|
| `home_pen_share`        | float | Proporção de gols de pênalti do mandante (últimos 10 jogos) |
| `away_pen_share`        | float | Proporção de gols de pênalti do visitante                   |
| `home_scorer_diversity` | float | Média de artilheiros distintos por jogo do mandante         |
| `away_scorer_diversity` | float | Média de artilheiros distintos por jogo do visitante        |

> **Nota sobre data leakage:** o placar (`home_score`, `away_score`) é usado apenas
> para criar o target e calcular features de jogos passados. Ele **nunca** entra como
> feature do jogo que está sendo previsto.
