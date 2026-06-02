"""API REST que serve o modelo de previsão de resultados.

Rodar localmente:
    uvicorn src.api:app --reload

Depois acesse a documentação interativa em http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import available_teams, predict_match

app = FastAPI(
    title="Previsão de Resultados de Futebol",
    description="Estima a probabilidade de Vitória/Empate/Derrota a partir do "
    "Elo e da forma recente das seleções.",
    version="1.0.0",
)


class MatchRequest(BaseModel):
    home_team: str = Field(..., examples=["Brazil"])
    away_team: str = Field(..., examples=["Argentina"])
    neutral: bool = Field(False, description="True se for em campo neutro")


@app.get("/")
def root():
    return {"status": "ok", "mensagem": "Use /docs para testar, /teams para a lista de seleções."}


@app.get("/teams")
def teams():
    times = available_teams()
    return {"total": len(times), "teams": times}


@app.post("/predict")
def predict(req: MatchRequest):
    try:
        return predict_match(req.home_team, req.away_team, req.neutral)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
