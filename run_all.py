"""Roda o pipeline inteiro de uma vez.

    python run_all.py

Executa: features + comparação de modelos -> estado das seleções -> figuras.
Ao final, o modelo, o team_state e os gráficos estão prontos, e a API pode subir.
"""
from src.modeling import run_comparison
from src.team_state import build_team_states
from src.responsible_ai import main as gerar_figuras


def main():
    print("=" * 60)
    print("ETAPA 1-2 | Features, split temporal e comparacao de modelos")
    print("=" * 60)
    run_comparison()

    print("\n" + "=" * 60)
    print("ETAPA 4 | Gerando o estado atual das selecoes (para a API)")
    print("=" * 60)
    estados = build_team_states()
    print(f"{len(estados)} selecoes salvas em models/team_state.json")

    print("\n" + "=" * 60)
    print("ETAPA 3 | Gerando as figuras de IA responsavel")
    print("=" * 60)
    gerar_figuras()

    print("\nTudo pronto! Agora voce pode subir a API com:")
    print("    uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
