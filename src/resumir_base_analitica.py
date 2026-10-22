
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_analitica_inicial.csv"
saida = raiz / "relatorios" / "resumo_base_analitica.csv"

df = pd.read_csv(arquivo)

resumo = (
    df.groupby("Ano Referencia")
    .agg(
        matrizes=("ID Matriz Curricular", "nunique"),
        estudantes=("Qtde Estudantes", "sum"),
        abandonos=("Qtde Abandono", "sum"),
        trancamentos=("Qtde Trancamentos", "sum"),
        concluintes=("Qtde Concluintes", "sum"),
        cursos=("Curso", "nunique"),
    )
    .reset_index()
)

saida.parent.mkdir(parents=True, exist_ok=True)
resumo.to_csv(saida, index=False, encoding="utf-8-sig")

print(resumo.to_string(index=False))
print(f"\nResumo salvo em: {saida}")

print("\nComparação exploratória:")

comparacao = (
    df.groupby("Ano Referencia")
    .agg(
        estudantes=("Qtde Estudantes", "sum"),
        abandonos=("Qtde Abandono", "sum"),
    )
    .reset_index()
)

comparacao["proporcao_exploratoria_%"] = (
    comparacao["abandonos"] / comparacao["estudantes"]
) * 100

print(comparacao.to_string(index=False))