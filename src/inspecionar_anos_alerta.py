
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_analitica_inicial.csv"

df = pd.read_csv(arquivo)

colunas = [
    "Ano Referencia",
    "Curso",
    "Qtde Estudantes",
    "Qtde Abandono",
    "Qtde Trancamentos",
    "Qtde Concluintes",
]

for ano in [2020, 2021, 2022]:
    recorte = df.loc[df["Ano Referencia"] == ano, colunas]

    print(f"\n===== ANO {ano} =====")
    print("Registros:", len(recorte))
    print("Estudantes:", recorte["Qtde Estudantes"].sum())
    print("Abandonos:", recorte["Qtde Abandono"].sum())
    print("Registros com abandono zero:",
          (recorte["Qtde Abandono"] == 0).sum())
    print("Registros com abandono positivo:",
          (recorte["Qtde Abandono"] > 0).sum())

    print("\nCinco maiores valores de abandono:")
    print(
        recorte.nlargest(5, "Qtde Abandono")[
            ["Curso", "Qtde Estudantes", "Qtde Abandono"]
        ].to_string(index=False)
    )
