
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_analitica_inicial.csv"

df = pd.read_csv(arquivo)

anos = sorted(df["Ano Referencia"].unique())

print("Pares de anos disponíveis:\n")

for ano in anos[:-1]:
    origem = df[df["Ano Referencia"] == ano]
    destino = df[df["Ano Referencia"] == ano + 1]

    matrizes_comuns = set(origem["ID Matriz Curricular"]) & set(
        destino["ID Matriz Curricular"]
    )

    print(
        f"{ano} -> {ano + 1}: "
        f"{len(origem)} registros de origem, "
        f"{len(destino)} registros de destino, "
        f"{len(matrizes_comuns)} matrizes em comum"
    )
