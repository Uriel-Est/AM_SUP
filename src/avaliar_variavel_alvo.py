
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_temporal_inicial.csv"

df = pd.read_csv(arquivo)

print("Quantidade de registros:", len(df))
print("\nResumo da variável-alvo:")
print(df["Abandono_destino"].describe().to_string())

print("\nValores iguais a zero:")
print((df["Abandono_destino"] == 0).sum())

print("\nValores ausentes:")
print(df["Abandono_destino"].isna().sum())

print("\nResumo por ano de destino:")
print(
    df.groupby("Ano_destino")["Abandono_destino"]
    .agg(["count", "mean", "median", "min", "max"])
    .to_string()
)

print("\nMaiores valores de abandono:")
print(
    df.nlargest(10, "Abandono_destino")[
        ["Ano_origem", "Ano_destino", "Curso", "Abandono_destino"]
    ].to_string(index=False)
)
