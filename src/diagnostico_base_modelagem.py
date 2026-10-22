
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_semestral_ano_corrigido.csv"

df = pd.read_csv(arquivo)

print("Dimensões:", df.shape)
print("\nColunas:", df.columns.tolist())
print("\nValores ausentes:")
print(df.isna().sum())

print("\nRegistros por ano e período:")
print(df.groupby(["Ano", "Período"]).size().unstack(fill_value=0))

print("\nTipos de dados:")
print(df.dtypes)
