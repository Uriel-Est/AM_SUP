
import pandas as pd
from pathlib import Path

arquivo = Path(
    "dados/bruto/discentes_de_cursos_de_graduacao_da_ufpb_2026-07.csv"
)

df = pd.read_csv(arquivo, encoding="utf-8-sig")

print("DIMENSÕES:", df.shape)

print("\nCOLUNAS:")
print(df.columns.tolist())

print("\nTIPOS:")
print(df.dtypes.to_string())

print("\nVALORES AUSENTES:")
print(df.isna().sum().to_string())

print("\nLINHAS COMPLETAMENTE DUPLICADAS:")
print(df.duplicated().sum())

print("\nPRIMEIRAS 10 LINHAS:")
print(df.head(10).to_string(index=False))

print("\nANOS DISPONÍVEIS:")
print(sorted(df["Ano Referencia"].unique().tolist()))

chave = ["Ano Referencia", "ID Matriz Curricular"]

print("\nLINHAS PERTENCENTES A GRUPOS REPETIDOS")
print("POR ANO + ID DA MATRIZ CURRICULAR:")
print(df.duplicated(subset=chave, keep=False).sum())

print("\nEXEMPLOS DE REPETIÇÕES NESSA CHAVE:")
repetidos = df[
    df.duplicated(subset=chave, keep=False)
].sort_values(chave)

print(repetidos.head(20).to_string(index=False))

print("\nREGISTROS SEM CÓDIGO INEP:")
print(
    df[df["Codigo Inep"].isna()].to_string(index=False)
)

print("\nQUANTIDADE DE MATRIZES POR ANO:")
print(
    df.groupby("Ano Referencia")["ID Matriz Curricular"]
    .nunique()
    .to_string()
)

print("\nABANDONO: ESTATÍSTICAS DESCRITIVAS")
print(df["Qtde Abandono"].describe().to_string())

print("\nESTUDANTES: ESTATÍSTICAS DESCRITIVAS")
print(df["Qtde Estudantes"].describe().to_string())