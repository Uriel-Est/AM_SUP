
import pandas as pd
from pathlib import Path

arquivo = Path(
    "dados/bruto/tabela_resumo___indicadores_graduacao_2026-10-16T19_42_23.516457508-03_00.csv"
)

# Ler a coluna Ano como texto para preservar valores como 2.010
df = pd.read_csv(
    arquivo,
    encoding="utf-8-sig",
    dtype={"Ano": "string"}
)

# Corrigir a representação do ano apenas na memória
ano_original = df["Ano"].str.strip()

df["Ano"] = (
    ano_original
    .str.replace(".", "", regex=False)
    .astype("Int64")
)

print("DIMENSÕES DA BASE:", df.shape)

print("\nANOS:")
print(sorted(df["Ano"].dropna().unique().tolist()))

print("\nREGISTROS POR ANO E PERÍODO:")
print(df.groupby(["Ano", "Período"]).size().to_string())

print("\nAUSENTES POR COLUNA:")
print(df.isna().sum().to_string())

print("\nCÓDIGO INEP AUSENTE:")
print(df[df["Código Inep"].isna()].to_string(index=False))

chave = ["Ano", "Período", "Código Inep"]
validos = df[df["Código Inep"].notna()]

print("\nLINHAS REPETIDAS NA CHAVE ANO + PERÍODO + CÓDIGO INEP:")
print(validos.duplicated(subset=chave, keep=False).sum())

chave2 = ["Ano", "Período", "Curso", "Turno", "Centro"]

print("\nLINHAS REPETIDAS NA CHAVE ANO + PERÍODO + CURSO + TURNO + CENTRO:")
print(df.duplicated(subset=chave2, keep=False).sum())
