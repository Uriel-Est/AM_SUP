from pathlib import Path
import pandas as pd

pasta = Path("dados/bruto")
saida = Path("dados/processado")
saida.mkdir(parents=True, exist_ok=True)

arquivo = next(pasta.glob("tabela_resumo*.csv"))
df = pd.read_csv(arquivo, encoding="utf-8-sig", dtype={"Ano": "string"})
df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# Corrige somente o padrão de ano observado no arquivo original.
ano = df["Ano"].str.strip()
mascara = ano.str.fullmatch(r"2\.0\d{2}", na=False)
ano.loc[mascara] = ano.loc[mascara].str.replace(".", "", regex=False)

df["Ano"] = pd.to_numeric(ano, errors="raise").astype("int64")

destino = saida / "base_semestral_ano_corrigido.csv"
df.to_csv(destino, index=False, encoding="utf-8-sig")

print("ARQUIVO GERADO:", destino)
print("DIMENSÕES:", df.shape)
print("\nANOS:")
print(sorted(df["Ano"].unique().tolist()))
print("\nREGISTROS POR ANO E PERÍODO:")
print(df.groupby(["Ano", "Período"]).size().to_string())
print("\nVALORES AUSENTES:")
print(df.isna().sum().to_string())
print("\nBASE ORIGINAL PRESERVADA.")
