from pathlib import Path
import pandas as pd

pasta = Path("dados/bruto")
arquivo = next(pasta.glob("tabela_resumo*.csv"))

print("ARQUIVO:", arquivo.name)

# Examina como o ano aparece no CSV original.
print("\nPRIMEIRAS LINHAS DO ARQUIVO ORIGINAL:")
with open(arquivo, "r", encoding="utf-8-sig") as f:
    for _ in range(5):
        print(repr(f.readline()))

# Lê o ano como texto para evitar conversão automática incorreta.
df = pd.read_csv(
    arquivo,
    encoding="utf-8-sig",
    sep=",",
    dtype={"Ano": "string"}
)
df.columns = df.columns.str.strip()

print("\nVALORES ORIGINAIS DO CAMPO ANO:")
print(df["Ano"].value_counts(dropna=False).sort_index().to_string())

# Converte anos que estejam escritos como 2010, 2010.0 ou 2.01.
ano_num = pd.to_numeric(df["Ano"], errors="coerce")
ano_corrigido = ano_num.copy()

# Se o ano foi interpretado como 2.01, 2.011 etc.,
# tenta reconstruí-lo apenas para o diagnóstico.
mascara = ano_num.between(2.0, 2.1, inclusive="left")
ano_corrigido.loc[mascara] = (
    ano_num.loc[mascara].mul(1000).round()
)

print("\nANOS APÓS A TENTATIVA DE CORREÇÃO:")
print(ano_corrigido.value_counts(dropna=False).sort_index().to_string())

print("\nREGISTROS POR ANO E PERÍODO:")
teste = df.assign(Ano_diagnostico=ano_corrigido)
print(
    teste.groupby(["Ano_diagnostico", "Período"], dropna=False)
    .size()
    .to_string()
)

print("\nNenhum CSV original foi alterado.")
