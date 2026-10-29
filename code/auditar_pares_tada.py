
from pathlib import Path
import pandas as pd

pasta = Path("data/silver/fluxo_educacao_superior")
arquivos = sorted(pasta.glob("*.parquet"))

colunas = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO",
    "NU_ANO_REFERENCIA",
    "TADA",
]

partes = []

for arquivo in arquivos:
    partes.append(pd.read_parquet(arquivo, columns=colunas))

base = pd.concat(partes, ignore_index=True)

chave = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO",
]

# Cada linha de origem representa as informações do ano t.
origem = base.copy()
origem["ANO_ALVO"] = origem["NU_ANO_REFERENCIA"] + 1

# Cada linha de destino fornece a TADA observada no ano seguinte.
destino = base[
    chave + ["NU_ANO_REFERENCIA", "TADA"]
].copy()

destino = destino.rename(
    columns={
        "NU_ANO_REFERENCIA": "ANO_ALVO",
        "TADA": "TADA_FUTURA",
    }
)

pares = origem.merge(
    destino,
    on=chave + ["ANO_ALVO"],
    how="left",
    validate="one_to_one",
)

pares_validos = pares.dropna(subset=["TADA_FUTURA"]).copy()

print("\n===== AUDITORIA DE TADA FUTURA =====")
print(f"Registros de origem: {len(base):,}")
print(f"Registros com ano seguinte correspondente: {len(pares_validos):,}")
print(f"Registros sem ano seguinte correspondente: {pares['TADA_FUTURA'].isna().sum():,}")

print("\nPares disponíveis por ano de origem:")
print(
    pares_validos.groupby("NU_ANO_REFERENCIA")
    .size()
    .sort_index()
    .to_string()
)

print("\nPares disponíveis por ano de ingresso:")
print(
    pares_validos.groupby("NU_ANO_INGRESSO")
    .size()
    .sort_index()
    .to_string()
)

print("\nResumo da TADA futura por ano-alvo:")
print(
    pares_validos.groupby("ANO_ALVO")["TADA_FUTURA"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
    .to_string()
)

print("\nAuditoria concluída. Nenhum arquivo foi alterado.")
