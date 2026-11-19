
from pathlib import Path
import pandas as pd

pasta = Path("data/silver/fluxo_educacao_superior")
arquivos = sorted(pasta.glob("*.parquet"))

if not arquivos:
    raise FileNotFoundError(
        f"Nenhum arquivo Parquet encontrado em: {pasta}"
    )

colunas_referencia = None
partes = []
problemas = []

for arquivo in arquivos:
    print(f"Lendo: {arquivo.name}")
    df = pd.read_parquet(arquivo)

    colunas = set(df.columns)

    if colunas_referencia is None:
        colunas_referencia = colunas
    elif colunas != colunas_referencia:
        problemas.append(
            f"Colunas diferentes em {arquivo.name}"
        )

    partes.append(df)

base = pd.concat(partes, ignore_index=True)

chave = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO",
    "NU_ANO_REFERENCIA",
]

faltantes = [c for c in chave + ["TADA"] if c not in base.columns]

print("\n===== RESULTADO DA AUDITORIA =====")
print(f"Arquivos lidos: {len(arquivos)}")
print(f"Total de registros: {len(base):,}")
print(f"Total de colunas: {len(base.columns)}")

if problemas:
    print("\nPROBLEMAS DE ESQUEMA:")
    for problema in problemas:
        print("-", problema)
else:
    print("Esquema das colunas: consistente")

if faltantes:
    print("\nColunas obrigatórias ausentes:", faltantes)
else:
    duplicados = base.duplicated(
        subset=chave,
        keep=False
    ).sum()

    print(f"Registros envolvidos em duplicatas da chave: {duplicados:,}")
    print(f"TADA ausente: {base['TADA'].isna().sum():,}")
    print(f"TADA fora de 0 a 100: {((base['TADA'] < 0) | (base['TADA'] > 100)).sum():,}")

    print("\nRegistros por ano de referência:")
    print(
        base.groupby("NU_ANO_REFERENCIA")
        .size()
        .sort_index()
        .to_string()
    )

    print("\nRegistros por ano de ingresso:")
    print(
        base.groupby("NU_ANO_INGRESSO")
        .size()
        .sort_index()
        .to_string()
    )

    print("\nAuditoria concluída. Nenhum arquivo foi alterado.")

if problemas or faltantes:
    raise ValueError(
        "A auditoria encontrou problemas. "
        "Não avance para a modelagem ainda."
    )
