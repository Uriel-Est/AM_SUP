
from pathlib import Path
import pandas as pd
import numpy as np

pasta = Path("data/silver/fluxo_educacao_superior")
arquivos = sorted(pasta.glob("*.parquet"))

chave = ["CO_IES", "CO_CURSO", "NU_ANO_INGRESSO"]

colunas = chave + [
    "NU_ANO_REFERENCIA",
    "TADA",
]

base = pd.concat(
    [pd.read_parquet(f, columns=colunas) for f in arquivos],
    ignore_index=True,
)

origem = base.copy()
origem["ANO_ALVO"] = origem["NU_ANO_REFERENCIA"] + 1

destino = base[
    chave + ["NU_ANO_REFERENCIA", "TADA"]
].rename(columns={
    "NU_ANO_REFERENCIA": "ANO_ALVO",
    "TADA": "TADA_FUTURA",
})

dados = origem.merge(
    destino,
    on=chave + ["ANO_ALVO"],
    how="inner",
    validate="one_to_one",
)

teste = dados[dados["NU_ANO_REFERENCIA"] == 2023].copy()

# Persistência: previsão igual à TADA do ano de origem.
teste["PREV_PERSISTENCIA"] = teste["TADA"]

# Erros absolutos e quadráticos.
teste["ERRO_PERSISTENCIA"] = (
    teste["TADA_FUTURA"] - teste["PREV_PERSISTENCIA"]
).abs()

# O JSON contém as métricas agregadas do Random Forest,
# mas não suas previsões individuais.
print("Registros no teste:", len(teste))
print("\nFaixas de TADA futura:")

teste["FAIXA_TADA"] = pd.cut(
    teste["TADA_FUTURA"],
    bins=[-0.001, 1, 5, 10, 20, 50, 100],
    labels=[
        "0 a 1",
        "acima de 1 a 5",
        "acima de 5 a 10",
        "acima de 10 a 20",
        "acima de 20 a 50",
        "acima de 50 a 100",
    ],
    include_lowest=True,
)

resumo = teste.groupby(
    "FAIXA_TADA",
    observed=False,
).agg(
    quantidade=("TADA_FUTURA", "size"),
    media_tada_futura=("TADA_FUTURA", "mean"),
    mae_persistencia=("ERRO_PERSISTENCIA", "mean"),
)

resumo = resumo.round(4)
print(resumo.to_string())

saida = Path("relatorios")
saida.mkdir(parents=True, exist_ok=True)

arquivo = saida / "erros_persistencia_por_faixa.csv"
resumo.to_csv(arquivo, encoding="utf-8-sig")

print(f"\nRelatório salvo em: {arquivo}")
print(
    "\nObservação: este relatório analisa a persistência. "
    "Para analisar o Random Forest por faixa, precisamos "
    "guardar suas previsões individuais."
)
