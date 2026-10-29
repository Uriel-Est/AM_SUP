
from pathlib import Path
import json
import gc
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PASTA = Path("data/silver/fluxo_educacao_superior")
RELATORIOS = Path("relatorios")
RELATORIOS.mkdir(parents=True, exist_ok=True)

CHAVE = ["CO_IES", "CO_CURSO", "NU_ANO_INGRESSO"]

COLUNAS = CHAVE + [
    "NU_ANO_REFERENCIA", "TADA",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "CO_REGIAO", "CO_UF", "CO_CINE_AREA_GERAL",
    "QT_INGRESSANTE",
]

print("1/5 - Lendo arquivos...", flush=True)

arquivos = sorted(PASTA.glob("*.parquet"))

base = pd.concat(
    [pd.read_parquet(f, columns=COLUNAS) for f in arquivos],
    ignore_index=True,
)

print("2/5 - Construindo TADA futura...", flush=True)

origem = base.copy()
origem["ANO_ALVO"] = origem["NU_ANO_REFERENCIA"] + 1

destino = base[
    CHAVE + ["NU_ANO_REFERENCIA", "TADA"]
].rename(
    columns={
        "NU_ANO_REFERENCIA": "ANO_ALVO",
        "TADA": "TADA_FUTURA",
    }
)

dados = origem.merge(
    destino,
    on=CHAVE + ["ANO_ALVO"],
    how="inner",
    validate="one_to_one",
)

del base, origem, destino
gc.collect()

categoricas = [
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "CO_REGIAO",
    "CO_UF",
    "CO_CINE_AREA_GERAL",
]

numericas = [
    "TADA",
    "NU_ANO_INGRESSO",
    "QT_INGRESSANTE",
]

preditoras = categoricas + numericas

treino = dados["NU_ANO_REFERENCIA"] <= 2022
teste = dados["NU_ANO_REFERENCIA"] == 2023

X_treino = dados.loc[treino, preditoras]
y_treino = dados.loc[treino, "TADA_FUTURA"]

X_teste = dados.loc[teste, preditoras]
y_teste = dados.loc[teste, "TADA_FUTURA"]

if len(X_treino) == 0 or len(X_teste) == 0:
    raise ValueError("Treino ou teste vazio.")

resultados = {
    "treino": "anos de origem 2010-2022",
    "teste": "origem 2023, alvo 2024",
    "quantidade_treino": int(len(X_treino)),
    "quantidade_teste": int(len(X_teste)),
    "modelos": {},
}

arquivo_json = RELATORIOS / "comparacao_tada_persistencia.json"
arquivo_csv = RELATORIOS / "previsoes_tada_teste.csv"


def salvar():
    arquivo_json.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def metricas(y_real, y_previsto):
    return {
        "MAE": float(mean_absolute_error(y_real, y_previsto)),
        "RMSE": float(mean_squared_error(y_real, y_previsto) ** 0.5),
        "R2": float(r2_score(y_real, y_previsto)),
    }


print(f"Pares: {len(dados):,}", flush=True)
print(
    f"Treino: {len(X_treino):,} | Teste: {len(X_teste):,}",
    flush=True,
)

print("3/5 - Avaliando persistência...", flush=True)

previsao_persistencia = X_teste["TADA"].fillna(
    X_treino["TADA"].median()
)

resultados["modelos"]["persistencia"] = metricas(
    y_teste,
    previsao_persistencia,
)

salvar()

for nome, valor in resultados["modelos"]["persistencia"].items():
    print(f"Persistência {nome}: {valor:.4f}", flush=True)

print("4/5 - Preparando Random Forest...", flush=True)

preprocessamento = ColumnTransformer([
    ("categoricas", Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("codificacao", OneHotEncoder(
            handle_unknown="ignore",
            max_categories=100,
        )),
    ]), categoricas),

    ("numericas", Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
    ]), numericas),
])

modelo = Pipeline([
    ("preprocessamento", preprocessamento),
    ("modelo", RandomForestRegressor(
        n_estimators=30,
        max_depth=15,
        min_samples_leaf=20,
        max_features=0.7,
        random_state=42,
        n_jobs=2,
    )),
])

print("Treinando Random Forest...", flush=True)

modelo.fit(X_treino, y_treino)

print("Calculando previsões...", flush=True)

previsoes = modelo.predict(X_teste)

resultados["modelos"]["random_forest"] = metricas(
    y_teste,
    previsoes,
)

salvar()

for nome, valor in resultados["modelos"]["random_forest"].items():
    print(f"Random Forest {nome}: {valor:.4f}", flush=True)

print("5/5 - Salvando previsões individuais...", flush=True)

predicoes = pd.DataFrame({
    "ANO_ORIGEM": dados.loc[teste, "NU_ANO_REFERENCIA"].to_numpy(),
    "ANO_ALVO": dados.loc[teste, "ANO_ALVO"].to_numpy(),
    "CO_IES": dados.loc[teste, "CO_IES"].to_numpy(),
    "CO_CURSO": dados.loc[teste, "CO_CURSO"].to_numpy(),
    "NU_ANO_INGRESSO": dados.loc[teste, "NU_ANO_INGRESSO"].to_numpy(),
    "TADA_ATUAL": X_teste["TADA"].to_numpy(),
    "TADA_REAL_FUTURA": y_teste.to_numpy(),
    "PREVISAO_PERSISTENCIA": previsao_persistencia.to_numpy(),
    "PREVISAO_RANDOM_FOREST": previsoes,
})

predicoes["ERRO_ABS_PERSISTENCIA"] = (
    predicoes["TADA_REAL_FUTURA"]
    - predicoes["PREVISAO_PERSISTENCIA"]
).abs()

predicoes["ERRO_ABS_RANDOM_FOREST"] = (
    predicoes["TADA_REAL_FUTURA"]
    - predicoes["PREVISAO_RANDOM_FOREST"]
).abs()

predicoes.to_csv(
    arquivo_csv,
    index=False,
    encoding="utf-8-sig",
)

print(f"Relatório salvo em: {arquivo_json}", flush=True)
print(f"Previsões salvas em: {arquivo_csv}", flush=True)
print(f"Total de previsões exportadas: {len(predicoes):,}", flush=True)
