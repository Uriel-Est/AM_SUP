from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RAIZ = Path.cwd()
ARQUIVO = RAIZ / "data/silver/fluxo_educacao_superior/indicadores_trajetoria_educacao_superior_2020_2024.parquet"
PASTA_GOLD = RAIZ / "data/gold/fluxo_educacao_superior"
PASTA_RELATORIOS = RAIZ / "relatorios"

PASTA_GOLD.mkdir(parents=True, exist_ok=True)
PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)

print("1. Lendo a base Silver...")
df = pd.read_parquet(ARQUIVO)

# Seleção de variáveis disponíveis antes de conhecer a taxa observada.
categoricas = [
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "CO_REGIAO",
    "CO_UF",
    "CO_CINE_AREA_GERAL",
]
numericas = ["NU_ANO_INGRESSO", "QT_INGRESSANTE"]
alvo = "TDA"

colunas = categoricas + numericas + ["NU_ANO_REFERENCIA", alvo]
base = df[colunas].copy()
base = base.dropna(subset=[alvo, "NU_ANO_REFERENCIA"])
base = base[(base[alvo] >= 0) & (base[alvo] <= 100)]

# A referência temporal não entra como preditora: serve para separar os períodos.
base_gold = base.copy()
base_gold.to_parquet(
    PASTA_GOLD / "base_modelagem_tda.parquet", index=False
)
print("Base Gold salva:", base_gold.shape)

treino = base[base["NU_ANO_REFERENCIA"] < 2024].copy()
teste = base[base["NU_ANO_REFERENCIA"] == 2024].copy()

if treino.empty or teste.empty:
    raise ValueError("Treino ou teste vazio. Verifique os anos de referência.")

X_treino = treino[categoricas + numericas]
y_treino = treino[alvo]
X_teste = teste[categoricas + numericas]
y_teste = teste[alvo]

preprocessamento = ColumnTransformer(
    transformers=[
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricas),
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
        ]), numericas),
    ]
)

modelos = {
    "baseline_media": DummyRegressor(strategy="mean"),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
}

resultados = []
print("\n2. Treinando e avaliando...")
for nome, estimador in modelos.items():
    pipeline = Pipeline([
        ("preprocessamento", preprocessamento),
        ("modelo", estimador),
    ])
    pipeline.fit(X_treino, y_treino)
    previsoes = pipeline.predict(X_teste)

    resultados.append({
        "modelo": nome,
        "treino_registros": int(len(treino)),
        "teste_registros": int(len(teste)),
        "MAE": float(mean_absolute_error(y_teste, previsoes)),
        "RMSE": float(np.sqrt(mean_squared_error(y_teste, previsoes))),
        "R2": float(r2_score(y_teste, previsoes)),
    })
    print(resultados[-1])

relatorio = {
    "objetivo": "Prever TDA em registros com ano de referência 2024",
    "variavel_alvo": alvo,
    "treino": "anos de referência anteriores a 2024",
    "teste": "ano de referência 2024",
    "preditoras": categoricas + numericas,
    "observacao": (
        "Experimento inicial. A separação por ano não impede que o mesmo "
        "curso ou coorte apareça nos dois conjuntos. A utilidade preditiva "
        "e a definição operacional da TDA precisam ser discutidas."
    ),
    "resultados": resultados,
}
caminho_relatorio = PASTA_RELATORIOS / "baseline_tda.json"
caminho_relatorio.write_text(
    json.dumps(relatorio, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("\nRelatório salvo em:", caminho_relatorio)
print("Experimento concluído.")
