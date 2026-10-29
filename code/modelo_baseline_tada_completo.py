
from pathlib import Path
import json
import gc
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PASTA = Path("data/silver/fluxo_educacao_superior")
RELATORIOS = Path("relatorios")
RELATORIOS.mkdir(parents=True, exist_ok=True)

CHAVE = [
    "CO_IES", "CO_CURSO", "NU_ANO_INGRESSO"
]

COLUNAS = [
    "CO_IES", "CO_CURSO", "NU_ANO_INGRESSO",
    "NU_ANO_REFERENCIA", "TADA",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "CO_REGIAO", "CO_UF", "CO_CINE_AREA_GERAL",
    "QT_INGRESSANTE",
]

arquivo_json = RELATORIOS / "baseline_tada_completo.json"

def salvar(resultados):
    arquivo_json.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Resultados salvos em: {arquivo_json}", flush=True)

print("Etapa 1/5: lendo arquivos...", flush=True)
arquivos = sorted(PASTA.glob("*.parquet"))
if not arquivos:
    raise FileNotFoundError(f"Nenhum Parquet encontrado em {PASTA}")

base = pd.concat(
    [pd.read_parquet(f, columns=COLUNAS) for f in arquivos],
    ignore_index=True,
)

if base.duplicated(CHAVE + ["NU_ANO_REFERENCIA"]).any():
    raise ValueError("Existem duplicatas na chave temporal.")

print(f"Registros carregados: {len(base):,}", flush=True)

print("Etapa 2/5: montando pares temporais...", flush=True)
origem = base.copy()
origem["ANO_ALVO"] = origem["NU_ANO_REFERENCIA"] + 1

destino = base[
    CHAVE + ["NU_ANO_REFERENCIA", "TADA"]
].rename(columns={
    "NU_ANO_REFERENCIA": "ANO_ALVO",
    "TADA": "TADA_FUTURA",
})

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
    "CO_REGIAO", "CO_UF", "CO_CINE_AREA_GERAL",
]
numericas = ["NU_ANO_INGRESSO", "QT_INGRESSANTE"]
preditoras = categoricas + numericas

treino = dados["NU_ANO_REFERENCIA"] <= 2022
teste = dados["NU_ANO_REFERENCIA"] == 2023

X_treino = dados.loc[treino, preditoras]
y_treino = dados.loc[treino, "TADA_FUTURA"]
X_teste = dados.loc[teste, preditoras]
y_teste = dados.loc[teste, "TADA_FUTURA"]

if len(X_treino) == 0 or len(X_teste) == 0:
    raise ValueError("Treino ou teste está vazio.")

resultados = {
    "registros_pares": int(len(dados)),
    "treino_registros": int(len(X_treino)),
    "teste_registros": int(len(X_teste)),
    "anos_treino_origem": "2010-2022",
    "ano_teste_origem": 2023,
    "ano_alvo_teste": 2024,
    "modelos": {},
}
salvar(resultados)

def avaliar(nome, modelo):
    print(f"\nTreinando {nome}...", flush=True)
    modelo.fit(X_treino, y_treino)
    print(f"Calculando previsões de {nome}...", flush=True)
    previsoes = modelo.predict(X_teste)

    metricas = {
        "MAE": float(mean_absolute_error(y_teste, previsoes)),
        "RMSE": float(mean_squared_error(y_teste, previsoes) ** 0.5),
        "R2": float(r2_score(y_teste, previsoes)),
    }

    resultados["modelos"][nome] = metricas
    salvar(resultados)

    print(nome, flush=True)
    for chave, valor in metricas.items():
        print(f"  {chave}: {valor:.4f}", flush=True)

print("Etapa 3/5: baseline trivial...", flush=True)
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_treino[["NU_ANO_INGRESSO"]], y_treino)
pred_dummy = dummy.predict(X_teste[["NU_ANO_INGRESSO"]])

resultados["modelos"]["dummy_media"] = {
    "MAE": float(mean_absolute_error(y_teste, pred_dummy)),
    "RMSE": float(mean_squared_error(y_teste, pred_dummy) ** 0.5),
    "R2": float(r2_score(y_teste, pred_dummy)),
}
salvar(resultados)

for chave, valor in resultados["modelos"]["dummy_media"].items():
    print(f"  {chave}: {valor:.4f}", flush=True)

del dummy, pred_dummy
gc.collect()

print("\nEtapa 4/5: preparando Random Forest...", flush=True)
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

rf = Pipeline([
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

avaliar("random_forest", rf)

print("\nEtapa 5/5: concluído.", flush=True)
print(f"Relatório final: {arquivo_json}", flush=True)
