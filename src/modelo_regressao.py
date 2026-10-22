
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

raiz = Path(__file__).resolve().parents[1]
entrada = raiz / "dados" / "processado" / "base_temporal_inicial.csv"
pasta_relatorios = raiz / "relatorios"
pasta_relatorios.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(entrada)

# Mesmo recorte temporal usado no teste de referência.
treino = df[df["Ano_destino"].isin([2022, 2023, 2024])].copy()
teste = df[df["Ano_destino"] == 2025].copy()

variaveis_numericas = [
    "Estudantes_origem",
    "Abandono_origem",
    "Trancamentos_origem",
    "Concluintes_origem",
]

variaveis_categoricas = [
    "Curso",
    "Turno",
    "Grau Academico",
    "Modalidade",
]

X_treino = treino[variaveis_numericas + variaveis_categoricas]
y_treino = treino["Abandono_destino"]

X_teste = teste[variaveis_numericas + variaveis_categoricas]
y_teste = teste["Abandono_destino"]

preparo_numerico = Pipeline([
    ("imputacao", SimpleImputer(strategy="median")),
    ("escala", StandardScaler()),
])

preparo_categorico = Pipeline([
    ("imputacao", SimpleImputer(strategy="most_frequent")),
    ("codificacao", OneHotEncoder(handle_unknown="ignore")),
])

preparo = ColumnTransformer([
    ("numericas", preparo_numerico, variaveis_numericas),
    ("categoricas", preparo_categorico, variaveis_categoricas),
])

modelo = Pipeline([
    ("preparo", preparo),
    ("regressao", Ridge(alpha=1.0)),
])

modelo.fit(X_treino, y_treino)
previsoes = np.maximum(0, modelo.predict(X_teste))

mae = mean_absolute_error(y_teste, previsoes)
rmse = np.sqrt(mean_squared_error(y_teste, previsoes))

referencia = teste["Abandono_origem"]
mae_referencia = mean_absolute_error(y_teste, referencia)
rmse_referencia = np.sqrt(
    mean_squared_error(y_teste, referencia)
)

resultado = (
    "MODELO DE REGRESSAO RIDGE\n"
    f"Registros de treino: {len(treino)}\n"
    f"Registros de teste: {len(teste)}\n"
    f"MAE Ridge: {mae:.2f}\n"
    f"RMSE Ridge: {rmse:.2f}\n\n"
    "COMPARACAO COM A REFERENCIA\n"
    f"MAE referencia: {mae_referencia:.2f}\n"
    f"RMSE referencia: {rmse_referencia:.2f}\n"
)

print(resultado)

saida = pasta_relatorios / "comparacao_modelos.txt"
saida.write_text(resultado, encoding="utf-8")
print("Relatório salvo em:", saida)
