
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


# ============================================================
# 1. CAMINHOS
# ============================================================

RAIZ = Path.cwd()

ARQUIVO = (
    RAIZ
    / "data"
    / "silver"
    / "fluxo_educacao_superior"
    / "indicadores_trajetoria_educacao_superior_2020_2024.parquet"
)

PASTA_GOLD = (
    RAIZ / "data" / "gold" / "fluxo_educacao_superior"
)

PASTA_RELATORIOS = RAIZ / "relatorios"

PASTA_GOLD.mkdir(parents=True, exist_ok=True)
PASTA_RELATORIOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LEITURA E CONFERENCIA DA BASE
# ============================================================

print("1. Lendo a base Silver...")

if not ARQUIVO.exists():
    raise FileNotFoundError(
        f"Arquivo Silver não encontrado: {ARQUIVO}\n"
        "Execute o script a partir da pasta raiz AM_SUP."
    )

df = pd.read_parquet(ARQUIVO)

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
    "NU_ANO_INGRESSO",
    "QT_INGRESSANTE",
]

chaves = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO",
]

ano = "NU_ANO_REFERENCIA"
alvo_original = "TADA"
alvo = "TADA_FUTURA"

# dict.fromkeys evita nomes de colunas duplicados.
colunas_necessarias = list(dict.fromkeys(
    chaves
    + categoricas
    + numericas
    + [ano, alvo_original]
))

colunas_ausentes = [
    coluna
    for coluna in colunas_necessarias
    if coluna not in df.columns
]

if colunas_ausentes:
    raise ValueError(
        f"Estas colunas não existem na base: {colunas_ausentes}"
    )

print("Dimensão da base Silver:", df.shape)
print("Colunas necessárias conferidas.")


# ============================================================
# 3. CRIAÇÃO DO ALVO FUTURO
# ============================================================

print("\n2. Preparando a TADA do ano seguinte...")

# Confere se cada chave identifica uma única linha por ano.
if df.duplicated(chaves + [ano]).any():
    raise ValueError(
        "Há duplicatas em instituição, curso, coorte e ano. "
        "A associação temporal precisa ser investigada."
    )

# Mantém uma única ocorrência de cada coluna.
atual = df[colunas_necessarias].copy()

atual[ano] = pd.to_numeric(
    atual[ano], errors="coerce"
)

atual[alvo_original] = pd.to_numeric(
    atual[alvo_original], errors="coerce"
)

# Constrói uma tabela com a TADA observada no ano seguinte.
futura = df[chaves + [ano, alvo_original]].copy()

futura[ano] = pd.to_numeric(
    futura[ano], errors="coerce"
)

futura[alvo_original] = pd.to_numeric(
    futura[alvo_original], errors="coerce"
)

# Exemplo: a observação de 2021 será associada ao ano atual 2020.
futura["ANO_PREDICAO"] = futura[ano] - 1

futura = futura.rename(
    columns={alvo_original: alvo}
)

# Junta pelo curso, instituição, coorte e ano de previsão.
base = atual.merge(
    futura[chaves + ["ANO_PREDICAO", alvo]],
    left_on=chaves + [ano],
    right_on=chaves + ["ANO_PREDICAO"],
    how="left",
    validate="one_to_one",
)

# Só podemos treinar quando existe TADA observada no ano seguinte.
base = base.dropna(
    subset=[ano, alvo]
).copy()

# A TADA é uma taxa percentual entre 0 e 100.
base = base[base[alvo].between(0, 100)].copy()

# Não usamos 2024 como ano de previsão porque não há TADA de 2025.
base = base[base[ano] <= 2023].copy()

if base.empty:
    raise ValueError(
        "Não há registros com TADA futura disponível."
    )

print("Registros disponíveis para modelagem:", len(base))
print("\nQuantidade de registros por ano de referência:")

print(
    base.groupby(ano)[alvo]
    .agg(["count", "min", "max"])
    .to_string()
)


# ============================================================
# 4. SALVAMENTO DA BASE GOLD
# ============================================================

print("\n3. Salvando a base Gold...")

colunas_gold = categoricas + numericas + [ano, alvo]

base_gold = base[colunas_gold].copy()

caminho_gold = (
    PASTA_GOLD / "base_modelagem_tada_futura.parquet"
)

base_gold.to_parquet(
    caminho_gold,
    index=False
)

print("Base Gold salva em:", caminho_gold)
print("Dimensão da base Gold:", base_gold.shape)


# ============================================================
# 5. SEPARAÇÃO TEMPORAL ENTRE TREINO E TESTE
# ============================================================

print("\n4. Separando treino e teste...")

# Treino: dados de 2020, 2021 e 2022.
# Teste: características de 2023 para prever a TADA de 2024.
treino = base_gold[
    base_gold[ano] < 2023
].copy()

teste = base_gold[
    base_gold[ano] == 2023
].copy()

if treino.empty or teste.empty:
    raise ValueError(
        "Treino ou teste vazio. Confira os anos disponíveis."
    )

X_treino = treino[categoricas + numericas]
y_treino = treino[alvo]

X_teste = teste[categoricas + numericas]
y_teste = teste[alvo]

print("Registros de treino:", len(treino))
print("Registros de teste:", len(teste))
print("Ano das características de teste: 2023")
print("Ano da TADA prevista no teste: 2024")


# ============================================================
# 6. PREPROCESSAMENTO
# ============================================================

preprocessamento = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            Pipeline([
                (
                    "imputar",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "codificar",
                    OneHotEncoder(handle_unknown="ignore")
                ),
            ]),
            categoricas,
        ),
        (
            "numericas",
            Pipeline([
                (
                    "imputar",
                    SimpleImputer(strategy="median")
                ),
            ]),
            numericas,
        ),
    ]
)


# ============================================================
# 7. MODELOS DE COMPARAÇÃO
# ============================================================

modelos = {
    "baseline_media": DummyRegressor(
        strategy="mean"
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# 8. TREINAMENTO E AVALIAÇÃO
# ============================================================

print("\n5. Treinando e avaliando os modelos...")

resultados = []

for nome, estimador in modelos.items():

    pipeline = Pipeline([
        ("preprocessamento", preprocessamento),
        ("modelo", estimador),
    ])

    # O pipeline aprende o preprocessamento apenas no treino.
    pipeline.fit(X_treino, y_treino)

    previsoes = pipeline.predict(X_teste)

    resultado = {
        "modelo": nome,
        "treino_registros": int(len(treino)),
        "teste_registros": int(len(teste)),
        "ano_caracteristicas_teste": 2023,
        "ano_tada_prevista": 2024,
        "MAE": float(
            mean_absolute_error(y_teste, previsoes)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_teste, previsoes)
            )
        ),
        "R2": float(
            r2_score(y_teste, previsoes)
        ),
    }

    resultados.append(resultado)

    print("\nModelo:", nome)
    print("MAE:", round(resultado["MAE"], 4))
    print("RMSE:", round(resultado["RMSE"], 4))
    print("R²:", round(resultado["R2"], 4))


# ============================================================
# 9. SALVAMENTO DO RELATÓRIO
# ============================================================

relatorio = {
    "objetivo": (
        "Prever a TADA do ano seguinte usando características "
        "disponíveis no ano de referência."
    ),
    "variavel_alvo": alvo,
    "definicao_temporal": (
        "TADA_FUTURA corresponde à TADA do mesmo curso, "
        "instituição e coorte no ano de referência seguinte."
    ),
    "chaves_associacao": chaves,
    "treino": "Anos de referência de 2020 a 2022",
    "teste": (
        "Características de 2023 para prever TADA observada em 2024"
    ),
    "preditoras": categoricas + numericas,
    "observacoes": [
        "NU_ANO_REFERENCIA é usado para separar os períodos, "
        "não como preditor.",
        "O pré-processamento é ajustado apenas nos dados de treino.",
        "Cursos e coortes podem aparecer em treino e teste.",
        "Os resultados medem a previsão temporal para registros "
        "das unidades representadas na base.",
        "A definição oficial e a interpretação da TADA devem "
        "ser confirmadas na documentação da fonte."
    ],
    "resultados": resultados,
}

caminho_relatorio = (
    PASTA_RELATORIOS / "baseline_tada_futura.json"
)

caminho_relatorio.write_text(
    json.dumps(
        relatorio,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8",
)

print("\n6. Relatório salvo em:", caminho_relatorio)
print("Experimento concluído.")
