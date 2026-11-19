from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "silver" / "fluxo_educacao_superior"
OUT_JSON = ROOT / "reports" / "metrics" / "comparacao_tada_persistencia.json"
OUT_CSV = ROOT / "reports" / "local_predictions" / "previsoes_tada_teste.csv"

KEYS = ["CO_IES", "CO_CURSO", "NU_ANO_INGRESSO"]
YEAR = "NU_ANO_REFERENCIA"
TARGET = "TADA"
CATEGORICAL = [
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "CO_REGIAO",
    "CO_UF",
    "CO_CINE_AREA_GERAL",
    "NU_ANO_INGRESSO",
]
NUMERICAL = ["TADA", "QT_INGRESSANTE", "ANO_ORIGEM"]
FEATURES = CATEGORICAL + NUMERICAL
VALIDATION_YEARS = [2021, 2022]
TEST_YEAR = 2023
MAX_TUNING_ROWS = 100_000
SEED = 45345
CONFIGS = [
    ("Ridge", 0.1), ("Ridge", 1.0), ("Ridge", 10.0), ("Ridge", 100.0),
    ("Lasso", 0.001), ("Lasso", 0.01), ("Lasso", 0.1),
    ("ElasticNet", 0.001), ("ElasticNet", 0.01), ("ElasticNet", 0.1),
]


def metricas(y_true, y_pred):
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, 100)
    y = np.asarray(y_true, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def criar_modelo(nome, alpha):
    if nome == "Ridge":
        estimator = Ridge(alpha=alpha)
    elif nome == "Lasso":
        estimator = Lasso(alpha=alpha, max_iter=5000)
    elif nome == "ElasticNet":
        estimator = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000)
    else:
        raise ValueError(f"Modelo desconhecido: {nome}")

    prep = ColumnTransformer(
        transformers=[
            ("categoricas", Pipeline([
                ("imputador", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
            ]), CATEGORICAL),
            ("numericas", Pipeline([
                ("imputador", SimpleImputer(strategy="median")),
                ("escalador", StandardScaler()),
            ]), NUMERICAL),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessamento", prep), ("modelo", estimator)])



def carregar_dados():
    arquivos = sorted(DATA_DIR.rglob("*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em: {DATA_DIR}")

    colunas_necessarias = list(dict.fromkeys(KEYS + [YEAR, TARGET] + CATEGORICAL + ["QT_INGRESSANTE"]))
    partes = []
    for arquivo in arquivos:
        print(f"Lendo: {arquivo.name}")
        df = pd.read_parquet(arquivo, columns=colunas_necessarias)
        partes.append(df)

    base = pd.concat(partes, ignore_index=True)
    base[YEAR] = pd.to_numeric(base[YEAR], errors="coerce")
    base[TARGET] = pd.to_numeric(base[TARGET], errors="coerce")
    base = base.dropna(subset=KEYS + [YEAR, TARGET])
    base[YEAR] = base[YEAR].astype(int)
    base = base.drop_duplicates(KEYS + [YEAR])
    base = base.sort_values(KEYS + [YEAR]).reset_index(drop=True)
    print(f"Base consolidada: {len(base):,} registros")
    return base


def criar_pares(base):
    origem = base.copy()
    destino = base[KEYS + [YEAR, TARGET]].copy()
    destino[YEAR] = destino[YEAR] - 1
    destino = destino.rename(columns={TARGET: "TADA_FUTURO"})
    pares = origem.merge(destino, on=KEYS + [YEAR], how="inner", validate="one_to_one")
    pares = pares.rename(columns={YEAR: "ANO_ORIGEM"})
    pares = pares.dropna(subset=[TARGET, "TADA_FUTURO"])
    pares = pares[(pares[TARGET].between(0, 100)) & (pares["TADA_FUTURO"].between(0, 100))]
    print(f"Pares temporais t -> t+1: {len(pares):,}")
    if pares.empty:
        raise ValueError("NÃ£o foram encontrados pares temporais vÃ¡lidos de TADA.")
    return pares


def amostrar(df):
    if len(df) <= MAX_TUNING_ROWS:
        return df
    return df.sample(n=MAX_TUNING_ROWS, random_state=SEED)



def main():
    base = carregar_dados()
    pares = criar_pares(base)
    treino_final = pares[pares["ANO_ORIGEM"] <= 2022].copy()
    validacoes = {}
    resultados_config = []

    for ano_val in VALIDATION_YEARS:
        treino = pares[pares["ANO_ORIGEM"] < ano_val].copy()
        valid = pares[pares["ANO_ORIGEM"] == ano_val].copy()
        if treino.empty or valid.empty:
            raise ValueError(f"Treino ou validaÃ§Ã£o vazios para ano {ano_val}.")
        treino_amostra = amostrar(treino)
        X_train = treino_amostra[FEATURES]
        y_train = treino_amostra["TADA_FUTURO"]
        X_valid = valid[FEATURES]
        y_valid = valid["TADA_FUTURO"]
        pers_valid = valid[TARGET]
        validacoes[ano_val] = {"persistencia": metricas(y_valid, pers_valid)}
        print(f"\nValidaÃ§Ã£o {ano_val}: treino={len(treino):,}, validaÃ§Ã£o={len(valid):,}")
        for nome, alpha in CONFIGS:
            model = criar_modelo(nome, alpha)
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            m = metricas(y_valid, pred)
            resultados_config.append({"modelo": nome, "alpha": alpha, "ano_validacao": ano_val, **m})
            print(f"{nome} alpha={alpha}: MAE={m['MAE']:.4f}")

    tabela = pd.DataFrame(resultados_config)
    resumo = tabela.groupby(["modelo", "alpha"], as_index=False).agg(
        MAE_validacao_medio=("MAE", "mean"), RMSE_validacao_medio=("RMSE", "mean"),
        R2_validacao_medio=("R2", "mean")
    ).sort_values("MAE_validacao_medio")
    melhor = resumo.iloc[0]
    melhor_nome = str(melhor["modelo"])
    melhor_alpha = float(melhor["alpha"])
    print(f"\nMelhor configuraÃ§Ã£o: {melhor_nome}, alpha={melhor_alpha}")

    # AvaliaÃ§Ã£o final: treinamento atÃ© origem 2022, teste em origem 2023 -> alvo 2024.
    teste = pares[pares["ANO_ORIGEM"] == TEST_YEAR].copy()
    if teste.empty:
        raise ValueError("NÃ£o hÃ¡ pares de teste para 2023 -> 2024.")
    treino_final_amostra = amostrar(treino_final)
    X_train_full = treino_final_amostra[FEATURES]
    y_train_full = treino_final_amostra["TADA_FUTURO"]
    X_test = teste[FEATURES]
    y_test = teste["TADA_FUTURO"]

    final_model = criar_modelo(melhor_nome, melhor_alpha)
    final_model.fit(X_train_full, y_train_full)
    pred_model = np.clip(final_model.predict(X_test), 0, 100)
    pred_persist = np.clip(teste[TARGET].to_numpy(dtype=float), 0, 100)

    metricas_teste = {
        "persistencia": metricas(y_test, pred_persist),
        f"{melhor_nome}_alpha_{melhor_alpha:g}": metricas(y_test, pred_model),
    }
    previsoes = teste[KEYS + ["ANO_ORIGEM", TARGET, "TADA_FUTURO"]].copy()
    previsoes = previsoes.rename(columns={TARGET: "TADA_ORIGEM", "TADA_FUTURO": "TADA_REAL_2024"})
    previsoes["PRED_PERSISTENCIA"] = pred_persist
    previsoes["PRED_MODELO"] = pred_model

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    relatorio = {
        "objetivo": "Prever TADA no ano seguinte usando apenas informaÃ§Ãµes do ano de origem.",
        "definicao_temporal": "origem t; alvo t+1; teste 2023 -> 2024",
        "n_pares_total": int(len(pares)),
        "n_treino_final": int(len(treino_final)),
        "n_treino_final_utilizado_modelo": int(len(treino_final_amostra)),
        "n_teste": int(len(teste)),
        "anos_validacao": VALIDATION_YEARS,
        "metricas_persistencia_validacao": validacoes,
        "configuracoes_validacao": resultados_config,
        "ranking_configuracoes": resumo.to_dict(orient="records"),
        "melhor_configuracao": {"modelo": melhor_nome, "alpha": melhor_alpha},
        "metricas_teste_2023_2024": metricas_teste,
        "features": FEATURES,
        "nota": "Teste final reservado: origem 2023 e alvo 2024. ConfiguraÃ§Ã£o escolhida apenas com validaÃ§Ãµes 2021 e 2022.",
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    previsoes.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n===== RESULTADO TESTE 2023 -> 2024 =====")
    print(json.dumps(metricas_teste, ensure_ascii=False, indent=2))
    print(f"\nRelatÃ³rio salvo: {OUT_JSON}")
    print(f"PrevisÃµes salvas: {OUT_CSV}")



if __name__ == "__main__":
    main()


