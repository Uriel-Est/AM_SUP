from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "silver" / "fluxo_educacao_superior"
OUT_JSON = ROOT / "reports" / "metrics" / "xgb_tada_experimento.json"
OUT_CSV = ROOT / "reports" / "local_predictions" / "xgb_tada_previsoes_teste.csv"

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
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
    {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.03, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10.0},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 2.0},
]



def metricas(y_true, y_pred):
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, 100)
    y = np.asarray(y_true, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }



def carregar_dados():
    arquivos = sorted(DATA_DIR.rglob("*.parquet"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em: {DATA_DIR}")
    colunas = list(dict.fromkeys(KEYS + [YEAR, TARGET] + CATEGORICAL + ["QT_INGRESSANTE"]))
    partes = []
    for arquivo in arquivos:
        print(f"Lendo: {arquivo.name}")
        partes.append(pd.read_parquet(arquivo, columns=colunas))
    base = pd.concat(partes, ignore_index=True)
    base[YEAR] = pd.to_numeric(base[YEAR], errors="coerce")
    base[TARGET] = pd.to_numeric(base[TARGET], errors="coerce")
    base = base.dropna(subset=KEYS + [YEAR, TARGET])
    base[YEAR] = base[YEAR].astype(int)
    base = base.drop_duplicates(KEYS + [YEAR]).sort_values(KEYS + [YEAR]).reset_index(drop=True)
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
    pares = pares[pares[TARGET].between(0, 100) & pares["TADA_FUTURO"].between(0, 100)]
    print(f"Pares temporais t -> t+1: {len(pares):,}")
    if pares.empty:
        raise ValueError("Nenhum par temporal vÃ¡lido encontrado.")
    return pares


def preparar_categorias(df, category_levels):
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col].astype("string"), categories=category_levels[col])
    for col in NUMERICAL:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def construir_modelo(config):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=-1,
        **config,
    )


def amostrar(df):
    if len(df) <= MAX_TUNING_ROWS:
        return df
    return df.sample(n=MAX_TUNING_ROWS, random_state=SEED)



def main():
    base = carregar_dados()
    pares = criar_pares(base)
    # Category levels are fixed across splits; only predictor categories are used.
    category_levels = {col: pd.Index(pares[col].dropna().astype(str).unique()).sort_values().tolist() for col in CATEGORICAL}
    treino_final = pares[pares["ANO_ORIGEM"] <= 2022].copy()
    resultados = []
    validacoes_persistencia = {}

    for ano_val in VALIDATION_YEARS:
        treino = pares[pares["ANO_ORIGEM"] < ano_val].copy()
        valid = pares[pares["ANO_ORIGEM"] == ano_val].copy()
        if treino.empty or valid.empty:
            raise ValueError(f"Treino ou validaÃ§Ã£o vazios no ano {ano_val}.")
        treino_s = amostrar(treino)
        X_train = preparar_categorias(treino_s, category_levels)
        y_train = treino_s["TADA_FUTURO"]
        X_valid = preparar_categorias(valid, category_levels)
        y_valid = valid["TADA_FUTURO"]
        validacoes_persistencia[ano_val] = metricas(y_valid, valid[TARGET])
        print(f"\nValidaÃ§Ã£o {ano_val}: treino={len(treino):,}; val={len(valid):,}")

        for i, config in enumerate(CONFIGS, start=1):
            model = construir_modelo(config)
            model.fit(X_train, y_train, verbose=False)
            pred = model.predict(X_valid)
            m = metricas(y_valid, pred)
            resultados.append({"config_id": i, "config": config, "ano_validacao": ano_val, **m})
            print(f"Config {i}: MAE={m['MAE']:.4f}; RMSE={m['RMSE']:.4f}; R2={m['R2']:.4f}")

    tabela = pd.DataFrame(resultados)
    resumo = tabela.groupby("config_id", as_index=False).agg(
        MAE_validacao_medio=("MAE", "mean"), RMSE_validacao_medio=("RMSE", "mean"), R2_validacao_medio=("R2", "mean")
    ).sort_values("MAE_validacao_medio")
    melhor_id = int(resumo.iloc[0]["config_id"])
    melhor_config = CONFIGS[melhor_id - 1]
    print(f"\nConfig selecionada: {melhor_id} | {melhor_config}")

    teste = pares[pares["ANO_ORIGEM"] == TEST_YEAR].copy()
    if teste.empty:
        raise ValueError("NÃ£o hÃ¡ pares de teste 2023 -> 2024.")
    treino_s = amostrar(treino_final)
    X_train_full = preparar_categorias(treino_s, category_levels)
    y_train_full = treino_s["TADA_FUTURO"]
    X_test = preparar_categorias(teste, category_levels)
    y_test = teste["TADA_FUTURO"]

    final_model = construir_modelo(melhor_config)
    final_model.fit(X_train_full, y_train_full, verbose=False)
    pred_xgb = np.clip(final_model.predict(X_test), 0, 100)
    pred_persist = np.clip(teste[TARGET].to_numpy(dtype=float), 0, 100)
    metricas_teste = {
        "persistencia": metricas(y_test, pred_persist),
        f"xgboost_config_{melhor_id}": metricas(y_test, pred_xgb),
    }

    previsoes = teste[KEYS + ["ANO_ORIGEM", TARGET, "TADA_FUTURO"]].copy()
    previsoes = previsoes.rename(columns={TARGET: "TADA_ORIGEM", "TADA_FUTURO": "TADA_REAL_2024"})
    previsoes["PRED_PERSISTENCIA"] = pred_persist
    previsoes["PRED_XGBOOST"] = pred_xgb

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    relatorio = {
        "objetivo": "Prever TADA no ano seguinte usando apenas caracterÃ­sticas do ano de origem.",
        "definicao_temporal": "origem t; alvo t+1; teste 2023 -> 2024",
        "n_pares_total": int(len(pares)),
        "n_treino_final": int(len(treino_final)),
        "n_treino_final_utilizado_modelo": int(len(treino_s)),
        "n_teste": int(len(teste)),
        "anos_validacao": VALIDATION_YEARS,
        "metricas_persistencia_validacao": validacoes_persistencia,
        "configuracoes": CONFIGS,
        "resultados_validacao": resultados,
        "ranking_configuracoes": resumo.to_dict(orient="records"),
        "melhor_configuracao": {"config_id": melhor_id, **melhor_config},
        "metricas_teste_2023_2024": metricas_teste,
        "features": FEATURES,
        "nota": "Teste final reservado: origem 2023 para alvo 2024; seleÃ§Ã£o baseada apenas nas validaÃ§Ãµes 2021 e 2022.",
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    previsoes.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n===== RESULTADO TESTE 2023 -> 2024 =====")
    print(json.dumps(metricas_teste, ensure_ascii=False, indent=2))
    print(f"RelatÃ³rio salvo: {OUT_JSON}")
    print(f"PrevisÃµes salvas: {OUT_CSV}")


if __name__ == "__main__":
    main()


