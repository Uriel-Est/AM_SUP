from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "silver" / "fluxo_educacao_superior"
OUT_JSON = ROOT / "reports" / "metrics" / "rf_tada_experimento.json"
OUT_CSV = ROOT / "reports" / "local_predictions" / "rf_tada_previsoes_teste.csv"

KEYS = ["CO_IES", "CO_CURSO", "NU_ANO_INGRESSO"]
YEAR_COL = "NU_ANO_REFERENCIA"
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
NUMERICAL = ["TADA", "QT_INGRESSANTE"]
FEATURES = CATEGORICAL + NUMERICAL
VALIDATION_YEARS = [2021, 2022]
TEST_ORIGIN_YEAR = 2023
MAX_TRAIN_ROWS = 80000
SEED = 45345
RF_CONFIGS = [
    {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"n_estimators": 60, "max_depth": 18, "min_samples_leaf": 20, "max_features": "sqrt"},
]



def metrics(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 100)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }



def build_model(config):
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, CATEGORICAL),
        ("num", num_pipe, NUMERICAL),
    ])
    model = RandomForestRegressor(
        **config,
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("rf", model)])



def sample_training(data, max_rows=MAX_TRAIN_ROWS):
    if len(data) <= max_rows:
        return data
    return data.sample(n=max_rows, random_state=SEED)



def load_pairs():
    files = sorted(DATA_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum Parquet encontrado em: {DATA_DIR}")
    frames = []
    needed = list(dict.fromkeys(KEYS + [YEAR_COL, TARGET] + CATEGORICAL + NUMERICAL))
    for file in files:
        print(f"Lendo: {file.name}", flush=True)
        frames.append(pd.read_parquet(file, columns=needed))
    base = pd.concat(frames, ignore_index=True)
    # Diagnostico anterior confirmou zero duplicatas nas chaves entre os arquivos.
    base = base.dropna(subset=KEYS + [YEAR_COL, TARGET])
    base[YEAR_COL] = pd.to_numeric(base[YEAR_COL], errors="coerce")
    base[TARGET] = pd.to_numeric(base[TARGET], errors="coerce")
    base = base.dropna(subset=[YEAR_COL, TARGET])
    base[YEAR_COL] = base[YEAR_COL].astype(int)
    base = base[base[TARGET].between(0, 100)]

    origin = base.copy().rename(columns={YEAR_COL: "ANO_ORIGEM"})
    future = base[KEYS + [YEAR_COL, TARGET]].copy()
    future["ANO_ORIGEM"] = future[YEAR_COL] - 1
    future = future.rename(columns={TARGET: "TADA_REAL"})
    future = future[KEYS + ["ANO_ORIGEM", "TADA_REAL"]]

    # Join by the exact prior year to avoid a many-to-many temporal merge.
    pairs = origin.merge(future, on=KEYS + ["ANO_ORIGEM"], how="inner", validate="one_to_one")
    pairs = pairs.dropna(subset=["TADA", "TADA_REAL"])
    pairs = pairs.sort_values("ANO_ORIGEM").reset_index(drop=True)
    print(f"Base: {len(base):,} linhas | pares t->t+1: {len(pairs):,}", flush=True)
    return pairs



def main():
    pairs = load_pairs()
    results = {"validations": {}, "configs": RF_CONFIGS, "test": {}}
    config_scores = {i: [] for i in range(len(RF_CONFIGS))}

    for val_year in VALIDATION_YEARS:
        train = pairs[pairs["ANO_ORIGEM"] < val_year]
        val = pairs[pairs["ANO_ORIGEM"] == val_year]
        if train.empty or val.empty:
            raise ValueError(f"Sem dados para validaÃ§Ã£o {val_year}: treino={len(train)}, validaÃ§Ã£o={len(val)}")
        train_sample = sample_training(train)
        print(f"\nValidaÃ§Ã£o {val_year}: treino={len(train):,} (amostra {len(train_sample):,}), validaÃ§Ã£o={len(val):,}", flush=True)
        val_x = val[FEATURES]
        val_y = val["TADA_REAL"]
        per_config = {}
        for idx, cfg in enumerate(RF_CONFIGS):
            print(f"Random Forest configuraÃ§Ã£o {idx+1}/{len(RF_CONFIGS)}: {cfg}", flush=True)
            model = build_model(cfg)
            model.fit(train_sample[FEATURES], train_sample["TADA_REAL"])
            pred = np.clip(model.predict(val_x), 0, 100)
            m = metrics(val_y, pred)
            per_config[str(idx)] = m
            config_scores[idx].append(m["MAE"])
            print(f"  MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | R2={m['R2']:.4f}", flush=True)
        results["validations"][str(val_year)] = per_config

    mean_scores = {i: float(np.mean(scores)) for i, scores in config_scores.items()}
    best_idx = min(mean_scores, key=mean_scores.get)
    best_config = RF_CONFIGS[best_idx]
    results["selected_config_index"] = int(best_idx)
    results["selected_config"] = best_config
    results["validation_mean_MAE_by_config"] = {str(i): score for i, score in mean_scores.items()}
    print(f"\nConfiguraÃ§Ã£o selecionada: {best_config} (MAE mÃ©dio validaÃ§Ãµes={mean_scores[best_idx]:.4f})", flush=True)

    train_final = pairs[pairs["ANO_ORIGEM"] < TEST_ORIGIN_YEAR]
    test = pairs[pairs["ANO_ORIGEM"] == TEST_ORIGIN_YEAR]
    if train_final.empty or test.empty:
        raise ValueError(f"Sem dados para teste: treino={len(train_final)}, teste={len(test)}")
    train_sample_final = sample_training(train_final)
    print(f"\nTeste {TEST_ORIGIN_YEAR}->{TEST_ORIGIN_YEAR+1}: treino={len(train_final):,} (amostra {len(train_sample_final):,}), teste={len(test):,}", flush=True)
    final_model = build_model(best_config)
    final_model.fit(train_sample_final[FEATURES], train_sample_final["TADA_REAL"])
    pred_rf = np.clip(final_model.predict(test[FEATURES]), 0, 100)
    pred_persistence = np.clip(test["TADA"].to_numpy(dtype=float), 0, 100)
    results["test"]["persistencia"] = metrics(test["TADA_REAL"], pred_persistence)
    results["test"]["random_forest"] = metrics(test["TADA_REAL"], pred_rf)
    results["test"]["n_train_full"] = int(len(train_final))
    results["test"]["n_train_sample"] = int(len(train_sample_final))
    results["test"]["n_test"] = int(len(test))
    print("\n=== TESTE FINAL ===")
    print(json.dumps(results["test"], indent=2, ensure_ascii=False))

    out = test[KEYS + ["ANO_ORIGEM", "TADA", "TADA_REAL"]].copy()
    out = out.rename(columns={"TADA": "TADA_ORIGEM"})
    out["PRED_PERSISTENCIA"] = pred_persistence
    out["PRED_RANDOM_FOREST"] = pred_rf
    out["ERRO_RF"] = np.abs(out["TADA_REAL"] - out["PRED_RANDOM_FOREST"])
    out["ERRO_PERSISTENCIA"] = np.abs(out["TADA_REAL"] - out["PRED_PERSISTENCIA"])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nRelatÃ³rio salvo: {OUT_JSON}")
    print(f"PrevisÃµes salvas: {OUT_CSV}")



if __name__ == "__main__":
    main()


