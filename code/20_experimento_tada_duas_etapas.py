from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "silver" / "fluxo_educacao_superior"
OUT_JSON = ROOT / "reports" / "metrics" / "tada_duas_etapas.json"
OUT_CSV = ROOT / "reports" / "local_predictions" / "tada_duas_etapas_previsoes.csv"
KEYS = ["CO_IES", "CO_CURSO", "NU_ANO_INGRESSO"]
YEAR = "NU_ANO_REFERENCIA"
TARGET = "TADA"
CAT = ["TP_CATEGORIA_ADMINISTRATIVA", "TP_ORGANIZACAO_ACADEMICA", "TP_GRAU_ACADEMICO", "TP_MODALIDADE_ENSINO", "CO_REGIAO", "CO_UF", "CO_CINE_AREA_GERAL", "NU_ANO_INGRESSO"]
NUM = ["TADA_ORIGEM", "QT_INGRESSANTE"]
FEATURES = CAT + NUM
SOURCE_NUM = ["QT_INGRESSANTE"]
VAL_YEARS = [2021, 2022]
TEST_YEAR = 2023
SEED = 45345
MAX_SAMPLE = 100_000


def metrics(y, pred):
    pred = np.clip(np.asarray(pred, dtype=float), 0, 100)
    y = np.asarray(y, dtype=float)
    return {"MAE": float(mean_absolute_error(y, pred)), "RMSE": float(mean_squared_error(y, pred) ** 0.5), "R2": float(r2_score(y, pred))}



def make_preprocessor():
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))])
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("cat", cat_pipe, CAT), ("num", num_pipe, NUM)], remainder="drop")



def make_classifier(C):
    return Pipeline([("prep", make_preprocessor()), ("model", LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=SEED))])


def make_regressor(alpha):
    return Pipeline([("prep", make_preprocessor()), ("model", Lasso(alpha=alpha, max_iter=5000))])


def sample_rows(df, limit, seed):
    if len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=seed)



def load_pairs():
    files = sorted(DATA_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet encontrado em {DATA_DIR}")
    parts = []
    required = list(dict.fromkeys(KEYS + [YEAR, TARGET] + CAT + SOURCE_NUM))
    for f in files:
        print(f"Lendo: {f.name}", flush=True)
        d = pd.read_parquet(f)
        missing = [c for c in required if c not in d.columns]
        if missing:
            raise ValueError(f"Colunas ausentes em {f.name}: {missing}")
        parts.append(d[required].copy())
    base = pd.concat(parts, ignore_index=True)
    if base.duplicated(KEYS + [YEAR]).any():
        raise ValueError("Duplicatas de chave/ano encontradas; interrompendo para evitar pareamento ambÃ­guo.")
    base[YEAR] = pd.to_numeric(base[YEAR], errors="coerce")
    base[TARGET] = pd.to_numeric(base[TARGET], errors="coerce")
    base = base.dropna(subset=KEYS + [YEAR, TARGET])
    future = base[KEYS + [YEAR, TARGET]].copy()
    future[YEAR] = future[YEAR] - 1
    future = future.rename(columns={YEAR: "ANO_ORIGEM", TARGET: "TADA_FUTURO"})
    origin = base.rename(columns={YEAR: "ANO_ORIGEM", TARGET: "TADA_ORIGEM"})
    # TADA_ORIGEM is constructed from the source-year TADA, never expected in raw Parquet.
    pairs = origin.merge(future, on=KEYS + ["ANO_ORIGEM"], how="inner", validate="one_to_one")
    pairs["TADA_FUTURO"] = pd.to_numeric(pairs["TADA_FUTURO"], errors="coerce")
    pairs = pairs.dropna(subset=["TADA_FUTURO"])
    pairs = pairs[pairs["TADA_FUTURO"].between(0, 100)]
    print(f"Base: {len(base):,} | pares t->t+1: {len(pairs):,}", flush=True)
    return pairs


def fit_predict(train, valid, c, alpha, use_expected):
    clf_train = sample_rows(train, MAX_SAMPLE, SEED)
    classifier = make_classifier(c)
    classifier.fit(clf_train[FEATURES], (clf_train["TADA_FUTURO"] > 0).astype(int))
    positive_train = train[train["TADA_FUTURO"] > 0]
    reg_train = sample_rows(positive_train, MAX_SAMPLE, SEED + 1)
    regressor = make_regressor(alpha)
    regressor.fit(reg_train[FEATURES], reg_train["TADA_FUTURO"])
    p_positive = classifier.predict_proba(valid[FEATURES])[:, 1]
    positive_value = np.clip(regressor.predict(valid[FEATURES]), 0, 100)
    pred_expected = p_positive * positive_value
    pred_threshold = np.where(p_positive >= 0.5, positive_value, 0.0)
    return pred_expected, pred_threshold


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    pairs = load_pairs()
    config_list = [(C, a) for C in [0.1, 1.0, 10.0] for a in [0.01, 0.1, 1.0]]
    validation_results = []
    for year in VAL_YEARS:
        train = pairs[pairs["ANO_ORIGEM"] < year]
        valid = pairs[pairs["ANO_ORIGEM"] == year]
        print(f"\nValidacao {year}: treino={len(train):,}, validacao={len(valid):,}", flush=True)
        for C, alpha in config_list:
            p_expected, p_threshold = fit_predict(train, valid, C, alpha, True)
            mae_expected = mean_absolute_error(valid["TADA_FUTURO"], np.clip(p_expected, 0, 100))
            mae_threshold = mean_absolute_error(valid["TADA_FUTURO"], np.clip(p_threshold, 0, 100))
            validation_results.append({"year": year, "C": C, "alpha": alpha, "mae_expected": float(mae_expected), "mae_threshold": float(mae_threshold)})
            print(f"C={C}, alpha={alpha}: MAE esperado={mae_expected:.4f}, MAE corte 0.5={mae_threshold:.4f}", flush=True)
    val_df = pd.DataFrame(validation_results)
    mean_scores = val_df.groupby(["C", "alpha"])[["mae_expected", "mae_threshold"]].mean()
    best_expected = mean_scores["mae_expected"].idxmin()
    best_threshold = mean_scores["mae_threshold"].idxmin()
    print(f"\nMelhor configuracao (valor esperado): C={best_expected[0]}, alpha={best_expected[1]}")
    print(f"Melhor configuracao (corte 0.5): C={best_threshold[0]}, alpha={best_threshold[1]}")

    train = pairs[pairs["ANO_ORIGEM"] < TEST_YEAR]
    test = pairs[pairs["ANO_ORIGEM"] == TEST_YEAR].copy()
    print(f"\nTeste {TEST_YEAR}->{TEST_YEAR+1}: treino={len(train):,}, teste={len(test):,}")
    candidates = []
    for mode, config in [("expected", best_expected), ("threshold", best_threshold)]:
        C, alpha = config
        preds = fit_predict(train, test, C, alpha, mode == "expected")
        pred = preds[0] if mode == "expected" else preds[1]
        candidates.append((mode, C, alpha, np.clip(pred, 0, 100)))
    # Choose expected-value vs threshold strictly from validation results, not test.
    options = [
        (float(mean_scores.loc[best_expected, "mae_expected"]), "expected", float(best_expected[0]), float(best_expected[1])),
        (float(mean_scores.loc[best_threshold, "mae_threshold"]), "threshold", float(best_threshold[0]), float(best_threshold[1])),
    ]
    _, mode, C, alpha = min(options, key=lambda x: x[0])
    pred = next(x[3] for x in candidates if x[0] == mode and x[1] == C and x[2] == alpha)
    pred_persist = np.clip(test["TADA_ORIGEM"].values, 0, 100)
    result = {"persistencia": metrics(test["TADA_FUTURO"], pred_persist), f"hurdle_{mode}_C{C}_alpha{alpha}": metrics(test["TADA_FUTURO"], pred), "selected_mode_by_validation": mode, "validation": validation_results, "n_train": int(len(train)), "n_test": int(len(test))}
    print("\n=== RESULTADO TESTE ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out = test[KEYS + ["ANO_ORIGEM", "TADA_ORIGEM", "TADA_FUTURO"]].copy()
    out = out.rename(columns={"TADA_FUTURO": "TADA_REAL_2024"})
    out["PRED_PERSISTENCIA"] = pred_persist
    out["PRED_HURDLE"] = pred
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nRelatorio salvo: {OUT_JSON}")
    print(f"Previsoes salvas: {OUT_CSV}")

if __name__ == "__main__":
    main()


