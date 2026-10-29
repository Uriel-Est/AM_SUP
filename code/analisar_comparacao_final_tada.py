
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PASTA = Path("relatorios")
ARQUIVO = PASTA / "previsoes_tada_teste.csv"

if not ARQUIVO.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {ARQUIVO}")

df = pd.read_csv(ARQUIVO)

obrigatorias = [
    "TADA_REAL_FUTURA",
    "PREVISAO_PERSISTENCIA",
    "PREVISAO_RANDOM_FOREST",
]

faltantes = [c for c in obrigatorias if c not in df.columns]
if faltantes:
    raise ValueError(f"Colunas ausentes: {faltantes}")

df = df.dropna(subset=obrigatorias).copy()

def classificar_faixa(valor):
    if valor <= 1:
        return "0 a 1"
    elif valor <= 5:
        return "Acima de 1 a 5"
    elif valor <= 10:
        return "Acima de 5 a 10"
    elif valor <= 20:
        return "Acima de 10 a 20"
    elif valor <= 50:
        return "Acima de 20 a 50"
    return "Acima de 50 a 100"

ordem = [
    "0 a 1",
    "Acima de 1 a 5",
    "Acima de 5 a 10",
    "Acima de 10 a 20",
    "Acima de 20 a 50",
    "Acima de 50 a 100",
]

df["FAIXA_TADA_FUTURA"] = df["TADA_REAL_FUTURA"].apply(
    classificar_faixa
)

linhas = []

for faixa in ordem:
    grupo = df[df["FAIXA_TADA_FUTURA"] == faixa]

    if grupo.empty:
        continue

    real = grupo["TADA_REAL_FUTURA"]
    persistencia = grupo["PREVISAO_PERSISTENCIA"]
    floresta = grupo["PREVISAO_RANDOM_FOREST"]

    linhas.append({
        "faixa_tada_futura": faixa,
        "quantidade": len(grupo),
        "percentual_teste": len(grupo) / len(df) * 100,
        "media_tada_real": real.mean(),
        "mae_persistencia": mean_absolute_error(real, persistencia),
        "mae_random_forest": mean_absolute_error(real, floresta),
        "rmse_persistencia": mean_squared_error(
            real, persistencia
        ) ** 0.5,
        "rmse_random_forest": mean_squared_error(
            real, floresta
        ) ** 0.5,
    })

tabela = pd.DataFrame(linhas)

tabela.to_csv(
    PASTA / "comparacao_erros_por_faixa_tada.csv",
    index=False,
    encoding="utf-8-sig",
)

metricas_gerais = {}

for nome, coluna in [
    ("persistencia", "PREVISAO_PERSISTENCIA"),
    ("random_forest", "PREVISAO_RANDOM_FOREST"),
]:
    real = df["TADA_REAL_FUTURA"]
    previsto = df[coluna]

    metricas_gerais[nome] = {
        "MAE": float(mean_absolute_error(real, previsto)),
        "RMSE": float(mean_squared_error(real, previsto) ** 0.5),
        "R2": float(r2_score(real, previsto)),
    }

relatorio = {
    "total_teste": int(len(df)),
    "ano_origem": 2023,
    "ano_alvo": 2024,
    "metricas_gerais": metricas_gerais,
    "faixas": tabela.to_dict(orient="records"),
}

(PASTA / "analise_final_tada.json").write_text(
    json.dumps(relatorio, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nMÉTRICAS GERAIS")
print(pd.DataFrame(metricas_gerais).T.round(4).to_string())

print("\nERROS POR FAIXA DE TADA FUTURA")
print(tabela.round(4).to_string(index=False))

print("\nArquivos gerados:")
print(PASTA / "comparacao_erros_por_faixa_tada.csv")
print(PASTA / "analise_final_tada.json")
