
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_temporal_inicial.csv"

df = pd.read_csv(arquivo)

# Usar anos anteriores para treino e o último ano para teste.
treino = df[df["Ano_destino"].isin([2022, 2023, 2024])].copy()
teste = df[df["Ano_destino"] == 2025].copy()

# Modelo de referência: previsão do próximo ano igual ao abandono do ano atual.
treino["Previsao"] = treino["Abandono_origem"]
teste["Previsao"] = teste["Abandono_origem"]

print("Modelo de referência: abandono do ano anterior")
print("Registros de treino:", len(treino))
print("Registros de teste:", len(teste))

if len(teste) > 0:
    mae = mean_absolute_error(
        teste["Abandono_destino"], teste["Previsao"]
    )
    rmse = np.sqrt(
        mean_squared_error(teste["Abandono_destino"], teste["Previsao"])
    )

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    saida = raiz / "relatorios" / "resultado_modelo_referencia.txt"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(
        "Modelo de referência: abandono do ano anterior\n"
        f"Registros de treino: {len(treino)}\n"
        f"Registros de teste: {len(teste)}\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n",
        encoding="utf-8",
    )
    print("Relatório salvo em:", saida)
