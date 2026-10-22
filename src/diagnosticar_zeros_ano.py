
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_temporal_inicial.csv"

df = pd.read_csv(arquivo)

resumo = (
    df.groupby("Ano_destino")["Abandono_destino"]
    .agg(
        registros="count",
        zeros=lambda serie: (serie == 0).sum(),
        positivos=lambda serie: (serie > 0).sum(),
    )
)

resumo["percentual_zeros"] = (
    resumo["zeros"] / resumo["registros"] * 100
).round(2)

print(resumo.to_string())
