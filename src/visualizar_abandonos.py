
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "relatorios" / "resumo_base_analitica.csv"

df = pd.read_csv(arquivo)

plt.figure(figsize=(9, 5))
plt.plot(
    df["Ano Referencia"],
    df["abandonos"],
    marker="o"
)

plt.title("Evolução dos abandonos registrados na UFPB")
plt.xlabel("Ano de referência")
plt.ylabel("Quantidade de abandonos")
plt.xticks(df["Ano Referencia"])
plt.grid(True)
plt.tight_layout()
plt.show()

