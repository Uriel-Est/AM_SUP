
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]

entrada = raiz / "dados" / "processado" / "base_analitica_inicial.csv"
saida = raiz / "dados" / "processado" / "base_temporal_inicial.csv"

df = pd.read_csv(entrada)

colunas = [
    "ID Matriz Curricular",
    "Ano Referencia",
    "Curso",
    "Turno",
    "Grau Academico",
    "Modalidade",
    "Qtde Estudantes",
    "Qtde Abandono",
    "Qtde Trancamentos",
    "Qtde Concluintes",
]

df = df[colunas].copy()
df["Ano Referencia"] = pd.to_numeric(df["Ano Referencia"]).astype(int)

origem = df.copy()
destino = df.copy()

origem = origem.rename(columns={
    "Ano Referencia": "Ano_origem",
    "Qtde Estudantes": "Estudantes_origem",
    "Qtde Abandono": "Abandono_origem",
    "Qtde Trancamentos": "Trancamentos_origem",
    "Qtde Concluintes": "Concluintes_origem",
})

destino = destino.rename(columns={
    "Ano Referencia": "Ano_destino",
    "Qtde Abandono": "Abandono_destino",
})

base = origem.merge(
    destino[
        ["ID Matriz Curricular", "Ano_destino", "Abandono_destino"]
    ],
    left_on="ID Matriz Curricular",
    right_on="ID Matriz Curricular",
    how="inner",
)

base = base[base["Ano_destino"] == base["Ano_origem"] + 1].copy()

base.to_csv(saida, index=False, encoding="utf-8-sig")

print("Base temporal criada.")
print("Registros:", len(base))
print("Pares por ano:")
print(base.groupby(["Ano_origem", "Ano_destino"]).size().to_string())
print("Arquivo salvo em:", saida)
