
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]
arquivo = raiz / "dados" / "processado" / "base_semestral_ano_corrigido.csv"
saida = raiz / "relatorios" / "avaliacao_base_modelagem.txt"

df = pd.read_csv(arquivo)

linhas = []
linhas.append(f"Total de registros: {len(df)}")
linhas.append(f"Total de colunas: {len(df.columns)}")
linhas.append(f"Anos disponíveis: {sorted(df['Ano'].unique())}")
linhas.append(f"Valores ausentes por coluna:\n{df.isna().sum().to_string()}")

for coluna in ["Ingressante", "Matriculado", "Trancado",
               "Cancelado", "Concluído"]:
    valores = pd.to_numeric(df[coluna], errors="coerce")
    linhas.append(
        f"\n{coluna}: "
        f"mínimo={valores.min()}, "
        f"mediana={valores.median()}, "
        f"máximo={valores.max()}, "
        f"valores negativos={(valores < 0).sum()}, "
        f"valores decimais={(valores.dropna() % 1 != 0).sum()}"
    )

relatorio = "\n".join(linhas)

saida.parent.mkdir(parents=True, exist_ok=True)
saida.write_text(relatorio, encoding="utf-8")

print(relatorio)
print(f"\nRelatório salvo em: {saida}")
