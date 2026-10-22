
from pathlib import Path
import pandas as pd

raiz = Path(__file__).resolve().parents[1]

# Localiza o arquivo anual original
pasta_bruta = raiz / "dados" / "bruto"
arquivos = list(pasta_bruta.glob("discentes_de_cursos_de_graduacao_da_ufpb_2026-07.csv"))

if not arquivos:
    raise FileNotFoundError(
        f"Arquivo anual não encontrado em: {pasta_bruta}"
    )

df = pd.read_csv(arquivos[0])

# Padroniza nomes das colunas para facilitar o uso no código
df.columns = df.columns.str.strip()

# Exclui 2026 da primeira análise por possível incompletude.
# Não exclui 2020 nem 2021, mas sinaliza esses anos para revisão.
df["Ano Referencia"] = pd.to_numeric(
    df["Ano Referencia"], errors="coerce"
)

base = df[df["Ano Referencia"].between(2020, 2025)].copy()

base["Ano_com_alerta"] = base["Ano Referencia"].isin([2020, 2021])

# Não remove registros sem Código Inep: preserva a matriz curricular.
# Não altera contagens nem preenche valores artificialmente.
saida = raiz / "dados" / "processado" / "base_analitica_inicial.csv"
saida.parent.mkdir(parents=True, exist_ok=True)
base.to_csv(saida, index=False, encoding="utf-8-sig")

print("Base analítica inicial criada.")
print("Dimensões:", base.shape)
print("Anos:", sorted(base["Ano Referencia"].dropna().unique()))
print("Registros com alerta:", int(base["Ano_com_alerta"].sum()))
print("Valores ausentes por coluna:")
print(base.isna().sum())
print("Arquivo salvo em:", saida)
