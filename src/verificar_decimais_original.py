from pathlib import Path
import pandas as pd

arquivo = Path(
    "dados/bruto/tabela_resumo___indicadores_graduacao_2026-10-16T19_42_23.516457508-03_00.csv"
)

if not arquivo.exists():
    print("Arquivo original nao encontrado. Confira o nome em dados/bruto.")
    raise SystemExit(1)

df = pd.read_csv(arquivo, encoding="utf-8-sig", dtype=str)

coluna = "Matriculado"

for c in ["Ano", "Período", "Curso", "Turno", "Centro"]:
    df[c] = df[c].fillna("")

valores = pd.to_numeric(df[coluna], errors="coerce")
mascara = valores.notna() & ((valores - valores.round()).abs() > 1e-9)

print("REGISTROS DECIMAIS NO CSV ORIGINAL")
print("Quantidade:", int(mascara.sum()))

print(
    df.loc[
        mascara,
        ["Ano", "Período", "Curso", "Turno", "Centro", coluna]
    ].to_string(index=False)
)

print("\nNenhum arquivo foi alterado.")
