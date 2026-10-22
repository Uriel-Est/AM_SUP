from pathlib import Path
import pandas as pd

arquivo = Path("dados/processado/base_semestral_ano_corrigido.csv")

if not arquivo.exists():
    print("ERRO: a base semestral corrigida nao foi encontrada.")
    raise SystemExit(1)

df = pd.read_csv(arquivo, encoding="utf-8-sig")

colunas = ["Ingressante", "Matriculado", "Trancado", "Cancelado", "Concluído"]

print("=" * 60)
print("DIAGNOSTICO DE VALORES DECIMAIS")
print("=" * 60)

for coluna in colunas:
    valores = pd.to_numeric(df[coluna], errors="coerce")
    mascara = valores.notna() & ((valores - valores.round()).abs() > 1e-9)

    print(f"\n{coluna}")
    print("Valores preenchidos:", valores.notna().sum())
    print("Valores decimais:", int(mascara.sum()))

    if mascara.any():
        print("Exemplos:")
        print(df.loc[mascara, ["Ano", "Período", "Curso", "Turno", "Centro", coluna]].head(10).to_string(index=False))

print("\nTIPOS DE DADOS")
print(df[colunas].dtypes.to_string())
print("\nNenhum CSV foi alterado.")
