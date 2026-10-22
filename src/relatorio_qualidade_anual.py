from pathlib import Path
import pandas as pd

arquivo = Path("dados/bruto/discentes_de_cursos_de_graduacao_da_ufpb_2026-07.csv")
df = pd.read_csv(arquivo, encoding="utf-8-sig")

df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

colunas_numericas = [
    "Ano Referencia",
    "ID Matriz Curricular",
    "Codigo Inep",
    "Qtde Estudantes",
    "Qtde Abandono",
    "Qtde Trancamentos",
    "Qtde Concluintes",
]

for coluna in colunas_numericas:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

print("=" * 65)
print("RELATORIO DE QUALIDADE - BASE ANUAL DA UFPB")
print("=" * 65)

print("\n1. DIMENSOES")
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")

print("\n2. ANOS DISPONIVEIS")
print(sorted(df["Ano Referencia"].dropna().unique().tolist()))

print("\n3. VALORES AUSENTES POR COLUNA")
print(df.isna().sum().to_string())

print("\n4. LINHAS DUPLICADAS EXATAS")
print(df.duplicated().sum())

chave = ["Ano Referencia", "ID Matriz Curricular"]
print("\n5. DUPLICIDADES NA CHAVE ANO + MATRIZ CURRICULAR")
print(df.duplicated(subset=chave, keep=False).sum())

print("\n6. REGISTROS SEM CODIGO INEP")
print(df[df["Codigo Inep"].isna()].to_string(index=False))

print("\n7. RESUMO POR ANO")
resumo = df.groupby("Ano Referencia").agg(
    registros=("Curso", "size"),
    estudantes=("Qtde Estudantes", "sum"),
    abandonos=("Qtde Abandono", "sum"),
    trancamentos=("Qtde Trancamentos", "sum"),
    concluintes=("Qtde Concluintes", "sum"),
    matrizes_com_zero_abandono=(
        "Qtde Abandono", lambda s: (s == 0).sum()
    ),
    matrizes_com_abandono_positivo=(
        "Qtde Abandono", lambda s: (s > 0).sum()
    ),
)
print(resumo.to_string())

print("\n8. MAIORES CONTAGENS DE ABANDONO POR ANO")
for ano in sorted(df["Ano Referencia"].dropna().unique()):
    recorte = df[df["Ano Referencia"] == ano].copy()
    recorte = recorte.sort_values("Qtde Abandono", ascending=False)

    print(f"\nANO {int(ano)}")
    print(
        recorte[
            [
                "Curso",
                "ID Matriz Curricular",
                "Qtde Estudantes",
                "Qtde Abandono",
            ]
        ].head(5).to_string(index=False)
    )

print("\n9. VERIFICACOES ADICIONAIS")
print("Estudantes com valor negativo:",
      (df["Qtde Estudantes"] < 0).sum())
print("Abandonos com valor negativo:",
      (df["Qtde Abandono"] < 0).sum())
print("Trancamentos com valor negativo:",
      (df["Qtde Trancamentos"] < 0).sum())
print("Concluintes com valor negativo:",
      (df["Qtde Concluintes"] < 0).sum())

print("\nATENCAO:")
print("- O ano de 2026 pode estar incompleto; o relatorio nao confirma sua completude.")
print("- Contagens de abandono nao devem ser tratadas automaticamente como taxas.")
print("- Nenhum registro foi excluido ou corrigido neste diagnostico.")
