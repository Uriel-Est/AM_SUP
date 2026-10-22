from pathlib import Path
import pandas as pd

bruto = Path("dados/bruto")
processado = Path("dados/processado")
relatorios = Path("relatorios")
relatorios.mkdir(parents=True, exist_ok=True)

arquivo_anual = bruto / "discentes_de_cursos_de_graduacao_da_ufpb_2026-07.csv"
arquivo_semestral = processado / "base_semestral_ano_corrigido.csv"

if not arquivo_anual.exists():
    raise FileNotFoundError(f"Base anual nao encontrada: {arquivo_anual}")

if not arquivo_semestral.exists():
    raise FileNotFoundError(
        "Base semestral corrigida nao encontrada. "
        "Execute: python src/preparar_base_semestral.py"
    )

anual = pd.read_csv(arquivo_anual, encoding="utf-8-sig")
semestral = pd.read_csv(arquivo_semestral, encoding="utf-8-sig")

anual.columns = anual.columns.str.strip()
semestral.columns = semestral.columns.str.strip()

for coluna in [
    "Ano Referencia", "Qtde Estudantes", "Qtde Abandono",
    "Qtde Trancamentos", "Qtde Concluintes"
]:
    anual[coluna] = pd.to_numeric(anual[coluna], errors="coerce")

for coluna in [
    "Ano", "Período", "Matriculado", "Trancado",
    "Cancelado", "Concluído", "Ingressante"
]:
    semestral[coluna] = pd.to_numeric(
        semestral[coluna], errors="coerce"
    )

print("=" * 65)
print("COMPARACAO DESCRITIVA DAS BASES DA UFPB")
print("=" * 65)

print("\n1. DIMENSOES")
print("Base anual:", anual.shape)
print("Base semestral:", semestral.shape)

print("\n2. ANOS DISPONIVEIS")
print("Anual:", sorted(anual["Ano Referencia"].dropna().unique().tolist()))
print("Semestral:", sorted(semestral["Ano"].dropna().unique().tolist()))

print("\n3. REGISTROS POR ANO E PERIODO NA BASE SEMESTRAL")
print(semestral.groupby(["Ano", "Período"]).size().to_string())

print("\n4. RESUMO ANUAL")
resumo_anual = anual.groupby("Ano Referencia").agg(
    registros_anuais=("Curso", "size"),
    estudantes=("Qtde Estudantes", "sum"),
    abandonos=("Qtde Abandono", "sum"),
    trancamentos=("Qtde Trancamentos", "sum"),
    concluintes=("Qtde Concluintes", "sum"),
)

print(resumo_anual.to_string())

print("\n5. RESUMO SEMESTRAL POR ANO")
resumo_semestral = semestral.groupby("Ano").agg(
    registros_semestrais=("Curso", "size"),
    matriculados=("Matriculado", "sum"),
    cancelados=("Cancelado", "sum"),
    trancados=("Trancado", "sum"),
    concluidos=("Concluído", "sum"),
)

print(resumo_semestral.to_string())

print("\n6. VALORES AUSENTES NA BASE ANUAL")
print(anual.isna().sum().to_string())

print("\n7. VALORES AUSENTES NA BASE SEMESTRAL")
print(semestral.isna().sum().to_string())

print("\n8. OBSERVACOES")
print("- Abandono e cancelamento nao foram considerados equivalentes.")
print("- As bases podem ter unidades de registro e definicoes diferentes.")
print("- Valores ausentes nao foram substituidos por zero.")
print("- O ano de 2026 pode estar incompleto.")

arquivo_relatorio = relatorios / "comparacao_consistencia_bases.txt"

with arquivo_relatorio.open("w", encoding="utf-8") as f:
    f.write("COMPARACAO DESCRITIVA DAS BASES DA UFPB\n\n")
    f.write("RESUMO ANUAL\n")
    f.write(resumo_anual.to_string())
    f.write("\n\nRESUMO SEMESTRAL POR ANO\n")
    f.write(resumo_semestral.to_string())
    f.write("\n\nAUSENTES NA BASE ANUAL\n")
    f.write(anual.isna().sum().to_string())
    f.write("\n\nAUSENTES NA BASE SEMESTRAL\n")
    f.write(semestral.isna().sum().to_string())
    f.write(
        "\n\nNota: abandono e cancelamento nao foram considerados equivalentes. "
        "O ano de 2026 pode estar incompleto.\n"
    )

print(f"\nRelatorio salvo em: {arquivo_relatorio}")
print("Nenhum CSV original foi alterado.")
