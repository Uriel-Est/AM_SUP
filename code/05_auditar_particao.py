"""
Auditoria de particao treino/teste do experimento TADA futura.

Objetivo:
1. Confirmar como a particao foi feita (por ANO_ALVO = 2024 no teste).
2. Medir sobreposicao de chave (CO_IES, CO_CURSO, NU_ANO_INGRESSO)
   entre treino e teste -> vazamento de particao.
3. Listar colunas da base Gold e sinalizar quais existem apenas
   no ano-alvo (candidatas a vazamento temporal).
4. Medir perda de observacoes (registros sem par).
"""

import os
import pandas as pd

# Ajuste o caminho se necessario
GOLD = os.path.join("data", "gold", "base_gold_tada.parquet")

if not os.path.exists(GOLD):
    # tenta alternativas comuns
    candidatos = [
        os.path.join("data", "gold", "base_gold_tada.parquet"),
        os.path.join("data", "gold", "gold_tada.parquet"),
        os.path.join("data", "gold", "base_gold.parquet"),
    ]
    for c in candidatos:
        if os.path.exists(c):
            GOLD = c
            break
    else:
        raise FileNotFoundError(
            "Nao encontrei a base Gold. Ajuste a variavel GOLD no script."
        )

print("=" * 70)
print("LENDO BASE GOLD:", GOLD)
df = pd.read_parquet(GOLD)
print("Shape:", df.shape)
print("Colunas:")
for c in df.columns:
    print("  -", c)

# ------------------------------------------------------------------
# 1. Identificar coluna de ano-alvo
# ------------------------------------------------------------------
col_ano_alvo = None
for cand in ["ANO_ALVO", "NU_ANO_ALVO", "ano_alvo"]:
    if cand in df.columns:
        col_ano_alvo = cand
        break

if col_ano_alvo is None:
    print("\n[AVISO] Nao encontrei coluna de ano-alvo. Colunas com 'ANO':")
    print([c for c in df.columns if "ANO" in c.upper()])
    raise SystemExit

print("\n" + "=" * 70)
print("DISTRIBUICAO DO ANO-ALVO")
print(df[col_ano_alvo].value_counts().sort_index())

# ------------------------------------------------------------------
# 2. Refazer a particao exatamente como no modelo
# ------------------------------------------------------------------
treino = df[df[col_ano_alvo] < 2024].copy()
teste = df[df[col_ano_alvo] == 2024].copy()
print("\nTreino:", len(treino), "| Teste:", len(teste))

# ------------------------------------------------------------------
# 3. Sobreposicao de chave
# ------------------------------------------------------------------
chave = ["CO_IES", "CO_CURSO", "NU_ANO_INGRESSO"]
faltando = [c for c in chave if c not in df.columns]
if faltando:
    print("\n[AVISO] Colunas de chave ausentes:", faltando)
    print("Colunas disponiveis:", list(df.columns))
else:
    k_treino = set(map(tuple, treino[chave].drop_duplicates().values))
    k_teste = set(map(tuple, teste[chave].drop_duplicates().values))
    inter = k_treino & k_teste
    print("\n" + "=" * 70)
    print("SOBREPOSICAO DE CHAVE (CO_IES, CO_CURSO, NU_ANO_INGRESSO)")
    print(f"  Chaves unicas no treino : {len(k_treino):,}")
    print(f"  Chaves unicas no teste  : {len(k_teste):,}")
    print(f"  Chaves em ambos         : {len(inter):,}")
    if len(k_teste) > 0:
        pct = 100 * len(inter) / len(k_teste)
        print(f"  % do teste que tambem esta no treino: {pct:.2f}%")
    print("\n  -> Se o percentual for alto, ha vazamento de particao:")
    print("     a mesma coorte/curso aparece em treino e teste.")

# ------------------------------------------------------------------
# 4. Colunas candidatas a vazamento temporal
#    (existem apenas para o ano-alvo / ano seguinte)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("COLUNAS: checagem rapida de nulos por ano-alvo")
print("(colunas 100% nulas no treino e nao nulas no teste sao suspeitas)")
nulos_treino = treino.isna().mean()
nulos_teste = teste.isna().mean()
suspeitas = []
for c in df.columns:
    if nulos_treino.get(c, 1) >= 0.999 and nulos_teste.get(c, 1) <= 0.001:
        suspeitas.append(c)
if suspeitas:
    print("Suspeitas (100% nulas no treino, preenchidas no teste):")
    for c in suspeitas:
        print("  -", c)
else:
    print("Nenhuma coluna com esse padrao exato.")
print("\nRevise manualmente as colunas de taxa/indicador e verifique")
print("se o ano de referencia de cada uma e <= ano do corte.")

# ------------------------------------------------------------------
# 5. Perda de observacoes
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("PERDA DE OBSERVACOES")
print("Registros de origem (do log): 2,731,155")
print("Com par (ano seguinte)      : 2,407,858")
print("Sem par                     : 323,297")
print(f"Perda relativa              : {100*323297/2731155:.2f}%")

print("\nAuditoria de particao concluida.")