"""
SILVER — Converte os .xlsx da bronze em .parquet limpos.

Uso no Positron: abra este .py e clique em Run Source.

Dependências:
    pip install pandas openpyxl pyarrow

Estrutura dos arquivos (confirmada nos 11 arquivos do bronze):
  linha 0..5   : cabeçalho institucional (Ministério, INEP, título)
  linha 6      : nomes legíveis das colunas
  linha 7      : sub-nomes das colunas 27-30 (TAP, TCA, TDA, TCAN)
  linha 8      : NOMES TÉCNICOS (CO_IES, NO_IES, ..., TAP, TCA, TDA, TCAN)
  linha 9..N-2 : dados
  linha N-1    : "Fonte: Censo da Educação Superior/Inep."

Ação: lê com header=8, descarta a linha "Fonte:...", converte para parquet.
Os arquivos permanecem SEPARADOS, um parquet por xlsx, mesmo nome.
"""
from __future__ import annotations

# ==================== CONFIGURAÇÕES — POSITRON ====================
import os
import logging
from pathlib import Path

BRONZE_DIR = Path.cwd() / 'data' / 'bronze' / 'fluxo_educacao_superior'
SILVER_DIR = Path.cwd() / 'data' / 'silver' / 'fluxo_educacao_superior'

# Linha 0-based que vira o header (nomes técnicos CO_IES, ...).
# Confirmado em todos os 11 arquivos. Se algum dia o INEP mudar o layout,
# só isto precisa ser ajustado.
HEADER_ROW = 8

SOBRESCREVER = False
# ================================================================

try:
    import pandas as pd
except ImportError as e:
    raise RuntimeError('Instale: pip install pandas openpyxl pyarrow') from e

LOG = logging.getLogger('silver-fluxo')


def converter(caminho: Path) -> str:
    destino = SILVER_DIR / caminho.with_suffix('.parquet').name
    if destino.exists() and not SOBRESCREVER:
        LOG.info('✔ Já existe, pulando: %s', destino.name)
        return 'pulado'

    try:
        df = pd.read_excel(caminho, header=HEADER_ROW)

        # Sanidade: se o layout mudou, falha aqui em vez de gerar lixo.
        if 'CO_IES' not in df.columns:
            raise ValueError(
                f"coluna 'CO_IES' ausente após header={HEADER_ROW}. "
                f"Primeiras colunas lidas: {list(df.columns)[:6]}"
            )

        # Descarta a(s) linha(s) de rodapé "Fonte: ..."
        rodape = df['CO_IES'].astype(str).str.contains(
            'Fonte', case=False, na=False,
        )
        n_rodape = int(rodape.sum())
        df = df[~rodape]

        # Descarta colunas totalmente vazias (a 31ª coluna "fantasma"
        # que aparece no shape mas não tem nome no header 8).
        df = df.dropna(axis=1, how='all')

        # Descarta eventuais linhas totalmente vazias.
        df = df.dropna(how='all').reset_index(drop=True)

        # Escrita atômica.
        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix('.parquet.tmp')
        df.to_parquet(tmp, engine='pyarrow', index=False)
        os.replace(tmp, destino)

        LOG.info(
            '✅ %s → %s | %d linhas × %d colunas | rodapés removidos: %d',
            caminho.name, destino.name, len(df), df.shape[1], n_rodape,
        )
        return 'convertido'
    except Exception as e:
        LOG.error('Falha em %s: %s', caminho.name, e)
        return 'falha'


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    LOG.info('BRONZE: %s', BRONZE_DIR)
    LOG.info('SILVER: %s', SILVER_DIR)

    if not BRONZE_DIR.is_dir():
        LOG.error('bronze não existe: %s', BRONZE_DIR)
        return

    arquivos = sorted(BRONZE_DIR.glob('*.xlsx'))
    LOG.info('%d arquivos .xlsx', len(arquivos))
    if not arquivos:
        return

    resumo = {'convertido': 0, 'pulado': 0, 'falha': 0}
    for f in arquivos:
        resumo[converter(f)] += 1
    LOG.info('RESUMO: %s', resumo)
    LOG.info('Parquets em: %s', SILVER_DIR)


if __name__ == '__main__':
    main()