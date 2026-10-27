
"""
SILVER — Converte os arquivos .xlsx da Bronze em arquivos .parquet limpos.

Uso:
    Execute a partir da raiz do repositório:
    python -u .\code\silver_fluxo_ensino_superior.py

Dependências:
    pandas, openpyxl, pyarrow

Os arquivos permanecem separados: um Parquet por arquivo Excel.
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

import pandas as pd


# ==================== CONFIGURAÇÕES ====================

BRONZE_DIR = (
    Path.cwd() / "data" / "bronze" / "fluxo_educacao_superior"
)
SILVER_DIR = (
    Path.cwd() / "data" / "silver" / "fluxo_educacao_superior"
)

HEADER_ROW = 8

# False preserva os Parquet já existentes.
# Mude para True somente se quiser recriar todos os arquivos.
SOBRESCREVER = False

LOG = logging.getLogger("silver-fluxo")


def converter(caminho: Path) -> str:
    """Converte um arquivo Excel em Parquet e retorna seu resultado."""

    destino = SILVER_DIR / caminho.with_suffix(".parquet").name

    if destino.exists() and not SOBRESCREVER:
        LOG.info("Já existe, pulando: %s", destino.name)
        return "pulado"

    try:
        LOG.info("Lendo Excel: %s", caminho.name)

        df = pd.read_excel(caminho, header=HEADER_ROW)

        if "CO_IES" not in df.columns:
            raise ValueError(
                f"Coluna 'CO_IES' ausente após header={HEADER_ROW}. "
                f"Colunas encontradas: {list(df.columns)}"
            )

        # Remove linhas de rodapé que contêm "Fonte".
        rodape = df["CO_IES"].astype(str).str.contains(
            "Fonte", case=False, na=False
        )
        n_rodape = int(rodape.sum())
        df = df.loc[~rodape].copy()

        # Remove colunas e linhas completamente vazias.
        df = df.dropna(axis=1, how="all")
        df = df.dropna(how="all").reset_index(drop=True)

        SILVER_DIR.mkdir(parents=True, exist_ok=True)

        # Grava primeiro em arquivo temporário para evitar
        # deixar um Parquet incompleto em caso de falha.
        tmp = destino.with_name(destino.stem + ".tmp.parquet")

        try:
            df.to_parquet(tmp, engine="pyarrow", index=False)
            os.replace(tmp, destino)
        finally:
            if tmp.exists():
                tmp.unlink()

        LOG.info(
            "CONVERTIDO: %s | %s | %s linhas x %s colunas | "
            "rodapés removidos: %s",
            caminho.name,
            destino.name,
            len(df),
            df.shape[1],
            n_rodape,
        )

        return "convertido"

    except Exception:
        LOG.error(
            "FALHA ao processar %s:\n%s",
            caminho.name,
            traceback.format_exc(),
        )
        return "falha"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    LOG.info("BRONZE: %s", BRONZE_DIR.resolve())
    LOG.info("SILVER: %s", SILVER_DIR.resolve())

    if not BRONZE_DIR.is_dir():
        LOG.error("A pasta Bronze não existe: %s", BRONZE_DIR.resolve())
        return

    arquivos = sorted(BRONZE_DIR.glob("*.xlsx"))
    LOG.info("Total de arquivos Excel encontrados: %d", len(arquivos))

    if not arquivos:
        LOG.warning("Nenhum arquivo .xlsx encontrado.")
        return

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    resumo = {
        "convertido": 0,
        "pulado": 0,
        "falha": 0,
    }

    for indice, arquivo in enumerate(arquivos, start=1):
        LOG.info(
            "PROGRESSO: arquivo %d de %d: %s",
            indice,
            len(arquivos),
            arquivo.name,
        )

        try:
            resultado = converter(arquivo)
            resumo[resultado] += 1
        except Exception:
            # Uma falha inesperada não impede os próximos arquivos.
            resumo["falha"] += 1
            LOG.exception("Erro inesperado em %s", arquivo.name)

    LOG.info("================ RESUMO FINAL ================")
    LOG.info("Arquivos Excel encontrados: %d", len(arquivos))
    LOG.info("Convertidos: %d", resumo["convertido"])
    LOG.info("Já existentes, ignorados: %d", resumo["pulado"])
    LOG.info("Falhas: %d", resumo["falha"])
    LOG.info("Pasta Silver: %s", SILVER_DIR.resolve())
    LOG.info("================================================")


if __name__ == "__main__":
    main()
