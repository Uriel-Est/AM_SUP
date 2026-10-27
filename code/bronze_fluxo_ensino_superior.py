"""
BRONZE — Indicadores de Trajetória / Fluxo Histórico da Educação Superior (INEP).

Execute no Positron: abra este .py e clique em Run Source.

Instalar uma única vez:
    pip install requests beautifulsoup4 tqdm truststore

Por coorte:
  1. Baixa o ZIP para _temporarios/
  2. Extrai SOMENTE a planilha que casa com PADRAO_PLANILHA
     e move direto para a raiz de BRONZE_DIR (sem subpastas)
  3. Descarta ZIP, _temporarios e todo o resto do conteúdo do ZIP

Controle de "já baixado": um glob pelo nome do arquivo. Sem manifest.
"""
from __future__ import annotations

# ==================== CONFIGURAÇÕES — POSITRON ====================
import os
from pathlib import Path

BRONZE_DIR = Path.cwd() / 'data' / 'bronze' / 'fluxo_educacao_superior'

BAIXAR_ARQUIVOS = True
SOBRESCREVER = False
CONSULTAR_CATALOGO = True
SONDAR_ANOS_NOVOS = True
LIMPAR_TEMPORARIOS = True
PERMITIR_FALLBACK_SSL_INEP = True
MAX_PAGINAS_CATALOGO = 55
# ================================================================

import hashlib
import logging
import re
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, unquote, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from tqdm import tqdm
    from urllib3.exceptions import InsecureRequestWarning
    from urllib3.util.retry import Retry
except ImportError as e:
    raise RuntimeError(
        'Dependência ausente. Instale no Python do Positron: '
        'pip install requests beautifulsoup4 tqdm truststore'
    ) from e

LOG = logging.getLogger('bronze-fluxo-superior')

BASE = 'https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/'

CATALOGOS = [
    'https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/'
    'indicadores-educacionais/indicadores-de-trajetoria-da-educacao-superior',
    'https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/'
    'indicadores-educacionais/indicadores-de-fluxo-da-educacao-superior',
]

URLS_HISTORICAS = [
    *(BASE + f'indicadores_trajetoria_es_{ano}_2024.zip' for ano in range(2020, 2014, -1)),
    BASE + 'indicadores_fluxo_es_2014-2023.zip',
    BASE + 'indicadores_fluxo_es_2013-2022.zip',
    BASE + 'Indicadores_Fluxo_ES_2012_2021.zip',
    BASE + 'Indicadores_Fluxo_ES_2011_2020.zip',
    BASE + 'Indicadores_Educacionais_Indicadores_Fluxo_Educacao_Superior_2010_2019.zip',
]

# Só estes nomes são extraídos do ZIP. Todo o resto é descartado.
PADRAO_PLANILHA = re.compile(
    r'^indicadores?_(?:trajetoria|fluxo)_'
    r'(?:educacao_superior|es)_'
    r'(20\d{2})[-_](20\d{2})\.xlsx$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Edicao:
    categoria: str       # 'trajetoria' | 'fluxo_historico'
    inicio: int
    fim: int
    url: str
    origem: str

    @property
    def periodo(self) -> str:
        return f'{self.inicio}_{self.fim}'


# --------------------------------------------------------- ident / sessão


def identificar(url: str) -> tuple[str, int, int] | None:
    nome = unquote(urlparse(url).path.rsplit('/', 1)[-1]).lower()
    if not nome.endswith('.zip') or not any(x in nome for x in ('fluxo', 'trajetoria')):
        return None
    achados = re.findall(r'(?<!\d)(20\d{2})[-_](20\d{2})(?!\d)', nome)
    if not achados:
        return None
    inicio, fim = map(int, achados[-1])
    if not (2000 <= inicio <= fim <= datetime.now().year + 1):
        return None
    categoria = 'trajetoria' if 'trajetoria' in nome else 'fluxo_historico'
    return categoria, inicio, fim


def sessao() -> requests.Session:
    try:
        import truststore  # type: ignore
        truststore.inject_into_ssl()
        LOG.info('Certificados: truststore ativado (store do SO)')
    except ImportError:
        LOG.info('Certificados: requests/certifi (truststore NÃO instalado)')
    s = requests.Session()
    s.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 Chrome/129 Safari/537.36'
        ),
        'Accept': '*/*',
    })
    retry = Retry(
        total=2, connect=2, read=1, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET', 'HEAD'],
    )
    s.mount('https://', HTTPAdapter(max_retries=retry))
    return s


def abrir(session: requests.Session, url: str, **kwargs):
    try:
        return session.get(url, **kwargs)
    except requests.exceptions.SSLError as erro:
        dominio = (urlparse(url).hostname or '').lower()
        if dominio != 'download.inep.gov.br' or not PERMITIR_FALLBACK_SSL_INEP:
            raise
        LOG.warning('Certificado inválido em %s. Fallback SSL: %s', url, erro)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            resposta = session.get(url, verify=False, **kwargs)
        if (urlparse(resposta.url).hostname or '').lower() != 'download.inep.gov.br':
            resposta.close()
            raise RuntimeError('Redirecionamento para outro host no fallback SSL')
        return resposta


# --------------------------------------------------------- catálogo


def descobrir(session: requests.Session) -> list[Edicao]:
    raizes = [c.rstrip('/') + '/' for c in CATALOGOS]
    fila: list[tuple[str, int]] = [(c, 0) for c in CATALOGOS]
    vistos: set[str] = set()
    encontrados: dict[tuple[str, int, int], Edicao] = {}
    ok_paginas = 0

    while fila and len(vistos) < MAX_PAGINAS_CATALOGO:
        pagina, nivel = fila.pop(0)
        c = pagina.rstrip('/')
        if c in vistos:
            continue
        vistos.add(c)
        try:
            with abrir(session, pagina, timeout=(12, 30)) as r:
                r.raise_for_status()
                html, pagina_final = r.text, r.url
            ok_paginas += 1
        except (requests.RequestException, RuntimeError) as erro:
            LOG.warning('Catálogo indisponível %s: %s', pagina, erro)
            continue

        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.select('a[href]'):
            url = urljoin(pagina_final, a['href'].strip()).split('#')[0]
            host = (urlparse(url).hostname or '').lower()
            ident = identificar(url)
            if host == 'download.inep.gov.br' and ident:
                cat, ini, fim = ident
                encontrados[(cat, ini, fim)] = Edicao(cat, ini, fim, url, 'catalogo')
                continue
            if host in ('www.gov.br', 'gov.br'):
                if any(url.rstrip('/').startswith(r) for r in raizes):
                    if nivel < 3 and (re.search(r'20\d{2}[-_]20\d{2}', url) or nivel == 0):
                        if url.rstrip('/') not in vistos:
                            fila.append((url, nivel + 1))

    LOG.info('Catálogo: %d páginas, %d ZIPs', ok_paginas, len(encontrados))
    return list(encontrados.values())


def periodos_pendentes_candidatos() -> list[Edicao]:
    ano_atual = datetime.now().year
    ultimos = [identificar(u)[2] for u in URLS_HISTORICAS if identificar(u)]
    ultimo = max(ultimos) if ultimos else ano_atual - 1
    saida: list[Edicao] = []
    for fim in range(ultimo + 1, ano_atual + 1):
        for inicio in range(max(2010, fim - 9), fim - 3):
            url = BASE + f'indicadores_trajetoria_es_{inicio}_{fim}.zip'
            saida.append(Edicao('trajetoria', inicio, fim, url, 'sondagem'))
    return saida


# --------------------------------------------------------- download / extração


def ja_baixado(edicao: Edicao) -> bool:
    """Só olha o nome do .xlsx no BRONZE_DIR. Sem manifest."""
    prefixo = 'indicadores_trajetoria' if edicao.categoria == 'trajetoria' else 'indicadores_fluxo'
    padrao = f'{prefixo}_*_{edicao.inicio}_{edicao.fim}.xlsx'
    return any(
        p.is_file() and p.stat().st_size > 0
        for p in BRONZE_DIR.glob(padrao)
    )


def baixar_zip(session: requests.Session, edicao: Edicao, destino: Path) -> str:
    hash_ = hashlib.sha256()
    with abrir(session, edicao.url, stream=True, timeout=(20, 120)) as r:
        if r.status_code in (403, 404, 410):
            raise FileNotFoundError(f'HTTP {r.status_code}: não publicado')
        r.raise_for_status()
        total = int(r.headers.get('content-length') or 0)
        with destino.open('wb') as f, tqdm(
            desc=edicao.periodo, total=total or None,
            unit='B', unit_scale=True, unit_divisor=1024, leave=False,
        ) as barra:
            for bloco in r.iter_content(chunk_size=256 * 1024):
                if bloco:
                    f.write(bloco)
                    hash_.update(bloco)
                    barra.update(len(bloco))
    if not zipfile.is_zipfile(destino):
        raise ValueError('Servidor retornou conteúdo que NÃO é ZIP')
    with zipfile.ZipFile(destino) as z:
        ruim = z.testzip()
        if ruim:
            raise ValueError(f'ZIP corrompido: {ruim}')
    return hash_.hexdigest()


def extrair_planilhas_para_temp(zip_path: Path, tmp_dir: Path) -> list[str]:
    extraidos: list[str] = []
    ignorados: list[str] = []
    tmp_resolved = tmp_dir.resolve()

    with zipfile.ZipFile(zip_path) as z:
        for entrada in z.infolist():
            if entrada.is_dir():
                continue
            rel = Path(entrada.filename.replace('\\', '/'))
            if not rel.parts or rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f'Caminho inseguro no ZIP: {entrada.filename}')
            nome = rel.name
            if not PADRAO_PLANILHA.match(nome):
                if nome.lower().endswith('.xlsx'):
                    ignorados.append(nome)
                continue
            alvo = (tmp_dir / nome).resolve()
            if not alvo.is_relative_to(tmp_resolved):
                raise ValueError(f'Destino fora da pasta: {entrada.filename}')
            with z.open(entrada) as src, alvo.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            extraidos.append(nome)

    if ignorados:
        LOG.warning('  .xlsx fora do padrão (IGNORADOS): %s', ignorados)
    return extraidos


def mover_para_bronze(nomes: list[str], tmp_dir: Path) -> list[str]:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    movidos: list[str] = []
    for nome in nomes:
        origem = tmp_dir / nome
        destino = BRONZE_DIR / nome
        if destino.exists() and not SOBRESCREVER:
            LOG.warning('  Já existe (mantido): %s', destino)
            movidos.append(nome)
            continue
        tmp_target = destino.with_name(destino.name + '.tmp')
        shutil.copyfile(origem, tmp_target)
        os.replace(tmp_target, destino)
        movidos.append(nome)
    return movidos


def processar(session: requests.Session, edicao: Edicao, tmp_root: Path) -> str:
    if not SOBRESCREVER and ja_baixado(edicao):
        LOG.info('✔ Já existe, pulando: %s (%s)', edicao.periodo, edicao.categoria)
        return 'pulado'

    token = f'{edicao.categoria}_{edicao.periodo}'
    tmp_zip = tmp_root / f'{token}.zip'
    tmp_extrai = tmp_root / f'{token}.extraindo'
    tmp_zip.unlink(missing_ok=True)
    if tmp_extrai.exists():
        shutil.rmtree(tmp_extrai)

    try:
        LOG.info('📥 %s | %s', token, edicao.url)
        baixar_zip(session, edicao, tmp_zip)  # hash calculado, mas não persistido
        tmp_extrai.mkdir(parents=True, exist_ok=True)
        extraidos = extrair_planilhas_para_temp(tmp_zip, tmp_extrai)
        if not extraidos:
            with zipfile.ZipFile(tmp_zip) as z:
                xlsx_no_zip = [n for n in z.namelist() if n.lower().endswith('.xlsx')]
            raise ValueError(
                'Nenhum arquivo casou com PADRAO_PLANILHA. '
                f'.xlsx no ZIP: {xlsx_no_zip or "(nenhum .xlsx)"}'
            )
        movidos = mover_para_bronze(extraidos, tmp_extrai)
        LOG.info('✅ Salvo: %s', movidos)
        return 'baixado'
    except FileNotFoundError as e:
        LOG.warning('Período não disponível (%s): %s', token, e)
        return 'inexistente'
    except (requests.RequestException, OSError, RuntimeError, ValueError,
            zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        LOG.error('Falha no período %s: %s', token, e)
        return 'falha'
    finally:
        tmp_zip.unlink(missing_ok=True)
        if tmp_extrai.exists():
            shutil.rmtree(tmp_extrai, ignore_errors=True)


# --------------------------------------------------------- main


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    LOG.info('PASTA DESTINO: %s', BRONZE_DIR)
    LOG.info('Working directory: %s', Path.cwd())
    LOG.info('Modo download=%s | sobrescrever=%s', BAIXAR_ARQUIVOS, SOBRESCREVER)

    historicos: list[Edicao] = []
    for url in URLS_HISTORICAS:
        ident = identificar(url)
        if ident:
            cat, ini, fim = ident
            historicos.append(Edicao(cat, ini, fim, url, 'historico'))

    temporarios = BRONZE_DIR / '_temporarios'
    resumo = {'baixado': 0, 'pulado': 0, 'falha': 0, 'inexistente': 0}

    try:
        with sessao() as session:
            encontrados = descobrir(session) if CONSULTAR_CATALOGO else []
            candidatos = periodos_pendentes_candidatos() if SONDAR_ANOS_NOVOS else []

            unicos: dict[tuple[str, int, int], Edicao] = {}
            for e in candidatos + historicos + encontrados:
                unicos[(e.categoria, e.inicio, e.fim)] = e
            edicoes = sorted(unicos.values(),
                             key=lambda v: (v.categoria, -v.fim, -v.inicio))
            LOG.info('%d períodos candidatos', len(edicoes))

            if not BAIXAR_ARQUIVOS:
                for e in edicoes:
                    LOG.info('[PRÉVIA] %s | %s | %s', e.periodo, e.origem, e.url)
                return

            BRONZE_DIR.mkdir(parents=True, exist_ok=True)
            temporarios.mkdir(parents=True, exist_ok=True)
            for e in edicoes:
                status = processar(session, e, temporarios)
                resumo[status] += 1
    finally:
        if LIMPAR_TEMPORARIOS and temporarios.exists():
            shutil.rmtree(temporarios)
            LOG.info('🧹 _temporarios EXCLUÍDA: %s', temporarios)
        LOG.info('RESUMO: %s', resumo)
        LOG.info('Arquivos finais em: %s', BRONZE_DIR)


if __name__ == '__main__':
    main()
