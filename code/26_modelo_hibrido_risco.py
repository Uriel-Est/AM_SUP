from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import traceback

from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy
import sklearn

from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit, xlogy

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import psutil
except ImportError:
    psutil = None


# ============================================================
# CONFIGURACOES
# ============================================================

VERSAO = "26_v1"

GOLD_DIR = None

OUT_DIR = Path(
    "./reports/modelo_hibrido_risco"
)

BASELINES_DIR = Path(
    "./reports/baselines_longitudinais"
)

BINOMIAL_DIR = Path(
    "./reports/modelo_binomial_regularizado"
)

LINEARES_DIR = Path(
    "./reports/modelos_lineares_regularizados_v2"
)

ANOS_VALIDACAO = [
    2020,
    2021,
    2022,
    2023,
]

ANO_RETROSPECTIVO = 2024

RANDOM_STATE = 42

OFFSET_FEATURE = "hist_ies_curso_idade_risco"

OFFSET_FALLBACKS = [
    "hist_curso_idade_risco",
    "hist_curso_risco",
    "hist_idade_risco",
    "hist_global_risco",
]

EPS_OFFSET = 1e-6
EPS_METRICAS = 1e-12

# ------------------------------------------------------------
# Tuning
# ------------------------------------------------------------

AMOSTRA_TUNING = 200_000

TUNING_SPECS = [
    "H3",
    "H5",
]

ALPHAS = [
    0.001,
    0.01,
    0.1,
]

GAMMAS_TUNING = [
    0.0,
    0.5,
    1.0,
]

# ------------------------------------------------------------
# Ablacao
# ------------------------------------------------------------

ABLACAO_SPECS = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
]

# Quantidade maxima de pares alpha/gamma
# selecionados para a ablação completa.

MAX_CONFIGS_ABLACAO = 3

# None utiliza todos os registros expostos.

LIMITE_TREINO_COMPLETO = None

# ------------------------------------------------------------
# Otimizacao
# ------------------------------------------------------------

MAX_ITER_TUNING = 100

MAX_ITER_FULL = 150

MAX_ITER_ADICIONAL = 150

GTOL = 1e-5

FTOL = 1e-8

# Limite adicional para identificar soluções
# aproximadamente estacionárias.

GRADIENTE_APROX_MAX = 1e-3

REFINAR_TUNING = True

REFINAR_FULL = True

LOG_ITERACOES = 10

HEARTBEAT_SEGUNDOS = 30

# ------------------------------------------------------------
# Saidas
# ------------------------------------------------------------

SALVAR_MODELOS = True

SALVAR_PREVISOES = True

TOP_N = 20

KEYS = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO",
    "ANO_ORIGEM",
]

TARGET_TADA = "TADA_FUTURA"

TARGET_D = "QT_DESISTENCIA_FUTURO"

TARGET_Q = "RISCO_DESISTENCIA_FUTURO"

TARGET_RECONSTRUIDA = "TADA_RECONSTRUIDA_FUTURA"

BENCHMARKS = {

    "baseline_23": {

        "validacao_mae":
            3.564263690214786,

        "validacao_rmse":
            7.62925703444876,

        "validacao_r2":
            0.3121081630771244,

        "retrospectivo_mae":
            1.935115045104516,

        "retrospectivo_rmse":
            4.799383562367058,

        "retrospectivo_r2":
            0.27151678313937044,

    },

    "ridge_24": {

        "validacao_mae":
            3.624371,

        "validacao_rmse":
            7.415902,

        "validacao_r2":
            0.349113,

        "retrospectivo_mae":
            2.143430,

        "retrospectivo_rmse":
            4.750548,

        "retrospectivo_r2":
            0.286267,

    },

    "binomial_25": {

        "validacao_mae":
            3.589993442751062,

        "validacao_rmse":
            7.43856352028672,

        "validacao_r2":
            0.34703700284376016,

        "retrospectivo_mae":
            1.976775294216166,

        "retrospectivo_rmse":
            4.705122550411347,

        "retrospectivo_r2":
            0.299850939322755,

        "retrospectivo_vies_desistentes":
            -96307.69086847932,

    },

}


# ============================================================
# CONFIGURACAO DO LOGGING
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for sub in [
    "modelos",
    "previsoes",
]:
    (OUT_DIR / sub).mkdir(
        parents=True,
        exist_ok=True,
    )

LOG_PATH = OUT_DIR / "26_execucao.log"

logger = logging.getLogger(
    "modelo_hibrido_risco"
)

logger.setLevel(logging.INFO)

logger.propagate = False

for handler in list(logger.handlers):

    logger.removeHandler(handler)

    try:
        handler.close()
    except Exception:
        pass

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

console_handler = logging.StreamHandler(
    sys.stdout
)

console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(
    LOG_PATH,
    mode="a",
    encoding="utf-8",
)

file_handler.setFormatter(formatter)

logger.addHandler(console_handler)

logger.addHandler(file_handler)

pl.Config.set_tbl_rows(30)

pl.Config.set_tbl_cols(-1)

pl.Config.set_tbl_width_chars(240)


# ============================================================
# MONITORAMENTO DE RECURSOS
# ============================================================

class MonitorExecucao:

    def __init__(self):

        self.lock = threading.Lock()

        self.stop_event = threading.Event()

        self.etapa = "Inicializacao"

        self.inicio_etapa = time.perf_counter()

        self.inicio_total = time.perf_counter()

        self.thread = None

        self.processo = None

        if psutil is not None:

            self.processo = psutil.Process(
                os.getpid()
            )

            self.processo.cpu_percent(
                interval=None
            )

    def recursos(self):

        if self.processo is None:

            return (
                "RAM/CPU indisponiveis "
                "(psutil nao instalado)"
            )

        try:

            memoria = (
                self.processo.memory_info().rss
                / 1024 ** 3
            )

            cpu = self.processo.cpu_percent(
                interval=None
            )

            virtual = psutil.virtual_memory()

            return (

                f"RAM Python={memoria:.2f} GiB "
                f"| RAM sistema={virtual.percent:.1f}% "
                f"| CPU processo={cpu:.1f}%"

            )

        except Exception as erro:

            return f"Recursos indisponiveis: {erro}"

    def atualizar(self, etapa):

        with self.lock:

            self.etapa = str(etapa)

            self.inicio_etapa = time.perf_counter()

    def snapshot(self):

        with self.lock:

            etapa = self.etapa

            segundos = (
                time.perf_counter()
                - self.inicio_etapa
            )

        return etapa, segundos

    def loop(self):

        while not self.stop_event.wait(
            HEARTBEAT_SEGUNDOS
        ):

            etapa, segundos = self.snapshot()

            logger.info(

                "HEARTBEAT | etapa=%s "
                "| tempo_etapa=%.1fs "
                "| tempo_sessao=%.1fmin | %s",

                etapa,

                segundos,

                (
                    time.perf_counter()
                    - self.inicio_total
                ) / 60,

                self.recursos(),

            )

    def iniciar(self):

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self.loop,
            daemon=True,
        )

        self.thread.start()

    def parar(self):

        self.stop_event.set()

        if self.thread is not None:

            self.thread.join(timeout=2)

    @contextmanager
    def operacao(self, nome):

        self.atualizar(nome)

        inicio = time.perf_counter()

        logger.info(

            "INICIO | %s | %s",

            nome,

            self.recursos(),

        )

        try:

            yield

        except BaseException:

            logger.exception(
                "FALHA | %s",
                nome,
            )

            raise

        finally:

            logger.info(

                "FIM | %s | %.1fs | %s",

                nome,

                time.perf_counter() - inicio,

                self.recursos(),

            )


MONITOR = MonitorExecucao()


# ============================================================
# FUNCOES UTILITARIAS
# ============================================================

def exigir(condicao, mensagem):

    if not bool(condicao):

        raise ValueError(mensagem)


def titulo(texto):

    logger.info("=" * 90)

    logger.info(texto)

    logger.info("=" * 90)


def unicos(lista):

    return list(
        dict.fromkeys(lista)
    )


def arr(df, coluna):

    return (

        df[coluna]

        .cast(pl.Float64)

        .to_numpy()

    )


def normalizar_json(obj):

    if isinstance(obj, dict):

        return {

            str(k): normalizar_json(v)

            for k, v in obj.items()

        }

    if isinstance(obj, (list, tuple)):

        return [

            normalizar_json(v)

            for v in obj

        ]

    if isinstance(obj, np.bool_):

        return bool(obj)

    if isinstance(obj, np.integer):

        return int(obj)

    if isinstance(obj, np.floating):

        obj = float(obj)

    if isinstance(obj, float):

        return (

            obj

            if np.isfinite(obj)

            else None

        )

    if isinstance(obj, np.ndarray):

        return normalizar_json(
            obj.tolist()
        )

    if isinstance(obj, Path):

        return str(obj)

    return obj


def json_canonico(obj):

    return json.dumps(

        normalizar_json(obj),

        sort_keys=True,

        ensure_ascii=False,

        allow_nan=False,

    )


def salvar_json(obj, caminho):

    caminho = Path(caminho)

    tmp = caminho.with_suffix(
        caminho.suffix + ".tmp"
    )

    tmp.write_text(

        json.dumps(

            normalizar_json(obj),

            indent=2,

            ensure_ascii=False,

            allow_nan=False,

        ),

        encoding="utf-8",

    )

    tmp.replace(caminho)


def salvar_csv(df, nome):

    caminho = OUT_DIR / f"{nome}.csv"

    tmp = caminho.with_suffix(
        ".csv.tmp"
    )

    df.write_csv(tmp)

    tmp.replace(caminho)


def salvar_parquet(df, caminho):

    caminho = Path(caminho)

    tmp = caminho.with_suffix(
        ".parquet.tmp"
    )

    df.write_parquet(

        tmp,

        compression="zstd",

    )

    tmp.replace(caminho)


def imprimir(df):

    logger.info(

        "\n%s",

        df.write_csv(

            separator="\t",

            float_precision=6,

        ),

    )


def hash_objeto(obj):

    return hashlib.sha256(

        json_canonico(obj).encode("utf-8")

    ).hexdigest()


def hash_arquivo(caminho):

    h = hashlib.sha256()

    with open(caminho, "rb") as f:

        for bloco in iter(

            lambda: f.read(8 * 1024 * 1024),

            b"",

        ):

            h.update(bloco)

    return h.hexdigest()


def memoria_mib(x):

    return float(

        (

            x.data.nbytes

            + x.indices.nbytes

            + x.indptr.nbytes

        ) / 1024 ** 2

    )


def media_segura(x):

    if len(x) == 0:

        return None

    return float(np.mean(x))


# ============================================================
# LOCALIZACAO UNIVERSAL DA GOLD
# ============================================================

def localizar_gold():

    arquivos = [

        "features_longitudinais.parquet",

        "alvos_longitudinais.parquet",

        "09_auditoria_vazamento.json",

    ]

    candidatos = []

    if GOLD_DIR is not None:

        candidatos.append(
            Path(GOLD_DIR)
        )

    raizes = [

        Path.cwd(),

        *[
            Path(p)
            for p in sys.path
            if p
        ],

    ]

    for raiz in raizes:

        candidatos.extend([

            raiz
            / "data"
            / "gold"
            / "fluxo_educacao_superior"
            / "modelagem_longitudinal",

            raiz
            / "gold"
            / "fluxo_educacao_superior"
            / "modelagem_longitudinal",

            raiz
            / "fluxo_educacao_superior"
            / "modelagem_longitudinal",

            raiz / "modelagem_longitudinal",

            raiz,

        ])

    for pasta in candidatos:

        if all(

            (pasta / nome).is_file()

            for nome in arquivos

        ):

            logger.info(
                "Gold localizada: %s",
                pasta.resolve(),
            )

            return pasta.resolve()

    raise FileNotFoundError(

        "Gold longitudinal nao encontrada "
        "a partir do cwd ou sys.path."

    )


# ============================================================
# CONTRATO DO SCRIPT 09
# ============================================================

def carregar_contrato(pasta):

    contrato = json.loads(

        (

            pasta
            / "09_auditoria_vazamento.json"

        ).read_text(
            encoding="utf-8"
        )

    )

    exigir(

        contrato["status"] == "APROVADO",

        "Auditoria do script 09 nao aprovada.",

    )

    exigir(

        contrato["chaves"] == KEYS,

        "Chaves incompativeis.",

    )

    exigir(

        contrato.get("teste_sintetico") is True,

        "Teste sintetico do 09 reprovado.",

    )

    testes = {

        t["corte"]: t

        for t in contrato[
            "testes_invariancia"
        ]

    }

    for corte in [2020, 2022, 2023]:

        exigir(

            corte in testes,

            f"Corte {corte} nao auditado.",

        )

        exigir(

            testes[corte]["truncamento"]

            and testes[corte]["perturbacao"],

            f"Invariancia reprovada em {corte}.",

        )

    autorizadas = set(

        contrato["features_autorizadas"]

    )

    numericas = set(
        contrato["numericas"]
    )

    categoricas = set(
        contrato["categoricas"]
    )

    proibidas = set(

        contrato[
            "alvos_e_controles_proibidos"
        ]

    ) | set(

        contrato[
            "metadados_historicos_proibidos"
        ]

    )

    exigir(

        autorizadas.isdisjoint(proibidas),

        "Features proibidas detectadas.",

    )

    exigir(

        numericas.isdisjoint(categoricas),

        "Classificacao de features invalida.",

    )

    exigir(

        autorizadas == numericas | categoricas,

        "Features sem classificacao.",

    )

    exigir(

        OFFSET_FEATURE in autorizadas,

        "Offset nao consta na allowlist.",

    )

    for coluna in OFFSET_FALLBACKS:

        exigir(

            coluna in autorizadas,

            f"Fallback nao autorizado: {coluna}",

        )

    logger.info(

        "Contrato aprovado | features=%s "
        "| numericas=%s | categoricas=%s",

        len(autorizadas),

        len(numericas),

        len(categoricas),

    )

    return contrato


# ============================================================
# ESPECIFICACOES DOS MODELOS HIBRIDOS
# ============================================================

def construir_especificacoes(contrato):

    autorizadas = contrato[
        "features_autorizadas"
    ]

    m0 = [

        "TADA",

        "QT_INGRESSANTE",

        "ANO_ORIGEM",

        "NU_ANO_INGRESSO",

        "CO_REGIAO",

        "CO_UF",

        "CO_CINE_AREA_GERAL",

        "TP_CATEGORIA_ADMINISTRATIVA",

        "TP_ORGANIZACAO_ACADEMICA",

        "TP_GRAU_ACADEMICO",

        "TP_MODALIDADE_ENSINO",

    ]

    m1 = m0 + [

        "idade_coorte",

        "exposicao",

        "zero_estrutural",

        "QT_PERMANENCIA",

        "log_QT_PERMANENCIA",

    ]

    m2 = m1 + [

        "TAP",

        "TDA",

        "TCA",

        "QT_DESISTENCIA",

        "QT_CONCLUINTE",

        "log_QT_INGRESSANTE",

        "log_QT_DESISTENCIA",

        "log_QT_CONCLUINTE",

        "risco_atual",

    ]

    dinamicas = [

        c

        for c in autorizadas

        if any(

            termo in c

            for termo in [

                "_lag1",

                "_lag2",

                "_lag3",

                "_delta1",

                "_media3",

                "_inclinacao2",

            ]

        )

    ]

    m4 = m2 + dinamicas + [

        "CO_MUNICIPIO",

        "CO_CINE_ROTULO",

    ]

    historicas = [

        c

        for c in autorizadas

        if c.startswith("hist_")

    ]

    m7 = m4 + historicas

    specs = {

        "H0": [],

        "H1": [],

        "H2": m2,

        "H3": m4,

        "H4": m7,

        "H5": [

            c

            for c in m7

            if c != OFFSET_FEATURE

        ],

    }

    for nome, colunas in specs.items():

        colunas = unicos(colunas)

        exigir(

            set(colunas).issubset(
                autorizadas
            ),

            f"{nome}: feature proibida.",

        )

        specs[nome] = colunas

        logger.info(

            "Especificacao %s | %d features",

            nome,

            len(colunas),

        )

    return specs


# ============================================================
# BASE E AUDITORIA TEMPORAL
# ============================================================

def carregar_base(pasta, contrato):

    lf_f = pl.scan_parquet(

        pasta / "features_longitudinais.parquet"

    )

    lf_y = pl.scan_parquet(

        pasta / "alvos_longitudinais.parquet"

    )

    features = contrato[
        "features_autorizadas"
    ]

    metadados = contrato[
        "metadados_historicos_proibidos"
    ]

    schema = lf_f.collect_schema().names()

    exigir(

        set(features + metadados).issubset(
            schema
        ),

        "Colunas ausentes na Gold.",

    )

    with MONITOR.operacao(

        "Auditoria dos anos dos historicos"

    ):

        auditoria_historica = (

            lf_f.select([

                (

                    pl.col(c)

                    > pl.col("ANO_ORIGEM")

                )

                .fill_null(False)

                .sum()

                .alias(c)

                for c in metadados

            ])

            .collect()

            .row(0, named=True)

        )

        exigir(

            sum(
                auditoria_historica.values()
            ) == 0,

            "Historicos com informacoes futuras.",

        )

    for lf, esperado, nome in [

        (

            lf_f,

            contrato["n_features_linhas"],

            "features",

        ),

        (

            lf_y,

            contrato["n_alvos"],

            "alvos",

        ),

    ]:

        with MONITOR.operacao(

            f"Auditoria de unicidade: {nome}"

        ):

            audit = (

                lf.select([

                    pl.len().alias("n"),

                    pl.struct(KEYS)
                    .n_unique()
                    .alias("chaves"),

                ])

                .collect()

                .row(0, named=True)

            )

            exigir(

                audit["n"] == esperado,

                f"Dimensao incompativel: {nome}",

            )

            exigir(

                audit["n"] == audit["chaves"],

                f"Duplicatas: {nome}",

            )

    base = (

        lf_f

        .select(
            unicos(KEYS + features)
        )

        .join(

            lf_y.select([

                *KEYS,

                "ANO_ALVO",

                TARGET_TADA,

                TARGET_D,

                TARGET_Q,

                TARGET_RECONSTRUIDA,

                "FLAG_TADA_ACIMA_EXPOSICAO",

            ]),

            on=KEYS,

            how="inner",

            validate="1:1",

        )

    )

    p = pl.col("QT_PERMANENCIA")

    d = pl.col(TARGET_D)

    n = pl.col("QT_INGRESSANTE")

    y = pl.col(TARGET_TADA)

    q = pl.col(TARGET_Q)

    e = pl.col("exposicao")

    with MONITOR.operacao(

        "Auditoria do join e das contagens binomiais"

    ):

        auditoria = (

            base.select([

                pl.len().alias("n"),

                (

                    pl.col("ANO_ALVO")

                    != pl.col("ANO_ORIGEM") + 1

                )

                .fill_null(True)

                .sum()

                .alias("horizonte"),

                (

                    p.is_null()

                    | d.is_null()

                    | n.is_null()

                    | ~p.is_finite()

                    | ~d.is_finite()

                    | ~n.is_finite()

                    | (p < 0)

                    | (d < 0)

                    | (d > p)

                    | (n <= 0)

                    | (p > n)

                    | (

                        (p - p.round(0)).abs()

                        > 1e-8

                    )

                    | (

                        (d - d.round(0)).abs()

                        > 1e-8

                    )

                )

                .fill_null(True)

                .sum()

                .alias("contagens_invalidas"),

                (

                    e.is_null()

                    | ~e.is_finite()

                    | (

                        (e - 100 * p / n).abs()

                        > 1e-4

                    )

                )

                .fill_null(True)

                .sum()

                .alias("exposicao_invalida"),

                (

                    y.is_null()

                    | ~y.is_finite()

                    | ~y.is_between(0, 100)

                )

                .fill_null(True)

                .sum()

                .alias("tada_invalida"),

                (

                    (p == 0)

                    & (

                        (d != 0)

                        | (y != 0)

                        | q.is_not_null()

                    )

                )

                .fill_null(True)

                .sum()

                .alias("violacao_zero"),

                (

                    (p > 0)

                    & (

                        q.is_null()

                        | ~q.is_finite()

                        | ~q.is_between(0, 1)

                        | (

                            (q - d / p).abs()

                            > 1e-5

                        )

                    )

                )

                .fill_null(False)

                .sum()

                .alias("risco_invalido"),

                (p == 0)

                .sum()

                .alias("n_zero_estrutural"),

                (p > 0)

                .sum()

                .alias("n_expostos"),

            ])

            .collect()

            .row(0, named=True)

        )

    exigir(

        auditoria["n"] == contrato["n_alvos"],

        "Perda de observacoes no join.",

    )

    for campo in [

        "horizonte",

        "contagens_invalidas",

        "exposicao_invalida",

        "tada_invalida",

        "violacao_zero",

        "risco_invalido",

    ]:

        exigir(

            auditoria[campo] == 0,

            f"Auditoria reprovada: {campo}",

        )

    auditoria["historicos"] = (
        auditoria_historica
    )

    salvar_json(

        auditoria,

        OUT_DIR / "00_auditoria_base.json",

    )

    logger.info(

        "Auditoria aprovada: %s",

        auditoria,

    )

    return base


# ============================================================
# OFFSET HISTORICO
# ============================================================

def risco_historico_raw(df):

    colunas = [

        OFFSET_FEATURE,

        *OFFSET_FALLBACKS,

    ]

    n = df.height

    risco = np.full(

        n,

        np.nan,

        dtype=np.float64,

    )

    origem = np.full(

        n,

        -1,

        dtype=np.int16,

    )

    for idx, coluna in enumerate(colunas):

        q = arr(df, coluna)

        validos = (

            np.isfinite(q)

            & (q >= 0)

            & (q <= 1)

            & ~np.isfinite(risco)

        )

        risco[validos] = q[validos]

        origem[validos] = idx

    return risco, origem


def preparar_offsets(treino, teste):

    # O fallback final e estimado exclusivamente
    # utilizando as contagens do treinamento.

    d = arr(treino, TARGET_D)

    p = arr(treino, "QT_PERMANENCIA")

    exigir(

        p.sum() > 0,

        "Treino sem exposicao positiva.",

    )

    risco_global_treino = float(

        d.sum() / p.sum()

    )

    def transformar(df):

        risco, origem = risco_historico_raw(df)

        faltantes = ~np.isfinite(risco)

        risco[faltantes] = risco_global_treino

        origem[faltantes] = len(

            [OFFSET_FEATURE, *OFFSET_FALLBACKS]

        )

        ativos = (

            arr(df, "QT_PERMANENCIA") > 0

        )

        clipping = (

            ativos

            & (

                (risco < EPS_OFFSET)

                | (risco > 1 - EPS_OFFSET)

            )

        )

        q_clip = np.clip(

            risco,

            EPS_OFFSET,

            1 - EPS_OFFSET,

        )

        offset = logit(q_clip)

        exigir(

            np.isfinite(offset).all(),

            "Offset nao finito.",

        )

        diagnostico = {

            "n": df.height,

            "n_expostos": int(ativos.sum()),

            "n_offset_principal": int(

                ((origem == 0) & ativos).sum()

            ),

            "n_fallback_historico": int(

                (

                    (origem > 0)

                    & (

                        origem

                        < len(

                            [

                                OFFSET_FEATURE,

                                *OFFSET_FALLBACKS,

                            ]

                        )

                    )

                    & ativos

                ).sum()

            ),

            "n_fallback_treino": int(

                (faltantes & ativos).sum()

            ),

            "n_clipping": int(clipping.sum()),

        }

        return risco, offset, diagnostico

    q_train, offset_train, diag_train = (

        transformar(treino)

    )

    q_test, offset_test, diag_test = (

        transformar(teste)

    )

    logger.info(

        "Offset treino: %s",

        diag_train,

    )

    logger.info(

        "Offset avaliacao: %s",

        diag_test,

    )

    diagnostico = {

        "risco_global_treino":
            risco_global_treino,

        "treino":
            diag_train,

        "avaliacao":
            diag_test,

    }

    return (

        q_train,

        q_test,

        offset_train,

        offset_test,

        diagnostico,

    )


# ============================================================
# PARTICOES TEMPORAIS
# ============================================================

def carregar_fold(

    base,

    contrato,

    features,

    ano,

    etapa,

):

    colunas = unicos([

        *KEYS,

        *features,

        OFFSET_FEATURE,

        *OFFSET_FALLBACKS,

        "ANO_ALVO",

        TARGET_TADA,

        TARGET_D,

        TARGET_Q,

        TARGET_RECONSTRUIDA,

        "QT_INGRESSANTE",

        "QT_PERMANENCIA",

        "exposicao",

        "idade_coorte",

    ])

    with MONITOR.operacao(

        f"Leitura da particao de treino: {ano}"

    ):

        treino_total = (

            base

            .filter(

                pl.col("ANO_ALVO") < ano

            )

            .select(colunas)

            .collect()

        )

    with MONITOR.operacao(

        f"Leitura da particao de avaliacao: {ano}"

    ):

        teste = (

            base

            .filter(

                pl.col("ANO_ALVO") == ano

            )

            .select(colunas)

            .collect()

            .sort(KEYS)

        )

    esperado = next(

        p

        for p in contrato["particoes"]

        if p["ano_alvo"] == ano

    )

    exigir(

        treino_total.height
        == esperado["n_treino"],

        "Treino incompativel com contrato.",

    )

    exigir(

        teste.height
        == esperado["n_avaliacao"],

        "Avaliacao incompativel com contrato.",

    )

    exigir(

        treino_total["ANO_ALVO"].max()

        < teste["ANO_ALVO"].min(),

        "Sobreposicao temporal.",

    )

    with MONITOR.operacao(

        "Selecao de observacoes expostas"

    ):

        treino = (

            treino_total

            .filter(

                pl.col("QT_PERMANENCIA") > 0

            )

            .sort(KEYS)

        )

        del treino_total

        limite = (

            AMOSTRA_TUNING

            if etapa == "tuning"

            else LIMITE_TREINO_COMPLETO

        )

        if (

            limite is not None

            and treino.height > limite

        ):

            treino = treino.sample(

                n=limite,

                seed=RANDOM_STATE,

                shuffle=True,

            )

    gc.collect()

    logger.info(

        "Fold %s | etapa=%s "
        "| treino=%s | avaliacao=%s",

        ano,

        etapa,

        f"{treino.height:,}",

        f"{teste.height:,}",

    )

    return treino, teste


# ============================================================
# PREPROCESSAMENTO
# ============================================================

def construir_preprocessador(

    features,

    contrato,

):

    nums = [

        c

        for c in features

        if c in contrato["numericas"]

    ]

    cats = [

        c

        for c in features

        if c in contrato["categoricas"]

    ]

    exigir(

        set(nums + cats) == set(features),

        "Feature sem classificacao.",

    )

    transformadores = []

    if nums:

        transformadores.append((

            "numericas",

            Pipeline([

                (

                    "imputer",

                    SimpleImputer(

                        strategy="median",

                        add_indicator=True,

                        keep_empty_features=True,

                    ),

                ),

                (

                    "scaler",

                    StandardScaler(

                        with_mean=True,

                    ),

                ),

            ]),

            nums,

        ))

    if cats:

        transformadores.append((

            "categoricas",

            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=True,

                dtype=np.float32,

            ),

            cats,

        ))

    pre = ColumnTransformer(

        transformadores,

        remainder="drop",

        sparse_threshold=1.0,

    )

    return pre, nums, cats


def converter_pandas(

    df,

    features,

    nums,

    cats,

):

    dados = {}

    nums = set(nums)

    cats = set(cats)

    for c in features:

        if c in nums:

            x = (

                df[c]

                .cast(

                    pl.Float32,

                    strict=False,

                )

                .to_numpy()

                .copy()

            )

            x[~np.isfinite(x)] = np.nan

            dados[c] = x

        elif c in cats:

            dados[c] = (

                df[c]

                .cast(pl.String)

                .fill_null("__AUSENTE__")

                .to_numpy()

            )

    return pd.DataFrame(

        dados,

        columns=features,

        copy=False,

    )


def csr_float32(x):

    if sparse.issparse(x):

        x = x.tocsr()

    else:

        x = sparse.csr_matrix(x)

    x = x.astype(

        np.float32,

        copy=False,

    )

    x.eliminate_zeros()

    x.sum_duplicates()

    x.sort_indices()

    return x


def preparar_matrizes(

    treino,

    teste,

    features,

    contrato,

):

    # H0 e H1 nao possuem covariaveis.
    # H1 estima somente o intercepto.

    if len(features) == 0:

        x = sparse.csr_matrix(

            (treino.height, 0),

            dtype=np.float32,

        )

        xv = sparse.csr_matrix(

            (teste.height, 0),

            dtype=np.float32,

        )

        return x, xv, None

    pre, nums, cats = (

        construir_preprocessador(

            features,

            contrato,

        )

    )

    with MONITOR.operacao(

        "Conversao do treino para pandas"

    ):

        x_pd = converter_pandas(

            treino,

            features,

            nums,

            cats,

        )

    with MONITOR.operacao(

        "FIT imputacao, escala e one-hot no treino"

    ):

        x = csr_float32(

            pre.fit_transform(x_pd)

        )

    del x_pd

    gc.collect()

    with MONITOR.operacao(

        "Conversao da avaliacao para pandas"

    ):

        x_pd = converter_pandas(

            teste,

            features,

            nums,

            cats,

        )

    with MONITOR.operacao(

        "TRANSFORM da avaliacao"

    ):

        xv = csr_float32(

            pre.transform(x_pd)

        )

    del x_pd

    gc.collect()

    exigir(

        x.shape[1] == xv.shape[1],

        "Dimensoes incompativeis.",

    )

    exigir(

        np.isfinite(x.data).all(),

        "Matriz de treinamento invalida.",

    )

    exigir(

        np.isfinite(xv.data).all(),

        "Matriz de avaliacao invalida.",

    )

    logger.info(

        "Matrizes prontas | treino=%s "
        "| avaliacao=%s | %.1f MiB",

        x.shape,

        xv.shape,

        memoria_mib(x) + memoria_mib(xv),

    )

    return x, xv, pre


# ============================================================
# MODELO HIBRIDO BINOMIAL
# ============================================================

class HibridoBinomial:

    def __init__(

        self,

        alpha,

        gamma,

        max_iter,

        refinar=True,

    ):

        self.alpha = float(alpha)

        self.gamma = float(gamma)

        self.max_iter = int(max_iter)

        self.refinar = bool(refinar)

    def fit(

        self,

        x,

        d,

        p,

        offset,

        inicial=None,

    ):

        inicio = time.perf_counter()

        d = np.asarray(

            d,

            dtype=np.float64,

        )

        p = np.asarray(

            p,

            dtype=np.float64,

        )

        offset = np.asarray(

            offset,

            dtype=np.float64,

        )

        exigir(

            len(d)

            == len(p)

            == len(offset)

            == x.shape[0],

            "Dimensoes incompativeis.",

        )

        exigir(

            (p > 0).all(),

            "Treino com exposicao zero.",

        )

        exigir(

            ((d >= 0) & (d <= p)).all(),

            "Contagens invalidas.",

        )

        exigir(

            np.isfinite(offset).all(),

            "Offset invalido.",

        )

        q_obs = d / p

        # gamma = 1:
        # verossimilhanca binomial convencional.
        #
        # gamma = 0:
        # todas as trajetorias com peso igual.
        #
        # gamma = 0.5:
        # ponderacao intermediaria.

        pesos = np.power(

            p,

            self.gamma,

        )

        pesos /= pesos.sum()

        k = x.shape[1]

        theta0 = np.zeros(

            k + 1,

            dtype=np.float64,

        )

        if inicial is not None:

            exigir(

                len(inicial) == k + 1,

                "Warm start incompativel.",

            )

            theta0 = np.asarray(

                inicial,

                dtype=np.float64,

            ).copy()

        alpha = self.alpha

        contador = 0

        ultimo = {

            "loss": None,

            "grad": None,

        }

        def objetivo(theta):

            beta = theta[1:]

            eta = (

                offset

                + theta[0]

                + x @ beta

            )

            prob = expit(eta)

            perda = np.dot(

                pesos,

                np.logaddexp(0, eta)

                - q_obs * eta,

            )

            perda += (

                0.5

                * alpha

                * np.dot(beta, beta)

            )

            residuo = (

                pesos

                * (prob - q_obs)

            )

            grad = np.empty_like(theta)

            grad[0] = residuo.sum()

            grad[1:] = (

                x.T @ residuo

                + alpha * beta

            )

            ultimo["loss"] = float(perda)

            ultimo["grad"] = float(

                np.max(np.abs(grad))

            )

            return float(perda), grad

        def callback(theta):

            nonlocal contador

            contador += 1

            if (

                contador == 1

                or contador % LOG_ITERACOES == 0

            ):

                logger.info(

                    "ITER | alpha=%g | gamma=%g "
                    "| iter=%d | loss=%.8f "
                    "| grad_avaliado=%.3e "
                    "| tempo=%.1fs",

                    self.alpha,

                    self.gamma,

                    contador,

                    ultimo["loss"],

                    ultimo["grad"],

                    time.perf_counter() - inicio,

                )

        def otimizar(theta, maxiter):

            return minimize(

                objetivo,

                theta,

                method="L-BFGS-B",

                jac=True,

                callback=callback,

                options={

                    "maxiter": int(maxiter),

                    "gtol": GTOL,

                    "ftol": FTOL,

                    "maxcor": 10,

                    "maxls": 30,

                },

            )

        logger.info(

            "Otimizacao iniciada "
            "| alpha=%g | gamma=%g "
            "| max_iter=%d",

            self.alpha,

            self.gamma,

            self.max_iter,

        )

        resultado = otimizar(

            theta0,

            self.max_iter,

        )

        n_iter = int(resultado.nit)

        n_fun = int(resultado.nfev)

        refinado = False

        gradiente = float(

            np.max(

                np.abs(resultado.jac)

            )

        )

        pode_refinar = (

            np.isfinite(resultado.x).all()

            and np.isfinite(resultado.fun)

            and resultado.status in [0, 1]

        )

        precisa_refinar = (

            not resultado.success

            or gradiente > GRADIENTE_APROX_MAX

        )

        if (

            self.refinar

            and pode_refinar

            and precisa_refinar

        ):

            refinado = True

            logger.info(

                "REFINAMENTO | iter_adicional=%d "
                "| gradiente_anterior=%.3e",

                MAX_ITER_ADICIONAL,

                gradiente,

            )

            anterior = resultado

            novo = otimizar(

                anterior.x,

                MAX_ITER_ADICIONAL,

            )

            n_iter += int(novo.nit)

            n_fun += int(novo.nfev)

            # Nao substitui uma solucao finita
            # por uma piora numerica.

            if (

                np.isfinite(novo.x).all()

                and np.isfinite(novo.fun)

                and novo.fun <= anterior.fun + 1e-10

            ):

                resultado = novo

        self.theta_ = np.asarray(

            resultado.x,

            dtype=np.float64,

        )

        self.intercept_ = float(

            self.theta_[0]

        )

        self.coef_ = self.theta_[1:]

        self.objetivo_ = float(

            resultado.fun

        )

        self.gradiente_max_ = float(

            np.max(

                np.abs(resultado.jac)

            )

        )

        self.convergiu_ = bool(

            resultado.success

        )

        self.estacionario_ = bool(

            self.gradiente_max_

            <= GRADIENTE_APROX_MAX

        )

        self.status_ = int(

            resultado.status

        )

        self.mensagem_ = str(

            resultado.message

        )

        self.n_iter_ = int(n_iter)

        self.n_fun_ = int(n_fun)

        self.refinado_ = bool(refinado)

        self.tempo_ = float(

            time.perf_counter() - inicio

        )

        self.elegivel_ = bool(

            np.isfinite(self.theta_).all()

            and np.isfinite(self.objetivo_)

            and self.status_ in [0, 1]

            and (

                self.convergiu_

                or self.estacionario_

            )

        )

        exigir(

            np.isfinite(self.theta_).all(),

            "Coeficientes nao finitos.",

        )

        logger.info(

            "MODELO CONCLUIDO "
            "| alpha=%g | gamma=%g "
            "| iter=%d | convergiu=%s "
            "| estacionario=%s "
            "| elegivel=%s | grad=%.3e "
            "| tempo=%.1fs",

            self.alpha,

            self.gamma,

            self.n_iter_,

            self.convergiu_,

            self.estacionario_,

            self.elegivel_,

            self.gradiente_max_,

            self.tempo_,

        )

        return self

    def predict_risco(self, x, offset):

        eta = (

            np.asarray(

                offset,

                dtype=np.float64,

            )

            + self.intercept_

            + x @ self.coef_

        )

        return expit(eta)


# ============================================================
# METRICAS
# ============================================================

def calcular_metricas(df, risco):

    risco = np.asarray(

        risco,

        dtype=np.float64,

    )

    p = arr(df, "QT_PERMANENCIA")

    n = arr(df, "QT_INGRESSANTE")

    d = arr(df, TARGET_D)

    y = arr(df, TARGET_TADA)

    y_rec = arr(

        df,

        TARGET_RECONSTRUIDA,

    )

    e = arr(df, "exposicao")

    ativos = p > 0

    exigir(

        len(risco) == len(y),

        "Previsoes desalinhadas.",

    )

    exigir(

        np.isfinite(risco).all(),

        "Risco nao finito.",

    )

    exigir(

        ((risco >= 0) & (risco <= 1)).all(),

        "Risco fora de [0, 1].",

    )

    tada = e * risco

    tada[~ativos] = 0

    d_pred = p * risco

    d_pred[~ativos] = 0

    exigir(

        (tada <= e + 1e-10).all(),

        "TADA prevista acima da exposicao.",

    )

    erro = tada - y

    abs_erro = np.abs(erro)

    erro2 = erro ** 2

    sst = np.sum(

        (y - y.mean()) ** 2

    )

    sst_exp = np.sum(

        (

            y[ativos]

            - y[ativos].mean()

        ) ** 2

    )

    positivos = y > 0

    pa = p[ativos]

    da = d[ativos]

    q_obs = da / pa

    q_pred = risco[ativos]

    q_clip = np.clip(

        q_pred,

        EPS_METRICAS,

        1 - EPS_METRICAS,

    )

    loglik = (

        xlogy(da, q_clip)

        + xlogy(

            pa - da,

            1 - q_clip,

        )

    )

    loglik_sat = (

        xlogy(da, q_obs)

        + xlogy(

            pa - da,

            1 - q_obs,

        )

    )

    deviance = float(

        2 * np.sum(

            loglik_sat - loglik

        )

    )

    brier = np.sum(

        da * (1 - q_pred) ** 2

        + (pa - da) * q_pred ** 2

    ) / pa.sum()

    total_obs = float(d.sum())

    total_pred = float(d_pred.sum())

    vies_total = (

        total_pred - total_obs

    )

    metricas = {

        "n":
            int(len(y)),

        "n_expostos":
            int(ativos.sum()),

        "n_zero_estrutural":
            int((~ativos).sum()),

        "mae":
            float(abs_erro.mean()),

        "rmse":
            float(np.sqrt(erro2.mean())),

        "r2":
            float(1 - erro2.sum() / sst)
            if sst > 0 else None,

        "vies":
            float(erro.mean()),

        "media_observada":
            float(y.mean()),

        "media_prevista":
            float(tada.mean()),

        "mae_expostos":
            media_segura(abs_erro[ativos]),

        "rmse_expostos":
            float(np.sqrt(erro2[ativos].mean())),

        "r2_expostos":

            float(

                1

                - erro2[ativos].sum()

                / sst_exp

            )

            if sst_exp > 0 else None,

        "mae_positivos":
            media_segura(abs_erro[positivos]),

        "mae_zero_estrutural":
            media_segura(abs_erro[~ativos]),

        "mae_ponderado_ingressantes":

            float(

                np.average(

                    abs_erro,

                    weights=n,

                )

            ),

        "mae_tada_reconstruida":

            float(

                np.abs(

                    tada - y_rec

                ).mean()

            ),

        "pct_pred_acima_exposicao":

            float(

                100 * (

                    tada > e + 1e-10

                ).mean()

            ),

        "pct_pred_zero":

            float(

                100 * (tada == 0).mean()

            ),

        "risco_mae":

            float(

                np.abs(

                    q_pred - q_obs

                ).mean()

            ),

        "risco_rmse":

            float(

                np.sqrt(

                    np.mean(

                        (q_pred - q_obs) ** 2

                    )

                )

            ),

        "risco_vies":

            float(

                (q_pred - q_obs).mean()

            ),

        "risco_real_ponderado":

            float(

                da.sum() / pa.sum()

            ),

        "risco_pred_ponderado":

            float(

                np.average(

                    q_pred,

                    weights=pa,

                )

            ),

        "logloss_binomial":

            float(

                -loglik.sum() / pa.sum()

            ),

        "brier_individual":

            float(brier),

        "deviance_binomial":

            deviance,

        "deviance_por_coorte":

            float(

                deviance / ativos.sum()

            ),

        "desistentes_observados":

            total_obs,

        "desistentes_previstos":

            total_pred,

        "vies_total_desistentes":

            vies_total,

        "vies_pct_desistentes":

            float(

                100 * vies_total / total_obs

            )

            if total_obs > 0 else None,

        "mae_contagem":

            float(

                np.abs(d_pred - d).mean()

            ),

    }

    return metricas, tada, d_pred


# ============================================================
# CHECKPOINT
# ============================================================

class Checkpoint:

    def __init__(self, caminho, assinatura):

        self.con = sqlite3.connect(

            caminho,

            timeout=30,

        )

        self.con.execute(

            "PRAGMA journal_mode=WAL"

        )

        self.con.execute(

            "PRAGMA synchronous=FULL"

        )

        self.con.execute("""

            CREATE TABLE IF NOT EXISTS metadata (

                chave TEXT PRIMARY KEY,

                valor TEXT NOT NULL

            )

        """)

        self.con.execute("""

            CREATE TABLE IF NOT EXISTS resultados (

                id TEXT PRIMARY KEY,

                etapa TEXT NOT NULL,

                resultado TEXT NOT NULL

            )

        """)

        anterior = self.con.execute("""

            SELECT valor

            FROM metadata

            WHERE chave = 'assinatura'

        """).fetchone()

        if anterior is None:

            with self.con:

                self.con.execute(

                    """

                    INSERT INTO metadata

                    VALUES ('assinatura', ?)

                    """,

                    (assinatura,),

                )

        else:

            exigir(

                anterior[0] == assinatura,

                "Checkpoint incompativel. "
                "Nao sobrescreva resultados anteriores.",

            )

    def existe(self, chave):

        return self.con.execute(

            """

            SELECT 1

            FROM resultados

            WHERE id = ?

            """,

            (chave,),

        ).fetchone() is not None

    def salvar(self, chave, etapa, resultado):

        with self.con:

            self.con.execute(

                """

                INSERT OR REPLACE

                INTO resultados

                VALUES (?, ?, ?)

                """,

                (

                    chave,

                    etapa,

                    json_canonico(resultado),

                ),

            )

    def carregar(self, etapa):

        linhas = self.con.execute(

            """

            SELECT resultado

            FROM resultados

            WHERE etapa = ?

            ORDER BY rowid

            """,

            (etapa,),

        ).fetchall()

        return [

            json.loads(linha[0])

            for linha in linhas

        ]

    def contar(self, etapa):

        return self.con.execute(

            """

            SELECT COUNT(*)

            FROM resultados

            WHERE etapa = ?

            """,

            (etapa,),

        ).fetchone()[0]

    def fechar(self):

        self.con.close()


# ============================================================
# ASSINATURA DA EXECUCAO
# ============================================================

def construir_assinatura(pasta, specs):

    arquivos = {}

    for nome in [

        "features_longitudinais.parquet",

        "alvos_longitudinais.parquet",

        "09_auditoria_vazamento.json",

    ]:

        with MONITOR.operacao(

            f"Hash: {nome}"

        ):

            arquivos[nome] = hash_arquivo(

                pasta / nome

            )

    configuracao = {

        "versao": VERSAO,

        "arquivos": arquivos,

        "offset": OFFSET_FEATURE,

        "fallbacks": OFFSET_FALLBACKS,

        "eps_offset": EPS_OFFSET,

        "eps_metricas": EPS_METRICAS,

        "anos_validacao": ANOS_VALIDACAO,

        "ano_retrospectivo": ANO_RETROSPECTIVO,

        "amostra_tuning": AMOSTRA_TUNING,

        "limite_treino_completo":
            LIMITE_TREINO_COMPLETO,

        "tuning_specs": TUNING_SPECS,

        "ablacao_specs": ABLACAO_SPECS,

        "alphas": ALPHAS,

        "gammas_tuning": GAMMAS_TUNING,

        "max_configs_ablacao":
            MAX_CONFIGS_ABLACAO,

        "max_iter_tuning": MAX_ITER_TUNING,

        "max_iter_full": MAX_ITER_FULL,

        "max_iter_adicional":
            MAX_ITER_ADICIONAL,

        "gtol": GTOL,

        "ftol": FTOL,

        "gradiente_aprox_max":
            GRADIENTE_APROX_MAX,

        "refinar_tuning": REFINAR_TUNING,

        "refinar_full": REFINAR_FULL,

        "seed": RANDOM_STATE,

        "salvar_modelos": SALVAR_MODELOS,

        "salvar_previsoes": SALVAR_PREVISOES,

        "especificacoes": specs,

        "bibliotecas": {

            "numpy": np.__version__,

            "pandas": pd.__version__,

            "polars": pl.__version__,

            "scipy": scipy.__version__,

            "sklearn": sklearn.__version__,

        },

    }

    return (

        hash_objeto(configuracao),

        configuracao,

    )


# ============================================================
# IDENTIFICACAO DOS AJUSTES
# ============================================================

def chave_experimento(

    etapa,

    ano,

    spec,

    alpha,

    gamma,

):

    return hash_objeto([

        etapa,

        int(ano),

        str(spec),

        float(alpha),

        float(gamma),

    ])


def mostrar_progresso(

    checkpoint,

    etapa,

    total,

    inicial,

    inicio,

):

    atual = checkpoint.contar(etapa)

    novos = atual - inicial

    restantes = total - atual

    if novos > 0:

        eta = (

            (

                time.perf_counter() - inicio

            )

            / novos

            * restantes

            / 60

        )

        eta_texto = f"{eta:.1f} min"

    else:

        eta_texto = "indisponivel"

    logger.info(

        "PROGRESSO | %s | %d/%d | %.1f%% "
        "| ETA aproximada=%s",

        etapa,

        atual,

        total,

        100 * atual / total,

        eta_texto,

    )


# ============================================================
# SALVAMENTO DOS FINALISTAS
# ============================================================

def salvar_finalista(

    modelo_id,

    modelo,

    pre,

    features,

    teste,

    q_hist,

    risco,

    tada,

    d_pred,

    ano,

    alpha,

    gamma,

    spec,

    diagnostico_offset,

):

    if SALVAR_MODELOS:

        pacote = {

            "tipo": "hibrido_binomial_offset",

            "especificacao": spec,

            "features": features,

            "preprocessador": pre,

            "offset_feature": OFFSET_FEATURE,

            "offset_fallbacks": OFFSET_FALLBACKS,

            "eps_offset": EPS_OFFSET,

            "risco_global_treino":

                diagnostico_offset[

                    "risco_global_treino"

                ],

            "coeficientes":

                modelo.coef_

                if modelo is not None

                else np.zeros(0),

            "intercepto":

                modelo.intercept_

                if modelo is not None

                else 0.0,

            "alpha": alpha,

            "gamma": gamma,

            "ano_maximo_treino": ano - 1,

        }

        caminho = (

            OUT_DIR

            / "modelos"

            / f"{modelo_id}.joblib"

        )

        tmp = caminho.with_suffix(

            ".joblib.tmp"

        )

        with MONITOR.operacao(

            f"Salvando modelo {modelo_id}"

        ):

            joblib.dump(

                pacote,

                tmp,

                compress=3,

            )

            tmp.replace(caminho)

    if SALVAR_PREVISOES:

        ativos = (

            arr(teste, "QT_PERMANENCIA") > 0

        )

        e = arr(teste, "exposicao")

        risco_coluna = pl.Series(

            "risco_previsto",

            risco,

            dtype=pl.Float64,

        ).set(

            pl.Series(~ativos),

            None,

        )

        risco_hist_coluna = pl.Series(

            "risco_historico",

            q_hist,

            dtype=pl.Float64,

        ).set(

            pl.Series(~ativos),

            None,

        )

        saida = teste.select([

            *KEYS,

            "ANO_ALVO",

            "idade_coorte",

            TARGET_TADA,

            TARGET_D,

            TARGET_Q,

            TARGET_RECONSTRUIDA,

            "QT_INGRESSANTE",

            "QT_PERMANENCIA",

            "exposicao",

        ])

        saida = saida.with_columns([

            risco_coluna,

            risco_hist_coluna,

            pl.Series(

                "tada_prevista",

                tada,

                dtype=pl.Float64,

            ),

            pl.Series(

                "tada_historica",

                e * q_hist,

                dtype=pl.Float64,

            ),

            pl.Series(

                "desistentes_previstos",

                d_pred,

                dtype=pl.Float64,

            ),

            pl.Series(

                "correcao_tada",

                tada - e * q_hist,

                dtype=pl.Float64,

            ),

        ])

        caminho = (

            OUT_DIR

            / "previsoes"

            / f"{modelo_id}.parquet"

        )

        with MONITOR.operacao(

            f"Salvando previsoes {modelo_id}"

        ):

            salvar_parquet(

                saida,

                caminho,

            )


# ============================================================
# EXECUCAO DAS ETAPAS
# ============================================================

def executar_etapa(

    base,

    contrato,

    specs,

    checkpoint,

    etapa,

    grupos,

):

    total = sum(

        len(configs)

        for _, _, configs in grupos

    )

    inicial = checkpoint.contar(etapa)

    inicio = time.perf_counter()

    mostrar_progresso(

        checkpoint,

        etapa,

        total,

        inicial,

        inicio,

    )

    for ano, spec, configs in grupos:

        pendentes = [

            (alpha, gamma)

            for alpha, gamma in configs

            if not checkpoint.existe(

                chave_experimento(

                    etapa,

                    ano,

                    spec,

                    alpha,

                    gamma,

                )

            )

        ]

        if not pendentes:

            logger.info(

                "CHECKPOINT | reaproveitando "
                "| %s | %s | %s",

                etapa,

                ano,

                spec,

            )

            continue

        titulo(

            f"{etapa.upper()} | {spec} | {ano}"

        )

        features = specs[spec]

        treino, teste = carregar_fold(

            base,

            contrato,

            features,

            ano,

            etapa,

        )

        with MONITOR.operacao(

            "Construcao do offset historico"

        ):

            (

                q_train,

                q_test,

                offset_train,

                offset_test,

                diagnostico_offset,

            ) = preparar_offsets(

                treino,

                teste,

            )

        x, xv, pre = preparar_matrizes(

            treino,

            teste,

            features,

            contrato,

        )

        d_train = arr(

            treino,

            TARGET_D,

        )

        p_train = arr(

            treino,

            "QT_PERMANENCIA",

        )

        n_treino = treino.height

        del treino, q_train

        gc.collect()

        max_iter = (

            MAX_ITER_TUNING

            if etapa == "tuning"

            else MAX_ITER_FULL

        )

        refinar = (

            REFINAR_TUNING

            if etapa == "tuning"

            else REFINAR_FULL

        )

        # Warm start apenas entre ajustes com
        # o mesmo gamma e a mesma matriz.

        warm_starts = {}

        for alpha, gamma in sorted(

            pendentes,

            key=lambda v: (v[1], -v[0]),

        ):

            chave = chave_experimento(

                etapa,

                ano,

                spec,

                alpha,

                gamma,

            )

            modelo_id = (

                f"{spec}_hibrido_"

                f"{chave[:12]}"

            )

            modelo = None

            if spec == "H0":

                # H0 recupera o risco historico
                # sem clipping logit.

                risco = q_test.copy()

                metadados_modelo = {

                    "convergiu": True,

                    "estacionario": True,

                    "elegivel": True,

                    "refinado": False,

                    "n_iter": 0,

                    "n_fun": 0,

                    "gradiente_max": 0.0,

                    "tempo_treino": 0.0,

                    "status_otimizador": 0,

                    "mensagem_otimizador":
                        "Baseline sem ajuste",

                    "objetivo": None,

                    "intercepto": 0.0,

                }

            else:

                with MONITOR.operacao(

                    f"Otimizacao {spec} {ano} "
                    f"alpha={alpha:g} gamma={gamma:g}"

                ):

                    modelo = HibridoBinomial(

                        alpha=alpha,

                        gamma=gamma,

                        max_iter=max_iter,

                        refinar=refinar,

                    ).fit(

                        x,

                        d_train,

                        p_train,

                        offset_train,

                        inicial=warm_starts.get(

                            gamma

                        ),

                    )

                if modelo.elegivel_:

                    warm_starts[gamma] = (

                        modelo.theta_.copy()

                    )

                risco = modelo.predict_risco(

                    xv,

                    offset_test,

                )

                metadados_modelo = {

                    "convergiu":

                        modelo.convergiu_,

                    "estacionario":

                        modelo.estacionario_,

                    "elegivel":

                        modelo.elegivel_,

                    "refinado":

                        modelo.refinado_,

                    "n_iter":

                        modelo.n_iter_,

                    "n_fun":

                        modelo.n_fun_,

                    "gradiente_max":

                        modelo.gradiente_max_,

                    "tempo_treino":

                        modelo.tempo_,

                    "status_otimizador":

                        modelo.status_,

                    "mensagem_otimizador":

                        modelo.mensagem_,

                    "objetivo":

                        modelo.objetivo_,

                    "intercepto":

                        modelo.intercept_,

                }

            with MONITOR.operacao(

                f"Avaliacao {modelo_id}"

            ):

                metricas, tada, d_pred = (

                    calcular_metricas(

                        teste,

                        risco,

                    )

                )

                metricas_hist, _, _ = (

                    calcular_metricas(

                        teste,

                        q_test,

                    )

                )

            resultado = {

                "etapa": etapa,

                "ano_alvo": int(ano),

                "especificacao": spec,

                "modelo_id": modelo_id,

                "alpha": float(alpha),

                "gamma": float(gamma),

                "n_treino": int(n_treino),

                "n_features_originais":

                    len(features),

                "n_features_transformadas":

                    int(x.shape[1]),

                "memoria_mib":

                    memoria_mib(x),

                "offset_n_fallback_treino":

                    diagnostico_offset["avaliacao"][

                        "n_fallback_treino"

                    ],

                "offset_n_fallback_historico":

                    diagnostico_offset["avaliacao"][

                        "n_fallback_historico"

                    ],

                "offset_n_clipping":

                    diagnostico_offset["avaliacao"][

                        "n_clipping"

                    ],

                "mae_historico":

                    metricas_hist["mae"],

                "rmse_historico":

                    metricas_hist["rmse"],

                "r2_historico":

                    metricas_hist["r2"],

                "delta_mae_historico":

                    metricas["mae"]

                    - metricas_hist["mae"],

                "delta_rmse_historico":

                    metricas["rmse"]

                    - metricas_hist["rmse"],

                "delta_r2_historico":

                    metricas["r2"]

                    - metricas_hist["r2"],

                **metadados_modelo,

                **metricas,

            }

            logger.info(

                "RESULTADO | %s | %s "
                "| alpha=%g | gamma=%g "
                "| MAE=%.6f | RMSE=%.6f "
                "| R2=%.6f | vies_desistentes=%.2f%%",

                spec,

                ano,

                alpha,

                gamma,

                resultado["mae"],

                resultado["rmse"],

                resultado["r2"],

                resultado["vies_pct_desistentes"],

            )

            if etapa == "retrospectivo":

                salvar_finalista(

                    modelo_id,

                    modelo,

                    pre,

                    features,

                    teste,

                    q_test,

                    risco,

                    tada,

                    d_pred,

                    ano,

                    alpha,

                    gamma,

                    spec,

                    diagnostico_offset,

                )

            checkpoint.salvar(

                chave,

                etapa,

                resultado,

            )

            salvar_csv(

                pl.DataFrame(

                    checkpoint.carregar(etapa),

                    infer_schema_length=None,

                ),

                f"{etapa}_parcial",

            )

            mostrar_progresso(

                checkpoint,

                etapa,

                total,

                inicial,

                inicio,

            )

            del modelo, risco, tada, d_pred

            gc.collect()

        del x, xv, pre, teste

        del d_train, p_train

        del offset_train, offset_test

        del q_test, warm_starts

        gc.collect()

    return pl.DataFrame(

        checkpoint.carregar(etapa),

        infer_schema_length=None,

    )


# ============================================================
# RANKING TEMPORAL
# ============================================================

def ranking_temporal(resultados):

    ranking = (

        resultados

        .group_by([

            "especificacao",

            "alpha",

            "gamma",

        ])

        .agg([

            pl.col("ano_alvo")

            .n_unique()

            .alias("n_validacoes"),

            pl.col("mae")

            .mean()

            .alias("mae_medio"),

            pl.col("mae")

            .std()

            .alias("mae_dp"),

            pl.col("rmse")

            .mean()

            .alias("rmse_medio"),

            pl.col("r2")

            .mean()

            .alias("r2_medio"),

            pl.col("vies")

            .mean()

            .alias("vies_medio"),

            pl.col("mae_expostos")

            .mean()

            .alias("mae_expostos_medio"),

            pl.col("mae_ponderado_ingressantes")

            .mean()

            .alias("mae_ponderado_medio"),

            pl.col("logloss_binomial")

            .mean()

            .alias("logloss_medio"),

            pl.col("brier_individual")

            .mean()

            .alias("brier_medio"),

            pl.col("vies_pct_desistentes")

            .mean()

            .alias("vies_pct_desistentes_medio"),

            pl.col("vies_pct_desistentes")

            .abs()

            .mean()

            .alias("erro_pct_agregado_medio"),

            pl.col("delta_mae_historico")

            .mean()

            .alias("delta_mae_historico_medio"),

            pl.col("delta_rmse_historico")

            .mean()

            .alias("delta_rmse_historico_medio"),

            pl.col("tempo_treino")

            .mean()

            .alias("tempo_medio"),

            pl.col("convergiu")

            .all()

            .alias("convergiu_todos"),

            pl.col("estacionario")

            .all()

            .alias("estacionario_todos"),

            pl.col("elegivel")

            .all()

            .alias("elegivel_todos"),

            pl.col("refinado")

            .sum()

            .alias("n_refinados"),

            pl.col("gradiente_max")

            .max()

            .alias("gradiente_max"),

            pl.col("n_features_transformadas")

            .max()

            .alias("n_features_transformadas"),

        ])

        .filter(

            pl.col("n_validacoes")

            == len(ANOS_VALIDACAO)

        )

        .with_columns([

            (

                pl.col("mae_medio")

                - BENCHMARKS["baseline_23"][

                    "validacao_mae"

                ]

            ).alias("delta_mae_vs_23"),

            (

                pl.col("rmse_medio")

                - BENCHMARKS["binomial_25"][

                    "validacao_rmse"

                ]

            ).alias("delta_rmse_vs_25"),

            (

                pl.col("r2_medio")

                - BENCHMARKS["binomial_25"][

                    "validacao_r2"

                ]

            ).alias("delta_r2_vs_25"),

        ])

        .sort([

            "mae_medio",

            "rmse_medio",

            "especificacao",

            "alpha",

            "gamma",

        ])

    )

    return ranking


# ============================================================
# SELECAO DE HIPERPARAMETROS
# ============================================================

def selecionar_configs(ranking):

    candidatos = ranking.filter(

        pl.col("elegivel_todos")

    )

    exigir(

        candidatos.height > 0,

        "Nenhuma configuracao elegivel.",

    )

    # Diferentemente do 25, nao restringe
    # globalmente a seleção às configurações
    # que receberam success=True.
    #
    # Soluções aproximadamente estacionárias
    # também podem competir.

    criterios = [

        ("mae", "mae_medio", False),

        ("rmse", "rmse_medio", False),

        (

            "calibracao_agregada",

            "erro_pct_agregado_medio",

            False,

        ),

        ("logloss", "logloss_medio", False),

    ]

    selecionados = {}

    configs = []

    for nome, coluna, desc in criterios:

        melhor = (

            candidatos

            .sort(

                [coluna, "mae_medio"],

                descending=[desc, False],

            )

            .row(0, named=True)

        )

        selecionados[nome] = melhor

        par = (

            float(melhor["alpha"]),

            float(melhor["gamma"]),

        )

        if par not in configs:

            if len(configs) < MAX_CONFIGS_ABLACAO:

                configs.append(par)

    exigir(

        len(configs) > 0,

        "Nenhuma configuracao selecionada.",

    )

    return configs, selecionados


# ============================================================
# SELECAO DOS FINALISTAS
# ============================================================

def selecionar_finalistas(ranking):

    candidatos = ranking.filter(

        pl.col("elegivel_todos")

        & (pl.col("especificacao") != "H0")

    )

    exigir(

        candidatos.height > 0,

        "Nenhum modelo hibrido elegivel.",

    )

    criterios = [

        ("mae", "mae_medio", False),

        ("rmse", "rmse_medio", False),

        ("r2", "r2_medio", True),

        ("logloss", "logloss_medio", False),

        (

            "calibracao_agregada",

            "erro_pct_agregado_medio",

            False,

        ),

    ]

    finalistas = {}

    for nome, coluna, desc in criterios:

        melhor = (

            candidatos

            .sort(

                [coluna, "mae_medio"],

                descending=[desc, False],

            )

            .row(0, named=True)

        )

        chave = (

            melhor["especificacao"],

            float(melhor["alpha"]),

            float(melhor["gamma"]),

        )

        if chave not in finalistas:

            finalistas[chave] = {

                "especificacao":

                    chave[0],

                "alpha":

                    chave[1],

                "gamma":

                    chave[2],

                "criterios": [],

                "metricas_validacao":

                    melhor,

            }

        finalistas[chave][

            "criterios"

        ].append(nome)

    return list(finalistas.values())


# ============================================================
# AUDITORIA DA REPRODUCAO DO BASELINE
# ============================================================

def auditar_baseline_h0(h0):

    esperado_2024 = (

        BENCHMARKS["baseline_23"][

            "retrospectivo_mae"

        ]

    )

    ranking = ranking_temporal(

        h0.filter(

            pl.col("ano_alvo")

            .is_in(ANOS_VALIDACAO)

        )

    )

    observado_validacao = float(

        ranking["mae_medio"][0]

    )

    esperado_validacao = (

        BENCHMARKS["baseline_23"][

            "validacao_mae"

        ]

    )

    teste_2024 = h0.filter(

        pl.col("ano_alvo")

        == ANO_RETROSPECTIVO

    )

    observado_2024 = float(

        teste_2024["mae"][0]

    )

    delta_validacao = abs(

        observado_validacao

        - esperado_validacao

    )

    delta_2024 = abs(

        observado_2024

        - esperado_2024

    )

    # Tolerancia para diferenças pequenas
    # de precisao numerica.

    tolerancia = 1e-5

    resultado = {

        "mae_validacao_observado":

            observado_validacao,

        "mae_validacao_esperado":

            esperado_validacao,

        "delta_validacao":

            delta_validacao,

        "mae_2024_observado":

            observado_2024,

        "mae_2024_esperado":

            esperado_2024,

        "delta_2024":

            delta_2024,

        "tolerancia":

            tolerancia,

        "reproduziu_baseline":

            bool(

                delta_validacao <= tolerancia

                and delta_2024 <= tolerancia

            ),

    }

    salvar_json(

        resultado,

        OUT_DIR / "00_auditoria_baseline.json",

    )

    logger.info(

        "Reproducao do baseline: %s",

        resultado,

    )

    exigir(

        resultado["reproduziu_baseline"],

        "H0 nao reproduziu o baseline 23. "
        "Verifique a construcao do offset "
        "antes de prosseguir.",

    )

    return resultado


# ============================================================
# COMPARACAO COM BENCHMARKS
# ============================================================

def tabela_benchmarks(retrospectivo):

    registros = []

    for nome, b in BENCHMARKS.items():

        registros.append({

            "modelo": nome,

            "mae": b["retrospectivo_mae"],

            "rmse": b["retrospectivo_rmse"],

            "r2": b["retrospectivo_r2"],

            "origem": "benchmark_anterior",

        })

    for row in retrospectivo.to_dicts():

        registros.append({

            "modelo": (

                f"{row['especificacao']}"

                f"_a{row['alpha']:g}"

                f"_g{row['gamma']:g}"

            ),

            "mae": row["mae"],

            "rmse": row["rmse"],

            "r2": row["r2"],

            "origem": "hibrido_26",

        })

    tabela = (

        pl.DataFrame(registros)

        .sort("mae")

    )

    salvar_csv(

        tabela,

        "08_comparacao_benchmarks_2024",

    )

    return tabela


# ============================================================
# CALIBRACAO POR IDADE DA COORTE
# ============================================================

def diagnosticar_calibracao():

    arquivos = sorted(

        (OUT_DIR / "previsoes")

        .glob("*.parquet")

    )

    resultados = []

    for caminho in arquivos:

        with MONITOR.operacao(

            f"Calibracao: {caminho.stem}"

        ):

            df = pl.read_parquet(caminho)

            tabela = (

                df

                .group_by("idade_coorte")

                .agg([

                    pl.len()

                    .alias("n_trajetorias"),

                    pl.col("QT_PERMANENCIA")

                    .sum()

                    .alias("total_expostos"),

                    pl.col(TARGET_D)

                    .sum()

                    .alias("desistentes_observados"),

                    pl.col("desistentes_previstos")

                    .sum(),

                    pl.col(TARGET_TADA)

                    .mean()

                    .alias("tada_observada"),

                    pl.col("tada_prevista")

                    .mean(),

                    (

                        pl.col("tada_prevista")

                        - pl.col(TARGET_TADA)

                    )

                    .abs()

                    .mean()

                    .alias("mae"),

                ])

                .with_columns([

                    pl.lit(caminho.stem)

                    .alias("modelo"),

                    (

                        pl.col("desistentes_previstos")

                        - pl.col(

                            "desistentes_observados"

                        )

                    )

                    .alias("vies_desistentes"),

                    pl.when(

                        pl.col(

                            "desistentes_observados"

                        ) > 0

                    )

                    .then(

                        100

                        * (

                            pl.col(

                                "desistentes_previstos"

                            )

                            - pl.col(

                                "desistentes_observados"

                            )

                        )

                        / pl.col(

                            "desistentes_observados"

                        )

                    )

                    .otherwise(None)

                    .alias("vies_pct_desistentes"),

                ])

                .sort("idade_coorte")

            )

            resultados.append(tabela)

    if resultados:

        salvar_csv(

            pl.concat(

                resultados,

                how="diagonal_relaxed",

            ),

            "10_calibracao_por_idade_2024",

        )


# ============================================================
# TESTES SINTETICOS
# ============================================================

def testes_sinteticos():

    titulo("TESTES SINTETICOS DO MODELO HIBRIDO")

    # --------------------------------------------------------
    # Teste 1: identidade do offset
    # --------------------------------------------------------

    q_hist = np.array([

        0.05,

        0.10,

        0.30,

        0.75,

    ])

    offset = logit(q_hist)

    exigir(

        np.allclose(

            expit(offset),

            q_hist,

            atol=1e-12,

        ),

        "Falha na identidade logit/logistica.",

    )

    # --------------------------------------------------------
    # Teste 2: correcao nula
    # --------------------------------------------------------

    e = np.array([

        0,

        10,

        50,

        100,

    ])

    exigir(

        np.allclose(

            e * expit(offset),

            e * q_hist,

        ),

        "Falha na recuperacao do baseline.",

    )

    # --------------------------------------------------------
    # Teste 3: recuperacao de parametro conhecido
    # --------------------------------------------------------

    rng = np.random.default_rng(

        RANDOM_STATE

    )

    z = rng.normal(

        size=5000

    )

    q0 = rng.uniform(

        0.1,

        0.5,

        size=len(z),

    )

    off = logit(q0)

    prob = expit(

        off

        + 0.3

        + 0.7 * z

    )

    p = rng.integers(

        10,

        100,

        size=len(z),

    )

    d = rng.binomial(

        p,

        prob,

    )

    x = sparse.csr_matrix(

        z.reshape(-1, 1)

        .astype(np.float32)

    )

    modelo = HibridoBinomial(

        alpha=0.0001,

        gamma=1,

        max_iter=100,

        refinar=False,

    ).fit(

        x,

        d,

        p,

        off,

    )

    exigir(

        abs(modelo.intercept_ - 0.3) < 0.1,

        "Falha na recuperacao do intercepto.",

    )

    exigir(

        abs(modelo.coef_[0] - 0.7) < 0.1,

        "Falha na recuperacao do coeficiente.",

    )

    # --------------------------------------------------------
    # Teste 4: H1, somente intercepto
    # --------------------------------------------------------

    x_vazio = sparse.csr_matrix(

        (4, 0),

        dtype=np.float32,

    )

    p = np.array([

        100.0,

        100.0,

        100.0,

        100.0,

    ])

    d = np.array([

        30.0,

        30.0,

        30.0,

        30.0,

    ])

    off = logit(

        np.full(4, 0.2)

    )

    modelo = HibridoBinomial(

        alpha=0.0,

        gamma=1.0,

        max_iter=100,

        refinar=False,

    ).fit(

        x_vazio,

        d,

        p,

        off,

    )

    exigir(

        np.allclose(

            modelo.predict_risco(

                x_vazio,

                off,

            ),

            0.3,

            atol=1e-4,

        ),

        "Falha na correcao por intercepto.",

    )

    # --------------------------------------------------------
    # Teste 5: imputacao e categorias desconhecidas
    # --------------------------------------------------------

    contrato = {

        "numericas": ["x"],

        "categoricas": ["curso"],

    }

    pre, _, _ = construir_preprocessador(

        ["x", "curso"],

        contrato,

    )

    treino = pd.DataFrame({

        "x": [1.0, 2.0, 3.0],

        "curso": ["A", "B", "A"],

    })

    teste = pd.DataFrame({

        "x": [100.0, np.nan],

        "curso": ["C", "A"],

    })

    xt = pre.fit_transform(treino)

    xv = pre.transform(teste)

    categorias = (

        pre.named_transformers_[

            "categoricas"

        ]

        .categories_[0]

    )

    mediana = (

        pre.named_transformers_[

            "numericas"

        ]

        .named_steps["imputer"]

        .statistics_[0]

    )

    exigir(

        "C" not in categorias,

        "Categoria futura aprendida.",

    )

    exigir(

        mediana == 2.0,

        "Imputacao contaminada.",

    )

    exigir(

        xt.shape[1] == xv.shape[1],

        "Dimensao incompativel.",

    )

    # --------------------------------------------------------
    # Teste 6: serializacao NumPy
    # --------------------------------------------------------

    teste_json = {

        "booleano": np.bool_(True),

        "inteiro": np.int64(10),

        "float": np.float64(0.5),

    }

    exigir(

        json.loads(

            json_canonico(teste_json)

        )["booleano"] is True,

        "Falha na serializacao JSON.",

    )

    logger.info(

        "Todos os testes sinteticos aprovados."

    )


# ============================================================
# MAIN
# ============================================================

def main():

    inicio_total = time.perf_counter()

    checkpoint = None

    manifesto = (

        OUT_DIR

        / "26_resumo_hibrido.json"

    )

    MONITOR.iniciar()

    try:

        titulo(

            "26 — MODELO HIBRIDO DE RISCO"

        )

        logger.info(

            "Diretorio atual: %s",

            Path.cwd(),

        )

        logger.info(

            "Log persistente: %s",

            LOG_PATH.resolve(),

        )

        logger.info(

            "Monitoramento psutil: %s",

            psutil is not None,

        )

        testes_sinteticos()

        pasta = localizar_gold()

        contrato = carregar_contrato(pasta)

        specs = construir_especificacoes(

            contrato

        )

        assinatura, configuracao = (

            construir_assinatura(

                pasta,

                specs,

            )

        )

        checkpoint = Checkpoint(

            OUT_DIR / "26_checkpoint.sqlite",

            assinatura,

        )

        salvar_json(

            configuracao,

            OUT_DIR

            / "00_configuracao_execucao.json",

        )

        salvar_json(

            specs,

            OUT_DIR

            / "00_especificacoes_features.json",

        )

        salvar_json(

            {

                "status": "EM_PROCESSAMENTO",

                "assinatura": assinatura,

            },

            manifesto,

        )

        base = carregar_base(

            pasta,

            contrato,

        )

        # ====================================================
        # ETAPA 0: REPRODUCAO DO BASELINE
        # ====================================================

        titulo(

            "ETAPA 0 — AUDITORIA DO OFFSET"

        )

        grupos_h0 = [

            (

                ano,

                "H0",

                [(0.0, 1.0)],

            )

            for ano in [

                *ANOS_VALIDACAO,

                ANO_RETROSPECTIVO,

            ]

        ]

        h0 = executar_etapa(

            base,

            contrato,

            specs,

            checkpoint,

            "baseline_h0",

            grupos_h0,

        )

        salvar_csv(

            h0,

            "00_metricas_baseline_h0",

        )

        auditoria_baseline = (

            auditar_baseline_h0(h0)

        )

        # ====================================================
        # ETAPA 1: TUNING
        # ====================================================

        titulo(

            "ETAPA 1 — TUNING HIBRIDO"

        )

        grade = [

            (alpha, gamma)

            for gamma in GAMMAS_TUNING

            for alpha in ALPHAS

        ]

        grupos_tuning = [

            (

                ano,

                spec,

                grade,

            )

            for spec in TUNING_SPECS

            for ano in ANOS_VALIDACAO

        ]

        tuning = executar_etapa(

            base,

            contrato,

            specs,

            checkpoint,

            "tuning",

            grupos_tuning,

        )

        ranking_tuning = (

            ranking_temporal(tuning)

        )

        salvar_csv(

            tuning,

            "01_tuning_metricas",

        )

        salvar_csv(

            ranking_tuning,

            "02_tuning_ranking",

        )

        imprimir(

            ranking_tuning.head(TOP_N)

        )

        configs, selecao = (

            selecionar_configs(

                ranking_tuning

            )

        )

        salvar_json(

            {

                "configs": configs,

                "selecao": selecao,

            },

            OUT_DIR

            / "03_configs_selecionadas.json",

        )

        logger.info(

            "Configuracoes selecionadas: %s",

            configs,

        )

        # ====================================================
        # ETAPA 2: ABLACAO COMPLETA
        # ====================================================

        titulo(

            "ETAPA 2 — ABLACAO HIBRIDA"

        )

        grupos_ablacao = []

        for spec in ABLACAO_SPECS:

            if spec == "H1":

                # H1 nao possui coeficientes
                # penalizados, portanto alpha
                # nao altera o ajuste.

                configs_spec = [

                    (0.0, gamma)

                    for gamma in sorted({

                        gamma

                        for _, gamma in configs

                    })

                ]

            else:

                configs_spec = configs

            for ano in ANOS_VALIDACAO:

                grupos_ablacao.append((

                    ano,

                    spec,

                    configs_spec,

                ))

        ablacao = executar_etapa(

            base,

            contrato,

            specs,

            checkpoint,

            "ablacao",

            grupos_ablacao,

        )

        validacao_h0 = h0.filter(

            pl.col("ano_alvo")

            .is_in(ANOS_VALIDACAO)

        )

        validacao_completa = pl.concat(

            [

                validacao_h0,

                ablacao,

            ],

            how="diagonal_relaxed",

        )

        ranking = ranking_temporal(

            validacao_completa

        )

        salvar_csv(

            validacao_completa,

            "04_ablacao_metricas",

        )

        salvar_csv(

            ranking,

            "05_ablacao_ranking",

        )

        titulo(

            "RANKING HIBRIDO — VALIDACOES"

        )

        imprimir(

            ranking.head(TOP_N)

        )

        # ====================================================
        # ETAPA 3: FINALISTAS
        # ====================================================

        finalistas = selecionar_finalistas(

            ranking

        )

        salvar_json(

            finalistas,

            OUT_DIR

            / "09_finalistas_congelados.json",

        )

        logger.info(

            "Finalistas selecionados: %d",

            len(finalistas),

        )

        grupos_finais = [

            (

                ANO_RETROSPECTIVO,

                f["especificacao"],

                [

                    (

                        f["alpha"],

                        f["gamma"],

                    )

                ],

            )

            for f in finalistas

        ]

        # ====================================================
        # ETAPA 4: RETROSPECTIVO
        # ====================================================

        titulo(

            "ETAPA 3 — RETROSPECTIVO 2024"

        )

        retrospectivo = executar_etapa(

            base,

            contrato,

            specs,

            checkpoint,

            "retrospectivo",

            grupos_finais,

        )

        retrospectivo = (

            retrospectivo

            .with_columns([

                (

                    pl.col("mae")

                    - BENCHMARKS["baseline_23"][

                        "retrospectivo_mae"

                    ]

                ).alias("delta_mae_vs_23"),

                (

                    pl.col("rmse")

                    - BENCHMARKS["binomial_25"][

                        "retrospectivo_rmse"

                    ]

                ).alias("delta_rmse_vs_25"),

                (

                    pl.col("r2")

                    - BENCHMARKS["binomial_25"][

                        "retrospectivo_r2"

                    ]

                ).alias("delta_r2_vs_25"),

                (

                    pl.col("mae")

                    - BENCHMARKS["binomial_25"][

                        "retrospectivo_mae"

                    ]

                ).alias("delta_mae_vs_25"),

            ])

            .sort("mae")

        )

        salvar_csv(

            retrospectivo,

            "06_metricas_retrospectivas",

        )

        imprimir(

            retrospectivo.select([

                "especificacao",

                "alpha",

                "gamma",

                "convergiu",

                "estacionario",

                "mae",

                "rmse",

                "r2",

                "vies",

                "vies_pct_desistentes",

                "delta_mae_vs_23",

                "delta_rmse_vs_25",

                "delta_r2_vs_25",

            ])

        )

        # ====================================================
        # ETAPA 5: CONSOLIDACAO
        # ====================================================

        titulo(

            "ETAPA 4 — CONSOLIDACAO"

        )

        comparacao = tabela_benchmarks(

            retrospectivo

        )

        imprimir(comparacao)

        diagnosticar_calibracao()

        resumo = {

            "status": "CONCLUIDO",

            "versao": VERSAO,

            "assinatura": assinatura,

            "gold": str(pasta),

            "contrato_09_aprovado": True,

            "testes_sinteticos_aprovados": True,

            "auditoria_baseline":

                auditoria_baseline,

            "n_features_autorizadas":

                len(

                    contrato[

                        "features_autorizadas"

                    ]

                ),

            "n_ajustes_baseline_h0":

                h0.height,

            "n_ajustes_tuning":

                tuning.height,

            "n_ajustes_ablacao":

                ablacao.height,

            "n_ajustes_retrospectivos":

                retrospectivo.height,

            "configs_selecionadas":

                configs,

            "finalistas":

                finalistas,

            "ranking_validacoes":

                ranking.head(TOP_N).to_dicts(),

            "resultados_retrospectivos":

                retrospectivo.to_dicts(),

            "benchmarks":

                BENCHMARKS,

            "tempo_sessao_segundos":

                float(

                    time.perf_counter()

                    - inicio_total

                ),

            "nota_metodologica": (

                "Modelo hibrido binomial "
                "com offset historico fixo. "

                "Regularizacao L2, "
                "intercepto nao penalizado. "

                "Gamma igual a 1 corresponde "
                "a verossimilhanca binomial. "

                "Gamma diferente de 1 corresponde "
                "a perda binomial reponderada. "

                "Offset construido a partir "
                "de features autorizadas "
                "do contrato longitudinal. "

                "Fallback final estimado "
                "exclusivamente no treinamento. "

                "Preprocessamento ajustado "
                "exclusivamente no treinamento. "

                "Selecao formal nas validacoes "
                "de 2020 a 2023. "

                "2024 e avaliacao retrospectiva "
                "previamente explorada, nao "
                "teste externo independente. "

                "Convergencia do otimizador e "
                "estacionariedade aproximada "
                "registradas separadamente. "

                "A hipotese binomial nao exclui "
                "sobredispersao ou dependencia "
                "entre estudantes."

            ),

        }

        salvar_json(

            resumo,

            manifesto,

        )

        titulo(

            "26 — MODELO HIBRIDO CONCLUIDO"

        )

        logger.info(

            "Tempo da sessao: %.2f minutos",

            (

                time.perf_counter()

                - inicio_total

            ) / 60,

        )

        logger.info(

            "Resultados: %s",

            OUT_DIR.resolve(),

        )

    except KeyboardInterrupt:

        logger.warning(

            "Execucao interrompida pelo usuario. "
            "Checkpoints preservados."

        )

        raise

    except Exception as erro:

        logger.exception(

            "ERRO FATAL NA EXECUCAO"

        )

        salvar_json(

            {

                "status": "ERRO",

                "erro": str(erro),

                "traceback":

                    traceback.format_exc(),

            },

            OUT_DIR / "26_ultimo_erro.json",

        )

        raise

    finally:

        if checkpoint is not None:

            checkpoint.fechar()

        MONITOR.parar()


if __name__ == "__main__":

    main()