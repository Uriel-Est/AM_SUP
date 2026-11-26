from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import threading
import time
import traceback

from pathlib import Path
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy
import sklearn
import catboost

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor, Pool

try:
    import psutil
except ImportError:
    psutil = None


# ============================================================
# 01. CONFIGURACOES
# ============================================================

VERSAO = "27_v1"

GOLD_DIR = None

OUT_DIR = Path("./reports/modelos_arvores")

REPORTS_DIR = Path("./reports")

ANOS_VALIDACAO = [2020, 2021, 2022, 2023]

ANO_RETROSPECTIVO = 2024

RANDOM_STATE = 42

AMOSTRA_TUNING = 200_000

# None utiliza todo o treinamento elegivel
# na confirmacao dos finalistas.

LIMITE_TREINO_COMPLETO = None

THREADS = max(
    1,
    min(6, os.cpu_count() or 1),
)

HEARTBEAT_SEGUNDOS = 30

SALVAR_MODELOS = True

INCLUIR_MODELOS_ANTERIORES = True

# ------------------------------------------------------------
# Especificacoes avaliadas
# ------------------------------------------------------------

ESPECIFICACOES = [
    "M4",
    "M7",
]

MODOS = [
    "direto",
    "risco",
]

# ------------------------------------------------------------
# Grades reduzidas para a Entrega 6
# ------------------------------------------------------------

GRADE_ARVORE = [

    {
        "max_depth": 8,
        "min_samples_leaf": 100,
    },

    {
        "max_depth": 16,
        "min_samples_leaf": 50,
    },

]

GRADE_CATBOOST = [

    {
        "iterations": 150,
        "depth": 5,
        "learning_rate": 0.10,
        "l2_leaf_reg": 5.0,
    },

    {
        "iterations": 250,
        "depth": 7,
        "learning_rate": 0.08,
        "l2_leaf_reg": 10.0,
    },

]

# Impede o uso de target statistics nas
# categorias das especificacoes M4 e M7.
#
# O script verifica a cardinalidade antes
# de iniciar cada ajuste do CatBoost.

CATBOOST_ONE_HOT_MAX_SIZE = 10_000

# ------------------------------------------------------------
# Diagnosticos
# ------------------------------------------------------------

N_CASOS_EXTREMOS = 50

N_MINIMO_GRUPO = 30

CENARIOS_CAPACIDADE = [
    0.01,
    0.05,
    0.10,
]

# ------------------------------------------------------------
# Contrato longitudinal
# ------------------------------------------------------------

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

OFFSET_FEATURE = "hist_ies_curso_idade_risco"

COLUNAS_GRUPOS = [

    "CO_REGIAO",
    "CO_UF",
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "CO_CINE_AREA_GERAL",
    "idade_coorte",

]

BENCHMARK_MAE_VALIDACAO = 3.564263690214786

BENCHMARK_MAE_2024 = 1.935115045104516

EPS = 1e-12


# ============================================================
# 02. LOGGING
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

LOG_PATH = OUT_DIR / "27_execucao.log"

logger = logging.getLogger(
    "modelos_arvores_27"
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

console = logging.StreamHandler(sys.stdout)

console.setFormatter(formatter)

arquivo_log = logging.FileHandler(

    LOG_PATH,

    mode="a",

    encoding="utf-8",

)

arquivo_log.setFormatter(formatter)

logger.addHandler(console)

logger.addHandler(arquivo_log)


def titulo(texto):

    logger.info("=" * 90)

    logger.info(texto)

    logger.info("=" * 90)


# ============================================================
# 03. MONITORAMENTO
# ============================================================

class Monitor:

    def __init__(self):

        self.stop = threading.Event()

        self.lock = threading.Lock()

        self.etapa = "Inicializacao"

        self.inicio = time.perf_counter()

        self.inicio_etapa = self.inicio

        self.thread = None

        self.proc = None

        if psutil is not None:

            self.proc = psutil.Process(
                os.getpid()
            )

            self.proc.cpu_percent(
                interval=None
            )

    def recursos(self):

        if self.proc is None:

            return "psutil indisponivel"

        try:

            ram = (
                self.proc.memory_info().rss
                / 1024**3
            )

            cpu = self.proc.cpu_percent(
                interval=None
            )

            sistema = (
                psutil.virtual_memory().percent
            )

            return (

                f"RAM Python={ram:.2f} GiB "
                f"| RAM sistema={sistema:.1f}% "
                f"| CPU={cpu:.1f}%"

            )

        except Exception:

            return "Recursos indisponiveis"

    def atualizar(self, etapa):

        with self.lock:

            self.etapa = etapa

            self.inicio_etapa = (
                time.perf_counter()
            )

    def loop(self):

        while not self.stop.wait(
            HEARTBEAT_SEGUNDOS
        ):

            with self.lock:

                etapa = self.etapa

                duracao = (
                    time.perf_counter()
                    - self.inicio_etapa
                )

            logger.info(

                "HEARTBEAT | etapa=%s "
                "| tempo_etapa=%.1fs "
                "| tempo_total=%.1fmin | %s",

                etapa,

                duracao,

                (
                    time.perf_counter()
                    - self.inicio
                ) / 60,

                self.recursos(),

            )

    def iniciar(self):

        self.stop.clear()

        self.thread = threading.Thread(

            target=self.loop,

            daemon=True,

        )

        self.thread.start()

    def parar(self):

        self.stop.set()

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


MONITOR = Monitor()


# ============================================================
# 04. UTILITARIOS
# ============================================================

def exigir(condicao, mensagem):

    if not bool(condicao):

        raise ValueError(mensagem)


def unicos(lista):

    return list(
        dict.fromkeys(lista)
    )


def arr(df, coluna):

    return (
        df[coluna]
        .cast(pl.Float64)
        .to_numpy()
        .copy()
    )


def limpar_json(obj):

    if isinstance(obj, dict):

        return {

            str(k): limpar_json(v)

            for k, v in obj.items()

        }

    if isinstance(obj, (list, tuple)):

        return [

            limpar_json(v)

            for v in obj

        ]

    if isinstance(obj, np.generic):

        return limpar_json(obj.item())

    if isinstance(obj, np.ndarray):

        return limpar_json(obj.tolist())

    if isinstance(obj, Path):

        return str(obj)

    if isinstance(obj, float):

        return obj if np.isfinite(obj) else None

    return obj


def json_canonico(obj):

    return json.dumps(

        limpar_json(obj),

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

            limpar_json(obj),

            indent=2,

            ensure_ascii=False,

            allow_nan=False,

        ),

        encoding="utf-8",

    )

    tmp.replace(caminho)


def salvar_csv(df, nome):

    caminho = OUT_DIR / f"{nome}.csv"

    tmp = caminho.with_suffix(".csv.tmp")

    df.to_csv(

        tmp,

        index=False,

        encoding="utf-8-sig",

    )

    tmp.replace(caminho)


def salvar_parquet(df, caminho):

    caminho = Path(caminho)

    tmp = caminho.with_suffix(".parquet.tmp")

    df.write_parquet(

        tmp,

        compression="zstd",

    )

    tmp.replace(caminho)


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


def imprimir(df):

    logger.info(

        "\n%s",

        df.to_string(index=False),

    )


# ============================================================
# 05. LOCALIZACAO UNIVERSAL DA GOLD
# ============================================================

def localizar_gold():

    arquivos = [

        "features_longitudinais.parquet",

        "alvos_longitudinais.parquet",

        "09_auditoria_vazamento.json",

    ]

    candidatos = []

    if GOLD_DIR is not None:

        candidatos.append(Path(GOLD_DIR))

    raizes = [

        Path.cwd(),

        *[Path(p) for p in sys.path if p],

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
# 06. CONTRATO DE FEATURES
# ============================================================

def carregar_contrato(pasta):

    contrato = json.loads(

        (

            pasta
            / "09_auditoria_vazamento.json"

        ).read_text(encoding="utf-8")

    )

    exigir(

        contrato["status"] == "APROVADO",

        "Contrato do script 09 reprovado.",

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

        for t in contrato["testes_invariancia"]

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

    proibidas = set(

        contrato["alvos_e_controles_proibidos"]

    ) | set(

        contrato["metadados_historicos_proibidos"]

    )

    nums = set(contrato["numericas"])

    cats = set(contrato["categoricas"])

    exigir(

        autorizadas.isdisjoint(proibidas),

        "Features proibidas detectadas.",

    )

    exigir(

        nums.isdisjoint(cats),

        "Classificacao de features invalida.",

    )

    exigir(

        autorizadas == nums | cats,

        "Features sem classificacao.",

    )

    logger.info(

        "Contrato aprovado | features=%d "
        "| numericas=%d | categoricas=%d",

        len(autorizadas),

        len(nums),

        len(cats),

    )

    return contrato


# ============================================================
# 07. ESPECIFICACOES M4 E M7
# ============================================================

def construir_specs(contrato):

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

        c for c in autorizadas

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

        c for c in autorizadas

        if c.startswith("hist_")

    ]

    m7 = m4 + historicas

    specs = {

        "M4": unicos(m4),

        "M7": unicos(m7),

    }

    for nome, cols in specs.items():

        exigir(

            set(cols).issubset(autorizadas),

            f"Features proibidas em {nome}.",

        )

        logger.info(

            "%s | %d features",

            nome,

            len(cols),

        )

    exigir(

        len(specs["M4"]) == 83,

        "M4 diferente da especificacao anterior.",

    )

    exigir(

        len(specs["M7"]) == 163,

        "M7 diferente da especificacao anterior.",

    )

    return specs


# ============================================================
# 08. BASE LONGITUDINAL E AUDITORIA
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

    meta = contrato[
        "metadados_historicos_proibidos"
    ]

    with MONITOR.operacao(

        "Auditoria dos metadados temporais"

    ):

        audit_hist = (

            lf_f

            .select([

                (

                    pl.col(c)

                    > pl.col("ANO_ORIGEM")

                )

                .fill_null(False)

                .sum()

                .alias(c)

                for c in meta

            ])

            .collect()

            .row(0, named=True)

        )

    exigir(

        sum(audit_hist.values()) == 0,

        "Informacao futura nos historicos.",

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

            f"Unicidade das chaves: {nome}"

        ):

            a = (

                lf.select([

                    pl.len().alias("n"),

                    pl.struct(KEYS)

                    .n_unique()

                    .alias("n_chaves"),

                    pl.any_horizontal([

                        pl.col(c).is_null()

                        for c in KEYS

                    ])

                    .sum()

                    .alias("chaves_nulas"),

                ])

                .collect()

                .row(0, named=True)

            )

        exigir(

            a["n"] == esperado,

            f"Dimensao incompativel: {nome}",

        )

        exigir(

            a["n"] == a["n_chaves"],

            f"Chaves duplicadas: {nome}",

        )

        exigir(

            a["chaves_nulas"] == 0,

            f"Chaves nulas: {nome}",

        )

    base = (

        lf_f

        .select(unicos(KEYS + features))

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

    p = pl.col("QT_PERMANENCIA").cast(pl.Float64)

    n = pl.col("QT_INGRESSANTE").cast(pl.Float64)

    d = pl.col(TARGET_D).cast(pl.Float64)

    y = pl.col(TARGET_TADA).cast(pl.Float64)

    q = pl.col(TARGET_Q).cast(pl.Float64)

    e = pl.col("exposicao").cast(pl.Float64)

    with MONITOR.operacao(

        "Auditoria longitudinal completa"

    ):

        audit = (

            base.select([

                pl.len().alias("n"),

                (

                    pl.col("ANO_ALVO")

                    != pl.col("ANO_ORIGEM") + 1

                )

                .fill_null(True)

                .sum()

                .alias("horizonte_invalido"),

                (

                    p.is_null()

                    | n.is_null()

                    | d.is_null()

                    | ~p.is_finite()

                    | ~n.is_finite()

                    | ~d.is_finite()

                    | (p < 0)

                    | (n <= 0)

                    | (p > n)

                    | (d < 0)

                    | (d > p)

                    | ((p - p.round(0)).abs() > 1e-8)

                    | ((d - d.round(0)).abs() > 1e-8)

                )

                .fill_null(True)

                .sum()

                .alias("contagens_invalidas"),

                (

                    y.is_null()

                    | ~y.is_finite()

                    | ~y.is_between(0, 100)

                )

                .fill_null(True)

                .sum()

                .alias("tada_invalida"),

                (

                    e.is_null()

                    | ~e.is_finite()

                    | ((e - 100 * p / n).abs() > 1e-4)

                )

                .fill_null(True)

                .sum()

                .alias("exposicao_invalida"),

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

                        | ((q - d / p).abs() > 1e-5)

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

        audit["n"] == contrato["n_alvos"],

        "Perda de observacoes no join.",

    )

    for c in [

        "horizonte_invalido",

        "contagens_invalidas",

        "tada_invalida",

        "exposicao_invalida",

        "violacao_zero",

        "risco_invalido",

    ]:

        exigir(

            audit[c] == 0,

            f"Auditoria reprovada: {c}",

        )

    with MONITOR.operacao(

        "Auditoria das particoes temporais"

    ):

        contagens = (

            base

            .group_by("ANO_ALVO")

            .agg([

                pl.len().alias("n"),

                (pl.col("QT_PERMANENCIA") > 0)

                .sum()

                .alias("expostos"),

            ])

            .sort("ANO_ALVO")

            .collect()

            .to_dicts()

        )

    particoes = {}

    for esperado in contrato["particoes"]:

        ano = int(esperado["ano_alvo"])

        treino = sum(

            r["n"]

            for r in contagens

            if r["ANO_ALVO"] < ano

        )

        treino_expostos = sum(

            r["expostos"]

            for r in contagens

            if r["ANO_ALVO"] < ano

        )

        avaliacao = sum(

            r["n"]

            for r in contagens

            if r["ANO_ALVO"] == ano

        )

        exigir(

            treino == esperado["n_treino"],

            f"Treino divergente: {ano}",

        )

        exigir(

            avaliacao == esperado["n_avaliacao"],

            f"Avaliacao divergente: {ano}",

        )

        particoes[ano] = {

            "n_treino": treino,

            "n_treino_expostos": treino_expostos,

            "n_avaliacao": avaliacao,

        }

    audit["historicos"] = audit_hist

    audit["particoes"] = particoes

    salvar_json(

        audit,

        OUT_DIR / "00_auditoria_base.json",

    )

    logger.info(

        "Auditoria aprovada: %s",

        audit,

    )

    return base, particoes


# ============================================================
# 09. CONFIGURACOES EXPERIMENTAIS
# ============================================================

def construir_grade():

    configs = []

    for familia, grade in [

        ("arvore", GRADE_ARVORE),

        ("catboost", GRADE_CATBOOST),

    ]:

        for spec in ESPECIFICACOES:

            for modo in MODOS:

                for params in grade:

                    cfg = {

                        "familia": familia,

                        "especificacao": spec,

                        "modo": modo,

                        "hiperparametros": params,

                    }

                    cfg["id_config"] = (

                        f"{familia}_{spec}_{modo}_"

                        + hash_objeto(cfg)[:10]

                    )

                    configs.append(cfg)

    return configs


# ============================================================
# 10. CHECKPOINT
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
                "Preserve a pasta anterior.",

            )

    def existe(self, chave):

        return self.con.execute(

            "SELECT 1 FROM resultados WHERE id=?",

            (chave,),

        ).fetchone() is not None

    def salvar(self, chave, etapa, registro):

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

                    json_canonico(registro),

                ),

            )

    def carregar(self, etapa):

        dados = self.con.execute(

            """

            SELECT resultado

            FROM resultados

            WHERE etapa=?

            ORDER BY rowid

            """,

            (etapa,),

        ).fetchall()

        return pd.DataFrame([

            json.loads(r[0])

            for r in dados

        ])

    def contar(self, etapa):

        return self.con.execute(

            """

            SELECT COUNT(*)

            FROM resultados

            WHERE etapa=?

            """,

            (etapa,),

        ).fetchone()[0]

    def fechar(self):

        self.con.close()


def construir_assinatura(pasta, specs, configs):

    arquivos = {}

    for nome in [

        "features_longitudinais.parquet",

        "alvos_longitudinais.parquet",

        "09_auditoria_vazamento.json",

    ]:

        with MONITOR.operacao(f"Hash: {nome}"):

            arquivos[nome] = hash_arquivo(

                pasta / nome

            )

    configuracao = {

        "versao": VERSAO,

        "arquivos": arquivos,

        "specs": specs,

        "configs": configs,

        "anos_validacao": ANOS_VALIDACAO,

        "ano_retrospectivo": ANO_RETROSPECTIVO,

        "amostra_tuning": AMOSTRA_TUNING,

        "limite_treino_completo":

            LIMITE_TREINO_COMPLETO,

        "seed": RANDOM_STATE,

        "threads": THREADS,

        "catboost_one_hot_max_size":

            CATBOOST_ONE_HOT_MAX_SIZE,

        "salvar_modelos": SALVAR_MODELOS,

        "bibliotecas": {

            "numpy": np.__version__,

            "pandas": pd.__version__,

            "polars": pl.__version__,

            "scipy": scipy.__version__,

            "sklearn": sklearn.__version__,

            "catboost": catboost.__version__,

        },

    }

    return hash_objeto(configuracao), configuracao


# ============================================================
# 11. CARREGAMENTO DAS PARTICOES
# ============================================================

def colunas_leitura(features):

    return unicos([

        *KEYS,

        *features,

        *COLUNAS_GRUPOS,

        "ANO_ALVO",

        TARGET_TADA,

        TARGET_D,

        TARGET_Q,

        TARGET_RECONSTRUIDA,

        "QT_INGRESSANTE",

        "QT_PERMANENCIA",

        "exposicao",

        OFFSET_FEATURE,

        "FLAG_TADA_ACIMA_EXPOSICAO",

    ])


def carregar_avaliacao(base, ano, features):

    with MONITOR.operacao(

        f"Leitura avaliacao {ano}"

    ):

        teste = (

            base

            .filter(pl.col("ANO_ALVO") == ano)

            .select(colunas_leitura(features))

            .collect()

            .sort(KEYS)

        )

    return teste


def carregar_fold(

    base,

    particoes,

    features,

    ano,

    etapa,

):

    limite = (

        AMOSTRA_TUNING

        if etapa == "tuning"

        else LIMITE_TREINO_COMPLETO

    )

    lf = (

        base

        .filter(

            (pl.col("ANO_ALVO") < ano)

            & (pl.col("QT_PERMANENCIA") > 0)

        )

    )

    n_disponivel = particoes[ano][

        "n_treino_expostos"

    ]

    # Amostra as chaves antes de materializar
    # todas as colunas na memoria.

    if (

        limite is not None

        and n_disponivel > limite

    ):

        with MONITOR.operacao(

            f"Amostragem das chaves {ano}"

        ):

            chaves = (

                lf.select(KEYS)

                .collect()

                .sort(KEYS)

                .sample(

                    n=limite,

                    seed=RANDOM_STATE,

                    shuffle=True,

                )

            )

        lf = lf.join(

            chaves.lazy(),

            on=KEYS,

            how="semi",

        )

    with MONITOR.operacao(

        f"Leitura treino {ano}"

    ):

        treino = (

            lf

            .select(

                colunas_leitura(features)

            )

            .collect()

            .sort([

                "ANO_ORIGEM",

                "CO_IES",

                "CO_CURSO",

                "NU_ANO_INGRESSO",

            ])

        )

    teste = carregar_avaliacao(

        base,

        ano,

        features,

    )

    n_esperado = (

        min(n_disponivel, limite)

        if limite is not None

        else n_disponivel

    )

    exigir(

        treino.height == n_esperado,

        "Dimensao do treino divergente.",

    )

    exigir(

        teste.height

        == particoes[ano]["n_avaliacao"],

        "Dimensao da avaliacao divergente.",

    )

    exigir(

        treino["ANO_ALVO"].max()

        < teste["ANO_ALVO"].min(),

        "Sobreposicao temporal.",

    )

    logger.info(

        "FOLD | %d | treino=%d "
        "| treino disponivel=%d | avaliacao=%d",

        ano,

        treino.height,

        n_disponivel,

        teste.height,

    )

    return treino, teste


# ============================================================
# 12. PREPROCESSAMENTO
# ============================================================

def converter_pandas(df, features, contrato):

    numericas = set(contrato["numericas"])

    dados = {}

    for c in features:

        if c in numericas:

            valores = (

                df[c]

                .cast(pl.Float32, strict=False)

                .to_numpy()

                .copy()

            )

            valores[~np.isfinite(valores)] = np.nan

            dados[c] = valores

        else:

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


def preparar_arvore(

    treino,

    teste,

    features,

    contrato,

):

    nums = [

        c for c in features

        if c in contrato["numericas"]

    ]

    cats = [

        c for c in features

        if c in contrato["categoricas"]

    ]

    transformadores = []

    if nums:

        transformadores.append((

            "numericas",

            SimpleImputer(

                strategy="median",

                add_indicator=True,

                keep_empty_features=True,

            ),

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

    with MONITOR.operacao(

        "Arvore: FIT preprocessamento no treino"

    ):

        train_pd = converter_pandas(

            treino,

            features,

            contrato,

        )

        x = pre.fit_transform(train_pd)

        del train_pd

    gc.collect()

    with MONITOR.operacao(

        "Arvore: TRANSFORM avaliacao"

    ):

        test_pd = converter_pandas(

            teste,

            features,

            contrato,

        )

        xv = pre.transform(test_pd)

        del test_pd

    # A arvore do sklearn utiliza CSC no fit
    # e CSR na predicao.

    if scipy.sparse.issparse(x):

        x = x.astype(

            np.float32,

            copy=False,

        ).tocsc()

        xv = xv.astype(

            np.float32,

            copy=False,

        ).tocsr()

    else:

        x = np.asarray(x, dtype=np.float32)

        xv = np.asarray(xv, dtype=np.float32)

    exigir(

        x.shape[1] == xv.shape[1],

        "Dimensoes incompativeis.",

    )

    logger.info(

        "Arvore | matriz treino=%s "
        "| matriz avaliacao=%s",

        x.shape,

        xv.shape,

    )

    return x, xv, pre, cats


def preparar_catboost(

    treino,

    teste,

    features,

    contrato,

):

    cats = [

        c for c in features

        if c in contrato["categoricas"]

    ]

    with MONITOR.operacao(

        "CatBoost: dados de treinamento"

    ):

        x = converter_pandas(

            treino,

            features,

            contrato,

        )

        cardinalidades = {

            c: int(x[c].nunique(dropna=False))

            for c in cats

        }

        exigir(

            all(

                v <= CATBOOST_ONE_HOT_MAX_SIZE

                for v in cardinalidades.values()

            ),

            "Categoria acima do limite de "
            "one-hot interno do CatBoost.",

        )

    with MONITOR.operacao(

        "CatBoost: dados de avaliacao"

    ):

        xv = converter_pandas(

            teste,

            features,

            contrato,

        )

    logger.info(

        "CatBoost | cardinalidades=%s",

        cardinalidades,

    )

    logger.info(

        "CatBoost | treino=%s | avaliacao=%s",

        x.shape,

        xv.shape,

    )

    return x, xv, None, cats


# ============================================================
# 13. CONSTRUCAO DOS MODELOS
# ============================================================

def construir_modelo(cfg):

    parametros = dict(

        cfg["hiperparametros"]

    )

    if cfg["familia"] == "arvore":

        return DecisionTreeRegressor(

            criterion="squared_error",

            splitter="best",

            random_state=RANDOM_STATE,

            **parametros,

        )

    if cfg["familia"] == "catboost":

        return CatBoostRegressor(

            loss_function="RMSE",

            eval_metric="MAE",

            random_seed=RANDOM_STATE,

            thread_count=THREADS,

            task_type="CPU",

            boosting_type="Plain",

            has_time=True,

            one_hot_max_size=(

                CATBOOST_ONE_HOT_MAX_SIZE

            ),

            allow_writing_files=False,

            verbose=50,

            **parametros,

        )

    raise ValueError(

        f"Familia desconhecida: {cfg['familia']}"

    )


# ============================================================
# 14. ALVOS E RESTRICOES ESTRUTURAIS
# ============================================================

def alvo_treinamento(treino, modo):

    if modo == "direto":

        y = arr(treino, TARGET_TADA)

    elif modo == "risco":

        # Risco calculado diretamente pelas
        # contagens auditadas.
        #
        # Cada trajetoria tem peso igual
        # na funcao de perda.

        d = arr(treino, TARGET_D)

        p = arr(treino, "QT_PERMANENCIA")

        exigir(

            (p > 0).all(),

            "Treino do risco sem exposicao.",

        )

        y = d / p

    else:

        raise ValueError(

            f"Modo desconhecido: {modo}"

        )

    exigir(

        np.isfinite(y).all(),

        "Alvo de treinamento nao finito.",

    )

    return y


def converter_previsao(teste, pred, modo):

    pred = np.asarray(

        pred,

        dtype=np.float64,

    ).reshape(-1)

    p = arr(teste, "QT_PERMANENCIA")

    e = arr(teste, "exposicao")

    exigir(

        len(pred) == teste.height,

        "Predicoes desalinhadas.",

    )

    exigir(

        np.isfinite(pred).all(),

        "Predicoes nao finitas.",

    )

    if modo == "direto":

        tada = np.clip(

            pred,

            0,

            e,

        )

    elif modo == "risco":

        tada = (

            e * np.clip(pred, 0, 1)

        )

    else:

        raise ValueError(modo)

    tada[p == 0] = 0.0

    exigir(

        (tada <= e + 1e-8).all(),

        "Predicao acima da exposicao.",

    )

    return tada


def risco_implicito(teste, tada):

    e = arr(teste, "exposicao")

    p = arr(teste, "QT_PERMANENCIA")

    q = np.divide(

        tada,

        e,

        out=np.zeros_like(tada),

        where=e > 0,

    )

    q = np.clip(q, 0, 1)

    q[p == 0] = 0

    return q


# ============================================================
# 15. METRICAS
# ============================================================

def calcular_metricas(

    y,

    pred,

    p,

    n,

    d,

    e,

):

    y = np.asarray(y, dtype=np.float64)

    pred = np.asarray(pred, dtype=np.float64)

    p = np.asarray(p, dtype=np.float64)

    n = np.asarray(n, dtype=np.float64)

    d = np.asarray(d, dtype=np.float64)

    e = np.asarray(e, dtype=np.float64)

    erro = pred - y

    ativos = p > 0

    positivos = y > 0

    q = np.divide(

        pred,

        e,

        out=np.zeros_like(pred),

        where=e > 0,

    )

    q = np.clip(q, 0, 1)

    q[~ativos] = 0

    d_pred = p * q

    sst = np.sum((y - y.mean()) ** 2)

    total_obs = float(d.sum())

    total_pred = float(d_pred.sum())

    m = {

        "n": int(len(y)),

        "n_expostos": int(ativos.sum()),

        "n_zero_estrutural":

            int((~ativos).sum()),

        "mae":

            float(np.abs(erro).mean()),

        "rmse":

            float(np.sqrt(np.mean(erro**2))),

        "r2":

            float(1 - np.sum(erro**2) / sst)

            if sst > 0 else None,

        "vies":

            float(erro.mean()),

        "media_observada":

            float(y.mean()),

        "media_prevista":

            float(pred.mean()),

        "mae_ponderado_ingressantes":

            float(

                np.average(

                    np.abs(erro),

                    weights=n,

                )

            ),

        "mae_expostos":

            float(np.abs(erro[ativos]).mean())

            if ativos.any() else None,

        "rmse_expostos":

            float(

                np.sqrt(

                    np.mean(erro[ativos]**2)

                )

            )

            if ativos.any() else None,

        "mae_positivos":

            float(

                np.abs(erro[positivos]).mean()

            )

            if positivos.any() else None,

        "mae_zero_estrutural":

            float(np.abs(erro[~ativos]).mean())

            if (~ativos).any() else None,

        "pct_pred_zero":

            float(100 * (pred == 0).mean()),

        "pct_pred_acima_exposicao":

            float(

                100 * (pred > e + 1e-8).mean()

            ),

        "desistentes_observados":

            total_obs,

        "desistentes_previstos":

            total_pred,

        "vies_total_desistentes":

            total_pred - total_obs,

        "vies_pct_desistentes":

            100 * (total_pred - total_obs) / total_obs

            if total_obs > 0 else None,

        "mae_contagem":

            float(np.abs(d_pred - d).mean()),

    }

    if ativos.any():

        pa = p[ativos]

        da = d[ativos]

        qa = q[ativos]

        qo = da / pa

        qc = np.clip(

            qa,

            EPS,

            1 - EPS,

        )

        m.update({

            "risco_mae":

                float(np.abs(qa - qo).mean()),

            "risco_real_ponderado":

                float(da.sum() / pa.sum()),

            "risco_pred_ponderado":

                float(np.average(qa, weights=pa)),

            "logloss_binomial":

                float(

                    -np.sum(

                        da * np.log(qc)

                        + (pa - da) * np.log1p(-qc)

                    )

                    / pa.sum()

                ),

            "brier_individual":

                float(

                    np.sum(

                        da * (1 - qa)**2

                        + (pa - da) * qa**2

                    )

                    / pa.sum()

                ),

        })

    else:

        m.update({

            "risco_mae": None,

            "risco_real_ponderado": None,

            "risco_pred_ponderado": None,

            "logloss_binomial": None,

            "brier_individual": None,

        })

    return m


def metricas_df(teste, pred):

    return calcular_metricas(

        arr(teste, TARGET_TADA),

        pred,

        arr(teste, "QT_PERMANENCIA"),

        arr(teste, "QT_INGRESSANTE"),

        arr(teste, TARGET_D),

        arr(teste, "exposicao"),

    )


# ============================================================
# 16. REPRODUCAO DO BASELINE HISTORICO
# ============================================================

def previsao_baseline(teste):

    q = arr(teste, OFFSET_FEATURE)

    p = arr(teste, "QT_PERMANENCIA")

    e = arr(teste, "exposicao")

    ativos = p > 0

    exigir(

        np.isfinite(q[ativos]).all(),

        "Historico ausente na avaliacao. "
        "Nenhum fallback sera estimado.",

    )

    exigir(

        (

            (q[ativos] >= 0)

            & (q[ativos] <= 1)

        ).all(),

        "Risco historico invalido.",

    )

    q[~ativos] = 0.0

    return e * q


def executar_baseline(base, particoes):

    resultados = []

    for ano in [

        *ANOS_VALIDACAO,

        ANO_RETROSPECTIVO,

    ]:

        teste = carregar_avaliacao(

            base,

            ano,

            [OFFSET_FEATURE],

        )

        exigir(

            teste.height

            == particoes[ano]["n_avaliacao"],

            "Baseline em amostra divergente.",

        )

        inicio = time.perf_counter()

        pred = previsao_baseline(teste)

        tempo_pred = (

            time.perf_counter() - inicio

        )

        m = metricas_df(teste, pred)

        resultados.append({

            "modelo": "baseline_23",

            "familia": "baseline",

            "ano_alvo": ano,

            "tempo_treino": 0.0,

            "tempo_predicao": tempo_pred,

            **m,

        })

        logger.info(

            "BASELINE | %d "
            "| MAE=%.6f | RMSE=%.6f "
            "| R2=%.6f",

            ano,

            m["mae"],

            m["rmse"],

            m["r2"],

        )

        del teste, pred

        gc.collect()

    df = pd.DataFrame(resultados)

    mae_val = df.loc[

        df["ano_alvo"].isin(ANOS_VALIDACAO),

        "mae",

    ].mean()

    mae_2024 = df.loc[

        df["ano_alvo"] == ANO_RETROSPECTIVO,

        "mae",

    ].iloc[0]

    exigir(

        abs(

            mae_val - BENCHMARK_MAE_VALIDACAO

        ) <= 1e-5,

        "Baseline das validacoes divergente.",

    )

    exigir(

        abs(

            mae_2024 - BENCHMARK_MAE_2024

        ) <= 1e-5,

        "Baseline retrospectivo divergente.",

    )

    salvar_csv(

        df,

        "00_metricas_baseline",

    )

    salvar_json(

        {

            "status": "APROVADO",

            "mae_validacao": float(mae_val),

            "mae_2024": float(mae_2024),

            "delta_validacao":

                float(

                    mae_val

                    - BENCHMARK_MAE_VALIDACAO

                ),

            "delta_2024":

                float(

                    mae_2024

                    - BENCHMARK_MAE_2024

                ),

        },

        OUT_DIR / "00_auditoria_baseline.json",

    )

    logger.info(

        "Baseline 23 reproduzido corretamente."

    )

    return df


# ============================================================
# 17. SALVAMENTO DE MODELOS E PREVISOES
# ============================================================

def salvar_finalista(

    cfg,

    modelo,

    pre,

    features,

    cats,

    teste,

    pred,

):

    nome = cfg["id_config"]

    q = risco_implicito(teste, pred)

    p = arr(teste, "QT_PERMANENCIA")

    ativos = p > 0

    d_pred = p * q

    risco_serie = pl.Series(

        "risco_previsto",

        q,

        dtype=pl.Float64,

    ).set(

        pl.Series(~ativos),

        None,

    )

    saida = teste.select([

        *KEYS,

        "ANO_ALVO",

        *COLUNAS_GRUPOS,

        TARGET_TADA,

        TARGET_D,

        TARGET_Q,

        TARGET_RECONSTRUIDA,

        "QT_PERMANENCIA",

        "QT_INGRESSANTE",

        "exposicao",

        "FLAG_TADA_ACIMA_EXPOSICAO",

    ])

    saida = saida.with_columns([

        pl.Series(

            "tada_prevista",

            pred,

            dtype=pl.Float64,

        ),

        risco_serie,

        pl.Series(

            "desistentes_previstos",

            d_pred,

            dtype=pl.Float64,

        ),

    ])

    with MONITOR.operacao(

        f"Salvamento previsoes {nome}"

    ):

        salvar_parquet(

            saida,

            OUT_DIR / "previsoes" / f"{nome}.parquet",

        )

    if SALVAR_MODELOS:

        pacote = {

            "configuracao": cfg,

            "modelo": modelo,

            "preprocessador": pre,

            "features": features,

            "categoricas": cats,

            "ano_maximo_alvo_treino":

                ANO_RETROSPECTIVO - 1,

            "regra_predicao":

                "Clip pela exposicao e zero estrutural",

        }

        caminho = (

            OUT_DIR

            / "modelos"

            / f"{nome}.joblib"

        )

        tmp = caminho.with_suffix(".joblib.tmp")

        with MONITOR.operacao(

            f"Salvamento modelo {nome}"

        ):

            joblib.dump(

                pacote,

                tmp,

                compress=3,

            )

            tmp.replace(caminho)


# ============================================================
# 18. PROGRESSO E IDENTIFICACAO
# ============================================================

def chave_experimento(etapa, ano, cfg):

    return hash_objeto([

        etapa,

        ano,

        cfg["id_config"],

    ])


def mostrar_progresso(

    cp,

    etapa,

    total,

    inicial,

    inicio,

):

    atual = cp.contar(etapa)

    novos = atual - inicial

    if novos > 0:

        eta = (

            (time.perf_counter() - inicio)

            / novos

            * max(total - atual, 0)

            / 60

        )

        eta_txt = f"{eta:.1f}min"

    else:

        eta_txt = "indisponivel"

    logger.info(

        "PROGRESSO | %s | %d/%d "
        "| %.1f%% | ETA aproximada=%s",

        etapa,

        atual,

        total,

        100 * atual / total,

        eta_txt,

    )


# ============================================================
# 19. EXECUCAO DOS EXPERIMENTOS
# ============================================================

def executar_etapa(

    base,

    particoes,

    contrato,

    specs,

    cp,

    etapa,

    configs,

    anos,

):

    total = len(configs) * len(anos)

    inicial = cp.contar(etapa)

    inicio_etapa = time.perf_counter()

    mostrar_progresso(

        cp,

        etapa,

        total,

        inicial,

        inicio_etapa,

    )

    for ano in anos:

        for familia in ["arvore", "catboost"]:

            for spec in ESPECIFICACOES:

                pendentes = [

                    cfg for cfg in configs

                    if (

                        cfg["familia"] == familia

                        and cfg["especificacao"] == spec

                        and not cp.existe(

                            chave_experimento(

                                etapa,

                                ano,

                                cfg,

                            )

                        )

                    )

                ]

                if not pendentes:

                    continue

                titulo(

                    f"{etapa.upper()} | "

                    f"{familia} | {spec} | {ano}"

                )

                features = specs[spec]

                treino, teste = carregar_fold(

                    base,

                    particoes,

                    features,

                    ano,

                    etapa,

                )

                inicio_pre = time.perf_counter()

                if familia == "arvore":

                    x, xv, pre, cats = (

                        preparar_arvore(

                            treino,

                            teste,

                            features,

                            contrato,

                        )

                    )

                else:

                    x, xv, pre, cats = (

                        preparar_catboost(

                            treino,

                            teste,

                            features,

                            contrato,

                        )

                    )

                tempo_pre = (

                    time.perf_counter()

                    - inicio_pre

                )

                n_treino = treino.height

                alvos = {

                    modo: alvo_treinamento(

                        treino,

                        modo,

                    )

                    for modo in {

                        cfg["modo"]

                        for cfg in pendentes

                    }

                }

                del treino

                gc.collect()

                pool_teste = None

                if familia == "catboost":

                    with MONITOR.operacao(

                        "CatBoost: Pool de avaliacao"

                    ):

                        pool_teste = Pool(

                            xv,

                            cat_features=cats,

                            thread_count=THREADS,

                        )

                # Reaproveita a preparacao dos dados
                # entre os diferentes hiperparametros.

                for modo in MODOS:

                    configs_modo = [

                        c for c in pendentes

                        if c["modo"] == modo

                    ]

                    if not configs_modo:

                        continue

                    y = alvos[modo]

                    pool_treino = None

                    if familia == "catboost":

                        with MONITOR.operacao(

                            f"CatBoost: Pool treino {modo}"

                        ):

                            pool_treino = Pool(

                                x,

                                label=y,

                                cat_features=cats,

                                thread_count=THREADS,

                            )

                    for cfg in configs_modo:

                        nome = cfg["id_config"]

                        chave = chave_experimento(

                            etapa,

                            ano,

                            cfg,

                        )

                        modelo = construir_modelo(cfg)

                        with MONITOR.operacao(

                            f"FIT {nome} | ano={ano}"

                        ):

                            inicio = time.perf_counter()

                            if familia == "arvore":

                                modelo.fit(x, y)

                            else:

                                # Nenhum eval_set da particao
                                # avaliada e usado durante fit.
                                #
                                # O numero de iteracoes e
                                # definido na grade.

                                modelo.fit(

                                    pool_treino,

                                    use_best_model=False,

                                )

                            tempo_fit = (

                                time.perf_counter()

                                - inicio

                            )

                        with MONITOR.operacao(

                            f"PREDICT {nome} | ano={ano}"

                        ):

                            inicio = time.perf_counter()

                            if familia == "arvore":

                                pred_raw = modelo.predict(xv)

                            else:

                                pred_raw = modelo.predict(

                                    pool_teste

                                )

                            pred = converter_previsao(

                                teste,

                                pred_raw,

                                modo,

                            )

                            tempo_pred = (

                                time.perf_counter()

                                - inicio

                            )

                        m = metricas_df(teste, pred)

                        extras = {}

                        if familia == "arvore":

                            extras = {

                                "n_nos":

                                    int(modelo.tree_.node_count),

                                "profundidade_real":

                                    int(modelo.get_depth()),

                                "n_folhas":

                                    int(modelo.get_n_leaves()),

                            }

                        else:

                            extras = {

                                "n_arvores":

                                    int(modelo.tree_count_),

                            }

                        registro = {

                            **cfg,

                            "hiperparametros":

                                json_canonico(

                                    cfg["hiperparametros"]

                                ),

                            "etapa": etapa,

                            "ano_alvo": ano,

                            "n_treino": n_treino,

                            "n_treino_disponivel":

                                particoes[ano][

                                    "n_treino_expostos"

                                ],

                            "n_features_originais":

                                len(features),

                            "n_features_transformadas":

                                int(x.shape[1]),

                            "tempo_preprocessamento_grupo":

                                tempo_pre,

                            "tempo_treino":

                                tempo_fit,

                            "tempo_predicao":

                                tempo_pred,

                            **extras,

                            **m,

                        }

                        logger.info(

                            "RESULTADO | %s "
                            "| ano=%d "
                            "| MAE=%.6f "
                            "| RMSE=%.6f "
                            "| R2=%.6f "
                            "| tempo=%.1fs",

                            nome,

                            ano,

                            m["mae"],

                            m["rmse"],

                            m["r2"],

                            tempo_fit,

                        )

                        if etapa == "retrospectivo":

                            salvar_finalista(

                                cfg,

                                modelo,

                                pre,

                                features,

                                cats,

                                teste,

                                pred,

                            )

                        cp.salvar(

                            chave,

                            etapa,

                            registro,

                        )

                        salvar_csv(

                            cp.carregar(etapa),

                            f"{etapa}_parcial",

                        )

                        mostrar_progresso(

                            cp,

                            etapa,

                            total,

                            inicial,

                            inicio_etapa,

                        )

                        del modelo, pred, pred_raw

                        gc.collect()

                    del pool_treino

                    gc.collect()

                del x, xv, pre, teste

                del alvos, pool_teste

                gc.collect()

    resultado = cp.carregar(etapa)

    exigir(

        len(resultado) == total,

        f"Etapa incompleta: {etapa}",

    )

    return resultado


# ============================================================
# 20. RANKING TEMPORAL
# ============================================================

def ranking_temporal(df):

    ranking = (

        df.groupby(

            [

                "id_config",

                "familia",

                "especificacao",

                "modo",

                "hiperparametros",

            ],

            as_index=False,

        )

        .agg(

            n_validacoes=(

                "ano_alvo",

                "nunique",

            ),

            mae_medio=(

                "mae",

                "mean",

            ),

            mae_dp=(

                "mae",

                "std",

            ),

            rmse_medio=(

                "rmse",

                "mean",

            ),

            r2_medio=(

                "r2",

                "mean",

            ),

            vies_medio=(

                "vies",

                "mean",

            ),

            mae_expostos_medio=(

                "mae_expostos",

                "mean",

            ),

            mae_ponderado_medio=(

                "mae_ponderado_ingressantes",

                "mean",

            ),

            vies_pct_desistentes_medio=(

                "vies_pct_desistentes",

                "mean",

            ),

            erro_pct_agregado_medio=(

                "vies_pct_desistentes",

                lambda s: s.abs().mean(),

            ),

            logloss_medio=(

                "logloss_binomial",

                "mean",

            ),

            tempo_treino_medio=(

                "tempo_treino",

                "mean",

            ),

            tempo_preprocessamento_medio=(

                "tempo_preprocessamento_grupo",

                "mean",

            ),

            tempo_predicao_medio=(

                "tempo_predicao",

                "mean",

            ),

            n_features_transformadas=(

                "n_features_transformadas",

                "max",

            ),

        )

    )

    ranking = ranking.loc[

        ranking["n_validacoes"]

        == len(ANOS_VALIDACAO)

    ].copy()

    ranking["delta_mae_vs_23"] = (

        ranking["mae_medio"]

        - BENCHMARK_MAE_VALIDACAO

    )

    return ranking.sort_values([

        "mae_medio",

        "rmse_medio",

        "id_config",

    ]).reset_index(drop=True)


def selecionar_finalistas(ranking, configs):

    mapa = {

        c["id_config"]: c

        for c in configs

    }

    finalistas = []

    for familia in ["arvore", "catboost"]:

        candidatos = ranking.loc[

            ranking["familia"] == familia

        ]

        exigir(

            len(candidatos) > 0,

            f"Nenhum candidato completo: {familia}",

        )

        melhor = candidatos.iloc[0]

        cfg = dict(

            mapa[melhor["id_config"]]

        )

        finalistas.append(cfg)

        logger.info(

            "FINALISTA | %s "
            "| MAE tuning=%.6f "
            "| RMSE tuning=%.6f "
            "| R2 tuning=%.6f",

            cfg["id_config"],

            melhor["mae_medio"],

            melhor["rmse_medio"],

            melhor["r2_medio"],

        )

    return finalistas


# ============================================================
# 21. PREVISOES DOS MODELOS ANTERIORES
# ============================================================

def carregar_predicao_anterior(

    teste,

    caminho,

):

    df = pl.read_parquet(caminho)

    exigir(

        set(KEYS).issubset(df.columns),

        "Previsoes anteriores sem chaves.",

    )

    exigir(

        "tada_prevista" in df.columns,

        "Coluna tada_prevista ausente.",

    )

    exigir(

        df.height == teste.height,

        "Amostra anterior divergente.",

    )

    exigir(

        df.select(

            pl.struct(KEYS).n_unique()

        ).item() == df.height,

        "Chaves duplicadas nas previsoes.",

    )

    if "ANO_ALVO" in df.columns:

        exigir(

            df["ANO_ALVO"].unique().to_list()

            == [ANO_RETROSPECTIVO],

            "Ano-alvo anterior divergente.",

        )

    colunas = [

        *KEYS,

        "tada_prevista",

    ]

    tem_alvo = TARGET_TADA in df.columns

    if tem_alvo:

        colunas.append(TARGET_TADA)

    anterior = df.select(colunas)

    if tem_alvo:

        anterior = anterior.rename({

            TARGET_TADA:

                "tada_observada_anterior",

        })

    alinhado = (

        teste.select([

            *KEYS,

            TARGET_TADA,

        ])

        .join(

            anterior,

            on=KEYS,

            how="inner",

            validate="1:1",

        )

        .sort(KEYS)

    )

    exigir(

        alinhado.height == teste.height,

        "Perda de linhas no alinhamento.",

    )

    if tem_alvo:

        exigir(

            np.allclose(

                arr(alinhado, TARGET_TADA),

                arr(

                    alinhado,

                    "tada_observada_anterior",

                ),

                atol=1e-8,

                rtol=0,

            ),

            "Alvos anteriores divergentes.",

        )

    pred = arr(

        alinhado,

        "tada_prevista",

    )

    exigir(

        np.isfinite(pred).all(),

        "Previsoes anteriores nao finitas.",

    )

    e = arr(teste, "exposicao")

    p = arr(teste, "QT_PERMANENCIA")

    exigir(

        (

            (pred >= -1e-8)

            & (pred <= e + 1e-8)

        ).all(),

        "Previsoes anteriores fora da exposicao.",

    )

    exigir(

        np.allclose(

            pred[p == 0],

            0,

            atol=1e-8,

            rtol=0,

        ),

        "Zeros estruturais anteriores divergentes.",

    )

    return pred


def reunir_previsoes(teste, finalistas):

    previsoes = {

        "baseline_23":

            previsao_baseline(teste),

    }

    for cfg in finalistas:

        nome = cfg["id_config"]

        caminho = (

            OUT_DIR

            / "previsoes"

            / f"{nome}.parquet"

        )

        exigir(

            caminho.is_file(),

            f"Arquivo finalista ausente: {caminho}",

        )

        previsoes[nome] = (

            carregar_predicao_anterior(

                teste,

                caminho,

            )

        )

    if not INCLUIR_MODELOS_ANTERIORES:

        return previsoes

    anteriores = {

        "ridge_m7_24": (

            REPORTS_DIR

            / "modelos_lineares_regularizados_v2"

            / "previsoes"

            / "M7_expostos_ridge_f091ce5aecda.parquet"

        ),

        "binomial_m7_25": (

            REPORTS_DIR

            / "modelo_binomial_regularizado"

            / "previsoes"

            / "M7_binomial_l2_d34604c6af27.parquet"

        ),

        "hibrido_h4_26_preliminar": (

            REPORTS_DIR

            / "modelo_hibrido_risco"

            / "previsoes"

            / "H4_hibrido_a6c07162a03f.parquet"

        ),

        "hibrido_h5_26_preliminar": (

            REPORTS_DIR

            / "modelo_hibrido_risco"

            / "previsoes"

            / "H5_hibrido_70275adbc9c0.parquet"

        ),

    }

    for nome, caminho in anteriores.items():

        if not caminho.is_file():

            logger.warning(

                "Modelo anterior nao encontrado: %s",

                caminho,

            )

            continue

        try:

            previsoes[nome] = (

                carregar_predicao_anterior(

                    teste,

                    caminho,

                )

            )

            logger.info(

                "Previsoes anteriores auditadas: %s",

                nome,

            )

        except Exception as erro:

            logger.warning(

                "Modelo anterior ignorado: %s | %s",

                nome,

                erro,

            )

    return previsoes


# ============================================================
# 22. DIAGNOSTICOS DE ERROS E EQUIDADE
# ============================================================

def diagnosticar_grupos(teste, previsoes):

    y = arr(teste, TARGET_TADA)

    p = arr(teste, "QT_PERMANENCIA")

    n = arr(teste, "QT_INGRESSANTE")

    d = arr(teste, TARGET_D)

    e = arr(teste, "exposicao")

    grupos = pd.DataFrame({

        c: (

            teste[c]

            .cast(pl.String)

            .fill_null("__AUSENTE__")

            .to_numpy()

        )

        for c in COLUNAS_GRUPOS

    })

    grupos["porte_coorte"] = pd.cut(

        n,

        bins=[

            0,

            10,

            30,

            100,

            300,

            np.inf,

        ],

        labels=[

            "1-10",

            "11-30",

            "31-100",

            "101-300",

            "301+",

        ],

        include_lowest=True,

    )

    grupos["faixa_tada_observada"] = pd.cut(

        y,

        bins=[

            -np.inf,

            0,

            1,

            5,

            10,

            20,

            100,

        ],

        labels=[

            "Zero",

            "0-1",

            "1-5",

            "5-10",

            "10-20",

            "20-100",

        ],

    )

    resultados = []

    for nome, pred in previsoes.items():

        logger.info(

            "Diagnostico por grupos: %s",

            nome,

        )

        grupos["faixa_tada_prevista"] = pd.cut(

            pred,

            bins=[

                -np.inf,

                0,

                1,

                5,

                10,

                20,

                100.000001,

            ],

            labels=[

                "Zero",

                "0-1",

                "1-5",

                "5-10",

                "10-20",

                "20-100",

            ],

        )

        for dimensao in grupos.columns:

            indices = grupos.groupby(

                dimensao,

                observed=True,

                dropna=False,

            ).indices

            for grupo, idx in indices.items():

                idx = np.asarray(

                    idx,

                    dtype=np.int64,

                )

                if len(idx) == 0:

                    continue

                m = calcular_metricas(

                    y[idx],

                    pred[idx],

                    p[idx],

                    n[idx],

                    d[idx],

                    e[idx],

                )

                resultados.append({

                    "modelo": nome,

                    "dimensao": dimensao,

                    "grupo": str(grupo),

                    "grupo_pequeno":

                        len(idx) < N_MINIMO_GRUPO,

                    "pct_zero_estrutural":

                        100 * (p[idx] == 0).mean(),

                    "total_ingressantes":

                        float(n[idx].sum()),

                    "total_expostos":

                        float(p[idx].sum()),

                    **m,

                })

    tabela = pd.DataFrame(resultados)

    salvar_csv(

        tabela,

        "11_erros_equidade_grupos_2024",

    )

    baseline = tabela.loc[

        tabela["modelo"] == "baseline_23",

        [

            "dimensao",

            "grupo",

            "mae",

            "rmse",

            "vies",

        ],

    ].rename(columns={

        "mae": "mae_baseline",

        "rmse": "rmse_baseline",

        "vies": "vies_baseline",

    })

    # Faixas de previsao diferem entre os
    # modelos e nao constituem grupos
    # diretamente comparaveis.

    comparacao = (

        tabela.loc[

            tabela["dimensao"]

            != "faixa_tada_prevista"

        ]

        .merge(

            baseline,

            on=[

                "dimensao",

                "grupo",

            ],

            how="left",

            validate="many_to_one",

        )

    )

    comparacao["delta_mae_baseline"] = (

        comparacao["mae"]

        - comparacao["mae_baseline"]

    )

    comparacao["delta_rmse_baseline"] = (

        comparacao["rmse"]

        - comparacao["rmse_baseline"]

    )

    salvar_csv(

        comparacao,

        "12_equidade_comparacao_baseline_2024",

    )

    # Disparidades descritivas.
    #
    # Nao constituem teste causal de
    # discriminacao ou injustica.

    disparidades = []

    dimensoes_equidade = [

        "CO_REGIAO",

        "CO_UF",

        "TP_CATEGORIA_ADMINISTRATIVA",

        "TP_MODALIDADE_ENSINO",

        "TP_GRAU_ACADEMICO",

        "idade_coorte",

        "porte_coorte",

    ]

    elegiveis = tabela.loc[

        ~tabela["grupo_pequeno"]

        & tabela["dimensao"].isin(

            dimensoes_equidade

        )

    ]

    for (nome, dim), g in elegiveis.groupby(

        ["modelo", "dimensao"]

    ):

        if len(g) < 2:

            continue

        melhor = g.loc[g["mae"].idxmin()]

        pior = g.loc[g["mae"].idxmax()]

        disparidades.append({

            "modelo": nome,

            "dimensao": dim,

            "n_grupos": len(g),

            "grupo_menor_mae": melhor["grupo"],

            "grupo_maior_mae": pior["grupo"],

            "mae_minimo": melhor["mae"],

            "mae_maximo": pior["mae"],

            "diferenca_mae":

                pior["mae"] - melhor["mae"],

            "razao_mae":

                pior["mae"] / melhor["mae"]

                if melhor["mae"] > 0

                else None,

        })

    salvar_csv(

        pd.DataFrame(disparidades),

        "13_disparidades_entre_grupos_2024",

    )

    return tabela


# ============================================================
# 23. CASOS PROBLEMATICOS
# ============================================================

def diagnosticar_extremos(teste, previsoes):

    y = arr(teste, TARGET_TADA)

    resultados = []

    atributos = unicos([

        *KEYS,

        *COLUNAS_GRUPOS,

        TARGET_TADA,

        TARGET_D,

        "QT_INGRESSANTE",

        "QT_PERMANENCIA",

        "exposicao",

        "FLAG_TADA_ACIMA_EXPOSICAO",

    ])

    dados = pd.DataFrame(

        teste.select(atributos).to_dict(

            as_series=False

        )

    )

    resumo = []

    for nome, pred in previsoes.items():

        erro = pred - y

        abs_erro = np.abs(erro)

        idx = np.argsort(

            -abs_erro,

            kind="stable",

        )[:N_CASOS_EXTREMOS]

        casos = dados.iloc[idx].copy()

        casos.insert(

            0,

            "modelo",

            nome,

        )

        casos["tada_prevista"] = pred[idx]

        casos["erro"] = erro[idx]

        casos["erro_absoluto"] = abs_erro[idx]

        casos["tipo_erro"] = np.where(

            erro[idx] > 0,

            "superestimacao",

            "subestimacao",

        )

        resultados.append(casos)

        resumo.append({

            "modelo": nome,

            "n": len(y),

            "n_subestimacoes":

                int((erro < -1e-8).sum()),

            "n_superestimacoes":

                int((erro > 1e-8).sum()),

            "n_exatas":

                int((abs_erro <= 1e-8).sum()),

            "n_erro_ge_10pp":

                int((abs_erro >= 10).sum()),

            "n_erro_ge_20pp":

                int((abs_erro >= 20).sum()),

            "n_erro_ge_50pp":

                int((abs_erro >= 50).sum()),

            "p90_erro_absoluto":

                float(np.quantile(abs_erro, .90)),

            "p95_erro_absoluto":

                float(np.quantile(abs_erro, .95)),

            "p99_erro_absoluto":

                float(np.quantile(abs_erro, .99)),

            "erro_maximo":

                float(abs_erro.max()),

        })

    salvar_csv(

        pd.concat(

            resultados,

            ignore_index=True,

        ),

        "14_casos_extremos_2024",

    )

    salvar_csv(

        pd.DataFrame(resumo),

        "15_resumo_erros_extremos_2024",

    )


# ============================================================
# 24. CENARIOS OPERACIONAIS DE TRIAGEM
# ============================================================

def diagnosticar_operacao(teste, previsoes):

    y = arr(teste, TARGET_TADA)

    d = arr(teste, TARGET_D)

    p = arr(teste, "QT_PERMANENCIA")

    elegiveis = np.flatnonzero(p > 0)

    total_desistentes = float(d.sum())

    resultados = []

    for fracao in CENARIOS_CAPACIDADE:

        k = max(

            1,

            math.ceil(

                len(elegiveis) * fracao

            ),

        )

        # Referencia retrospectiva idealizada.
        # Nao e modelo operacional.

        top_real = elegiveis[

            np.argsort(

                -y[elegiveis],

                kind="stable",

            )[:k]

        ]

        conjunto_real = set(

            top_real.tolist()

        )

        for nome, pred in previsoes.items():

            top = elegiveis[

                np.argsort(

                    -pred[elegiveis],

                    kind="stable",

                )[:k]

            ]

            capturados = float(d[top].sum())

            sobreposicao = len(

                conjunto_real.intersection(

                    top.tolist()

                )

            )

            cursos = (

                teste.select([

                    "CO_IES",

                    "CO_CURSO",

                ])

                .to_pandas()

                .iloc[top]

                .drop_duplicates()

                .shape[0]

            )

            resultados.append({

                "modelo": nome,

                "fracao_capacidade": fracao,

                "n_elegiveis": len(elegiveis),

                "k_trajetorias": k,

                "n_cursos_distintos": cursos,

                "limiar_score_top_k":

                    float(pred[top].min()),

                "tada_observada_media_selecionados":

                    float(y[top].mean()),

                "tada_prevista_media_selecionados":

                    float(pred[top].mean()),

                "desistentes_observados_selecionados":

                    capturados,

                "pct_desistentes_observados":

                    100 * capturados / total_desistentes

                    if total_desistentes > 0 else None,

                "sobreposicao_top_k_real":

                    sobreposicao,

                "pct_sobreposicao_top_k_real":

                    100 * sobreposicao / k,

            })

    tabela = pd.DataFrame(resultados)

    baseline = tabela.loc[

        tabela["modelo"] == "baseline_23",

        [

            "fracao_capacidade",

            "desistentes_observados_selecionados",

            "pct_sobreposicao_top_k_real",

        ],

    ].rename(columns={

        "desistentes_observados_selecionados":

            "desistentes_selecionados_baseline",

        "pct_sobreposicao_top_k_real":

            "pct_sobreposicao_baseline",

    })

    tabela = tabela.merge(

        baseline,

        on="fracao_capacidade",

        how="left",

        validate="many_to_one",

    )

    tabela["delta_desistentes_selecionados"] = (

        tabela["desistentes_observados_selecionados"]

        - tabela["desistentes_selecionados_baseline"]

    )

    tabela["delta_sobreposicao_pp"] = (

        tabela["pct_sobreposicao_top_k_real"]

        - tabela["pct_sobreposicao_baseline"]

    )

    salvar_csv(

        tabela,

        "16_cenarios_operacionais_top_k_2024",

    )

    salvar_json(

        {

            "natureza": "Cenarios ilustrativos",

            "capacidade_real_validada": False,

            "custos_validados": False,

            "regra": (

                "Priorizar trajetorias expostas "
                "pela TADA futura prevista."

            ),

            "fracao_capacidade":

                CENARIOS_CAPACIDADE,

            "observacao": (

                "Desistentes observados em "
                "trajetorias selecionadas nao "
                "equivalem a desistencias evitadas. "
                "Nao ha estimacao de efeito causal."

            ),

        },

        OUT_DIR / "16_nota_operacional.json",

    )

    return tabela


# ============================================================
# 25. COMPARACAO CONSOLIDADA
# ============================================================

def consolidar_2024(teste, previsoes):

    registros = []

    for nome, pred in previsoes.items():

        m = metricas_df(

            teste,

            pred,

        )

        registros.append({

            "modelo": nome,

            **m,

        })

    tabela = (

        pd.DataFrame(registros)

        .sort_values("mae")

        .reset_index(drop=True)

    )

    baseline = tabela.loc[

        tabela["modelo"] == "baseline_23"

    ].iloc[0]

    tabela["delta_mae_baseline"] = (

        tabela["mae"]

        - baseline["mae"]

    )

    tabela["delta_rmse_baseline"] = (

        tabela["rmse"]

        - baseline["rmse"]

    )

    tabela["delta_r2_baseline"] = (

        tabela["r2"]

        - baseline["r2"]

    )

    tabela["reducao_mae_pct"] = (

        100

        * (

            baseline["mae"]

            - tabela["mae"]

        )

        / baseline["mae"]

    )

    salvar_csv(

        tabela,

        "08_comparacao_modelos_2024",

    )

    imprimir(

        tabela[[

            "modelo",

            "n",

            "mae",

            "rmse",

            "r2",

            "vies",

            "vies_pct_desistentes",

        ]]

    )

    return tabela


def quadro_obrigatorio(

    baseline,

    ranking_full,

    retrospectivo,

):

    b = baseline.loc[

        baseline["ano_alvo"]

        == ANO_RETROSPECTIVO

    ].iloc[0]

    registros = [{

        "abordagem": "Baseline operacional",

        "familia": "baseline",

        "modelo": "hist_ies_curso_idade_risco",

        "hiperparametros": "Historico suavizado",

        "mae_2024": b["mae"],

        "rmse_2024": b["rmse"],

        "r2_2024": b["r2"],

        "tempo_treino_2024": 0.0,

        "tempo_predicao_2024":

            b["tempo_predicao"],

        "n_avaliacao": b["n"],

    }]

    for _, row in retrospectivo.iterrows():

        cfg_id = row["id_config"]

        validacao = ranking_full.loc[

            ranking_full["id_config"] == cfg_id

        ].iloc[0]

        registros.append({

            "abordagem":

                "Arvore de decisao"

                if row["familia"] == "arvore"

                else "Ensemble CatBoost",

            "familia": row["familia"],

            "modelo": cfg_id,

            "especificacao":

                row["especificacao"],

            "modo": row["modo"],

            "hiperparametros":

                row["hiperparametros"],

            "mae_validacao":

                validacao["mae_medio"],

            "rmse_validacao":

                validacao["rmse_medio"],

            "r2_validacao":

                validacao["r2_medio"],

            "mae_2024": row["mae"],

            "rmse_2024": row["rmse"],

            "r2_2024": row["r2"],

            "tempo_treino_2024":

                row["tempo_treino"],

            "tempo_predicao_2024":

                row["tempo_predicao"],

            "tempo_preprocessamento_2024":

                row["tempo_preprocessamento_grupo"],

            "n_avaliacao": row["n"],

        })

    tabela = pd.DataFrame(registros)

    salvar_csv(

        tabela,

        "09_quadro_obrigatorio_entrega_6",

    )

    return tabela


# ============================================================
# 26. TESTES SINTETICOS
# ============================================================

def testes_sinteticos():

    titulo("TESTES SINTETICOS")

    teste = pl.DataFrame({

        "QT_PERMANENCIA": [0, 10, 20, 100],

        "QT_INGRESSANTE": [100, 100, 100, 100],

        "exposicao": [0.0, 10.0, 20.0, 100.0],

    })

    pred = converter_previsao(

        teste,

        [-5, 20, -1, 150],

        "direto",

    )

    exigir(

        np.allclose(

            pred,

            [0, 10, 0, 100],

        ),

        "Falha nas restricoes da previsao direta.",

    )

    pred = converter_previsao(

        teste,

        [0.9, 0.5, -0.2, 1.5],

        "risco",

    )

    exigir(

        np.allclose(

            pred,

            [0, 5, 0, 100],

        ),

        "Falha nas restricoes de risco.",

    )

    contrato = {

        "numericas": ["x"],

        "categoricas": ["cat"],

    }

    treino = pl.DataFrame({

        "x": [1.0, 2.0, 3.0, 4.0],

        "cat": ["A", "B", "A", "B"],

    })

    avaliacao = pl.DataFrame({

        "x": [100.0, None],

        "cat": ["C", "A"],

    })

    x, xv, pre, _ = preparar_arvore(

        treino,

        avaliacao,

        ["x", "cat"],

        contrato,

    )

    categorias = (

        pre.named_transformers_[

            "categoricas"

        ].categories_[0]

    )

    mediana = (

        pre.named_transformers_[

            "numericas"

        ].statistics_[0]

    )

    exigir(

        "C" not in categorias,

        "Categoria de avaliacao aprendida.",

    )

    exigir(

        mediana == 2.5,

        "Imputacao contaminada.",

    )

    exigir(

        x.shape[1] == xv.shape[1],

        "Dimensao sintetica divergente.",

    )

    # Pequeno teste de funcionamento
    # dos dois algoritmos.

    arvore = DecisionTreeRegressor(

        max_depth=2,

        random_state=RANDOM_STATE,

    )

    arvore.fit(

        x,

        np.array([1.0, 2.0, 3.0, 4.0]),

    )

    exigir(

        np.isfinite(

            arvore.predict(xv)

        ).all(),

        "Falha sintetica na arvore.",

    )

    train_pd = converter_pandas(

        treino,

        ["x", "cat"],

        contrato,

    )

    test_pd = converter_pandas(

        avaliacao,

        ["x", "cat"],

        contrato,

    )

    cb = CatBoostRegressor(

        iterations=5,

        depth=2,

        loss_function="RMSE",

        one_hot_max_size=10,

        has_time=True,

        random_seed=RANDOM_STATE,

        thread_count=THREADS,

        allow_writing_files=False,

        verbose=False,

    )

    cb.fit(

        train_pd,

        np.array([1.0, 2.0, 3.0, 4.0]),

        cat_features=["cat"],

    )

    exigir(

        np.isfinite(

            cb.predict(test_pd)

        ).all(),

        "Falha sintetica no CatBoost.",

    )

    logger.info(

        "Todos os testes sinteticos aprovados."

    )


# ============================================================
# 27. MAIN
# ============================================================

def main():

    inicio = time.perf_counter()

    cp = None

    MONITOR.iniciar()

    try:

        titulo(

            "27 — ARVORE DE DECISAO E CATBOOST"

        )

        logger.info(

            "Diretorio atual: %s",

            Path.cwd(),

        )

        logger.info(

            "Saida: %s",

            OUT_DIR.resolve(),

        )

        logger.info(

            "Threads CatBoost: %d",

            THREADS,

        )

        logger.info(

            "Amostra tuning: %s",

            AMOSTRA_TUNING,

        )

        testes_sinteticos()

        pasta = localizar_gold()

        contrato = carregar_contrato(pasta)

        specs = construir_specs(contrato)

        configs = construir_grade()

        assinatura, configuracao = (

            construir_assinatura(

                pasta,

                specs,

                configs,

            )

        )

        cp = Checkpoint(

            OUT_DIR / "27_checkpoint.sqlite",

            assinatura,

        )

        salvar_json(

            configuracao,

            OUT_DIR / "00_configuracao_execucao.json",

        )

        salvar_json(

            specs,

            OUT_DIR / "00_especificacoes_features.json",

        )

        salvar_json(

            {

                "status": "EM_PROCESSAMENTO",

                "assinatura": assinatura,

            },

            OUT_DIR / "27_resumo_arvores.json",

        )

        base, particoes = carregar_base(

            pasta,

            contrato,

        )

        # ====================================================
        # BASELINE
        # ====================================================

        titulo(

            "ETAPA 0 — REPRODUCAO DO BASELINE"

        )

        baseline = executar_baseline(

            base,

            particoes,

        )

        # ====================================================
        # TUNING
        # ====================================================

        titulo(

            "ETAPA 1 — TUNING TEMPORAL"

        )

        tuning = executar_etapa(

            base,

            particoes,

            contrato,

            specs,

            cp,

            "tuning",

            configs,

            ANOS_VALIDACAO,

        )

        ranking_tuning = ranking_temporal(

            tuning

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

            ranking_tuning[[

                "id_config",

                "mae_medio",

                "rmse_medio",

                "r2_medio",

                "tempo_treino_medio",

            ]]

        )

        # ====================================================
        # SELECAO SEM CONSULTAR 2024
        # ====================================================

        finalistas = selecionar_finalistas(

            ranking_tuning,

            configs,

        )

        salvar_json(

            finalistas,

            OUT_DIR / "03_finalistas_congelados.json",

        )

        # ====================================================
        # CONFIRMACAO NA BASE COMPLETA
        # ====================================================

        titulo(

            "ETAPA 2 — VALIDACOES COMPLETAS"

        )

        full = executar_etapa(

            base,

            particoes,

            contrato,

            specs,

            cp,

            "validacao_completa",

            finalistas,

            ANOS_VALIDACAO,

        )

        ranking_full = ranking_temporal(full)

        salvar_csv(

            full,

            "04_validacao_completa_metricas",

        )

        salvar_csv(

            ranking_full,

            "05_validacao_completa_ranking",

        )

        imprimir(

            ranking_full[[

                "id_config",

                "mae_medio",

                "rmse_medio",

                "r2_medio",

                "tempo_treino_medio",

            ]]

        )

        # Os finalistas permanecem congelados.
        # A confirmacao em base completa nao
        # consulta o retrospectivo para selecionar.

        # ====================================================
        # AVALIACAO RETROSPECTIVA
        # ====================================================

        titulo(

            "ETAPA 3 — RETROSPECTIVO 2024"

        )

        retrospectivo = executar_etapa(

            base,

            particoes,

            contrato,

            specs,

            cp,

            "retrospectivo",

            finalistas,

            [ANO_RETROSPECTIVO],

        )

        salvar_csv(

            retrospectivo,

            "06_metricas_retrospectivas",

        )

        imprimir(

            retrospectivo[[

                "id_config",

                "n",

                "mae",

                "rmse",

                "r2",

                "vies",

                "tempo_treino",

            ]]

        )

        quadro = quadro_obrigatorio(

            baseline,

            ranking_full,

            retrospectivo,

        )

        # ====================================================
        # DIAGNOSTICOS SEM TREINAMENTO ADICIONAL
        # ====================================================

        titulo(

            "ETAPA 4 — ERROS, EQUIDADE E OPERACAO"

        )

        teste = carregar_avaliacao(

            base,

            ANO_RETROSPECTIVO,

            [],

        )

        previsoes = reunir_previsoes(

            teste,

            finalistas,

        )

        comparacao = consolidar_2024(

            teste,

            previsoes,

        )

        with MONITOR.operacao(

            "Diagnosticos por grupo"

        ):

            equidade = diagnosticar_grupos(

                teste,

                previsoes,

            )

        with MONITOR.operacao(

            "Diagnosticos de erros extremos"

        ):

            diagnosticar_extremos(

                teste,

                previsoes,

            )

        with MONITOR.operacao(

            "Cenarios operacionais"

        ):

            operacao = diagnosticar_operacao(

                teste,

                previsoes,

            )

        resumo = {

            "status": "CONCLUIDO",

            "versao": VERSAO,

            "assinatura": assinatura,

            "gold": str(pasta),

            "contrato_09_aprovado": True,

            "testes_sinteticos_aprovados": True,

            "baseline_reproduzido": True,

            "n_ajustes_tuning":

                len(tuning),

            "n_ajustes_validacao_completa":

                len(full),

            "n_ajustes_retrospectivos":

                len(retrospectivo),

            "n_registros_diagnostico_grupos":

                len(equidade),

            "n_cenarios_operacionais":

                len(operacao),

            "finalistas": finalistas,

            "ranking_validacoes":

                ranking_full.to_dict(

                    orient="records"

                ),

            "resultados_retrospectivos":

                retrospectivo.to_dict(

                    orient="records"

                ),

            "comparacao_consolidada":

                comparacao.to_dict(

                    orient="records"

                ),

            "quadro_entrega_6":

                quadro.to_dict(

                    orient="records"

                ),

            "modelos_com_previsoes_auditadas":

                list(previsoes),

            "tempo_sessao_segundos":

                time.perf_counter() - inicio,

            "nota_metodologica": (

                "Arvore de decisao individual e "
                "ensemble CatBoost avaliados com "
                "as mesmas particoes longitudinais. "

                "Treinamento apenas em trajetorias "
                "com exposicao positiva. "

                "Avaliacao na amostra completa. "

                "Selecao por MAE medio no tuning "
                "temporal de 2020 a 2023, "
                "com desempate por RMSE. "

                "Confirmacao dos finalistas "
                "na base completa. "

                "2024 nao constitui teste externo "
                "independente, pois foi previamente "
                "explorado durante o projeto. "

                "CatBoost sem target statistics "
                "nas categorias de treinamento, "
                "mediante one-hot interno e "
                "auditoria da cardinalidade. "

                "Modelos de risco minimizam erro "
                "quadratico do risco por trajetoria, "
                "nao verossimilhanca binomial. "

                "Previsoes respeitam exposicao "
                "e zeros estruturais. "

                "Diagnosticos de equidade sao "
                "descritivos, nao causais. "

                "Resultados hibridos anteriores "
                "permanecem preliminares devido "
                "ao fallback temporal pendente."

            ),

        }

        salvar_json(

            resumo,

            OUT_DIR / "27_resumo_arvores.json",

        )

        titulo(

            "27 — MODELOS DE ARVORES CONCLUIDOS"

        )

        logger.info(

            "Tempo da sessao: %.2f minutos",

            (

                time.perf_counter() - inicio

            ) / 60,

        )

        logger.info(

            "Resultados: %s",

            OUT_DIR.resolve(),

        )

    except KeyboardInterrupt:

        logger.warning(

            "Execucao interrompida. "
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

            OUT_DIR / "27_ultimo_erro.json",

        )

        raise

    finally:

        if cp is not None:

            cp.fechar()

        MONITOR.parar()


if __name__ == "__main__":

    main()