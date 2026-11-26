from __future__ import annotations

import gc
import hashlib
import json
import logging
import sqlite3
import sys
import time

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import scipy
import sklearn

from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, xlogy

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURACOES
# ============================================================

VERSAO = "25_v1"

GOLD_DIR = None

OUT_DIR = Path(
    "./reports/modelo_binomial_regularizado"
)

BASELINES_DIR = Path(
    "./reports/baselines_longitudinais"
)

ANOS_VALIDACAO = [
    2020,
    2021,
    2022,
    2023,
]

ANO_RETROSPECTIVO = 2024

RANDOM_STATE = 42

AMOSTRA_TUNING = 200_000

# None = todos os registros expostos.
LIMITE_TREINO_FINAL = None

TUNING_SPECS = [
    "M2",
    "M7",
]

ABLACAO_SPECS = [
    "M0",
    "M2",
    "M4",
    "M7",
    "M8",
    "M9",
]

ALPHAS = [
    0.0001,
    0.001,
    0.01,
    0.1,
]

MAX_ITER_TUNING = 100
MAX_ITER_FULL = 150

GTOL = 1e-5
FTOL = 1e-8

SALVAR_MODELOS = True
SALVAR_PREVISOES = True

TOP_N = 20

EPS = 1e-10

KEYS = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO",
    "ANO_ORIGEM",
]

TARGET_TADA = "TADA_FUTURA"
TARGET_D = "QT_DESISTENCIA_FUTURO"
TARGET_Q = "RISCO_DESISTENCIA_FUTURO"

BENCHMARKS = {

    "baseline_23_mae_validacao":
        3.564263690214786,

    "baseline_23_rmse_validacao":
        7.62925703444876,

    "baseline_23_r2_validacao":
        0.3121081630771244,

    "ridge_24_mae_validacao":
        3.624371,

    "ridge_24_rmse_validacao":
        7.415902,

    "ridge_24_r2_validacao":
        0.349113,

    "baseline_23_mae_2024":
        1.935115045104516,

    "baseline_23_rmse_2024":
        4.799383562367058,

    "baseline_23_r2_2024":
        0.27151678313937044,

    "ridge_m7_mae_2024":
        2.143430,

    "ridge_m7_rmse_2024":
        4.750548,

    "ridge_m7_r2_2024":
        0.286267,

}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

logger = logging.getLogger(
    "modelo_binomial_regularizado"
)

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)


# ============================================================
# UTILITARIOS
# ============================================================

def exigir(condicao, mensagem):

    if not condicao:
        raise ValueError(mensagem)


def titulo(texto):

    print("\n" + "=" * 110)
    print(texto)
    print("=" * 110)


def imprimir(df):

    print(
        df.write_csv(
            separator="\t",
            float_precision=6,
        )
    )


def unicos(lista):

    return list(dict.fromkeys(lista))


def arr(df, coluna):

    return (
        df[coluna]
        .cast(pl.Float64)
        .to_numpy()
    )


def converter_json_numpy(obj):

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    raise TypeError(
        f"Tipo não serializável: {type(obj).__name__}"
    )


def json_canonico(objeto):

    return json.dumps(
        objeto,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=converter_json_numpy,
    )


def salvar_json(objeto, caminho):

    caminho = Path(caminho)

    temporario = caminho.with_suffix(
        caminho.suffix + ".tmp"
    )

    temporario.write_text(
        json.dumps(
            objeto,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=converter_json_numpy,
        ),
        encoding="utf-8",
    )

    temporario.replace(caminho)


def salvar_csv(df, nome):

    caminho = OUT_DIR / f"{nome}.csv"

    temporario = caminho.with_suffix(
        ".csv.tmp"
    )

    df.write_csv(temporario)

    temporario.replace(caminho)


def hash_arquivo(caminho):

    h = hashlib.sha256()

    with open(caminho, "rb") as f:

        for bloco in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):

            h.update(bloco)

    return h.hexdigest()


def hash_objeto(objeto):

    return hashlib.sha256(
        json_canonico(objeto).encode("utf-8")
    ).hexdigest()


def memoria_mib(x):

    return (

        x.data.nbytes

        + x.indices.nbytes

        + x.indptr.nbytes

    ) / 1024 ** 2


# ============================================================
# LOCALIZACAO DA GOLD
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
            (pasta / a).is_file()
            for a in arquivos
        ):

            return pasta.resolve()

    raise FileNotFoundError(
        "Gold longitudinal não encontrada."
    )


# ============================================================
# CONTRATO E ESPECIFICACOES
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
        "Auditoria do 09 não aprovada.",
    )

    exigir(
        contrato["chaves"] == KEYS,
        "Chaves incompatíveis.",
    )

    exigir(
        contrato.get("teste_sintetico") is True,
        "Teste sintético do 09 não aprovado.",
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
            f"Corte {corte} não auditado.",
        )

        exigir(
            testes[corte]["truncamento"]
            and testes[corte]["perturbacao"],
            "Teste de invariância reprovado.",
        )

    autorizadas = set(
        contrato["features_autorizadas"]
    )

    proibidas = set(
        contrato["alvos_e_controles_proibidos"]
    ) | set(
        contrato["metadados_historicos_proibidos"]
    )

    exigir(
        autorizadas.isdisjoint(proibidas),
        "Interseção entre features e alvos.",
    )

    return contrato


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

    m3 = m2 + dinamicas

    m4 = m3 + [

        "CO_MUNICIPIO",
        "CO_CINE_ROTULO",

    ]

    m5 = m4 + ["CO_IES"]

    m6 = m5 + ["CO_CURSO"]

    historicas = [

        c for c in autorizadas

        if c.startswith("hist_")

    ]

    m7 = m4 + historicas

    m8 = m6 + historicas

    m9 = list(autorizadas)

    especificacoes = {

        "M0": m0,
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "M4": m4,
        "M5": m5,
        "M6": m6,
        "M7": m7,
        "M8": m8,
        "M9": m9,

    }

    for nome, cols in especificacoes.items():

        cols = unicos(cols)

        exigir(

            set(cols).issubset(
                autorizadas
            ),

            f"{nome}: feature proibida.",

        )

        especificacoes[nome] = cols

    return especificacoes


# ============================================================
# BASE LONGITUDINAL
# ============================================================

def carregar_base(pasta, contrato):

    lf_f = pl.scan_parquet(

        pasta
        / "features_longitudinais.parquet"

    )

    lf_y = pl.scan_parquet(

        pasta
        / "alvos_longitudinais.parquet"

    )

    features = contrato[
        "features_autorizadas"
    ]

    metadados = contrato[
        "metadados_historicos_proibidos"
    ]

    schema_f = lf_f.collect_schema()

    exigir(

        set(features + metadados)
        .issubset(schema_f.names()),

        "Colunas obrigatórias ausentes.",

    )

    violacoes = (

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

    )

    exigir(
        sum(violacoes.row(0)) == 0,
        "Históricos com informação futura.",
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

        auditoria = (

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
            auditoria["n"] == esperado,
            f"Dimensão incompatível: {nome}",
        )

        exigir(
            auditoria["n"] == auditoria["chaves"],
            f"Duplicatas em {nome}.",
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

                "TADA_RECONSTRUIDA_FUTURA",

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

    e = pl.col("exposicao")

    q = pl.col(TARGET_Q)

    y = pl.col(TARGET_TADA)

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

                | ((p - p.round(0)).abs() > EPS)

                | ((d - d.round(0)).abs() > EPS)

            )

            .fill_null(True)

            .sum()

            .alias("contagens_invalidas"),

            (

                e.is_null()

                | ~e.is_finite()

                | ((e - 100 * p / n).abs() > 1e-4)

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
        auditoria["n"] == contrato["n_alvos"],
        "Perda de observações.",
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
            f"Falha na auditoria: {campo}",
        )

    logger.info(
        "Auditoria binomial: %s",
        auditoria,
    )

    salvar_json(

        auditoria,

        OUT_DIR
        / "00_auditoria_binomial.json",

    )

    return base


# ============================================================
# CHECKPOINT SQLITE
# ============================================================

class Checkpoint:

    def __init__(
        self,
        caminho,
        assinatura,
    ):

        self.con = sqlite3.connect(
            caminho
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

        anterior = self.con.execute(

            """
            SELECT valor
            FROM metadata
            WHERE chave = 'assinatura'
            """

        ).fetchone()

        if anterior is None:

            self.con.execute(

                """
                INSERT INTO metadata
                VALUES ('assinatura', ?)
                """,

                (assinatura,),

            )

            self.con.commit()

        else:

            exigir(

                anterior[0] == assinatura,

                "Checkpoint incompatível. "
                "Utilize outra pasta de saída "
                "para a nova configuração.",

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

    def salvar(
        self,
        chave,
        etapa,
        resultado,
    ):

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

def construir_assinatura(
    pasta,
    especificacoes,
):

    arquivos = {}

    for nome in [

        "features_longitudinais.parquet",

        "alvos_longitudinais.parquet",

        "09_auditoria_vazamento.json",

    ]:

        logger.info(
            "Verificando hash: %s",
            nome,
        )

        arquivos[nome] = hash_arquivo(
            pasta / nome
        )

    configuracao = {

        "versao": VERSAO,

        "arquivos": arquivos,

        "anos_validacao": ANOS_VALIDACAO,

        "ano_retrospectivo": ANO_RETROSPECTIVO,

        "amostra_tuning": AMOSTRA_TUNING,

        "limite_treino_final":
            LIMITE_TREINO_FINAL,

        "tuning_specs": TUNING_SPECS,

        "ablacao_specs": ABLACAO_SPECS,

        "alphas": ALPHAS,

        "max_iter_tuning": MAX_ITER_TUNING,

        "max_iter_full": MAX_ITER_FULL,

        "gtol": GTOL,

        "ftol": FTOL,

        "seed": RANDOM_STATE,

        "salvar_modelos": SALVAR_MODELOS,

        "salvar_previsoes": SALVAR_PREVISOES,

        "especificacoes": especificacoes,

        "bibliotecas": {

            "numpy": np.__version__,

            "polars": pl.__version__,

            "pandas": pd.__version__,

            "scipy": scipy.__version__,

            "sklearn": sklearn.__version__,

        },

    }

    assinatura = hash_objeto(
        configuracao
    )

    return assinatura, configuracao


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

        "ANO_ALVO",

        TARGET_TADA,

        TARGET_D,

        TARGET_Q,

        "QT_PERMANENCIA",

        "QT_INGRESSANTE",

        "exposicao",

        "idade_coorte",

        "TADA_RECONSTRUIDA_FUTURA",

    ])

    treino_total = (

        base

        .filter(
            pl.col("ANO_ALVO") < ano
        )

        .select(colunas)

        .collect()

    )

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

        p for p in contrato["particoes"]

        if p["ano_alvo"] == ano

    )

    exigir(

        treino_total.height
        == esperado["n_treino"],

        "Treino incompatível com contrato.",

    )

    exigir(

        teste.height
        == esperado["n_avaliacao"],

        "Avaliação incompatível com contrato.",

    )

    exigir(

        treino_total["ANO_ALVO"].max()

        < teste["ANO_ALVO"].min(),

        "Sobreposição temporal.",

    )

    # O filtro utiliza exclusivamente a exposição
    # observada no ano de origem.

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

        else LIMITE_TREINO_FINAL

    )

    if limite is not None:

        if treino.height > limite:

            treino = treino.sample(

                n=limite,

                seed=RANDOM_STATE,

                shuffle=True,

            )

    exigir(
        treino.height > 0,
        "Treinamento sem expostos.",
    )

    logger.info(

        "%s | ano=%s | treino=%s | avaliação=%s",

        etapa,

        ano,

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

        c for c in features

        if c in contrato["numericas"]

    ]

    cats = [

        c for c in features

        if c in contrato["categoricas"]

    ]

    exigir(

        set(nums + cats) == set(features),

        "Feature sem classificação.",

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

    pre, nums, cats = (

        construir_preprocessador(

            features,

            contrato,

        )

    )

    x_pd = converter_pandas(

        treino,

        features,

        nums,

        cats,

    )

    # FIT somente no treinamento.

    x = csr_float32(
        pre.fit_transform(x_pd)
    )

    del x_pd

    gc.collect()

    x_pd = converter_pandas(

        teste,

        features,

        nums,

        cats,

    )

    # TRANSFORM somente na avaliação.

    xv = csr_float32(
        pre.transform(x_pd)
    )

    del x_pd

    gc.collect()

    exigir(
        x.shape[1] == xv.shape[1],
        "Dimensões incompatíveis.",
    )

    exigir(
        np.isfinite(x.data).all(),
        "Matriz de treino inválida.",
    )

    exigir(
        np.isfinite(xv.data).all(),
        "Matriz de avaliação inválida.",
    )

    logger.info(

        "CSR treino=%s | teste=%s | %.0f MiB",

        x.shape,

        xv.shape,

        memoria_mib(x) + memoria_mib(xv),

    )

    return x, xv, pre


# ============================================================
# VEROSSIMILHANCA BINOMIAL PENALIZADA
# ============================================================

class BinomialRegularizado:

    def __init__(
        self,
        alpha=0.001,
        max_iter=100,
    ):

        self.alpha = float(alpha)

        self.max_iter = int(max_iter)

    def fit(
        self,
        x,
        desistentes,
        expostos,
        inicial=None,
    ):

        d = np.asarray(
            desistentes,
            dtype=np.float64,
        )

        p = np.asarray(
            expostos,
            dtype=np.float64,
        )

        exigir(
            len(d) == len(p) == x.shape[0],
            "Dimensões incompatíveis.",
        )

        exigir(
            np.isfinite(d).all()
            and np.isfinite(p).all(),
            "Contagens não finitas.",
        )

        exigir(
            (p > 0).all(),
            "Treino contém exposição zero.",
        )

        exigir(
            ((d >= 0) & (d <= p)).all(),
            "Contagens binomiais inválidas.",
        )

        total_p = float(p.sum())

        n_features = x.shape[1]

        q_global = np.clip(

            d.sum() / total_p,

            EPS,

            1 - EPS,

        )

        theta0 = np.zeros(

            n_features + 1,

            dtype=np.float64,

        )

        theta0[0] = np.log(

            q_global / (1 - q_global)

        )

        if inicial is not None:

            exigir(

                len(inicial) == len(theta0),

                "Warm start incompatível.",

            )

            theta0 = np.asarray(

                inicial,

                dtype=np.float64,

            ).copy()

        alpha = self.alpha

        # Função objetivo:
        #
        # sum(P*log(1+exp(eta)) - D*eta)
        # --------------------------------
        #              sum(P)
        #
        # + alpha/2 * ||beta||^2
        #
        # Intercepto sem regularização.

        def objetivo(theta):

            intercepto = theta[0]

            beta = theta[1:]

            eta = (

                x @ beta

                + intercepto

            )

            q = expit(eta)

            perda = (

                np.dot(

                    p,

                    np.logaddexp(
                        0.0,
                        eta,
                    ),

                )

                - np.dot(d, eta)

            ) / total_p

            perda += (

                0.5

                * alpha

                * np.dot(beta, beta)

            )

            residuo = (

                p * q - d

            ) / total_p

            grad = np.empty_like(theta)

            grad[0] = residuo.sum()

            grad[1:] = (

                x.T @ residuo

                + alpha * beta

            )

            return float(perda), grad

        inicio = time.perf_counter()

        resultado = minimize(

            objetivo,

            theta0,

            method="L-BFGS-B",

            jac=True,

            options={

                "maxiter": self.max_iter,

                "gtol": GTOL,

                "ftol": FTOL,

                "maxcor": 10,

                "maxls": 30,

            },

        )

        self.tempo_ = (

            time.perf_counter()

            - inicio

        )

        self.theta_ = np.asarray(
            resultado.x
        )

        self.intercept_ = float(
            self.theta_[0]
        )

        self.coef_ = self.theta_[1:]

        self.convergiu_ = bool(
            resultado.success
        )

        self.n_iter_ = int(
            resultado.nit
        )

        self.n_fun_ = int(
            resultado.nfev
        )

        self.status_ = int(
            resultado.status
        )

        self.mensagem_ = str(
            resultado.message
        )

        self.objetivo_ = float(
            resultado.fun
        )

        self.gradiente_max_ = float(

            np.max(

                np.abs(resultado.jac)

            )

        )

        self.atingiu_limite_ = (

            not self.convergiu_

            and self.n_iter_

            >= self.max_iter

        )

        exigir(
            np.isfinite(self.theta_).all(),
            "Coeficientes não finitos.",
        )

        exigir(
            np.isfinite(self.objetivo_),
            "Função objetivo inválida.",
        )

        # Não admite falhas de linha de busca
        # ou erros numéricos como convergência.

        self.elegivel_ = (

            self.convergiu_

            or (

                self.atingiu_limite_

                and self.status_ == 1

                and np.isfinite(
                    self.gradiente_max_
                )

            )

        )

        logger.info(

            "Binomial | alpha=%g | iter=%s "
            "| convergiu=%s | grad=%.3e "
            "| tempo=%.1fs",

            self.alpha,

            self.n_iter_,

            self.convergiu_,

            self.gradiente_max_,

            self.tempo_,

        )

        return self

    def predict_risco(self, x):

        eta = (

            x @ self.coef_

            + self.intercept_

        )

        return expit(eta)


# ============================================================
# METRICAS
# ============================================================

def media_segura(x):

    return (

        float(np.mean(x))

        if len(x)

        else None

    )


def calcular_metricas(
    teste,
    risco,
):

    p = arr(
        teste,
        "QT_PERMANENCIA",
    )

    n = arr(
        teste,
        "QT_INGRESSANTE",
    )

    d = arr(
        teste,
        TARGET_D,
    )

    y = arr(
        teste,
        TARGET_TADA,
    )

    e = arr(
        teste,
        "exposicao",
    )

    y_reconstruida = arr(

        teste,

        "TADA_RECONSTRUIDA_FUTURA",

    )

    ativos = p > 0

    risco = np.asarray(

        risco,

        dtype=np.float64,

    )

    exigir(
        len(risco) == len(y),
        "Previsões desalinhadas.",
    )

    exigir(
        np.isfinite(risco).all(),
        "Riscos não finitos.",
    )

    exigir(

        ((risco >= 0) & (risco <= 1)).all(),

        "Risco fora do intervalo válido.",

    )

    tada = e * risco

    tada[~ativos] = 0

    exigir(

        (tada <= e + EPS).all(),

        "Previsão acima da exposição.",

    )

    erro = tada - y

    abs_erro = np.abs(erro)

    erro2 = erro ** 2

    sst = np.sum(

        (y - y.mean()) ** 2

    )

    positivos = y > 0

    q_real = d[ativos] / p[ativos]

    q_pred = risco[ativos]

    q_clip = np.clip(

        q_pred,

        EPS,

        1 - EPS,

    )

    da = d[ativos]

    pa = p[ativos]

    # Log-loss por estudante exposto.

    logloss = -np.sum(

        xlogy(da, q_clip)

        + xlogy(

            pa - da,

            1 - q_clip,

        )

    ) / pa.sum()

    # Deviance binomial relativa ao modelo saturado.

    loglik_saturado = (

        xlogy(da, q_real)

        + xlogy(

            pa - da,

            1 - q_real,

        )

    )

    loglik_modelo = (

        xlogy(da, q_clip)

        + xlogy(

            pa - da,

            1 - q_clip,

        )

    )

    deviance = 2 * np.sum(

        loglik_saturado

        - loglik_modelo

    )

    # Brier individual recuperado das contagens,
    # sem expansão em registros de estudantes.

    brier = np.sum(

        da * (1 - q_pred) ** 2

        + (pa - da) * q_pred ** 2

    ) / pa.sum()

    desistentes_pred = p * risco

    desistentes_pred[~ativos] = 0

    sst_expostos = np.sum(

        (

            y[ativos]

            - y[ativos].mean()

        ) ** 2

    )

    resultado = {

        "n": int(len(y)),

        "n_expostos": int(
            ativos.sum()
        ),

        "n_zero_estrutural": int(
            (~ativos).sum()
        ),

        "mae": float(
            abs_erro.mean()
        ),

        "rmse": float(
            np.sqrt(erro2.mean())
        ),

        "r2": (

            float(

                1 - erro2.sum() / sst

            )

            if sst > 0 else None

        ),

        "vies": float(
            erro.mean()
        ),

        "media_observada": float(
            y.mean()
        ),

        "media_prevista": float(
            tada.mean()
        ),

        "mae_expostos": media_segura(
            abs_erro[ativos]
        ),

        "rmse_expostos": float(

            np.sqrt(
                erro2[ativos].mean()
            )

        ),

        "r2_expostos": (

            float(

                1

                - erro2[ativos].sum()

                / sst_expostos

            )

            if sst_expostos > 0

            else None

        ),

        "mae_positivos": media_segura(
            abs_erro[positivos]
        ),

        "mae_zero_estrutural": media_segura(
            abs_erro[~ativos]
        ),

        "mae_ponderado_ingressantes": float(

            np.average(

                abs_erro,

                weights=n,

            )

        ),

        "pct_pred_zero": float(

            100 * (tada == 0).mean()

        ),

        "pct_pred_acima_exposicao": float(

            100 * (tada > e + EPS).mean()

        ),

        "mae_tada_reconstruida": float(

            np.abs(

                tada - y_reconstruida

            ).mean()

        ),

        "risco_mae": float(

            np.abs(q_pred - q_real).mean()

        ),

        "risco_rmse": float(

            np.sqrt(

                np.mean(

                    (q_pred - q_real) ** 2

                )

            )

        ),

        "risco_vies": float(

            (q_pred - q_real).mean()

        ),

        "risco_real_ponderado": float(

            da.sum() / pa.sum()

        ),

        "risco_pred_ponderado": float(

            np.average(

                q_pred,

                weights=pa,

            )

        ),

        "logloss_binomial": float(logloss),

        "brier_individual": float(brier),

        "deviance_binomial": float(deviance),

        "deviance_por_coorte": float(

            deviance / ativos.sum()

        ),

        "desistentes_observados": float(
            d.sum()
        ),

        "desistentes_previstos": float(
            desistentes_pred.sum()
        ),

        "vies_total_desistentes": float(

            desistentes_pred.sum()

            - d.sum()

        ),

        "mae_contagem": float(

            np.abs(

                desistentes_pred - d

            ).mean()

        ),

    }

    return resultado, tada, desistentes_pred


# ============================================================
# IDENTIFICACAO DOS EXPERIMENTOS
# ============================================================

def chave_experimento(
    etapa,
    ano,
    spec,
    alpha,
):

    return hash_objeto([

        etapa,

        int(ano),

        spec,

        float(alpha),

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

    restante = total - atual

    if novos > 0:

        decorrido = (

            time.perf_counter()

            - inicio

        )

        eta = (

            decorrido

            / novos

            * restante

            / 60

        )

        eta_texto = f"{eta:.1f} min"

    else:

        eta_texto = "indisponível"

    logger.info(

        "%s: %s/%s (%.1f%%) "
        "| ETA aproximada: %s",

        etapa,

        atual,

        total,

        100 * atual / total,

        eta_texto,

    )


# ============================================================
# TREINAMENTO E AVALIACAO
# ============================================================

def executar_etapa(
    base,
    contrato,
    especificacoes,
    checkpoint,
    etapa,
    grupos,
):

    total = sum(

        len(alphas)

        for _, _, alphas in grupos

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

    for ano, spec, alphas in grupos:

        pendentes = [

            alpha

            for alpha in alphas

            if not checkpoint.existe(

                chave_experimento(

                    etapa,

                    ano,

                    spec,

                    alpha,

                )

            )

        ]

        if not pendentes:

            logger.info(

                "Reaproveitando: %s | %s | %s",

                etapa,

                ano,

                spec,

            )

            continue

        features = especificacoes[spec]

        treino, teste = carregar_fold(

            base,

            contrato,

            features,

            ano,

            etapa,

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

        del treino

        gc.collect()

        max_iter = (

            MAX_ITER_TUNING

            if etapa == "tuning"

            else MAX_ITER_FULL

        )

        # Warm start entre os alphas do mesmo fold
        # e da mesma especificação.
        #
        # Não reaproveita coeficientes entre
        # diferentes períodos de avaliação.

        inicial_theta = None

        for alpha in sorted(

            pendentes,

            reverse=True,

        ):

            chave = chave_experimento(

                etapa,

                ano,

                spec,

                alpha,

            )

            logger.info(

                "Ajustando | %s | %s "
                "| %s | alpha=%g",

                etapa,

                ano,

                spec,

                alpha,

            )

            modelo = BinomialRegularizado(

                alpha=alpha,

                max_iter=max_iter,

            )

            modelo.fit(

                x,

                d_train,

                p_train,

                inicial=inicial_theta,

            )

            # A retomada de uma grade interrompida
            # não depende do caminho de warm start.
            # Os modelos convergentes devem alcançar
            # soluções compatíveis dentro da tolerância.

            if modelo.convergiu_:

                inicial_theta = (

                    modelo.theta_.copy()

                )

            risco = modelo.predict_risco(xv)

            metricas, tada, d_pred = (

                calcular_metricas(

                    teste,

                    risco,

                )

            )

            modelo_id = (

                f"{spec}_binomial_l2_"

                f"{chave[:12]}"

            )

            resultado = {

                "etapa": etapa,

                "ano_alvo": int(ano),

                "especificacao": spec,

                "modelo_id": modelo_id,

                "alpha": float(alpha),

                "n_treino": n_treino,

                "n_features_originais":

                    len(features),

                "n_features_transformadas":

                    int(x.shape[1]),

                "memoria_mib":

                    float(memoria_mib(x)),

                "convergiu":

                    modelo.convergiu_,

                "elegivel":

                    modelo.elegivel_,

                "atingiu_limite":

                    modelo.atingiu_limite_,

                "n_iter":

                    modelo.n_iter_,

                "n_fun":

                    modelo.n_fun_,

                "status_otimizador":

                    modelo.status_,

                "mensagem_otimizador":

                    modelo.mensagem_,

                "objetivo":

                    modelo.objetivo_,

                "gradiente_max":

                    modelo.gradiente_max_,

                "tempo_treino":

                    modelo.tempo_,

                **metricas,

            }

            if etapa == "retrospectivo":

                if SALVAR_MODELOS:

                    # Armazena parâmetros diretamente,
                    # evitando depender da importação
                    # futura de uma classe __main__.

                    pacote = {

                        "preprocessador": pre,

                        "coeficientes":
                            modelo.coef_,

                        "intercepto":
                            modelo.intercept_,

                        "features":
                            features,

                        "alpha":
                            float(alpha),

                        "ano_maximo_treino":
                            ano - 1,

                        "tipo":
                            "binomial_l2",

                    }

                    caminho = (

                        OUT_DIR

                        / "modelos"

                        / f"{modelo_id}.joblib"

                    )

                    tmp = caminho.with_suffix(
                        ".joblib.tmp"
                    )

                    joblib.dump(

                        pacote,

                        tmp,

                        compress=3,

                    )

                    tmp.replace(caminho)

                if SALVAR_PREVISOES:

                    ativos = (

                        arr(

                            teste,

                            "QT_PERMANENCIA",

                        ) > 0

                    )

                    risco_salvar = [

                        float(v)

                        if ativo else None

                        for v, ativo in zip(

                            risco,

                            ativos,

                        )

                    ]

                    saida = teste.select([

                        *KEYS,

                        "ANO_ALVO",

                        "idade_coorte",

                        TARGET_TADA,

                        TARGET_D,

                        TARGET_Q,

                        "QT_INGRESSANTE",

                        "QT_PERMANENCIA",

                        "exposicao",

                        "TADA_RECONSTRUIDA_FUTURA",

                    ])

                    saida = saida.with_columns([

                        pl.Series(

                            "risco_previsto",

                            risco_salvar,

                            dtype=pl.Float64,

                        ),

                        pl.Series(

                            "tada_prevista",

                            tada,

                            dtype=pl.Float64,

                        ),

                        pl.Series(

                            "desistentes_previstos",

                            d_pred,

                            dtype=pl.Float64,

                        ),

                    ])

                    caminho = (

                        OUT_DIR

                        / "previsoes"

                        / f"{modelo_id}.parquet"

                    )

                    tmp = caminho.with_suffix(
                        ".parquet.tmp"
                    )

                    saida.write_parquet(

                        tmp,

                        compression="zstd",

                    )

                    tmp.replace(caminho)

            # Registra o ajuste somente depois
            # de salvar os artefatos obrigatórios.

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

            del modelo

            gc.collect()

        del x, xv, pre, teste

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

            pl.col("risco_mae")

            .mean()

            .alias("risco_mae_medio"),

            pl.col("tempo_treino")

            .mean()

            .alias("tempo_medio"),

            pl.col("convergiu")

            .all()

            .alias("convergiu_todos"),

            pl.col("elegivel")

            .all()

            .alias("elegivel_todos"),

            pl.col("atingiu_limite")

            .sum()

            .alias("n_limite"),

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

                - BENCHMARKS[
                    "baseline_23_mae_validacao"
                ]

            ).alias("delta_mae_vs_23"),

            (

                pl.col("rmse_medio")

                - BENCHMARKS[
                    "baseline_23_rmse_validacao"
                ]

            ).alias("delta_rmse_vs_23"),

            (

                pl.col("r2_medio")

                - BENCHMARKS[
                    "baseline_23_r2_validacao"
                ]

            ).alias("delta_r2_vs_23"),

        ])

        .sort("mae_medio")

    )

    return ranking


# ============================================================
# SELECAO DE HIPERPARAMETROS
# ============================================================

def selecionar_alphas(ranking):

    elegiveis = ranking.filter(

        pl.col("elegivel_todos")

    )

    exigir(

        elegiveis.height > 0,

        "Nenhuma configuração binomial elegível.",

    )

    convergentes = elegiveis.filter(

        pl.col("convergiu_todos")

    )

    if convergentes.height > 0:

        candidatos = convergentes

    else:

        candidatos = elegiveis

        logger.warning(

            "Seleção baseada em soluções "
            "que atingiram o orçamento "
            "de iterações."

        )

    selecionados = {}

    criterios = [

        ("mae", "mae_medio"),

        ("rmse", "rmse_medio"),

        ("logloss", "logloss_medio"),

    ]

    for nome, coluna in criterios:

        melhor = (

            candidatos

            .sort(coluna)

            .row(0, named=True)

        )

        selecionados[nome] = melhor

    alphas = sorted({

        float(m["alpha"])

        for m in selecionados.values()

    })

    return alphas, selecionados


# ============================================================
# FINALISTAS
# ============================================================

def selecionar_finalistas(ranking):

    elegiveis = ranking.filter(

        pl.col("elegivel_todos")

    )

    exigir(

        elegiveis.height > 0,

        "Nenhum finalista elegível.",

    )

    convergentes = elegiveis.filter(

        pl.col("convergiu_todos")

    )

    candidatos = (

        convergentes

        if convergentes.height > 0

        else elegiveis

    )

    criterios = [

        ("mae", "mae_medio", False),

        ("rmse", "rmse_medio", False),

        ("r2", "r2_medio", True),

        ("logloss", "logloss_medio", False),

        ("brier", "brier_medio", False),

    ]

    grupos = {}

    for criterio, coluna, desc in criterios:

        melhor = (

            candidatos

            .sort(

                coluna,

                descending=desc,

            )

            .row(0, named=True)

        )

        chave = (

            melhor["especificacao"],

            float(melhor["alpha"]),

        )

        if chave not in grupos:

            grupos[chave] = {

                "especificacao":
                    chave[0],

                "alpha":
                    chave[1],

                "criterios": [],

                "metricas_validacao":
                    melhor,

            }

        grupos[chave]["criterios"].append(
            criterio
        )

    # Garante uma referência parcimoniosa M7,
    # desde que haja configuração elegível.

    m7 = candidatos.filter(

        pl.col("especificacao") == "M7"

    )

    if m7.height > 0:

        melhor = (

            m7.sort("mae_medio")

            .row(0, named=True)

        )

        chave = (

            "M7",

            float(melhor["alpha"]),

        )

        if chave not in grupos:

            grupos[chave] = {

                "especificacao": "M7",

                "alpha": chave[1],

                "criterios": [],

                "metricas_validacao":
                    melhor,

            }

        grupos[chave]["criterios"].append(
            "parcimonia_M7"
        )

    return list(grupos.values())


# ============================================================
# COMPARACAO COM BASELINES
# ============================================================

def comparar_baselines(
    validacao,
    retrospectivo,
):

    caminho = (

        BASELINES_DIR

        / "01_metricas_completas.csv"

    )

    if not caminho.is_file():

        logger.warning(

            "CSV do script 23 não encontrado. "
            "Comparação por ano indisponível."

        )

        return

    baselines = (

        pl.read_csv(caminho)

        .filter(

            pl.col("modelo").is_in([

                "hist_ies_curso_idade_risco",

                "hist_curso_risco",

            ])

        )

        .select([

            "ano_alvo",

            pl.col("modelo")
            .alias("baseline"),

            pl.col("n")
            .alias("n_baseline"),

            pl.col("mae")
            .alias("mae_baseline"),

            pl.col("rmse")
            .alias("rmse_baseline"),

            pl.col("r2")
            .alias("r2_baseline"),

        ])

    )

    resultados = pl.concat(

        [

            validacao,

            retrospectivo,

        ],

        how="diagonal_relaxed",

    )

    comparacao = (

        resultados

        .join(

            baselines,

            on="ano_alvo",

            how="inner",

        )

        .with_columns([

            (

                pl.col("mae")

                - pl.col("mae_baseline")

            ).alias("delta_mae"),

            (

                pl.col("rmse")

                - pl.col("rmse_baseline")

            ).alias("delta_rmse"),

            (

                pl.col("r2")

                - pl.col("r2_baseline")

            ).alias("delta_r2"),

        ])

    )

    exigir(

        comparacao.filter(

            pl.col("n")

            != pl.col("n_baseline")

        ).height == 0,

        "Amostras incompatíveis com script 23.",

    )

    salvar_csv(

        comparacao,

        "08_comparacao_baselines",

    )


# ============================================================
# TESTES SINTETICOS
# ============================================================

def testes_sinteticos():

    titulo(
        "TESTES SINTETICOS"
    )

    # Modelo somente com intercepto.
    # O risco estimado deve ser igual a sum(D)/sum(P).

    x = sparse.csr_matrix(

        np.zeros(

            (4, 1),

            dtype=np.float32,

        )

    )

    p = np.array([

        10.0,

        20.0,

        30.0,

        40.0,

    ])

    d = np.array([

        1.0,

        4.0,

        9.0,

        16.0,

    ])

    esperado = d.sum() / p.sum()

    modelo = BinomialRegularizado(

        alpha=0.001,

        max_iter=100,

    ).fit(x, d, p)

    q = modelo.predict_risco(x)

    exigir(

        np.allclose(

            q,

            esperado,

            atol=1e-5,

        ),

        "Falha no teste de intercepto.",

    )

    # Gradiente e otimização em um problema
    # com covariável conhecida.

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    z = rng.normal(
        size=5000
    )

    prob = expit(

        -1.0

        + 0.7 * z

    )

    p2 = rng.integers(

        10,

        100,

        size=len(z),

    )

    d2 = rng.binomial(

        p2,

        prob,

    )

    x2 = sparse.csr_matrix(

        z.reshape(-1, 1)

        .astype(np.float32)

    )

    modelo2 = BinomialRegularizado(

        alpha=0.0001,

        max_iter=100,

    ).fit(x2, d2, p2)

    exigir(

        abs(modelo2.coef_[0] - 0.7)

        < 0.1,

        "Falha na recuperação do coeficiente.",

    )

    # Verifica categoria desconhecida e
    # imputação exclusivamente pelo treinamento.

    contrato = {

        "numericas": ["x"],

        "categoricas": ["curso"],

    }

    pre, _, _ = construir_preprocessador(

        ["x", "curso"],

        contrato,

    )

    treino = pd.DataFrame({

        "x": [

            1.0,

            2.0,

            3.0,

        ],

        "curso": [

            "A",

            "B",

            "A",

        ],

    })

    teste = pd.DataFrame({

        "x": [

            100.0,

            np.nan,

        ],

        "curso": [

            "C",

            "A",

        ],

    })

    xt = csr_float32(

        pre.fit_transform(treino)

    )

    xv = csr_float32(

        pre.transform(teste)

    )

    categorias = (

        pre.named_transformers_[
            "categoricas"
        ]

        .categories_[0]

    )

    exigir(
        "C" not in categorias,
        "Categoria futura aprendida.",
    )

    mediana = (

        pre.named_transformers_[
            "numericas"
        ]

        .named_steps["imputer"]

        .statistics_[0]

    )

    exigir(
        mediana == 2.0,
        "Imputação contaminada.",
    )

    exigir(

        xt.shape[1] == xv.shape[1],

        "Encoding incompatível.",

    )

    # Exposição e zeros estruturais.

    e = np.array([

        0.0,

        10.0,

        50.0,

    ])

    q = np.array([

        0.7,

        0.5,

        0.2,

    ])

    pred = e * q

    exigir(

        np.allclose(

            pred,

            [0.0, 5.0, 10.0],

        ),

        "Falha na reconstrução da TADA.",

    )

    logger.info(
        "Testes sintéticos aprovados."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    inicio_total = time.perf_counter()

    titulo(

        "25 — REGRESSAO BINOMIAL REGULARIZADA"

    )

    OUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )

    for sub in [

        "modelos",

        "previsoes",

    ]:

        (

            OUT_DIR / sub

        ).mkdir(

            parents=True,

            exist_ok=True,

        )

    testes_sinteticos()

    pasta = localizar_gold()

    contrato = carregar_contrato(pasta)

    especificacoes = (

        construir_especificacoes(

            contrato

        )

    )

    assinatura, configuracao = (

        construir_assinatura(

            pasta,

            especificacoes,

        )

    )

    checkpoint = Checkpoint(

        OUT_DIR / "25_checkpoint.sqlite",

        assinatura,

    )

    salvar_json(

        configuracao,

        OUT_DIR
        / "00_configuracao_execucao.json",

    )

    salvar_json(

        especificacoes,

        OUT_DIR
        / "00_especificacoes_features.json",

    )

    manifesto = (

        OUT_DIR

        / "25_resumo_binomial.json"

    )

    salvar_json(

        {

            "status": "EM_PROCESSAMENTO",

            "assinatura": assinatura,

        },

        manifesto,

    )

    try:

        base = carregar_base(

            pasta,

            contrato,

        )

        # ====================================================
        # ETAPA 1: TUNING
        # ====================================================

        titulo(
            "ETAPA 1 — TUNING BINOMIAL"
        )

        grupos_tuning = [

            (

                ano,

                spec,

                ALPHAS,

            )

            for spec in TUNING_SPECS

            for ano in ANOS_VALIDACAO

        ]

        tuning = executar_etapa(

            base,

            contrato,

            especificacoes,

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

        alphas, selecao = (

            selecionar_alphas(

                ranking_tuning

            )

        )

        salvar_json(

            {

                "alphas": alphas,

                "selecao": selecao,

            },

            OUT_DIR

            / "03_alphas_selecionados.json",

        )

        logger.info(

            "Alphas selecionados: %s",

            alphas,

        )

        # ====================================================
        # ETAPA 2: ABLACAO
        # ====================================================

        titulo(

            "ETAPA 2 — ABLACAO BINOMIAL"

        )

        grupos_ablacao = [

            (

                ano,

                spec,

                alphas,

            )

            for spec in ABLACAO_SPECS

            for ano in ANOS_VALIDACAO

        ]

        ablacao = executar_etapa(

            base,

            contrato,

            especificacoes,

            checkpoint,

            "ablacao",

            grupos_ablacao,

        )

        ranking = ranking_temporal(
            ablacao
        )

        salvar_csv(

            ablacao,

            "04_ablacao_metricas",

        )

        salvar_csv(

            ranking,

            "05_ablacao_ranking",

        )

        titulo(

            "RANKING BINOMIAL — VALIDACOES"

        )

        imprimir(

            ranking.head(TOP_N)

        )

        # ====================================================
        # ETAPA 3: CONGELAMENTO DOS FINALISTAS
        # ====================================================

        finalistas = (

            selecionar_finalistas(

                ranking

            )

        )

        salvar_json(

            finalistas,

            OUT_DIR

            / "09_finalistas_congelados.json",

        )

        grupos_finais = [

            (

                ANO_RETROSPECTIVO,

                f["especificacao"],

                [f["alpha"]],

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

            especificacoes,

            checkpoint,

            "retrospectivo",

            grupos_finais,

        )

        retrospectivo = (

            retrospectivo

            .with_columns([

                (

                    pl.col("mae")

                    - BENCHMARKS[
                        "baseline_23_mae_2024"
                    ]

                ).alias("delta_mae_vs_23"),

                (

                    pl.col("rmse")

                    - BENCHMARKS[
                        "baseline_23_rmse_2024"
                    ]

                ).alias("delta_rmse_vs_23"),

                (

                    pl.col("r2")

                    - BENCHMARKS[
                        "baseline_23_r2_2024"
                    ]

                ).alias("delta_r2_vs_23"),

                (

                    pl.col("mae")

                    - BENCHMARKS[
                        "ridge_m7_mae_2024"
                    ]

                ).alias("delta_mae_vs_ridge"),

                (

                    pl.col("rmse")

                    - BENCHMARKS[
                        "ridge_m7_rmse_2024"
                    ]

                ).alias("delta_rmse_vs_ridge"),

                (

                    pl.col("r2")

                    - BENCHMARKS[
                        "ridge_m7_r2_2024"
                    ]

                ).alias("delta_r2_vs_ridge"),

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

                "convergiu",

                "mae",

                "rmse",

                "r2",

                "vies",

                "logloss_binomial",

                "brier_individual",

                "delta_mae_vs_23",

                "delta_rmse_vs_23",

                "delta_r2_vs_23",

            ])

        )

        # ====================================================
        # CONSOLIDACAO
        # ====================================================

        comparar_baselines(

            ablacao,

            retrospectivo,

        )

        resumo = {

            "status": "CONCLUIDO",

            "versao": VERSAO,

            "assinatura": assinatura,

            "gold": str(pasta),

            "contrato_09_aprovado": True,

            "testes_sinteticos_aprovados": True,

            "n_features_autorizadas":

                len(

                    contrato[
                        "features_autorizadas"
                    ]

                ),

            "n_ajustes_tuning":

                tuning.height,

            "n_ajustes_ablacao":

                ablacao.height,

            "n_ajustes_retrospectivos":

                retrospectivo.height,

            "alphas_selecionados":

                alphas,

            "finalistas":

                finalistas,

            "ranking_validacoes":

                ranking.head(TOP_N)

                .to_dicts(),

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

                "Regressão binomial agregada "
                "com regularização L2. "

                "Verossimilhança ponderada "
                "pela população exposta. "

                "Intercepto sem penalização. "

                "Otimização L-BFGS-B "
                "com gradiente analítico. "

                "Imputação, escala e encoding "
                "ajustados exclusivamente "
                "no treinamento. "

                "Treinamento apenas entre "
                "observações expostas; "
                "avaliação na amostra completa. "

                "Seleção pelas validações "
                "de 2020 a 2023. "

                "2024 constitui avaliação "
                "retrospectiva previamente explorada. "

                "Soluções que atingiram o limite "
                "de iterações são identificadas. "

                "A hipótese binomial não exclui "
                "a possibilidade de sobredispersão "
                "e dependência entre estudantes."

            ),

        }

        salvar_json(

            resumo,

            manifesto,

        )

        titulo(

            "25 — REGRESSAO BINOMIAL CONCLUIDA"

        )

        print(

            json.dumps(

                resumo,

                indent=2,

                ensure_ascii=False,

            )

        )

        logger.info(

            "Concluído: %s",

            OUT_DIR.resolve(),

        )

    finally:

        checkpoint.fechar()


if __name__ == "__main__":

    main()