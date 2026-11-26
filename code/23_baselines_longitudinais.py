from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl


# ============================================================
# CONFIGURACOES
# ============================================================

GOLD_DIR = None
OUT_DIR = Path("./reports/baselines_longitudinais")

ANOS_VALIDACAO = [2020, 2021, 2022, 2023]
ANO_RETROSPECTIVO = 2024

ANOS_AVALIACAO = ANOS_VALIDACAO + [ANO_RETROSPECTIVO]

TOL = 0.01

SALVAR_PREVISOES = True
TOP_N = 15

KEYS = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO",
    "ANO_ORIGEM",
]

TARGET = "TADA_FUTURA"

NIVEIS_HISTORICOS = [
    "global",
    "idade",
    "ies",
    "curso",
    "ies_curso",
    "ies_idade",
    "curso_idade",
    "ies_curso_idade",
    "cine_idade",
    "modalidade_idade",
]

REFERENCIA_08 = {
    "ano_alvo": 2024,
    "mae": 1.935115387746275,
    "rmse": 4.799383209904068,
    "r2": 0.27151689013784663,
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

logger = logging.getLogger("baselines_longitudinais")

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)


# ============================================================
# LOCALIZACAO DOS DADOS
# ============================================================

def localizar_gold():

    nomes = [
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

        if all((pasta / nome).is_file() for nome in nomes):
            return pasta.resolve()

    raise FileNotFoundError(
        "Arquivos da Gold longitudinal não encontrados."
    )


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

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


def exigir(condicao, mensagem):

    if not condicao:
        raise ValueError(mensagem)


def salvar_json(objeto, caminho):

    with open(caminho, "w", encoding="utf-8") as f:

        json.dump(
            objeto,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def salvar_csv(df, nome):

    caminho = OUT_DIR / f"{nome}.csv"

    df.write_csv(caminho)

    logger.info(
        "%s: %s linhas",
        nome,
        f"{df.height:,}",
    )


def array(df, coluna):

    return (
        df[coluna]
        .cast(pl.Float64)
        .to_numpy()
    )


def preencher(x, referencia):

    x = np.asarray(x, dtype=np.float64)

    return np.where(
        np.isfinite(x),
        x,
        referencia,
    )


def limitar(pred, exposicao):

    return np.minimum(
        np.maximum(pred, 0),
        exposicao,
    )


def aplicar_zero(pred, exposicao):

    return np.where(
        exposicao == 0,
        0,
        np.clip(pred, 0, 100),
    )


def media_segura(x):

    return (
        float(np.mean(x))
        if len(x)
        else None
    )


# ============================================================
# CONTRATO DE DADOS
# ============================================================

def carregar_contrato(pasta):

    caminho = pasta / "09_auditoria_vazamento.json"

    with open(caminho, encoding="utf-8") as f:
        contrato = json.load(f)

    exigir(
        contrato["status"] == "APROVADO",
        "A auditoria do script 09 não está aprovada.",
    )

    exigir(
        contrato["chaves"] == KEYS,
        "Chaves incompatíveis com o contrato.",
    )

    exigir(
        contrato.get("teste_sintetico") is True,
        "Teste sintético não aprovado.",
    )

    testes = contrato.get("testes_invariancia", [])

    exigir(
        bool(testes),
        "Testes de invariância não encontrados.",
    )

    for teste in testes:

        exigir(
            teste.get("truncamento") is True
            and teste.get("perturbacao") is True,
            f"Falha na auditoria do corte {teste['corte']}.",
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
        not autorizadas.intersection(proibidas),
        "Há variáveis simultaneamente permitidas e proibidas.",
    )

    logger.info("Contrato de dados aprovado.")

    return contrato


# ============================================================
# CARREGAMENTO DOS PARQUETS
# ============================================================

def carregar_base(pasta, contrato):

    lf_features = pl.scan_parquet(
        pasta / "features_longitudinais.parquet"
    )

    lf_alvos = pl.scan_parquet(
        pasta / "alvos_longitudinais.parquet"
    )

    schema_features = lf_features.collect_schema()
    schema_alvos = lf_alvos.collect_schema()

    autorizadas = set(
        contrato["features_autorizadas"]
    )

    necessarias = [
        "TADA",
        "TADA_lag1",
        "TADA_lag2",
        "TADA_lag3",
        "TADA_delta1",
        "TADA_media3",
        "TADA_inclinacao2",
        "QT_INGRESSANTE",
        "QT_PERMANENCIA",
        "idade_coorte",
        "exposicao",
        "zero_estrutural",
        "risco_atual",
    ]

    historicas = []

    for nivel in NIVEIS_HISTORICOS:

        for sufixo in [
            "n",
            "expostos",
            "ativos",
            "risco",
            "risco_bruto",
            "tada",
            "tada_bruta",
        ]:

            coluna = f"hist_{nivel}_{sufixo}"

            if coluna in schema_features:
                historicas.append(coluna)

    necessarias += historicas

    exigir(
        set(necessarias).issubset(autorizadas),
        "Feature não autorizada solicitada.",
    )

    faltantes = (
        set(KEYS + necessarias)
        - set(schema_features.names())
    )

    exigir(
        not faltantes,
        f"Features ausentes: {faltantes}",
    )

    colunas_alvos = [
        *KEYS,
        "ANO_ALVO",
        "TADA_FUTURA",
        "QT_DESISTENCIA_FUTURO",
        "RISCO_DESISTENCIA_FUTURO",
        "TADA_RECONSTRUIDA_FUTURA",
        "FLAG_TADA_ACIMA_EXPOSICAO",
    ]

    exigir(
        set(colunas_alvos).issubset(
            schema_alvos.names()
        ),
        "Colunas obrigatórias ausentes nos alvos.",
    )

    # Metadados históricos utilizados exclusivamente
    # para verificar a disponibilidade temporal.

    metadados = contrato[
        "metadados_historicos_proibidos"
    ]

    exigir(
        set(metadados).issubset(
            schema_features.names()
        ),
        "Metadados históricos de auditoria ausentes.",
    )

    logger.info("Verificando fronteiras temporais.")

    violacoes = (
        lf_features
        .select([
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

    n_violacoes = sum(
        int(v or 0)
        for v in violacoes.row(0)
    )

    exigir(
        n_violacoes == 0,
        f"Violações temporais: {n_violacoes}",
    )

    logger.info("Carregando features selecionadas.")

    features = (
        lf_features
        .select(KEYS + necessarias)
        .collect()
    )

    alvos = (
        lf_alvos
        .select(colunas_alvos)
        .collect()
    )

    exigir(
        features.height
        == contrato["n_features_linhas"],
        "Quantidade de features incompatível.",
    )

    exigir(
        alvos.height == contrato["n_alvos"],
        "Quantidade de alvos incompatível.",
    )

    exigir(
        not features.select(KEYS)
        .is_duplicated().any(),
        "Features com chaves duplicadas.",
    )

    exigir(
        not alvos.select(KEYS)
        .is_duplicated().any(),
        "Alvos com chaves duplicadas.",
    )

    logger.info("Relacionando features e alvos.")

    base = features.join(
        alvos,
        on=KEYS,
        how="inner",
        validate="1:1",
    )

    exigir(
        base.height == alvos.height,
        "Perda de observações durante o join.",
    )

    exigir(
        base.filter(
            pl.col("ANO_ALVO")
            != pl.col("ANO_ORIGEM") + 1
        ).height == 0,
        "Horizonte preditivo inválido.",
    )

    exigir(
        base.filter(
            pl.col(TARGET).is_null()
            | ~pl.col(TARGET).is_finite()
            | ~pl.col(TARGET).is_between(0, 100)
        ).height == 0,
        "Alvos inválidos.",
    )

    exigir(
        base.filter(
            pl.col("exposicao").is_null()
            | ~pl.col("exposicao").is_finite()
            | ~pl.col("exposicao").is_between(0, 100)
        ).height == 0,
        "Exposição inválida.",
    )

    exigir(
        base.filter(
            pl.col("QT_INGRESSANTE").is_null()
            | (pl.col("QT_INGRESSANTE") <= 0)
        ).height == 0,
        "Ingressantes inválidos.",
    )

    exigir(
        base.filter(
            (
                pl.col("exposicao")
                - 100
                * pl.col("QT_PERMANENCIA")
                / pl.col("QT_INGRESSANTE")
            ).abs() > 1e-6
        ).height == 0,
        "Exposição incompatível com as contagens.",
    )

    exigir(
        base.filter(
            (
                pl.col("zero_estrutural")
                .cast(pl.Boolean)
                != (pl.col("QT_PERMANENCIA") == 0)
            )
        ).height == 0,
        "Indicador de zero estrutural incompatível.",
    )

    exigir(
        base.filter(
            (pl.col("exposicao") == 0)
            & (pl.col(TARGET) > 0)
        ).height == 0,
        "Violação da regra de zeros estruturais.",
    )

    del features, alvos

    gc.collect()

    logger.info(
        "Base de avaliação: %s pares.",
        f"{base.height:,}",
    )

    return base


# ============================================================
# METRICAS
# ============================================================

def metricas(
    y,
    pred,
    exposicao,
    ingressantes,
):

    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    exigir(
        len(y) == len(pred),
        "Dimensões incompatíveis.",
    )

    exigir(
        np.isfinite(pred).all(),
        "Previsões não finitas.",
    )

    erro = pred - y

    abs_erro = np.abs(erro)
    sq_erro = erro ** 2

    sst = np.sum(
        (y - y.mean()) ** 2
    )

    expostos = exposicao > 0
    positivos = y > 0
    estruturais = exposicao == 0

    zeros_expostos = (
        expostos & (y == 0)
    )

    return {

        "n": int(len(y)),

        "mae": float(abs_erro.mean()),

        "rmse": float(
            np.sqrt(sq_erro.mean())
        ),

        "r2": (
            float(1 - sq_erro.sum() / sst)
            if sst > 0 else None
        ),

        "vies": float(erro.mean()),

        "media_observada": float(y.mean()),

        "media_prevista": float(pred.mean()),

        "mae_expostos": media_segura(
            abs_erro[expostos]
        ),

        "rmse_expostos": (
            float(
                np.sqrt(
                    sq_erro[expostos].mean()
                )
            )
            if expostos.any()
            else None
        ),

        "mae_positivos": media_segura(
            abs_erro[positivos]
        ),

        "mae_zero_estrutural": media_segura(
            abs_erro[estruturais]
        ),

        "mae_zero_exposto": media_segura(
            abs_erro[zeros_expostos]
        ),

        "mae_ponderado_ingressantes": float(
            np.average(
                abs_erro,
                weights=ingressantes,
            )
        ),

        "pct_pred_zero": float(
            (pred == 0).mean() * 100
        ),

        "pct_pred_acima_exposicao": float(
            (
                pred > exposicao + TOL
            ).mean() * 100
        ),

        "pct_observado_acima_exposicao": float(
            (
                y > exposicao + TOL
            ).mean() * 100
        ),

    }


# ============================================================
# BASELINES CONVENCIONAIS
# ============================================================

def construir_baselines(treino, teste):

    predicoes = {}
    coberturas = {}

    e = array(teste, "exposicao")

    def registrar(
        nome,
        pred,
        cobertura=1.0,
    ):

        pred = np.asarray(
            pred,
            dtype=np.float64,
        )

        predicoes[nome] = np.clip(
            pred,
            0,
            100,
        )

        coberturas[nome] = float(cobertura)

    def registrar_variantes(
        nome,
        pred,
        cobertura=1.0,
    ):

        registrar(
            nome,
            pred,
            cobertura,
        )

        registrar(
            nome + "_zero_estrutural",
            aplicar_zero(pred, e),
            cobertura,
        )

        registrar(
            nome + "_limitado",
            limitar(pred, e),
            cobertura,
        )

    y_treino = array(treino, TARGET)

    media_global = float(
        y_treino.mean()
    )

    mediana_global = float(
        np.median(y_treino)
    )

    persistencia = array(teste, "TADA")

    # --------------------------------------------------------
    # Convencionais
    # --------------------------------------------------------

    registrar(
        "zero",
        np.zeros(teste.height),
    )

    registrar_variantes(
        "media_treino",
        np.full(
            teste.height,
            media_global,
        ),
    )

    registrar_variantes(
        "mediana_treino",
        np.full(
            teste.height,
            mediana_global,
        ),
    )

    registrar_variantes(
        "persistencia",
        persistencia,
    )

    # --------------------------------------------------------
    # Defasagens e dinamica
    # --------------------------------------------------------

    for coluna in [
        "TADA_lag1",
        "TADA_lag2",
        "TADA_lag3",
        "TADA_media3",
    ]:

        valores = array(
            teste,
            coluna,
        )

        cobertura = np.isfinite(
            valores
        ).mean()

        registrar_variantes(
            coluna.lower(),
            preencher(
                valores,
                persistencia,
            ),
            cobertura,
        )

    delta = array(
        teste,
        "TADA_delta1",
    )

    registrar_variantes(
        "tendencia_delta1",
        persistencia + preencher(delta, 0),
        np.isfinite(delta).mean(),
    )

    inclinacao = array(
        teste,
        "TADA_inclinacao2",
    )

    registrar_variantes(
        "tendencia_inclinacao2",
        persistencia
        + preencher(inclinacao, 0),
        np.isfinite(inclinacao).mean(),
    )

    # --------------------------------------------------------
    # Históricos por idade ajustados somente no treino
    # --------------------------------------------------------

    hist_idade = (
        treino
        .group_by("idade_coorte")
        .agg([

            pl.col(TARGET)
            .mean()
            .alias("_media_idade"),

            pl.col(TARGET)
            .median()
            .alias("_mediana_idade"),

        ])
    )

    tmp = teste.select(
        ["idade_coorte"]
    ).join(
        hist_idade,
        on="idade_coorte",
        how="left",
        validate="m:1",
        maintain_order="left",
    )

    registrar_variantes(
        "media_idade_treino",
        preencher(
            array(tmp, "_media_idade"),
            media_global,
        ),
    )

    registrar_variantes(
        "mediana_idade_treino",
        preencher(
            array(tmp, "_mediana_idade"),
            mediana_global,
        ),
    )

    # ========================================================
    # HISTORICOS DA GOLD
    # ========================================================

    referencia_tada = preencher(
        array(teste, "hist_global_tada"),
        media_global,
    )

    referencia_idade = preencher(
        array(teste, "hist_idade_tada"),
        referencia_tada,
    )

    # Risco global calculado com informações
    # disponíveis exclusivamente no treinamento.

    ativos = treino.filter(
        pl.col("QT_PERMANENCIA") > 0
    )

    total_expostos = (
        ativos["QT_PERMANENCIA"].sum()
    )

    exigir(
        total_expostos > 0,
        "Treinamento sem população exposta.",
    )

    risco_global_treino = (
        ativos["QT_DESISTENCIA_FUTURO"].sum()
        / total_expostos
    )

    exigir(
        0 <= risco_global_treino <= 1,
        "Risco global incompatível.",
    )

    risco_global = preencher(
        array(teste, "hist_global_risco"),
        risco_global_treino,
    )

    risco_idade = preencher(
        array(teste, "hist_idade_risco"),
        risco_global,
    )

    registrar(
        "risco_global_treino",
        e * risco_global_treino,
    )

    # --------------------------------------------------------
    # Risco atual da própria trajetória
    # --------------------------------------------------------

    risco_atual = array(
        teste,
        "risco_atual",
    )

    registrar(
        "risco_atual_persistencia",
        e * np.clip(
            preencher(
                risco_atual,
                risco_idade,
            ),
            0,
            1,
        ),
        np.isfinite(risco_atual).mean(),
    )

    # --------------------------------------------------------
    # Históricos institucionais e acadêmicos
    # --------------------------------------------------------

    for nivel in NIVEIS_HISTORICOS:

        prefixo = f"hist_{nivel}"

        n_col = prefixo + "_n"
        p_col = prefixo + "_expostos"

        n_hist = (
            array(teste, n_col)
            if n_col in teste.columns
            else np.zeros(teste.height)
        )

        p_hist = (
            array(teste, p_col)
            if p_col in teste.columns
            else np.zeros(teste.height)
        )

        cobertura_tada = (
            np.isfinite(n_hist)
            & (n_hist > 0)
        ).mean()

        cobertura_risco = (
            np.isfinite(p_hist)
            & (p_hist > 0)
        ).mean()

        for sufixo in [
            "tada",
            "tada_bruta",
        ]:

            coluna = (
                prefixo + "_" + sufixo
            )

            if coluna not in teste.columns:
                continue

            valores = array(
                teste,
                coluna,
            )

            valores = preencher(
                valores,
                referencia_idade,
            )

            registrar_variantes(
                coluna,
                valores,
                cobertura_tada,
            )

        for sufixo in [
            "risco",
            "risco_bruto",
        ]:

            coluna = (
                prefixo + "_" + sufixo
            )

            if coluna not in teste.columns:
                continue

            valores = array(
                teste,
                coluna,
            )

            finitos = valores[
                np.isfinite(valores)
            ]

            exigir(
                (
                    (finitos >= -1e-8)
                    & (finitos <= 1 + 1e-8)
                ).all(),
                f"Histórico de risco inválido: {coluna}",
            )

            risco = preencher(
                valores,
                risco_idade,
            )

            registrar(
                coluna,
                e * np.clip(risco, 0, 1),
                cobertura_risco,
            )

    return predicoes, coberturas


# ============================================================
# AVALIACAO TEMPORAL
# ============================================================

def avaliar_ano(
    base,
    ano_alvo,
    contrato,
):

    treino = base.filter(
        pl.col("ANO_ALVO") < ano_alvo
    )

    teste = base.filter(
        pl.col("ANO_ALVO") == ano_alvo
    ).sort(KEYS)

    exigir(
        treino.height > 0,
        "Partição de treino vazia.",
    )

    exigir(
        teste.height > 0,
        "Partição de avaliação vazia.",
    )

    exigir(
        treino["ANO_ALVO"].max()
        < teste["ANO_ALVO"].min(),
        "Sobreposição temporal.",
    )

    particao = next(
        (
            p for p in contrato["particoes"]
            if p["ano_alvo"] == ano_alvo
        ),
        None,
    )

    exigir(
        particao is not None,
        f"Partição {ano_alvo} ausente.",
    )

    exigir(
        treino.height == particao["n_treino"],
        "Número de registros de treino incompatível.",
    )

    exigir(
        teste.height
        == particao["n_avaliacao"],
        "Número de registros de avaliação incompatível.",
    )

    logger.info(
        "Ano-alvo %s | treino=%s | avaliação=%s",
        ano_alvo,
        f"{treino.height:,}",
        f"{teste.height:,}",
    )

    y = array(teste, TARGET)
    e = array(teste, "exposicao")
    n = array(teste, "QT_INGRESSANTE")

    predicoes, coberturas = (
        construir_baselines(
            treino,
            teste,
        )
    )

    resultados = []

    for nome, pred in predicoes.items():

        resultado = metricas(
            y,
            pred,
            e,
            n,
        )

        resultados.append({

            "ano_alvo": ano_alvo,

            "papel": (
                "retrospectivo"
                if ano_alvo == ANO_RETROSPECTIVO
                else "validacao"
            ),

            "modelo": nome,

            **resultado,

            "cobertura_historico": (
                coberturas[nome]
            ),

        })

    resultados = pl.DataFrame(
        resultados
    )

    logger.info(
        "Ano-alvo %s: %s baselines avaliados.",
        ano_alvo,
        resultados.height,
    )

    # Todas as previsões permanecem associadas
    # à mesma chave longitudinal.

    if SALVAR_PREVISOES:

        saida = teste.select([

            *KEYS,

            "ANO_ALVO",
            "idade_coorte",
            TARGET,

            "exposicao",
            "QT_INGRESSANTE",
            "QT_PERMANENCIA",

            "zero_estrutural",

        ])

        saida = saida.with_columns([

            pl.Series(
                nome,
                pred,
                dtype=pl.Float64,
            )

            for nome, pred in predicoes.items()

        ])

        caminho = (
            OUT_DIR
            / "previsoes"
            / f"baselines_{ano_alvo}.parquet"
        )

        saida.write_parquet(
            caminho,
            compression="zstd",
        )

        logger.info(
            "Previsões salvas: %s",
            caminho,
        )

        del saida

    titulo(
        f"ANO-ALVO {ano_alvo} — "
        f"{TOP_N} MELHORES MAE"
    )

    imprimir(
        resultados
        .sort("mae")
        .head(TOP_N)
    )

    titulo(
        f"ANO-ALVO {ano_alvo} — "
        f"{TOP_N} MELHORES R2"
    )

    imprimir(
        resultados
        .sort("r2", descending=True)
        .head(TOP_N)
    )

    del treino, teste, predicoes

    gc.collect()

    return resultados


# ============================================================
# CONSOLIDACAO DOS RESULTADOS
# ============================================================

def consolidar(resultados):

    validacao = resultados.filter(
        pl.col("ano_alvo")
        .is_in(ANOS_VALIDACAO)
    )

    retrospectivo = resultados.filter(
        pl.col("ano_alvo")
        == ANO_RETROSPECTIVO
    )

    ranking = (
        validacao
        .group_by("modelo")
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

            pl.col("mae")
            .max()
            .alias("mae_pior_ano"),

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

            pl.col("mae_positivos")
            .mean()
            .alias("mae_positivos_medio"),

            pl.col("mae_ponderado_ingressantes")
            .mean()
            .alias("mae_ponderado_medio"),

            pl.col("pct_pred_acima_exposicao")
            .mean()
            .alias("pct_acima_exposicao_medio"),

        ])
        .filter(
            pl.col("n_validacoes")
            == len(ANOS_VALIDACAO)
        )
        .sort("mae_medio")
    )

    exigir(
        ranking.height > 0,
        "Nenhum baseline completou as validações.",
    )

    melhor_mae = (
        ranking
        .sort("mae_medio")
        .row(0, named=True)
    )

    melhor_rmse = (
        ranking
        .sort("rmse_medio")
        .row(0, named=True)
    )

    melhor_r2 = (
        ranking
        .sort(
            "r2_medio",
            descending=True,
        )
        .row(0, named=True)
    )

    # A escolha dos baselines ocorre pelas
    # validações, nunca pelo ano de 2024.

    selecionados = {

        "melhor_mae_validacao":
            melhor_mae["modelo"],

        "melhor_rmse_validacao":
            melhor_rmse["modelo"],

        "melhor_r2_validacao":
            melhor_r2["modelo"],

    }

    comparacoes = []

    for criterio, modelo in selecionados.items():

        linha = (
            retrospectivo
            .filter(
                pl.col("modelo") == modelo
            )
            .row(0, named=True)
        )

        comparacoes.append({

            "criterio_selecao": criterio,

            "modelo": modelo,

            "mae_2024": linha["mae"],

            "rmse_2024": linha["rmse"],

            "r2_2024": linha["r2"],

            "vies_2024": linha["vies"],

            "mae_expostos_2024":
                linha["mae_expostos"],

            "delta_mae_vs_08": (
                linha["mae"]
                - REFERENCIA_08["mae"]
            ),

            "delta_rmse_vs_08": (
                linha["rmse"]
                - REFERENCIA_08["rmse"]
            ),

            "delta_r2_vs_08": (
                linha["r2"]
                - REFERENCIA_08["r2"]
            ),

        })

    comparacoes = pl.DataFrame(
        comparacoes
    )

    estabilidade = (
        validacao
        .pivot(

            on="ano_alvo",
            index="modelo",
            values="mae",

            aggregate_function="first",

        )
        .join(

            ranking.select([
                "modelo",
                "mae_medio",
                "mae_dp",
            ]),

            on="modelo",
            how="inner",

        )
        .sort("mae_medio")
    )

    salvar_csv(
        resultados,
        "01_metricas_completas",
    )

    salvar_csv(
        ranking,
        "02_ranking_validacoes_mae",
    )

    salvar_csv(
        ranking.sort("rmse_medio"),
        "03_ranking_validacoes_rmse",
    )

    salvar_csv(
        ranking.sort(
            "r2_medio",
            descending=True,
        ),
        "04_ranking_validacoes_r2",
    )

    salvar_csv(
        retrospectivo.sort("mae"),
        "05_ranking_retrospectivo_2024",
    )

    salvar_csv(
        comparacoes,
        "06_comparacao_benchmark_08",
    )

    salvar_csv(
        estabilidade,
        "07_estabilidade_temporal",
    )

    titulo(
        "RANKING FINAL — VALIDACOES 2020 A 2023"
    )

    imprimir(
        ranking.head(TOP_N)
    )

    titulo(
        "BASELINES SELECIONADOS PELAS VALIDACOES"
    )

    imprimir(comparacoes)

    titulo(
        "ESTABILIDADE TEMPORAL DOS MELHORES BASELINES"
    )

    imprimir(
        estabilidade.head(TOP_N)
    )

    return {

        "selecionados": selecionados,

        "melhor_mae_validacao": melhor_mae,

        "melhor_rmse_validacao": melhor_rmse,

        "melhor_r2_validacao": melhor_r2,

        "comparacao_2024":
            comparacoes.to_dicts(),

    }


# ============================================================
# TESTE SINTETICO
# ============================================================

def teste_sintetico():

    e = np.array([
        0.0,
        5.0,
        10.0,
        20.0,
    ])

    pred = np.array([
        8.0,
        -2.0,
        15.0,
        10.0,
    ])

    esperado = np.array([
        0.0,
        0.0,
        10.0,
        10.0,
    ])

    exigir(
        np.allclose(
            limitar(pred, e),
            esperado,
        ),
        "Falha no teste de restrição.",
    )

    risco = np.array([
        0.2,
        0.3,
        0.5,
        0.8,
    ])

    previsto = e * risco

    exigir(
        (previsto <= e).all(),
        "Falha na reconstrução por risco.",
    )

    exigir(
        previsto[0] == 0,
        "Falha na regra de zero estrutural.",
    )

    logger.info(
        "Testes sintéticos aprovados."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    titulo(
        "23 — BASELINES LONGITUDINAIS DA TADA"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SALVAR_PREVISOES:

        (
            OUT_DIR / "previsoes"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    manifesto = (
        OUT_DIR
        / "23_resumo_baselines.json"
    )

    salvar_json(
        {
            "status": "EM_PROCESSAMENTO"
        },
        manifesto,
    )

    teste_sintetico()

    pasta = localizar_gold()

    logger.info(
        "Gold: %s",
        pasta,
    )

    logger.info(
        "Saída: %s",
        OUT_DIR.resolve(),
    )

    contrato = carregar_contrato(
        pasta
    )

    base = carregar_base(
        pasta,
        contrato,
    )

    resultados = []

    for ano_alvo in ANOS_AVALIACAO:

        resultado = avaliar_ano(
            base,
            ano_alvo,
            contrato,
        )

        resultados.append(
            resultado
        )

    resultados = pl.concat(
        resultados,
        how="vertical_relaxed",
    )

    resumo_modelos = consolidar(
        resultados
    )

    resumo = {

        "status": "CONCLUIDO",

        "gold": str(pasta),

        "n_pares": base.height,

        "n_baselines": (
            resultados["modelo"]
            .n_unique()
        ),

        "n_resultados": resultados.height,

        "anos_validacao": ANOS_VALIDACAO,

        "ano_retrospectivo":
            ANO_RETROSPECTIVO,

        "referencia_script_08":
            REFERENCIA_08,

        "teste_sintetico": True,

        "contrato_09_aprovado": True,

        **resumo_modelos,

        "nota": (
            "Históricos provenientes do script 09. "
            "Nenhum histórico institucional foi "
            "recalculado sobre a base completa. "
            "Seleção realizada pelas validações "
            "temporais de 2020 a 2023. "
            "2024 constitui avaliação retrospectiva "
            "previamente explorada."
        ),

    }

    salvar_json(
        resumo,
        manifesto,
    )

    titulo(
        "BASELINES LONGITUDINAIS CONCLUIDOS"
    )

    print(
        json.dumps(
            resumo,
            ensure_ascii=False,
            indent=2,
        )
    )

    logger.info(
        "Concluído: %s",
        OUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()