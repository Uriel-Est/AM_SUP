
"""

Corte simulado: indicadores observados no ano t -> TDA no ano t+1.
Com o parquet 2020-2024, a coorte e 2020 e a avaliacao final e 2024.

IMPORTANTE: é uma validação retrospectiva; as datas de publicação/revisão
histórica pelo Inep precisam ser verificadas antes de alegar previsão real.

Dependencias: pandas numpy pyarrow scikit-learn matplotlib joblib
"""

from pathlib import Path
import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==================== CONFIGURACOES POSITRON ====================

NOME_SILVER = "indicadores_trajetoria_educacao_superior_2020_2024.parquet"

ANO_TESTE = 2024

SEMENTE = 42

N_ESTIMATORS = 120
MIN_FOLHA = 15
PROFUNDIDADE_MAXIMA = 16

# One-hot de curso possui cardinalidade elevada.
# Mantemos desativado no primeiro experimento por custo computacional.
# Ativar por conta e risco.
INCLUIR_DUMMY_CURSO = False

SALVAR_MODELO = True

# ================================================================


LOG = logging.getLogger("modelo-baseline-tda")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ==================== DIRETORIOS ====================

def localizar_raiz():

    candidatos = [Path.cwd(), *Path.cwd().parents]

    if "__file__" in globals():
        script = Path(__file__).resolve()
        candidatos = [script.parent, *script.parents] + candidatos

    for p in dict.fromkeys(candidatos):
        caminho = (
            p / "data" / "silver" / "fluxo_educacao_superior" / NOME_SILVER
        )

        if caminho.is_file():
            return p

    raise FileNotFoundError(
        "Silver nao encontrada. Confira NOME_SILVER e abra o projeto na IDE. "
        "Esperado: data/silver/fluxo_educacao_superior/" + NOME_SILVER
    )


RAIZ = localizar_raiz()

ARQUIVO = (
    RAIZ / "data/silver/fluxo_educacao_superior" / NOME_SILVER
)

GOLD = RAIZ / "data/gold/fluxo_educacao_superior"
REL = RAIZ / "relatorios"

GOLD.mkdir(parents=True, exist_ok=True)
REL.mkdir(parents=True, exist_ok=True)


# ==================== VARIAVEIS ====================

CHAVE = [
    "CO_IES",
    "CO_CURSO",
    "NU_ANO_INGRESSO"
]

ANO = "NU_ANO_REFERENCIA"

ALVO = "TDA_PROXIMO_ANO"

CATEGORICAS = [
    "TP_CATEGORIA_ADMINISTRATIVA",
    "TP_ORGANIZACAO_ACADEMICA",
    "TP_GRAU_ACADEMICO",
    "TP_MODALIDADE_ENSINO",
    "CO_REGIAO",
    "CO_UF",
    "CO_CINE_AREA_GERAL",
    "NU_ANO_INGRESSO",
]

NUMERICAS = [
    "QT_INGRESSANTE",
    "TDA_ATUAL",
    "DURACAO_ACOMPANHAMENTO"
]

if INCLUIR_DUMMY_CURSO:
    CATEGORICAS.append("CO_CURSO")

PREDITORAS = CATEGORICAS + NUMERICAS


# ==================== SILVER -> GOLD ====================

def carregar_e_criar_pares():

    df = pd.read_parquet(ARQUIVO)

    exigidas = set(
        CHAVE
        + [
            ANO,
            "TDA",
            "CO_MUNICIPIO",
            "CO_REGIAO",
            "CO_UF"
        ]
        + [
            c for c in PREDITORAS
            if c not in (
                "TDA_ATUAL",
                "DURACAO_ACOMPANHAMENTO"
            )
        ]
    )

    ausentes = sorted(exigidas - set(df.columns))

    if ausentes:
        raise ValueError(
            "Colunas ausentes na Silver: " + ", ".join(ausentes)
        )

    LOG.info(
        "Silver: %s linhas, %s colunas",
        *df.shape
    )

    diagnostico = {
        "linhas_silver": len(df),
        "colunas_silver": len(df)
    }

    diagnostico["ausentes_municipio_silver"] = int(
        df["CO_MUNICIPIO"].isna().sum()
    )

    diagnostico["coortes"] = sorted(
        pd.to_numeric(
            df["NU_ANO_INGRESSO"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    diagnostico["municipio_ausente_nunca_imputado"] = True

    # Normalizacao dos tipos das variaveis essenciais.
    for col in CHAVE + [ANO, "TDA", "QT_INGRESSANTE"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Auditoria das chaves.
    if df[CHAVE + [ANO]].isna().any().any():
        raise ValueError(
            "Chaves incompletas: audite antes de modelar."
        )

    if df.duplicated(CHAVE + [ANO]).any():
        raise ValueError(
            "Duplicatas por curso, coorte e ano na Silver."
        )

    if (
        df["TDA"].isna().any()
        or not df["TDA"].between(0, 100).all()
    ):
        raise ValueError(
            "TDA ausente ou fora de [0,100] na Silver."
        )

    # Ordenacao longitudinal.
    df = (
        df.sort_values(CHAVE + [ANO])
        .reset_index(drop=True)
    )

    grp = df.groupby(
        CHAVE,
        sort=False,
        dropna=False
    )

    # Criacao do alvo futuro no MESMO curso/coorte.
    df["ANO_ALVO"] = grp[ANO].shift(-1)

    df[ALVO] = grp["TDA"].shift(-1)

    # Variaveis disponiveis no ano atual.
    df["TDA_ATUAL"] = df["TDA"]

    df["DURACAO_ACOMPANHAMENTO"] = (
        df[ANO] - df["NU_ANO_INGRESSO"]
    )

    # Apenas pares de anos consecutivos.
    valido = (
        (df["ANO_ALVO"] == df[ANO] + 1)
        & df[ALVO].notna()
    )

    pares = df.loc[valido].copy()

    pares["ANO_ALVO"] = pares["ANO_ALVO"].astype(int)

    if pares.empty:
        raise ValueError(
            "Nao foi possivel criar pares de anos consecutivos."
        )

    diagnostico["pares_validos"] = len(pares)

    diagnostico["pares_por_ano_alvo"] = {
        str(k): int(v)
        for k, v in (
            pares["ANO_ALVO"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    diagnostico["variacoes_tda_negativas"] = int(
        (pares[ALVO] < pares["TDA_ATUAL"]).sum()
    )

    diagnostico["coorte_unica"] = (
        len(diagnostico["coortes"]) == 1
    )

    if diagnostico["variacoes_tda_negativas"]:

        LOG.warning(
            "Encontradas %d quedas de TDA acumulada: "
            "revisar revisoes/regras do Inep.",
            diagnostico["variacoes_tda_negativas"]
        )

    # Identificadores preservados para auditoria.
    cols = list(dict.fromkeys(
        CHAVE
        + [
            "CO_MUNICIPIO",
            "NO_CURSO",
            "NO_IES",
            ANO,
            "ANO_ALVO",
            "TDA",
            ALVO
        ]
        + PREDITORAS
    ))

    pares = pares[
        [c for c in cols if c in pares.columns]
    ]

    destino = GOLD / "base_modelagem_tda_t_mais_1.parquet"

    pares.to_parquet(
        destino,
        index=False
    )

    LOG.info(
        "Gold criada: %s (%d pares)",
        destino,
        len(pares)
    )

    return pares, diagnostico


# ==================== PIPELINE ====================

def criar_pipeline():

    # Imputacao e codificacao ajustadas SOMENTE no treino.
    #
    # -1 representa categoria ausente, nao codigo geografico real.

    categoricas = Pipeline([
        (
            "imputar",
            SimpleImputer(
                strategy="constant",
                fill_value=-1
            )
        ),
        (
            "one_hot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    numericas = Pipeline([
        (
            "imputar",
            SimpleImputer(
                strategy="median"
            )
        )
    ])

    transformador = ColumnTransformer([
        (
            "categoricas",
            categoricas,
            CATEGORICAS
        ),
        (
            "numericas",
            numericas,
            NUMERICAS
        )
    ])

    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=PROFUNDIDADE_MAXIMA,
        min_samples_leaf=MIN_FOLHA,
        random_state=SEMENTE,
        n_jobs=-1
    )

    return Pipeline([
        ("preparar", transformador),
        ("modelo", rf)
    ])


# ==================== METRICAS ====================

def calcular_metricas(y, pred, nome, particao):

    y = np.asarray(y, dtype=float)

    pred = np.clip(
        np.asarray(pred, dtype=float),
        0,
        100
    )

    # Residuo positivo: modelo subestimou a desistência.
    erro = y - pred

    return {
        "particao": particao,
        "modelo": nome,
        "n": int(len(y)),

        "MAE": float(
            mean_absolute_error(y, pred)
        ),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(y, pred)
            )
        ),

        "R2": (
            float(r2_score(y, pred))
            if len(y) > 1
            else None
        ),

        "vies_medio_observado_menos_previsto": float(
            np.mean(erro)
        ),

        "fracao_subestimacoes": float(
            np.mean(erro > 0)
        ),
    }


# ==================== AVALIACAO ====================

def avaliar_periodo(treino, validacao, rotulo):

    if treino.empty or validacao.empty:
        raise ValueError(
            f"Conjuntos vazios em {rotulo}"
        )

    # Baseline trivial.
    dummy = DummyRegressor(
        strategy="mean"
    )

    dummy.fit(
        treino[PREDITORAS],
        treino[ALVO]
    )

    # Modelo supervisionado.
    rf = criar_pipeline()

    rf.fit(
        treino[PREDITORAS],
        treino[ALVO]
    )

    # Todos avaliados na MESMA particao.
    predicoes = {
        "baseline_media": dummy.predict(
            validacao[PREDITORAS]
        ),

        "baseline_persistencia": (
            validacao["TDA_ATUAL"].to_numpy()
        ),

        "random_forest": rf.predict(
            validacao[PREDITORAS]
        ),
    }

    metricas = []

    registros = validacao[
        [
            *CHAVE,
            "CO_MUNICIPIO",
            ANO,
            "ANO_ALVO",
            "TDA_ATUAL",
            ALVO
        ]
    ].copy()

    for nome, pred in predicoes.items():

        pred = np.clip(
            np.asarray(pred, dtype=float),
            0,
            100
        )

        metricas.append(
            calcular_metricas(
                validacao[ALVO],
                pred,
                nome,
                rotulo
            )
        )

        registros[f"pred_{nome}"] = pred

        registros[f"residuo_{nome}"] = (
            validacao[ALVO].to_numpy() - pred
        )

    return metricas, registros, rf


# ==================== DIAGNOSTICOS ====================

def gerar_figuras(previsoes):

    obs = previsoes[ALVO].to_numpy(
        dtype=float
    )

    pred = previsoes[
        "pred_random_forest"
    ].to_numpy(dtype=float)

    resid = obs - pred

    rng = np.random.default_rng(SEMENTE)

    amostra = rng.choice(
        len(obs),
        size=min(10000, len(obs)),
        replace=False
    )

    # Figura 1: observado versus previsto.
    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        obs[amostra],
        pred[amostra],
        s=5,
        alpha=0.16
    )

    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        color="black",
        linewidth=1
    )

    ax.set(
        xlabel=f"TDA observada em {ANO_TESTE} (%)",
        ylabel="TDA prevista (%)",
        title="Observado x previsto - Random Forest"
    )

    fig.tight_layout()

    fig.savefig(
        REL / f"tda_observado_previsto_{ANO_TESTE}.png",
        dpi=170
    )

    plt.close(fig)

    # Figura 2: diagnostico dos residuos.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4)
    )

    axes[0].scatter(
        pred[amostra],
        resid[amostra],
        s=5,
        alpha=0.16
    )

    axes[0].axhline(
        0,
        linestyle="--",
        color="black",
        linewidth=1
    )

    axes[0].set(
        xlabel="TDA prevista (%)",
        ylabel="Observado - previsto (p.p.)",
        title="Residuos vs. previsao"
    )

    axes[1].hist(
        resid,
        bins=50
    )

    axes[1].set(
        xlabel="Residuo (p.p.)",
        ylabel="Frequencia",
        title="Distribuicao dos residuos"
    )

    fig.tight_layout()

    fig.savefig(
        REL / f"tda_diagnostico_residuos_{ANO_TESTE}.png",
        dpi=170
    )

    plt.close(fig)


# ==================== ORQUESTRACAO ====================

def main():

    LOG.info(
        "Lendo %s",
        ARQUIVO
    )

    pares, auditoria = carregar_e_criar_pares()

    # Teste final separado ANTES da validacao.
    treino = pares.loc[
        pares["ANO_ALVO"] < ANO_TESTE
    ].copy()

    teste = pares.loc[
        pares["ANO_ALVO"] == ANO_TESTE
    ].copy()

    if treino.empty or teste.empty:
        raise ValueError(
            "Treino ou teste vazio. "
            "Confira ANO_TESTE e o periodo da Silver."
        )

    LOG.info(
        "Treino: %d pares; Teste: %d pares",
        len(treino),
        len(teste)
    )

    # ==================== VALIDACAO TEMPORAL ====================
    #
    # Validacao cruzada temporal EXPANSIVA:
    #
    # Fold 1:
    #   Treino -> alvo 2021
    #   Validacao -> alvo 2022
    #
    # Fold 2:
    #   Treino -> alvos 2021-2022
    #   Validacao -> alvo 2023
    #
    # Teste final:
    #   Treino -> alvos 2021-2023
    #   Teste -> alvo 2024
    #
    # Nunca utilizamos alvos futuros no treinamento de uma dobra.

    folds = []

    for ano_validacao in sorted(
        treino["ANO_ALVO"].unique()
    ):

        t = treino.loc[
            treino["ANO_ALVO"] < ano_validacao
        ]

        v = treino.loc[
            treino["ANO_ALVO"] == ano_validacao
        ]

        if t.empty or v.empty:
            continue

        LOG.info(
            "CV temporal: treino ate %s -> validacao %s",
            ano_validacao - 1,
            ano_validacao
        )

        metricas, _, _ = avaliar_periodo(
            t,
            v,
            f"cv_{ano_validacao}"
        )

        folds.extend(metricas)

    if not folds:
        raise ValueError(
            "Historico insuficiente para validacao temporal: "
            "necessario >= 3 anos-alvo."
        )

    pd.DataFrame(folds).to_csv(
        REL / "validacao_temporal_tda.csv",
        index=False
    )

    # ==================== TESTE FINAL ====================

    LOG.info(
        "Teste final, ANO_ALVO=%d",
        ANO_TESTE
    )

    metricas_teste, previsoes, modelo_final = avaliar_periodo(
        treino,
        teste,
        f"teste_{ANO_TESTE}"
    )

    previsoes.to_csv(
        REL / f"previsoes_tda_{ANO_TESTE}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Identificacao dos maiores erros individuais.
    erros = previsoes.copy()

    erros["erro_absoluto_rf"] = (
        erros["residuo_random_forest"].abs()
    )

    erros.nlargest(
        100,
        "erro_absoluto_rf"
    ).to_csv(
        REL / f"maiores_erros_tda_{ANO_TESTE}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Figuras de diagnostico.
    gerar_figuras(previsoes)

    # Persistencia do modelo treinado.
    if SALVAR_MODELO:
        joblib.dump(
            modelo_final,
            GOLD / "random_forest_tda_t_mais_1.joblib"
        )

    # ==================== RELATORIO JSON ====================

    resumo = {
        "objetivo": (
            "Prever TDA acumulada do mesmo curso/coorte "
            "no ano seguinte"
        ),

        "unidade": (
            "curso x instituicao x coorte x ano base "
            "(municipio como atributo)"
        ),

        "alvo": ALVO,
        "ano_teste": ANO_TESTE,

        "periodo_treino_anos_alvo": sorted(
            treino["ANO_ALVO"]
            .unique()
            .astype(int)
            .tolist()
        ),

        "n_treino": len(treino),
        "n_teste": len(teste),

        "preditoras": PREDITORAS,

        "auditoria": auditoria,

        "validacao_temporal": folds,

        "resultados_teste": metricas_teste,

        "observacoes": [
            "As linhas da Gold sao pares t->t+1, "
            "nao registros independentes.",

            "Um mesmo curso/coorte no treino e teste e "
            "intencional: somente passado alimenta cada previsao.",

            "Datas de publicacao e revisoes historicas "
            "nao foram auditadas: validacao retrospectiva, "
            "nao prospectiva comprovada.",

            "Apenas a coorte 2020 aparece se usada a Silver "
            "2020_2024; o one-hot da coorte fica constante.",

            "CO_CURSO identifica as unidades, mas nao e one-hot "
            "no RF por padrao devido a alta cardinalidade.",

            "Ausencias geograficas sao preservadas; -1 e "
            "categoria interna de imputacao, nao municipio real.",

            "TDA = Taxa de Desistencia Acumulada (nao Anual).",

            "A decisao operacional, custos B/C e a capacidade "
            "de atendimento devem ser definidos no Canvas."
        ]
    }

    caminho_json = REL / "baseline_tda.json"

    caminho_json.write_text(
        json.dumps(
            resumo,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    (REL / "auditoria_modelagem_tda.json").write_text(
        json.dumps(
            auditoria,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ==================== RESULTADOS ====================

    print("\n=== VALIDACAO TEMPORAL (folds) ===")

    print(
        pd.DataFrame(folds)
        .round(3)
        .to_string(index=False)
    )

    print("\n=== TESTE FINAL ===")

    print(
        pd.DataFrame(metricas_teste)
        .round(3)
        .to_string(index=False)
    )

    print("\nGold:", GOLD)
    print("Relatorios:", REL)

    print(
        "Modelo inicial concluido. "
        "Inspecione os residuos antes de interpretar o desempenho."
    )


if __name__ == "__main__":
    main()
