# Predição da Taxa de Desistência Anual (TADA) no Ensino Superior

## Sobre o projeto

Projeto acadêmico de aprendizado de máquina supervisionado para investigar a capacidade de prever a Taxa de Desistência Anual (TADA) dos cursos de graduação a partir de dados de trajetória da educação superior disponibilizados pelo INEP.

O objetivo é avaliar se informações de um determinado ano permitem estimar a TADA do ano seguinte, utilizando características institucionais, acadêmicas, territoriais e temporais.

## Objetivo

Desenvolver e comparar modelos de regressão para previsão da TADA em uma estrutura temporal, utilizando dados do ano de origem para prever o indicador do ano subsequente.

O projeto busca avaliar o desempenho dos modelos, compará-los com uma referência simples e investigar as limitações da previsão de indicadores de trajetória educacional.

## Fonte dos dados

- Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP).
- Indicadores de trajetória da educação superior.
- Dados anuais organizados para construção de observações temporais.

Os arquivos originais não estão incluídos neste repositório.

## Estrutura temporal

A base foi organizada em pares de anos, considerando:

- Ano de origem (t): variáveis utilizadas como preditoras.
- Ano alvo (t+1): TADA que se pretende prever.

A avaliação foi realizada respeitando a temporalidade dos dados.

| Etapa | Ano de origem | Ano alvo |
|---|---:|---:|
| Validação | 2021 | 2022 |
| Validação | 2022 | 2023 |
| Teste final | 2023 | 2024 |

O período de teste final foi reservado para avaliação da capacidade de generalização temporal.

## Base analítica

- Pares temporais: 2.407.858
- Treinamento (anos de origem até 2022): 2.217.370
- Teste final (2023 → 2024): 190.488

Identificação dos cursos:

- `CO_IES`
- `CO_CURSO`
- `NU_ANO_INGRESSO`

## Variáveis utilizadas

Foram utilizadas informações relacionadas à caracterização institucional, acadêmica, territorial e temporal, incluindo:

- Categoria administrativa
- Organização acadêmica
- Grau acadêmico
- Modalidade de ensino
- Região
- Unidade da Federação
- Área geral do curso (CINE)
- Ano de ingresso
- TADA no ano de origem
- Quantidade de ingressantes
- Ano de origem

A variável-alvo corresponde à TADA do ano seguinte.

## Modelos avaliados

- Lasso
- Persistência (baseline)
- XGBoost
- Hurdle (modelo de duas etapas)
- Random Forest

A persistência utiliza o valor do ano anterior como previsão, funcionando como referência para avaliar se os modelos conseguem superar uma abordagem simples.

## Resultados

Resultados no conjunto de teste final (origem 2023 → alvo 2024):

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Lasso | 3,0614 | 5,5861 | 0,0131 |
| Persistência | 4,0331 | 9,1792 | -1,6648 |
| XGBoost | 3,7303 | 5,9239 | -0,1098 |
| Hurdle (duas etapas) | 5,2294 | 6,6063 | -0,3803 |
| Random Forest | 6,3036 | 7,5196 | -0,7883 |

### Interpretação

O Lasso apresentou o melhor desempenho entre os modelos testados, com menores valores de MAE e RMSE e maior R². Também superou o baseline de persistência nas métricas avaliadas.

Entretanto, o R² próximo de zero indica capacidade limitada de explicar a variabilidade da TADA no conjunto de teste. Os resultados sugerem que a previsão de um ano para o outro apresenta desafios e que modelos mais complexos não necessariamente oferecem melhor desempenho.

Os resultados são exploratórios e não representam validação de uma ferramenta pronta para uso operacional.

## Métricas

- **MAE:** erro absoluto médio das previsões.
- **RMSE:** raiz do erro quadrático médio, penalizando erros maiores.
- **R²:** coeficiente de determinação, relacionado à capacidade explicativa do modelo em comparação com uma referência baseada na média.

## Pipeline de dados

O projeto está organizado em camadas:

- **Bronze:** dados de origem.
- **Silver:** dados processados, padronizados e armazenados em Parquet.
- **Gold:** bases estruturadas para análises e modelagem.

O pipeline inclui preparação, auditorias de consistência, construção de pares temporais, treinamento, avaliação e análise de erros.

## Estrutura do repositório

```text
AM_SUP/
├── code/
│   ├── 01_bronze_fluxo_ensino_superior.py
│   ├── 02_silver_fluxo_ensino_superior.py
│   ├── 03_auditar_base_completa.py
│   ├── 04_auditar_pares_tada.py
│   ├── 05_auditar_particao.py
│   ├── 06_analisar_erros_tada.py
│   ├── 07_analisar_comparacao_final_tada.py
│   ├── 10_modelo_baseline_tda.py
│   ├── 11_modelo_baseline_tada_completo.py
│   ├── 12_modelo_baseline_tada_futura.py
│   ├── 13_modelo_tada_persistencia.py
│   ├── 20_experimento_tada_duas_etapas.py
│   ├── 21_experimento_tada_random_forest.py
│   └── 22_experimento_tada_xgboost.py
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── reports/
│   ├── audit/
│   ├── metrics/
│   │   └── historico/
│   ├── local_predictions/
│   └── summaries/
├── .gitignore
└── README.md
```

## Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Parquet

## Como executar

### Requisitos

- Python 3
- Bibliotecas listadas nas dependências do projeto
- Arquivos de dados originais do INEP, armazenados localmente

### Pipeline

Os scripts numerados em `code/` representam as etapas de preparação, processamento, auditoria e modelagem.

A execução deve respeitar a sequência das etapas de preparação de dados antes dos scripts de modelagem. Os experimentos individuais podem ser executados separadamente, desde que suas bases de entrada já estejam disponíveis.

## Limitações

- Baixa capacidade explicativa no teste final.
- Generalização temporal limitada.
- Diferentes algoritmos apresentaram desempenhos distintos.
- O aumento de complexidade não garantiu melhora.
- A TADA pode ser influenciada por fatores não contemplados nas variáveis utilizadas.
- O desempenho observado em um período não garante resultados equivalentes em outros.

## Considerações finais

O experimento demonstra a importância da validação temporal e da comparação com modelos de referência em problemas de previsão educacional.

Embora o Lasso tenha apresentado o melhor desempenho neste conjunto de experimentos, sua capacidade explicativa foi limitada. Os resultados reforçam a necessidade de avaliar modelos com base em evidências empíricas e considerar a natureza dos dados e as limitações de generalização.

Projeto de caráter acadêmico e exploratório.