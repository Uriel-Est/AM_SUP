# AM_SUP
## Previsão de abandono em cursos de graduação da UFPB

Projeto desenvolvido para a disciplina de Aprendizagem de Máquina Supervisionada. O objetivo é investigar o uso de dados históricos agregados da Universidade Federal da Paraíba (UFPB) para prever a quantidade de abandonos registrada por matriz curricular no ano seguinte.

## Objetivo

Construir um fluxo reproduzível de preparação, análise e modelagem dos dados, comparando um modelo de referência com um modelo de regressão.

A unidade de análise atual é a matriz curricular por ano. Portanto, o projeto não prevê a evasão individual de estudantes.

## Fontes de dados

As bases utilizadas incluem:

- `discentes_de_cursos_de_graduacao_da_ufpb_2026-07.csv`: dados anuais agregados de graduação.
- `tabela_resumo___indicadores_graduacao_2026-10-16T19_42_23.516457508-03_00.csv`: indicadores semestrais utilizados nas etapas de diagnóstico.
- Dicionário de variáveis da base de discentes da graduação da UFPB.

Os arquivos originais devem ser preservados em `dados/bruto`. As bases derivadas são armazenadas em `dados/processado`.

## Metodologia

1. Diagnóstico da estrutura e da qualidade das bases.
2. Verificação de valores ausentes, duplicações, consistência temporal e valores atípicos.
3. Preparação da base analítica anual.
4. Construção de pares temporais entre anos consecutivos, relacionando as variáveis de origem ao abandono registrado no ano seguinte.
5. Avaliação de um modelo de referência.
6. Treinamento de uma regressão Ridge e comparação das métricas.

## Base temporal

A base `dados/processado/base_temporal_inicial.csv` contém 796 pares de registros entre anos consecutivos, de 2020 a 2025.

A variável-alvo é `Abandono_destino`, que representa a quantidade de abandonos registrada no ano seguinte para a matriz curricular correspondente.

As variáveis preditoras utilizadas no experimento Ridge incluem quantidade de estudantes, abandonos, trancamentos e concluintes do ano de origem, além de curso, turno, grau acadêmico e modalidade.

## Modelos e resultados iniciais

O conjunto de teste utiliza registros com ano de destino 2025. O treinamento utiliza registros com anos de destino 2022, 2023 e 2024.

| Modelo | MAE | RMSE |
|---|---:|---:|
| Referência: abandono do ano anterior | 27,15 | 51,16 |
| Regressão Ridge | 23,59 | 43,40 |

O MAE representa o erro absoluto médio, enquanto o RMSE penaliza mais fortemente erros elevados.

Neste teste inicial, o Ridge reduziu o MAE em aproximadamente 13,1% e o RMSE em aproximadamente 15,2% em relação à referência.

Esses resultados são preliminares. A comparação foi realizada em um único ano de teste e ainda precisa de avaliação temporal adicional para verificar a capacidade de generalização.

## Alertas de qualidade dos dados

Foram identificadas diferenças expressivas na variável de abandono:

- Em 2020, os 164 registros apresentam abandono igual a zero.
- Em 2021, foram registrados apenas 17 abandonos em 164 registros.
- Em 2022, foram registrados 6.454 abandonos em 164 registros.

Essas diferenças podem decorrer de mudanças reais, metodológicas ou de registro. A causa ainda não foi confirmada.

Por isso:

- Os arquivos brutos são preservados.
- Os valores não são corrigidos sem documentação.
- Os zeros não são interpretados automaticamente como ausência real de abandono.
- Os resultados dos modelos devem ser interpretados com cautela.

A variável `Qtde Abandono` também não deve ser confundida com uma taxa oficial de evasão. A definição do indicador e o denominador apropriado precisam ser validados.

## Estrutura do projeto

```text
AM_SUP/
├── dados/
│   ├── bruto/
│   └── processado/
├── src/
├── relatorios/
├── docs/
└── README.md
```

- `dados/bruto/`: arquivos originais.
- `dados/processado/`: bases derivadas e preparadas.
- `src/`: scripts Python de diagnóstico, preparação e modelagem.
- `relatorios/`: resultados e registros das análises.
- `docs/`: documentação complementar.

## Execução

Execute os scripts a partir da raiz do repositório, no terminal:

```powershell
python src/preparar_base_analitica.py
python src/construir_base_temporal.py
python src/avaliar_variavel_alvo.py
python src/modelo_referencia.py
python src/modelo_regressao.py
```

Os scripts de preparação devem ser executados antes dos scripts que dependem de suas saídas.

## Dependências

Python 3.13 e bibliotecas:

- pandas
- numpy
- scikit-learn

Instalação:

```powershell
python -m pip install pandas numpy scikit-learn
```

## Próximas etapas

- Verificar a definição e a comparabilidade histórica do indicador de abandono.
- Ampliar a avaliação temporal dos modelos.
- Investigar a influência de valores extremos e de registros iguais a zero.
- Documentar a execução completa e os resultados dos experimentos.
- Avaliar a adequação das variáveis disponíveis no momento real da previsão.

## Limitações

O histórico anual disponível abrange poucos anos e contém alertas de qualidade. A avaliação atual não demonstra, por si só, que o modelo seja adequado para decisões institucionais reais.

O projeto tem caráter acadêmico e exploratório. Qualquer uso operacional depende de validação adicional dos dados, da metodologia e do desempenho preditivo.
