# AM_SUP | Aprendizagem de Máquina Supervisionada

Repositório do projeto aplicado da disciplina Aprendizagem de Máquina
Supervisionada, UFPB/CCSA, período 2026.2.

## Problema e objetivo

O projeto explora dados públicos de indicadores de trajetória da educação
superior para avaliar a possibilidade de prever a Taxa de Desistência Anual
(TDA). O experimento atual é exploratório e não deve ser interpretado como
um sistema validado para decisões operacionais.

## Fonte e cobertura

A base utilizada é formada por arquivos de indicadores de trajetória da
educação superior, com dados organizados por instituição, curso, coorte de
ingresso e ano de referência. A base Silver auditada contém os anos de
referência de 2020 a 2024, com 174.240 registros e 31 colunas.

Os arquivos brutos são mantidos localmente e ignorados pelo Git por causa
do tamanho. Antes da entrega, a equipe deve confirmar e registrar o endereço
oficial exato da fonte, a data de obtenção e a licença/condições de uso.
O script de obtenção existente deve ser consultado para reproduzir o download.

## Estrutura

- `code/bronze_fluxo_ensino_superior.py`: obtenção dos arquivos brutos.
- `code/silver_fluxo_ensino_superior.py`: conversão para Parquet.
- `code/modelo_baseline_tda.py`: preparação da base Gold e experimento.
- `data/bronze/`: dados originais locais, ignorados pelo Git.
- `data/silver/`: dados tratados locais, ignorados pelo Git.
- `data/gold/`: base para modelagem, ignorada pelo Git.
- `notebooks/`: notebook de auditoria e modelagem.
- `docs/`: contrato de dados e diagnóstico de vazamento.
- `relatorios/`: relatórios de execução e avaliação.

## Como executar no Windows

Abra o terminal na raiz do repositório e instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Execute os scripts na ordem necessária:

```powershell
python code/bronze_fluxo_ensino_superior.py
python code/silver_fluxo_ensino_superior.py
python code/modelo_baseline_tda.py
```

Os scripts usam caminhos relativos à raiz do repositório. A etapa Bronze
pode baixar arquivos grandes e depende da disponibilidade da fonte. A etapa
de modelagem depende da existência do arquivo Silver esperado em
`data/silver/fluxo_educacao_superior/`.

Para executar o notebook, abra `notebooks/auditoria_baseline_tda.ipynb`
no VS Code com a extensão Jupyter e selecione o ambiente Python com as
dependências instaladas.

## Contrato de dados resumido

- Unidade da linha: registro associado à trajetória de uma coorte de ingresso,
  curso/instituição e ano de referência, conforme a estrutura da fonte.
- Chave candidata auditada: `CO_IES`, `CO_CURSO`, `NU_ANO_INGRESSO`,
  `NU_ANO_REFERENCIA`.
- Alvo do experimento: `TDA`, variável numérica entre 0 e 100 na base auditada.
- Cobertura da base auditada: anos de referência 2020 a 2024.
- Corte do experimento: treino em anos anteriores a 2024; teste em 2024.
- Atualização da fonte: deve ser confirmada na publicação oficial.
- Privacidade: a base usada é agregada por indicadores; confirmar a licença
  e as condições de redistribuição antes de publicar dados derivados.

## Experimento inicial

O script compara um `DummyRegressor` que prevê a média do treino com um
`RandomForestRegressor`. O pré-processamento está dentro de um `Pipeline`,
com imputação e codificação ajustadas durante o treino. Categorias não vistas
são tratadas com `handle_unknown="ignore"`.

Resultados registrados no experimento atual:

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline de média | 27,67 | 32,82 | -0,853 |
| Random Forest | 23,81 | 27,75 | -0,324 |

O Random Forest reduziu MAE e RMSE em relação ao baseline de média treinado.
Entretanto, o R² negativo indica desempenho limitado. O experimento não
demonstra, por si só, que o modelo seja adequado para uso real.

## Limitações e riscos

1. A separação temporal não impede que o mesmo curso ou coorte apareça em
   treino e teste.
2. A definição e o momento de disponibilidade da TDA precisam ser confirmados
   na documentação oficial da fonte.
3. A seleção de variáveis precisa ser validada contra o instante real em que
   uma previsão seria feita.
4. O experimento atual não implementa um baseline operacional baseado na
   regra usada hoje por uma equipe decisora.
5. Não há evidência neste repositório, por si só, de que o modelo tenha sido
   validado por especialistas ou avaliado em produção.

## Uso de IA generativa

Ferramenta utilizada: ChatGPT. Finalidade: apoio à organização da documentação,
estruturação de código, auditoria e explicação dos resultados. A equipe é
responsável por executar, conferir e compreender os arquivos antes da entrega.

## Integrantes e contribuições

Preencher com os nomes completos de todos os integrantes e descrever a
contribuição real de cada um antes da submissão.

## Referências e rastreabilidade

Registrar aqui o link exato da fonte oficial dos dados, a data de obtenção,
a versão dos arquivos e eventuais parâmetros de download. Não afirmar uma
fonte ou licença sem conferência documental.
