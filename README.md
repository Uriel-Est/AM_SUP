# Previsão da Taxa de Desistência Anual na Educação Superior

## 1. Apresentação

Este projeto, desenvolvido na disciplina de Aprendizagem de Máquina Supervisionada, tem como objetivo estruturar uma base de dados e desenvolver um modelo para previsão da Taxa de Desistência Anual (TDA) na educação superior, utilizando dados públicos disponibilizados pelo Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (INEP).

O desenvolvimento será organizado em etapas de coleta, armazenamento, transformação, auditoria, construção da base analítica e modelagem preditiva. A arquitetura seguirá o fluxo **Bronze → Silver → Gold**, utilizado nos scripts de processamento compartilhados por Uriel.

## 2. Fonte dos dados

**Fonte principal:** INEP, Indicadores de Trajetória da Educação Superior.

Página oficial:
https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/indicadores-de-trajetoria-da-educacao-superior

Os dados são disponibilizados em planilhas Excel, distribuídas em arquivos que abrangem diferentes períodos históricos. O script Bronze também consulta o catálogo do INEP e considera endereços históricos de arquivos de indicadores de fluxo da educação superior.

Os períodos disponíveis podem se sobrepor. Por isso, os arquivos originais serão mantidos separados durante a ingestão e a conversão, evitando a concatenação automática de registros potencialmente repetidos.

## 3. Arquitetura e estrutura planejada

```text
AM_SUP/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── bronze/
│   │   └── fluxo_educacao_superior/
│   │       └── arquivos originais .xlsx
│   ├── silver/
│   │   └── fluxo_educacao_superior/
│   │       └── arquivos tratados .parquet
│   └── gold/
│       └── fluxo_educacao_superior/
│           └── base analítica consolidada .parquet
├── src/
│   ├── bronze_fluxo_ensino_superior.py
│   ├── silver_fluxo_ensino_superior.py
│   ├── auditoria_dados.py
│   ├── transformacao.py
│   ├── validacao.py
│   └── construcao_gold.py
├── notebooks/
│   ├── 01_auditoria_bronze.ipynb
│   ├── 02_transformacao_silver.ipynb
│   ├── 03_construcao_gold.ipynb
│   ├── 04_qualidade_e_vazamento.ipynb
│   └── 05_particao_e_baselines.ipynb
├── docs/
│   ├── contrato_dados.md
│   ├── relatorio_qualidade.md
│   ├── analise_vazamento.md
│   ├── estrategia_baselines.md
│   └── contribuicoes.md
└── relatorios/
    └── resultados e registros das validações
```

## 4. Camadas de dados

### 4.1. Bronze: dados originais

Diretório: `data/bronze/fluxo_educacao_superior/`

Responsável por armazenar as planilhas originais do INEP, extraídas dos arquivos ZIP disponibilizados nos catálogos oficiais.

O script `src/bronze_fluxo_ensino_superior.py` realiza as seguintes operações:

- Consulta os catálogos de indicadores de trajetória e de fluxo da educação superior.
- Considera endereços históricos e períodos candidatos a atualização.
- Baixa os arquivos ZIP disponíveis.
- Extrai somente as planilhas Excel que correspondem ao padrão de nomes esperado.
- Mantém os arquivos extraídos separados e evita sobrescrever arquivos existentes por padrão.

Os arquivos originais serão preservados como referência para auditoria e reprodução das transformações.

### 4.2. Silver: dados tratados

Diretório: `data/silver/fluxo_educacao_superior/`

Responsável por converter as planilhas Bronze em arquivos Parquet, mantendo um arquivo por planilha de origem.

O script `src/silver_fluxo_ensino_superior.py`:

- Lê as planilhas utilizando a linha 8 como cabeçalho técnico (`header=8`).
- Verifica a presença da coluna `CO_IES`.
- Remove linhas de rodapé identificadas pela indicação de fonte.
- Remove colunas totalmente vazias e linhas totalmente vazias.
- Salva os resultados em formato Parquet.

A conversão facilita o processamento dos dados e reduz a necessidade de reler repetidamente as planilhas Excel.

**Limitação conhecida:** essa etapa não consolida os diferentes períodos nem resolve, por si só, a possível duplicidade decorrente da sobreposição entre arquivos históricos. Essas verificações serão realizadas nas etapas seguintes.

### 4.3. Gold: base analítica para modelagem

Diretório: `data/gold/fluxo_educacao_superior/`

Esta camada será construída a partir dos arquivos Silver após auditoria e definição das regras de consolidação.

A base deverá reunir os registros necessários para a análise temporal e a previsão da TDA, com documentação de sua unidade de análise, chave de identificação, cobertura temporal e variáveis utilizadas.

Antes da consolidação, serão avaliados:

- A compatibilidade entre os períodos e as versões das planilhas.
- A existência de registros repetidos e a regra para tratá-los.
- A consistência dos identificadores institucionais e demais dimensões disponíveis.
- A presença de valores ausentes, códigos inválidos e mudanças no layout.
- A disponibilidade da variável-alvo e das variáveis explicativas.
- A possibilidade de vazamento de informação entre os dados de entrada e o período previsto.

A estrutura Gold está planejada, mas sua construção e validação ainda fazem parte das próximas etapas do projeto.

## 5. Definição da previsão

A variável de interesse é a Taxa de Desistência Anual (TDA), disponibilizada nos indicadores de trajetória da educação superior do INEP.

A proposta é investigar a previsão da taxa para um período futuro, utilizando informações disponíveis em períodos anteriores. A definição final do horizonte de previsão, da unidade de análise e das variáveis explicativas dependerá da auditoria dos arquivos e da confirmação da comparabilidade temporal.

A modelagem deverá respeitar a ordem cronológica dos dados, evitando utilizar informações futuras para prever períodos anteriores.

## 6. Etapas metodológicas

1. **Ingestão:** obtenção dos arquivos públicos do INEP e armazenamento na camada Bronze.
2. **Transformação:** leitura, limpeza estrutural e conversão das planilhas para Parquet na camada Silver.
3. **Auditoria:** inspeção de colunas, tipos, períodos, valores ausentes, duplicidades e inconsistências.
4. **Construção da Gold:** definição da unidade de análise, da chave e das regras de consolidação.
5. **Análise de vazamento:** avaliação das variáveis que podem conter informação indisponível no momento da previsão.
6. **Preparação da modelagem:** definição das partições temporais de treino, validação e teste.
7. **Baselines:** implementação de modelos de referência para estabelecer um ponto de comparação.
8. **Avaliação:** comparação dos resultados com métricas adequadas ao problema e documentação das limitações.

O pré-processamento que exige aprendizado dos dados, como imputação e padronização, deverá ser ajustado apenas com os dados de treino e aplicado às demais partições sem utilizar informações futuras.

## 7. Reprodutibilidade e dependências

Os scripts compartilhados indicam as seguintes dependências iniciais:

Para a camada Bronze:

```bash
pip install requests beautifulsoup4 tqdm truststore
```

Para a camada Silver:

```bash
pip install pandas openpyxl pyarrow
```

As dependências serão reunidas no arquivo `requirements.txt` à medida que os demais componentes forem implementados.

A execução dos scripts depende do diretório de trabalho. Os caminhos relativos foram definidos para uma execução a partir da raiz do repositório `AM_SUP`.

## 8. Documentação e entregáveis

A documentação será organizada para registrar:

- O contrato da base de dados e a definição da unidade de análise.
- A origem, a cobertura temporal e as transformações aplicadas.
- Os resultados da auditoria e os problemas de qualidade encontrados.
- A análise de vazamento de dados.
- A estratégia de particionamento temporal e os modelos de referência.
- As contribuições dos integrantes da equipe.

Os relatórios e notebooks serão atualizados conforme as análises forem executadas, de modo que as conclusões documentadas correspondam aos resultados efetivamente obtidos.

## 9. Equipe

- Vivian Germano
- Uriel Holanda
