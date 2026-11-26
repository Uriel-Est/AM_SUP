# Predição longitudinal da desistência no ensino superior

Projeto aplicado de Aprendizagem de Máquina Supervisionada — UFPB/CCSA, 2026.2.

## Equipe C

- Vivian Germano da Silva
- Uriel Holanda

## 1. Sobre o projeto

O projeto investiga a capacidade de prever a Taxa de Desistência Anual (TADA) dos cursos de graduação brasileiros, utilizando os Indicadores de Trajetória da Educação Superior disponibilizados pelo Inep.

A proposta consiste em desenvolver uma ferramenta de apoio às coordenações de cursos, permitindo a priorização de trajetórias acadêmicas que demandem investigação adicional.

O sistema não realiza decisões individuais sobre estudantes nem estima efeitos causais de intervenções.

## 2. Situação atual

Entrega 6 — Pacote da Avaliação 2.

Atualização: 26 de novembro de 2026.

Etapas concluídas:

- [x] Coleta e padronização dos dados
- [x] Auditoria longitudinal
- [x] Reformulação do problema preditivo
- [x] Engenharia temporal de variáveis
- [x] Baselines históricos
- [x] Modelos lineares regularizados
- [x] Regressão binomial regularizada
- [x] Modelos híbridos de risco
- [x] Árvore de decisão
- [x] CatBoost
- [x] Diagnóstico de erros e equidade
- [x] Cenários de capacidade operacional
- [x] Figuras do relatório preliminar
- [ ] Refinamento dos modelos híbridos
- [ ] LightGBM
- [ ] Ablação consolidada
- [ ] Auditoria final
- [ ] Implementação do MVP

## 3. Fonte dos dados

Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep).

[Indicadores de Trajetória da Educação Superior](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/indicadores-de-trajetoria-da-educacao-superior).

Os arquivos originais e as bases processadas de grande dimensão não são distribuídos pelo repositório.

## 4. Cobertura da base longitudinal

| Indicador | Quantidade |
|---|---:|
| Coortes de ingresso | 2010–2020 |
| Último ano observado | 2024 |
| Registros Silver | 2.731.155 |
| Trajetórias distintas | 323.297 |
| Pares consecutivos | 2.407.858 |
| Pares com exposição positiva | 1.909.740 |
| Zeros estruturais | 498.118 |
| Variáveis autorizadas | 166 |
| Variáveis numéricas | 154 |
| Variáveis categóricas | 12 |

A unidade de análise corresponde à combinação de instituição, curso, coorte de ingresso e ano de referência.

Chaves:

- `CO_IES`
- `CO_CURSO`
- `NU_ANO_INGRESSO`
- `ANO_ORIGEM`

## 5. Formulação estatística

O objetivo é prever a TADA do ano subsequente utilizando informações disponíveis até o ano de origem.

A exposição é definida por:

$$
E_{i,t}=100\frac{P_{i,t}}{N_i}
$$

O risco condicional de desistência corresponde a:

$$
q_{i,t+1}=\frac{D_{i,t+1}}{P_{i,t}}
$$

A TADA reconstruída é dada por:

$$
\widetilde{Y}_{i,t+1}=E_{i,t}q_{i,t+1}
$$

Em que:

- N representa a quantidade inicial de ingressantes;
- P representa a população permanente no ano de origem;
- D representa as novas desistências no ano subsequente.

As previsões dos modelos recentes respeitam a população exposta e os zeros estruturais.

A avaliação utiliza a TADA publicada pelo Inep.

## 6. Engenharia de variáveis

As variáveis incluem indicadores contemporâneos, atributos institucionais, características acadêmicas, defasagens, tendências e históricos suavizados.

| Especificação | Variáveis | Informação |
|---|---:|---|
| M0 | 11 | Atributos iniciais |
| M1 | 16 | Exposição e idade da coorte |
| M2 | 25 | Contagens, taxas e risco corrente |
| M3 | 81 | Defasagens e dinâmica temporal |
| M4 | 83 | Atributos territoriais e CINE detalhado |
| M5 | 84 | Identificação institucional |
| M6 | 85 | Identificação do curso |
| M7 | 163 | M4 e históricos agregados |
| M8 | 165 | M6 e históricos agregados |
| M9 | 166 | Especificação completa |

Os históricos são construídos utilizando informações observadas até o ano de origem da previsão.

## 7. Validação temporal

A avaliação utiliza partições temporais expansivas.

| Ano-alvo | Treino total | Treino exposto | Avaliação |
|---|---:|---:|---:|
| 2020 | 1.220.977 | 1.052.670 | 262.526 |
| 2021 | 1.483.503 | 1.254.296 | 271.408 |
| 2022 | 1.754.911 | 1.462.469 | 244.577 |
| 2023 | 1.999.488 | 1.642.975 | 217.882 |
| 2024 | 2.217.370 | 1.792.698 | 190.488 |

A seleção de hiperparâmetros utiliza as validações de 2020 a 2023.

O ano de 2024 constitui uma avaliação retrospectiva, não um teste externo independente, pois seus resultados já foram explorados ao longo do desenvolvimento.

## 8. Modelos avaliados

### Experimentos iniciais

- Persistência
- Lasso
- Hurdle
- Random Forest
- XGBoost

### Experimentos longitudinais

- Baselines históricos suavizados
- Ridge
- Lasso regularizado
- ElasticNet
- Regressão binomial regularizada
- Modelo híbrido com offset histórico
- Árvore de decisão
- CatBoost

O histórico dos experimentos iniciais é preservado no repositório.

## 9. Resultados das validações temporais

Médias das quatro avaliações anuais, de 2020 a 2023.

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline histórico | 3,564264 | 7,656723 | 0,307926 |
| Ridge M7 | 3,638464 | 7,418202 | 0,349113 |
| Binomial M7 | 3,589993 | 7,439246 | 0,346910 |
| Híbrido H4 (preliminar) | 3,529856 | 7,363797 | 0,359081 |
| Árvore M7 | 3,576397 | 7,501734 | 0,334685 |
| CatBoost M7 | 3,506535 | 7,304137 | 0,369289 |

O CatBoost apresenta o menor MAE médio entre os candidatos avaliados na base completa até esta etapa.

O ganho frente ao baseline histórico corresponde a aproximadamente 1,62% no MAE médio.

## 10. Avaliação retrospectiva de 2024

Amostra comum: 190.488 observações.

| Modelo | MAE | RMSE | R² | Viés agregado |
|---|---:|---:|---:|---:|
| Baseline histórico | 1,935115 | 4,799384 | 0,271517 | -33,12% |
| Ridge M7 | 2,143430 | 4,750548 | 0,286267 | — |
| Binomial M7 | 1,976775 | 4,705123 | 0,299851 | -19,78% |
| Híbrido H4 (preliminar) | 1,951596 | 4,621455 | 0,324530 | -16,29% |
| Híbrido H5 (preliminar) | 1,930224 | 4,651296 | 0,315779 | -18,41% |
| Árvore M7 | 1,972975 | 4,763520 | 0,282363 | -5,43% |
| CatBoost M7 | 1,978691 | 4,627684 | 0,322708 | -12,63% |

MAE e RMSE são expressos em pontos percentuais.

O viés agregado corresponde à diferença percentual entre as quantidades previstas e observadas de desistentes.

As métricas do Ridge foram recuperadas da execução do script 24. Sua previsão ainda requer revalidação no consolidado do script 27, devido a uma incompatibilidade no nome da coluna.

Os resultados dos híbridos permanecem preliminares.

## 11. Calibração agregada

Em 2024, foram observadas 486.989 desistências.

| Modelo | Desistentes previstos | Subestimação |
|---|---:|---:|
| Baseline | 325.688 | 33,12% |
| Binomial | 390.681 | 19,78% |
| Híbrido H5 | 397.312 | 18,41% |
| Híbrido H4 | 407.654 | 16,29% |
| CatBoost | 425.467 | 12,63% |
| Árvore | 460.531 | 5,43% |

As contagens previstas representam valores esperados, arredondados nesta tabela.

## 12. Capacidade operacional

Foram simulados cenários de priorização de trajetórias com exposição positiva, ordenadas pela TADA futura prevista.

| Capacidade | Trajetórias | Baseline | Árvore | CatBoost |
|---|---:|---:|---:|---:|
| 1% | 1.171 | 12.062 | 55.471 | 21.808 |
| 5% | 5.853 | 59.627 | 106.357 | 53.719 |
| 10% | 11.705 | 103.151 | 155.043 | 102.625 |

Os valores representam desistentes observados nas trajetórias retrospectivamente selecionadas.

Não representam desistências evitadas.

As capacidades são ilustrativas e não correspondem a compromissos institucionais documentados.

## 13. Erros e equidade

Os diagnósticos incluem:

- Região e unidade da Federação
- Modalidade de ensino
- Categoria administrativa
- Organização acadêmica
- Grau acadêmico
- Área de formação
- Idade da coorte
- Porte da coorte
- Faixas da TADA observada e prevista
- Observações com erros extremos

O CatBoost apresentou 1.918 erros absolutos de pelo menos 20 pontos percentuais, contra 2.248 do baseline.

As diferenças de desempenho entre grupos são interpretadas descritivamente, sem atribuição causal de discriminação.

## 14. Auditoria e prevenção de vazamento

- Separação física das preditoras e dos alvos.
- Lista explícita de variáveis autorizadas.
- Construção temporal dos históricos.
- Testes de invariância por truncamento e perturbação.
- Pré-processamento ajustado nas partições de treinamento.
- Seleção de hiperparâmetros anterior à avaliação retrospectiva.
- Verificação de unicidade das chaves.
- Auditoria das restrições de exposição.

Os testes realizados não identificaram uso dos alvos da avaliação no treinamento.

Entretanto, o preenchimento do offset histórico em 25.714 observações antigas de treinamento do modelo híbrido pode incorporar informações posteriores à origem dessas observações. A correção está prevista para a etapa final.

A disponibilidade histórica dos arquivos do Inep, considerando possíveis revisões posteriores, também permanece como limitação.

## 15. Organização do pipeline

As etapas anteriores são preservadas, mantendo-se a numeração dos scripts.

| Etapa | Scripts |
|---|---|
| Coleta e processamento | 01–02 |
| Auditorias iniciais | 03–07 |
| Auditoria longitudinal | 08 |
| Engenharia longitudinal | 09 |
| Experimentos iniciais | 10–22 |
| Baselines longitudinais | 23 |
| Modelos lineares regularizados | 24 |
| Regressão binomial | 25 |
| Modelo híbrido | 26 |
| Árvore de decisão e CatBoost | 27 |
| Figuras do relatório parcial | 31 |

As etapas 28–30 estão reservadas para experimentos complementares, ablação e avaliação consolidada.

## 16. Estrutura de dados e resultados

```text
data/
├── bronze/
├── silver/
└── gold/
    └── fluxo_educacao_superior/
        └── modelagem_longitudinal/
            ├── features_longitudinais.parquet
            ├── alvos_longitudinais.parquet
            └── 09_auditoria_vazamento.json

reports/
├── audit/
├── baselines_longitudinais/
├── modelos_lineares_regularizados_v2/
├── modelo_binomial_regularizado/
├── modelo_hibrido_risco/
├── modelos_arvores/
└── entrega_06/
    └── figuras/
```

Essa estrutura representa os diretórios locais produzidos pelos scripts, não necessariamente arquivos versionados no GitHub.

## 17. Ambiente computacional

Versões registradas no experimento 27:

| Biblioteca | Versão |
|---|---|
| NumPy | 2.3.2 |
| Pandas | 2.3.1 |
| Polars | 1.35.1 |
| SciPy | 1.16.1 |
| Scikit-learn | 1.7.2 |
| CatBoost | 1.2.10 |

Dependências adicionais utilizadas pelo projeto incluem Matplotlib, Joblib, XGBoost e Psutil.

Instalação das principais dependências:

```bash
python -m pip install numpy==2.3.2 pandas==2.3.1 polars==1.35.1 scipy==1.16.1 scikit-learn==1.7.2 catboost==1.2.10 matplotlib joblib xgboost psutil
```

## 18. Como reproduzir

1. Obter os arquivos públicos do Inep.
2. Executar as etapas Bronze e Silver.
3. Executar as auditorias e a engenharia longitudinal.
4. Executar os scripts 23 a 27, respeitando as dependências.
5. Gerar as figuras com o script 31.

Os scripts utilizam caminhos locais configuráveis e dependem da disponibilidade das bases de entrada.

O script 27 utiliza semente 42, seis threads no CatBoost e 200 mil observações de treinamento por ajuste durante o tuning.

Foram realizados 74 treinamentos, com duração total aproximada de 44 minutos no ambiente utilizado.

Os experimentos recentes utilizam manifestos de configuração, logs, arquivos de previsão e checkpoints para rastreabilidade.

## 19. Produto

O MVP consiste em um painel de monitoramento de trajetórias acadêmicas, com filtros por instituição, curso, coorte, modalidade e período.

A recomendação preliminar é utilizar o CatBoost como candidato à previsão de taxas, preservando a árvore de decisão como alternativa para triagem operacional.

A escolha definitiva considera capacidade, custos, calibração e desempenho por grupo.

## 20. Limitações

- Avaliação retrospectiva, sem teste externo independente.
- Possíveis revisões históricas dos indicadores.
- Correção pendente do offset dos modelos híbridos.
- Incerteza estatística dos ganhos ainda não avaliada.
- Generalização para cursos inteiramente novos não demonstrada.
- Subestimação de trajetórias com desistência elevada.
- Heterogeneidade de desempenho entre grupos.
- Ausência de estimação causal dos efeitos de intervenções.

## 21. Contribuições

Uriel Holanda é responsável predominantemente pela elaboração dos relatórios e documentação metodológica, participando também do desenvolvimento computacional.

Vivian Germano da Silva participa da implementação, revisão e aprimoramento dos modelos e bases analíticas.

O desenvolvimento ocorre de maneira colaborativa e iterativa, com aprimoramentos sucessivos das implementações realizadas por ambos os integrantes.

## 22. Transparência sobre inteligência artificial

O ChatGPT foi utilizado como apoio à programação, revisão textual, estruturação metodológica, produção das figuras e organização dos diagnósticos.

Os integrantes são responsáveis pela conferência dos resultados e pelas decisões metodológicas e analíticas.

---

Projeto acadêmico e exploratório — Equipe C, UFPB/CCSA.
