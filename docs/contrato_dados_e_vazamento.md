# Contrato de dados e auditoria

## 1. Objetivo

O experimento investiga a previsão da Taxa de Desistência Acumulada (TADA) futura em registros de trajetória da educação superior. A unidade de análise é um par de observações da mesma instituição, curso e coorte de ingresso em anos de referência consecutivos.

## 2. Fonte e arquivos de origem

- **Instituição responsável:** Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep).
- **Conjunto de dados:** Indicadores de Trajetória da Educação Superior.
- **Página oficial:** https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/indicadores-de-trajetoria-da-educacao-superior
- **Diretório dos arquivos originais:** `data/bronze/fluxo_educacao_superior/`.
- **Formato original identificado:** Excel (`.xlsx`).
- **Formato processado:** Parquet (`.parquet`), armazenado em `data/silver/fluxo_educacao_superior/`.

Foram identificados os seguintes arquivos locais de origem:

- `indicadores_trajetoria_educacao_superior_2010_2019.xlsx`
- `indicadores_trajetoria_educacao_superior_2011_2020.xlsx`
- `indicadores_trajetoria_educacao_superior_2012_2021.xlsx`
- `indicadores_trajetoria_educacao_superior_2013_2022.xlsx`
- `indicadores_trajetoria_educacao_superior_2014_2023.xlsx`
- `indicadores_trajetoria_educacao_superior_2015_2024.xlsx`
- `indicadores_trajetoria_educacao_superior_2016_2024.xlsx`
- `indicadores_trajetoria_educacao_superior_2017_2024.xlsx`
- `indicadores_trajetoria_educacao_superior_2018_2024.xlsx`
- `indicadores_trajetoria_educacao_superior_2019_2024.xlsx`
- `indicadores_trajetoria_educacao_superior_2020_2024.xlsx`

A presença desses arquivos foi confirmada no ambiente local. A data de download, a versão exata publicada pelo Inep e as condições de uso aplicáveis ainda precisam ser documentadas a partir dos arquivos e da fonte original.

## 3. Chave e construção dos pares

A chave de identificação da trajetória é composta por:

- `CO_IES`: código da instituição de ensino superior.
- `CO_CURSO`: código do curso.
- `NU_ANO_INGRESSO`: ano de ingresso da coorte.

A observação anual é identificada adicionalmente por `NU_ANO_REFERENCIA`. A auditoria não encontrou duplicatas na combinação da chave da trajetória com o ano de referência.

Cada observação de origem é associada à observação da mesma trajetória no ano de referência seguinte. O alvo `TADA_FUTURA` corresponde à TADA observada no ano seguinte.

## 4. Cobertura, volume e partição

A auditoria da base Silver examinada registrou 2.731.155 linhas, com cobertura de coortes de ingresso de 2010 a 2020.

A construção dos pares produziu 2.407.858 observações:

- **Treinamento:** 2.217.370 pares, com ano de origem até 2022.
- **Teste:** 190.488 pares, com ano de origem 2023 e alvo referente a 2024.

A separação temporal simula a previsão de um período futuro em relação ao treinamento. Entretanto, não impede a presença das mesmas instituições, cursos ou coortes nos dois conjuntos. Portanto, não mede exclusivamente a generalização para entidades inéditas.

## 5. Alvo e diagnóstico de vazamento

O alvo principal é `TADA_FUTURA`, com valores entre 0 e 100. As variáveis preditoras devem representar informações disponíveis no momento da previsão, considerando o ano de origem.

### Classificação de variáveis

| Variável ou grupo | Classificação | Justificativa |
|---|---|---|
| `NU_ANO_INGRESSO` | Permitida, sujeita à definição do cenário | Informa o ano de ingresso da coorte e pode estar disponível no momento da previsão. |
| `QT_INGRESSANTE` | Permitida, se referente ao período de origem | Representa informação potencialmente disponível no início ou durante o acompanhamento da trajetória. |
| `TP_CATEGORIA_ADMINISTRATIVA`, `TP_ORGANIZACAO_ACADEMICA`, `TP_GRAU_ACADEMICO` e `TP_MODALIDADE_ENSINO` | Permitidas, se disponíveis na origem | Caracterizam a instituição ou o curso, mas sua disponibilidade temporal deve ser confirmada. |
| `TADA_FUTURA` ou qualquer variável calculada a partir do alvo futuro | Vazamento direto, proibida como preditora | Contém a própria informação que o modelo deve prever. |
| `QT_CONCLUINTE` ou `QT_DESISTENCIA` referente ao ano futuro | Vazamento temporal, proibida como preditora | Representa acontecimentos posteriores ao momento em que a previsão seria realizada. |
| Variáveis registradas somente após uma decisão ou acontecimento futuro | Vazamento temporal ou de consequência, proibidas como preditoras | Não estariam disponíveis no cenário operacional de previsão. |

A classificação é condicional à referência temporal de cada coluna. Antes da modelagem definitiva, deve-se confirmar se as variáveis são extraídas do ano de origem ou do ano futuro. O nome de uma coluna, isoladamente, não comprova sua disponibilidade temporal.

### TADA atual e persistência

A TADA observada no ano de origem pode ser usada como preditora quando estiver disponível naquele momento. A estratégia de persistência utiliza a TADA atual como previsão da TADA futura e funciona como referência operacional.

Essa estratégia não deve ser confundida com um modelo que recebe informações do ano futuro. A validade da comparação depende de a TADA atual estar disponível no cenário previsto.

## 6. Qualidade e limitações

As auditorias realizadas registraram:

- Nenhum valor ausente em TADA na base Silver examinada.
- Nenhum valor de TADA fora do intervalo de 0 a 100.
- Nenhuma duplicata na chave candidata auditada.
- Nenhuma duplicata na chave utilizada para construir os pares.

Esses resultados não significam que todas as variáveis estejam completas. A auditoria do arquivo `indicadores_trajetoria_educacao_superior_2020_2024.parquet`, por exemplo, registrou 26.230 ausências em cada uma das colunas `CO_REGIAO`, `CO_UF` e `CO_MUNICIPIO`, em uma base com 174.240 linhas.

O tratamento de ausências deve ser definido conforme os preditores utilizados. A imputação, codificação e demais transformações devem ser ajustadas somente com os dados de treinamento e aplicadas posteriormente ao teste.

## 7. Resultados dos baselines

O relatório de qualidade registra a comparação entre persistência e Random Forest para 190.488 pares de teste, com origem em 2023 e alvo em 2024.

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistência | 4,0331 | 9,1792 | -1,6648 |
| Random Forest | 6,4396 | 8,1779 | -1,1151 |

A persistência apresenta menor erro absoluto médio, enquanto o Random Forest apresenta menor RMSE e R² menos negativo. Ambos apresentam R² negativo, o que indica desempenho inferior ao baseline de referência baseado na média do alvo do teste, conforme a definição usual dessa métrica.

Esses números pertencem ao experimento documentado em `analise_final_tada.json` e `comparacao_tada_persistencia.json`. Os resultados de `baseline_tada_completo.json` e `baseline_tada_futura.json` correspondem a outras execuções e não devem ser combinados com essa tabela sem uma justificativa metodológica e a confirmação de que os dados, alvos, partições e preditores são comparáveis.

## 8. Reprodutibilidade

Os scripts de auditoria, construção de pares, modelagem e análise estão no diretório `code/`. Os relatórios e resultados estão no diretório `relatorios/`.

Os diretórios `data/bronze/`, `data/silver/` e `data/gold/` são locais e ignorados pelo Git. A reprodução exige que os arquivos necessários estejam disponíveis no ambiente de execução.

Para completar a documentação da aquisição, devem ser registrados a data de obtenção dos arquivos, a versão da fonte, as etapas de transformação e os comandos utilizados para gerar os resultados. Os scripts e os relatórios precisam identificar claramente a execução a que cada métrica pertence.

## 9. Limitações e próximos passos

- Confirmar a versão e a data de obtenção dos arquivos originais.
- Registrar os critérios de seleção das variáveis preditoras e sua correspondência com o cenário de previsão.
- Manter separados os resultados dos diferentes experimentos.
- Documentar o pré-processamento efetivamente aplicado em cada pipeline.
- Interpretar os resultados considerando a concentração de valores baixos de TADA futura e a sobreposição de instituições, cursos e coortes entre treino e teste.
