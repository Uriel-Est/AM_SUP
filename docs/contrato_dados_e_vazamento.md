# Contrato de dados e auditoria

## Objetivo

O experimento investiga a previsão da Taxa de Desistência Acumulada (TADA) futura em registros de trajetória da educação superior. A unidade de análise é um par de observações da mesma instituição, curso e coorte de ingresso em anos de referência consecutivos.

## Fonte dos dados

- **Instituição responsável:** Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira (Inep).
- **Conjunto de dados:** Indicadores de Trajetória da Educação Superior.
- **Página oficial:** https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/indicadores-de-trajetoria-da-educacao-superior
- **Natureza da fonte:** dados educacionais disponibilizados publicamente pelo Inep.
- **Dados utilizados:** base processada de trajetórias da educação superior, armazenada localmente no projeto.
- **Cobertura identificada na base Silver:** anos de coorte de ingresso de 2010 a 2020.
- **Versão e data de obtenção:** registrar o nome do arquivo original utilizado e a data em que foi obtido. A página oficial disponibiliza diferentes recortes temporais, portanto o endereço geral não identifica, por si só, a versão exata dos dados.
- **Condições de uso:** verificar as condições aplicáveis ao arquivo específico utilizado antes de declarar uma licença ou restrição específica.

## Chave e construção dos pares

Chave de identificação da trajetória:

- `CO_IES`: instituição de ensino superior.
- `CO_CURSO`: curso.
- `NU_ANO_INGRESSO`: ano de ingresso da coorte.

Cada observação de origem é associada à observação da mesma chave no ano de referência seguinte. O alvo é `TADA_FUTURA`, correspondente à TADA da observação futura.

A chave candidata inclui também `NU_ANO_REFERENCIA` para identificar uma observação anual. A auditoria da base Silver não encontrou duplicatas nessa combinação.

## Cobertura e volume

A base Silver é formada por arquivos anuais de coortes de ingresso, de 2010 a 2020, com 2.731.155 linhas no total.

A construção dos pares produziu 2.407.858 observações:

- **Treinamento:** 2.217.370 pares, com ano de origem até 2022.
- **Teste:** 190.488 pares, com ano de origem 2023 e alvo referente a 2024.

## Alvo e partição temporal

Alvo: `TADA_FUTURA`, com valores no intervalo de 0 a 100.

A partição separa os pares pelo ano de origem. O teste utiliza origem em 2023 para avaliar a previsão do valor futuro referente a 2024. A separação temporal evita treinar com pares de origem posteriores ao período de teste, mas não representa necessariamente uma avaliação de generalização para instituições, cursos ou coortes nunca vistos.

## Qualidade e limitações

As auditorias realizadas indicaram:

- Nenhum valor ausente em TADA na base Silver examinada.
- Nenhum valor de TADA fora do intervalo de 0 a 100.
- Nenhuma duplicata na chave candidata auditada.
- Nenhuma duplicata na chave utilizada para construir os pares.

Essas verificações devem ser interpretadas dentro do escopo das bases e dos scripts executados. A existência de dados ausentes em outras variáveis deve ser verificada conforme o conjunto de preditores usado em cada modelo.

A cobertura de 2010 a 2020 refere-se aos anos de coorte de ingresso identificados na base Silver, e não necessariamente ao período completo de referência das observações. A identificação exata do arquivo de origem e de sua versão é necessária para reproduzir a aquisição dos dados.

## Risco de vazamento e interpretação

A construção do alvo futuro exige garantir que as variáveis preditoras estejam disponíveis no momento em que a previsão seria feita. Variáveis que só se tornam conhecidas após o período de origem não devem ser utilizadas como preditoras.

A inclusão da TADA atual como preditora permite comparar o modelo com uma estratégia de persistência, que utiliza o valor atual como previsão do futuro. A validade dessa comparação depende de a TADA atual estar disponível no momento previsto para uso.

A separação por ano de origem não impede que a mesma instituição, curso ou coorte apareça em treino e teste. Portanto, os resultados não devem ser interpretados como evidência de generalização para entidades inéditas.

O diagnóstico de vazamento deve ser complementado com a classificação explícita das variáveis efetivamente utilizadas nos modelos, distinguindo preditores disponíveis no momento da previsão, variáveis temporalmente indisponíveis e variáveis que representem consequências posteriores ao período de origem.

## Reprodutibilidade

Os scripts de auditoria, construção de pares, modelagem e análise estão no diretório `code/`. Os dados locais estão organizados em `data/bronze/`, `data/silver/` e `data/gold/`, quando aplicável.

Esses diretórios são ignorados pelo Git; a execução depende da presença dos arquivos de dados necessários. Para permitir a reprodução, devem ser documentados o nome e a versão do arquivo original, a data de obtenção, as etapas de processamento e os comandos utilizados para executar as auditorias e os modelos.
