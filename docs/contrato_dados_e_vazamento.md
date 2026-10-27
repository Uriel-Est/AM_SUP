# Contrato de dados e auditoria

## Objetivo e unidade

O experimento atual utiliza a variável `TDA` como alvo e registros de
trajetória da educação superior. A chave candidata examinada é
`CO_IES`, `CO_CURSO`, `NU_ANO_INGRESSO` e `NU_ANO_REFERENCIA`.
A unicidade foi verificada nessa combinação na base Silver de 2020–2024.

## Cobertura conhecida

Base auditada: `indicadores_trajetoria_educacao_superior_2020_2024.parquet`.
Dimensão observada: 174.240 linhas e 31 colunas; 34.848 registros por ano
de referência de 2020 a 2024.

## Alvo e corte

Alvo: `TDA`, valores observados entre 0 e 100. A definição operacional
completa, a fórmula oficial, a data em que o indicador se torna disponível
e a frequência de atualização devem ser confirmadas na documentação oficial.

O experimento usa anos de referência anteriores a 2024 para treino e 2024
para teste. Isso simula uma previsão sobre registros do ano de referência
2024 a partir de registros de anos anteriores, mas não garante separação
por instituição, curso ou coorte.

## Qualidade já verificada

- Ausências em `TDA`: zero.
- Valores de `TDA` fora do intervalo [0, 100]: zero.
- Duplicatas na chave candidata: zero.
- A base Silver auditada tem 31 colunas.

A ausência de valores em outras variáveis existe e deve ser quantificada
na execução do notebook. Não interpretar ausência como zero sem justificativa.

## Variáveis do modelo atual

Preditoras categóricas:
`TP_CATEGORIA_ADMINISTRATIVA`, `TP_ORGANIZACAO_ACADEMICA`,
`TP_GRAU_ACADEMICO`, `TP_MODALIDADE_ENSINO`, `CO_REGIAO`, `CO_UF`,
`CO_CINE_AREA_GERAL`.

Preditoras numéricas:
`NU_ANO_INGRESSO`, `QT_INGRESSANTE`.

Alvo: `TDA`. `NU_ANO_REFERENCIA` é usado para definir a partição, não como
preditor no script atual.

## Diagnóstico preliminar de vazamento

| Variável | Decisão preliminar | Justificativa e ressalva |
|---|---|---|
| `TP_CATEGORIA_ADMINISTRATIVA` | Potencialmente permitida | Pode ser conhecida antes da previsão; confirmar estabilidade temporal. |
| `TP_ORGANIZACAO_ACADEMICA` | Potencialmente permitida | Deve estar disponível no corte e corresponder à classificação histórica. |
| `CO_UF` | Potencialmente permitida | Localização pode ser conhecida antes da previsão. |
| `NU_ANO_INGRESSO` | Potencialmente permitida | Coorte conhecida, mas a utilidade depende do cenário de previsão. |
| `QT_INGRESSANTE` | Condicional | Usar apenas se o total estiver disponível no instante de previsão e não for calculado posteriormente. |
| `NU_ANO_REFERENCIA` | Não usar como preditora no modelo atual | É usado para definir treino e teste; seu uso direto pode permitir que o modelo explore tendências temporais específicas. |
| `TDA` | Proibida como preditora | É o próprio alvo. |
| `QT_DESISTENCIA` | Proibida até verificar o corte | Pode ser parte do cálculo ou consequência do processo que gera a TDA. |
| `NU_ANO_MAXIMO_ACOMPANHAMENTO` | Condicional/provável risco temporal | Pode conter informação do horizonte futuro de acompanhamento. |
| `QT_CONCLUINTE` | Condicional/provável risco temporal | Pode refletir resultados observados após o momento de previsão. |

A classificação é preliminar. Para a entrega, a equipe deve verificar a
documentação das variáveis e definir um instante operacional de previsão.
A variável proibida por ser consequência da decisão precisa ser descrita
com base no cenário real do projeto, não apenas presumida.

## Limitações remanescentes

- Confirmar fonte oficial, licença, data de extração e atualização.
- Definir a decisão concreta que a previsão apoiaria e o corte operacional.
- Avaliar partição por grupo/coorte ou uma estratégia temporal mais rigorosa.
- Definir baseline operacional e custos/capacidade a partir do Canvas da equipe.
- Documentar imputações, categorias, contagens antes/depois e transformações.
