# Relatório de qualidade dos dados

## Base examinada

Arquivo Silver de indicadores de trajetória da educação superior,
referência 2020–2024. Dimensão observada: 174.240 × 31.

## Resultados de auditoria disponíveis

| Verificação | Resultado | Decisão/impacto |
|---|---|---|
| Quantidade de linhas | 174.240 | Base usada na auditoria. |
| Quantidade de colunas | 31 | Estrutura preservada na Silver. |
| Registros por ano de referência | 34.848 em cada ano de 2020 a 2024 | Distribuição uniforme por ano nessa extração. |
| Ausências em `TDA` | 0 | Nenhuma linha excluída por alvo ausente na auditoria conhecida. |
| `TDA` fora de [0,100] | 0 | Nenhuma exclusão por faixa inválida na auditoria conhecida. |
| Duplicatas na chave candidata | 0 | Chave candidata: instituição, curso, coorte e ano de referência. |
| Valores ausentes em outras colunas | Existem | O notebook deve quantificar e a equipe deve justificar imputação. |

## Modelagem

A base Gold criada pelo script possui 174.240 linhas e 11 colunas. O alvo é
`TDA`; registros sem alvo/ano de referência e valores de alvo fora de [0,100]
são excluídos pelo código, embora a auditoria prévia tenha encontrado zero
casos nessas condições.

Treino: 139.392 registros, anos de referência anteriores a 2024.
Teste: 34.848 registros, ano de referência 2024.

## Métricas iniciais

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline de média | 27,6676 | 32,8217 | -0,8527 |
| Random Forest | 23,8144 | 27,7510 | -0,3244 |

O Random Forest teve MAE e RMSE menores que o baseline de média. Ambos os
R² são negativos, portanto o desempenho é limitado e não sustenta alegação
de boa capacidade preditiva. O R² pode ser negativo quando o erro do modelo
é maior que o erro de uma previsão constante igual à média do conjunto de teste.

## Problemas e decisões ainda necessários

1. Registrar a URL oficial da fonte, data de download, versão e licença.
2. Quantificar ausências por variável e documentar o que significam.
3. Verificar a disponibilidade temporal de cada preditora.
4. Avaliar sobreposição de cursos/coortes entre treino e teste.
5. Justificar o cenário real, a métrica e o baseline operacional conforme o Canvas.
6. Confirmar se a definição de TDA é compatível com a decisão que o projeto deseja apoiar.

Este relatório é uma versão inicial baseada nas auditorias e execuções já
registradas. Atualize-o após executar o notebook completo.
