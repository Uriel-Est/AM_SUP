# Relatório de qualidade dos dados e resultados iniciais

## 1. Escopo

O experimento utiliza dados de trajetória da educação superior organizados em arquivos anuais no formato Parquet. O objetivo é avaliar a previsão da TADA futura a partir de pares de observações de anos consecutivos.

## 2. Construção dos pares

A base Silver examinada contém 2.731.155 linhas distribuídas entre arquivos anuais de coortes de ingresso de 2010 a 2020.

A construção de pares, usando instituição, curso e ano de ingresso como chave da trajetória e associando anos de referência consecutivos, produziu 2.407.858 pares.

| Etapa | Quantidade |
|---|---:|
| Total de pares | 2.407.858 |
| Treinamento, origem até 2022 | 2.217.370 |
| Teste, origem 2023 e alvo 2024 | 190.488 |

A auditoria dos pares não identificou duplicatas na chave utilizada. Essa verificação não substitui a validação semântica de cada variável nem a análise de possíveis vieses da base.

## 3. Distribuição temporal do alvo

A média da TADA futura apresenta redução ao longo dos anos observados, de aproximadamente 15,971 em 2011 para 2,326 em 2024. A mediana chega a zero em 2023.

Essa distribuição indica forte concentração de valores baixos no período mais recente. Por isso, métricas globais devem ser interpretadas junto à distribuição dos erros por faixa do alvo.

## 4. Partição temporal

O treinamento inclui pares com ano de origem até 2022. O teste inclui pares com ano de origem 2023, cujo alvo corresponde a 2024.

Essa estratégia avalia um período posterior ao treinamento. Entretanto, não garante que instituições, cursos ou coortes do teste sejam inéditos em relação ao treinamento.

## 5. Resultados iniciais dos modelos

Os resultados abaixo correspondem à comparação entre persistência e Random Forest no conjunto de teste informado pelo script de comparação final.

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistência | 4,0331 | 9,1792 | -1,6648 |
| Random Forest | 6,4396 | 8,1779 | -1,1151 |

A persistência apresenta menor MAE, enquanto o Random Forest apresenta menor RMSE e R² menos negativo. Ambos os modelos apresentam R² negativo, portanto os resultados não sustentam afirmar que os modelos tenham bom desempenho preditivo global.

O MAE menor da persistência indica menor erro absoluto médio no conjunto avaliado. Já o RMSE menor do Random Forest indica desempenho relativamente melhor nessa métrica, que penaliza mais fortemente erros grandes. As métricas respondem a aspectos diferentes dos erros e não devem ser interpretadas isoladamente.

## 6. Erros por faixa da TADA futura

| Faixa do alvo | Registros | Percentual do teste | MAE persistência | MAE Random Forest |
|---|---:|---:|---:|---:|
| 0 a 1 | 122.766 | 64,45% | 2,7283 | 6,7074 |
| Acima de 1 até 5 | 41.810 | 21,95% | 4,1995 | 5,5880 |
| Acima de 5 até 10 | 14.977 | 7,86% | 5,9406 | 3,6722 |
| Acima de 10 até 20 | 7.417 | 3,89% | 9,3895 | 4,5708 |
| Acima de 20 até 50 | 3.229 | 1,70% | 25,1161 | 20,0720 |
| Acima de 50 até 100 | 289 | 0,15% | 62,3793 | 54,9662 |

A maior parte do teste, 64,45%, concentra-se na faixa de 0 a 1. A persistência apresenta menor MAE nas duas primeiras faixas, que juntas representam aproximadamente 86,40% do teste. O Random Forest apresenta menor MAE nas faixas acima de 5.

As faixas elevadas contêm poucos registros, especialmente acima de 50. Os erros nessas faixas merecem atenção, mas as conclusões devem considerar a pequena quantidade de casos e a distribuição assimétrica do alvo.

## 7. Limitações e próximos passos

- Verificar a disponibilidade temporal de cada preditora para evitar vazamento de informação.
- Comparar os modelos com baselines adequados e manter as mesmas partições para todas as comparações.
- Investigar o desempenho por faixa do alvo, além das métricas globais.
- Examinar a influência da concentração de valores próximos de zero.
- Considerar a sobreposição de instituições, cursos e coortes entre treinamento e teste.
- Documentar as versões dos dados, scripts e parâmetros utilizados para reproduzir cada resultado.

## 8. Arquivos relacionados

Os scripts de construção, auditoria, modelagem e comparação estão em `code/`. Os resultados tabulares e os relatórios gerados estão em `relatorios/`. Os dados locais são ignorados pelo Git e precisam estar disponíveis para reproduzir a execução.
