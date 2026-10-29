# AM_SUP | Aprendizagem de Máquina Supervisionada

Repositório do projeto aplicado da disciplina Aprendizagem de Máquina Supervisionada, UFPB/CCSA, período 2026.2.

## Objetivo

Investigar a previsão da Taxa de Desistência Anual (TADA) futura a partir de indicadores públicos da trajetória da educação superior. O projeto tem caráter exploratório e não constitui um sistema validado para decisões operacionais.

## Dados e preparação

A base Silver contém arquivos Parquet organizados por coorte de ingresso e ano de referência. A auditoria considera a chave candidata `CO_IES`, `CO_CURSO`, `NU_ANO_INGRESSO` e `NU_ANO_REFERENCIA`.

A construção dos pares temporais resultou em 2.407.858 observações para prever a TADA do ano seguinte.

- Treinamento: anos de origem até 2022, com 2.217.370 registros.
- Teste: ano de origem 2023, para prever o alvo de 2024, com 190.488 registros.
- Variável-alvo: `TADA_FUTURA`.
- Modelos comparados: persistência e Random Forest.

A fonte oficial exata, a data de obtenção e as condições de uso dos dados ainda precisam ser registradas após conferência documental.

## Estrutura do repositório

- `code/`: scripts de preparação, auditoria, modelagem e análise.
- `data/bronze/`: dados originais, quando disponíveis.
- `data/silver/`: dados tratados em Parquet.
- `data/gold/`: dados preparados para modelagem, quando aplicável.
- `notebooks/`: notebooks de exploração e análise, incluindo experimentos históricos.
- `docs/`: documentação metodológica e contratos de dados.
- `relatorios/`: métricas, auditorias e análises finais.

Os diretórios locais de dados são ignorados pelo Git devido ao tamanho. A execução depende da presença dos arquivos de dados necessários.

## Como executar

No PowerShell, abra o terminal na raiz do repositório e instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Para reproduzir a comparação final, com os dados Silver e os arquivos necessários já disponíveis, execute:

```powershell
python code/modelo_tada_persistencia.py
python code/analisar_comparacao_final_tada.py
```

A construção da base depende das etapas anteriores de preparação dos dados. O script Bronze pode baixar arquivos grandes; execute-o apenas se precisar obter novamente os dados originais. Os scripts usam caminhos relativos à raiz do repositório.

## Resultados da avaliação

| Métrica | Persistência | Random Forest |
|---|---:|---:|
| MAE | 4,0331 | 6,4396 |
| RMSE | 9,1792 | 8,1779 |
| R² | -1,6648 | -1,1151 |

A persistência apresentou menor erro absoluto médio (MAE), enquanto a Random Forest apresentou menor RMSE e R² menos negativo. Os dois modelos tiveram R² negativo. Isso indica que, na avaliação realizada, ambos foram inferiores ao preditor de referência baseado na média do alvo, segundo essa métrica.

Na análise por faixa, a persistência apresentou menor MAE para TADA futura até 5. Nas faixas acima de 5, a Random Forest apresentou menor MAE. Os erros permaneceram elevados nos valores extremos.

Arquivos de resultados:

- `relatorios/comparacao_tada_persistencia.json`
- `relatorios/comparacao_erros_por_faixa_tada.csv`
- `relatorios/analise_final_tada.json`

O arquivo `relatorios/previsoes_tada_teste.csv` contém previsões individuais e é mantido localmente, fora do versionamento, devido ao tamanho.

## Limitações

1. A avaliação considera um único ano de teste.
2. A distribuição da TADA futura é concentrada em valores baixos, com poucos casos extremos.
3. A divisão temporal não garante independência entre cursos ou instituições nos conjuntos de treino e teste.
4. A disponibilidade temporal e a definição oficial das variáveis devem ser verificadas para evitar vazamento de informação.
5. Os resultados não demonstram adequação para uso operacional ou tomada de decisão.
6. A fonte oficial e as condições de uso precisam ser documentadas antes da entrega final.

## Uso de IA generativa

O ChatGPT foi utilizado como apoio à organização da documentação, estruturação de código e interpretação dos resultados. A equipe é responsável pela execução, verificação e compreensão dos arquivos entregues.

## Integrantes e contribuições

Preencher com os nomes completos dos integrantes e as contribuições efetivamente realizadas por cada pessoa.

## Referências e rastreabilidade

Registrar o endereço oficial da fonte dos dados, a data de obtenção, a versão dos arquivos e as condições de uso após verificação documental.
