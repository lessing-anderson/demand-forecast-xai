# Previsao de Demanda com IA Explicavel

Projeto de previsao de demanda baseado no dataset **M5 Forecasting Accuracy**.
O repositorio transforma os dados originais da competicao em tabelas analiticas,
cria features temporais, treina modelos de previsao, produz explicacoes locais
com LIME e SHAP e avalia essas explicacoes.

## O que e o projeto

O projeto implementa um fluxo experimental de ponta a ponta para o dataset M5:

1. explora os dados brutos e o contexto da competicao;
2. converte os CSVs em tabelas Parquet dimensionais e factuais;
3. cria features temporais, de preco, eventos e SNAP;
4. treina e compara um baseline sazonal com um modelo LightGBM;
5. analisa previsoes e importancias de features;
6. explica instancias selecionadas com LIME e SHAP;
7. mede fidelidade local, estabilidade e custo computacional das explicacoes.

Os notebooks orquestram os experimentos, enquanto a logica reutilizavel esta
organizada em modulos Python dentro de `src/`.

## Objetivo do TCC

Investigar a aplicacao de tecnicas de IA explicavel em previsao de demanda,
comparando como LIME e SHAP justificam as previsoes de um modelo LightGBM em
cenarios de menor e maior erro de previsao.

O estudo avalia a qualidade das explicacoes sob tres perspectivas: fidelidade
ao comportamento local do modelo, estabilidade diante de pequenas
perturbacoes e custo computacional.

## Dados

O projeto usa os arquivos da competicao [M5 Forecasting
Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
Para a aplicacao de XAI, `sales_train_evaluation.csv` e usado como fonte de
vendas. O periodo `d_1` a `d_1913` funciona como treino e `d_1914` a `d_1941`
como janela de teste/auditoria, permitindo calcular os erros com os valores
reais disponiveis.

Os arquivos originais devem ser mantidos em `data/raw/`:

- `calendar.csv`;
- `sell_prices.csv`;
- `sales_train_evaluation.csv`;
- `sales_train_validation.csv`;
- `sample_submission.csv`.

## Tecnologias

- Python 3.12.11;
- Pandas, NumPy e PyArrow para transformacao e persistencia em Parquet;
- LightGBM para previsao de demanda;
- scikit-learn para metricas de previsao;
- LIME e SHAP para explicabilidade;
- SciPy para a correlacao usada na estabilidade;
- Matplotlib, Seaborn e Plotly para visualizacao;
- Jupyter/IPython para execucao dos experimentos.

As versoes usadas pelo experimento estao fixadas em
[`requirements.txt`](requirements.txt). O projeto declara Python `>=3.12` em
[`pyproject.toml`](pyproject.toml).

## Pipeline

### Processamento dos dados

O notebook `02_data_processing.ipynb` carrega os CSVs e gera em
`data/processed/`:

- `dim_calendar.parquet`;
- `dim_location.parquet`;
- `dim_prices.parquet`;
- `bridge_snap.parquet`;
- `fact_sales.parquet`.

O notebook `03_feature_engineering.ipynb` consolida essas tabelas e salva
`data/features/features.parquet`. Entre as features criadas estao calendario,
eventos, preco, `sales_lag_7`, `sales_lag_28`, medias moveis de 7 e 28 dias e
medias moveis ancoradas em `sales_lag_28`.

### Modelos e avaliacao

O `SeasonalNaiveModel` usa `sales_lag_7` como previsao e serve como baseline
de referencia. O `LightGBMModel` implementa regressao com early stopping e
exposicao de importancia por `gain` ou `split`.

As metricas de previsao disponiveis incluem erro absoluto, MAE, RMSE, MAPE,
RMSSE e WRMSSE nos 12 niveis hierarquicos do M5. As avaliacoes de XAI usam:

- **fidelidade:** variacao media da previsao ao substituir as features mais
  importantes por valores amostrados da referencia;
- **estabilidade:** correlacao de Spearman entre importancias da instancia
  original e de uma instancia perturbada;
- **custo computacional:** tempo de execucao da funcao avaliada.

## Estrutura das pastas

```text
├── data/
│   ├── raw/                         # CSVs originais do M5
│   ├── processed/                   # Tabelas Parquet intermediarias
│   └── features/                    # Features finais para modelagem
├── experiments/
│   ├── exp_000_seasonal_naive/
│   │   └── artifacts/               # Artefatos do baseline sazonal
│   └── exp_001_lgbm/
│       └── artifacts/               # Modelo, previsoes, metricas e XAI
├── notebooks/
│   ├── 01_raw_data_exploration.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_seasonal_naive_model.ipynb
│   ├── 05_naive_analysis.ipynb
│   ├── 06_lightgbm_model.ipynb
│   ├── 07_lightgbm_analysis.ipynb
│   ├── 09_LIME_explainer.ipynb
│   ├── 10_SHAP_explainer.ipynb
│   ├── 11_fidelity_measuring.ipynb
│   ├── 12_stability_measuring.ipynb
│   └── 13_computational_cost_measuring.ipynb
├── src/
│   ├── data/                        # Carga, processamento e features
│   ├── explainers/                  # Integracoes LIME e SHAP
│   ├── evaluation/                  # Fidelidade, estabilidade e custo
│   ├── models/                      # Modelos e split temporal
│   └── utils/                       # Metricas e funcoes auxiliares
├── requirements.txt
└── pyproject.toml
```

## Como executar

### 1. Preparar o ambiente

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para abrir os notebooks fora de uma IDE com suporte Jupyter:

```powershell
python -m pip install notebook
python -m notebook
```

### 2. Obter os dados

Baixe os dados da competicao M5 e coloque os arquivos listados acima em
`data/raw/` com os nomes originais.

### 3. Executar os notebooks

Execute os notebooks a partir da pasta `notebooks/`, nesta ordem:

```text
01_raw_data_exploration.ipynb       # opcional
02_data_processing.ipynb
03_feature_engineering.ipynb
04_seasonal_naive_model.ipynb
05_naive_analysis.ipynb
06_lightgbm_model.ipynb
07_lightgbm_analysis.ipynb
09_LIME_explainer.ipynb
10_SHAP_explainer.ipynb
11_fidelity_measuring.ipynb
12_stability_measuring.ipynb
13_computational_cost_measuring.ipynb
```

Os notebooks usam caminhos relativos a `notebooks/`. Ao executa-los por outra
interface, ajuste o diretorio de trabalho ou os caminhos relativos.

### 4. Resultados

Os artefatos sao organizados por experimento:

- `experiments/exp_000_seasonal_naive/artifacts/` contem o modelo sazonal,
  dados de treino e previsoes;
- `experiments/exp_001_lgbm/artifacts/` contem o modelo LightGBM, dados de
  treino/teste, previsoes, importancias, explicacoes LIME/SHAP e resultados de
  fidelidade, estabilidade e custo computacional.

Os nomes dos arquivos incluem o recorte de series utilizado, como
`CA_1_TX_1`.
