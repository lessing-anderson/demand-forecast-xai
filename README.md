# Previsão de Demanda com IA Explicável

Projeto de previsão de demanda baseado no dataset **M5 Forecasting Accuracy**.
O repositório transforma os dados originais da competição em uma base analítica,
treina um modelo LightGBM, produz explicações locais com LIME e SHAP e as avalia.

## O que é o projeto

O projeto implementa um fluxo experimental de ponta a ponta para o
dataset M5:

1. explora e processa os dados brutos;
2. cria features temporais, de preço e de eventos;
3. treina e avalia um baseline LightGBM;
4. seleciona previsões com erros baixos e altos;
5. explica esses casos com LIME e SHAP.
6. calcula métricas de xAI em cada um dos métodos.

O fluxo é executado por notebooks, enquanto a lógica reutilizável está
organizada em módulos Python em `src/`.

## Objetivo do TCC

Investigar a aplicação de técnicas de IA explicável em previsão de demanda,
comparando como LIME e SHAP justificam as previsões de um modelo LightGBM,
especialmente em cenários de maior e menor erro de previsão.

O estudo estabelece uma base para avaliar a qualidade das explicações quanto a
fidelidade, estabilidade e custo computacional.

## Tecnologias

- Python 3.12.11;
- Pandas, NumPy e PyArrow para transformação e persistência de dados;
- LightGBM para previsão de demanda;
- scikit-learn para métricas;
- LIME e SHAP para explicabilidade;
- Matplotlib, Seaborn e Plotly para visualização;
- Jupyter/IPython para execução dos experimentos.

As versões usadas pelo experimento estão fixadas em
[`requirements.txt`](requirements.txt).

## Estrutura das pastas

```text
├── data/
│   ├── raw/                         # CSVs originais do M5
│   ├── processed/                   # Tabelas Parquet intermediárias
│   └── features/                    # Dataset final para modelagem
├── docs/                            # Documentação técnica completa
│   ├── 01_architecture/             # Arquitetura e decisões
│   ├── 02_data/                     # Schemas, contrato e linhagem
│   ├── 04_notebooks/                # Guias de execução dos notebooks (01 a 09)
│   ├── 05_modeling/                 # LightGBM, validação temporal e métricas
│   ├── 06_explainability/           # LIME, SHAP e amostragem por cauda de erro
│   ├── 07_operations/               # Instalação, reprodutibilidade e runbook
│   ├── 08_development/              # Organização do código, testes e contribuição
│   ├── 09_reference/                # Referência dos módulos Python em src/
│   └── 10_adr/                      # Architecture Decision Records (ADRs)
├── experiments/
│   └── exp_001_baseline_lgbm/
│       └── artifacts/               # Modelo, métricas, previsões e xAI
├── notebooks/
│   ├── 01_raw_data_exploration.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_lightgbm_baseline.ipynb
│   ├── 05_LIME_explainer.ipynb
│   ├── 06_SHAP_explainer.ipynb
│   ├── 07_faithfulness_measuring.ipynb
│   ├── 08_stability_measuring.ipynb
│   └── 09_computational_cost_measuring.ipynb
├── src/
│   ├── data/                        # Carga, processamento e features
│   ├── explainers/                  # Implementações LIME, SHAP e métricas de fidelidade, estabilidade e custo
│   ├── models/                      # Contrato base, LightGBM e split temporal
│   └── utils/                       # Métricas de negócio/previsão e serialização
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

Para abrir os notebooks fora de uma IDE com suporte Jupyter, instale uma
interface Jupyter no ambiente:

```powershell
python -m pip install notebook
python -m notebook
```

### 2. Obter os dados

Baixe os dados da competição
[M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
e mantenha-os em `data/raw/` com os nomes originais:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sample_submission.csv`

### 3. Executar os notebooks

Abra a pasta `notebooks/` e execute os notebooks nesta ordem:

```text
02_data_processing.ipynb
03_feature_engineering.ipynb
04_lightgbm_baseline.ipynb
05_LIME_explainer.ipynb
06_SHAP_explainer.ipynb
07_faithfulness_measuring.ipynb
08_stability_measuring.ipynb
09_computational_cost_measuring.ipynb
```

`01_raw_data_exploration.ipynb` é opcional e pode ser executado antes do
processamento. Os notebooks assumem `notebooks/` como diretório de trabalho;
por isso, execute-os a partir dessa pasta ou ajuste os caminhos relativos.

### 4. Resultados

Ao final da execução será possível visualizar o experimento `exp_001_baseline_lgbm` 
com os resultados em `experiments/exp_001_baseline_lgbm/artifacts/`.

O baseline gera `lightgbm_CA_1.pkl`, previsões do holdout, métricas e
importâncias. Os notebooks LIME e SHAP reutilizam esses artefatos e geram as
explicações locais correspondentes. E os notebooks de measuring geram as métricas das explicações.

## Documentação Técnica

A documentação técnica do repositório está centralizada na pasta [`docs/`](docs/README.md):

- [01. Arquitetura](docs/01_architecture/01_overview.md) (Visão Geral, Fluxo de Execução, Dependências e Decisões)
- [02. Dados](docs/02_data/01_m5-source-data.md) (Fonte M5, Schemas Processados, Contrato de Features e Linhagem)
- [04. Notebooks](docs/04_notebooks/01_raw-data-exploration.md) (Guias detalhados de execução para os notebooks 01 a 09)
- [05. Modelagem](docs/05_modeling/01_lightgbm-baseline.md) (LightGBM Baseline, Estratégia de Validação Temporal e Métricas)
- [06. Explicabilidade](docs/06_explainability/01_lime.md) (LIME, SHAP, Amostragem por Cauda de Erro e Protocolo xAI)
- [07. Operações](docs/07_operations/01_environment-and-installation.md) (Instalação, Reprodutibilidade, Catálogo de Artefatos e Runbook)
- [08. Desenvolvimento](docs/08_development/01_code-organization.md) (Estrutura do Código, Estratégia de Testes e Guia de Contribuição)
- [09. Referência de Código](docs/09_reference/index.md) (Documentação da API dos módulos Python em `src/`)
- [10. ADRs](docs/10_adr/README.md) (Architecture Decision Records)
