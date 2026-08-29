# Environment and Installation Guide

## Overview

This guide details the system prerequisites, python dependencies, and step-by-step setup instructions for configuring the execution environment for **demand-forecast-xai**.

## Prerequisites

- **Operating System**: Windows (PowerShell / CMD), Linux, or macOS.
- **Python**: Python **3.12.11** (specified in `.python-version`).
- **RAM**: Minimum 16 GB recommended (to handle unpivoting of the M5 evaluation sales dataset in memory).

## Dependency Stack

The project dependencies are fixed in `requirements.txt`:

| Category | Package | Purpose |
|---|---|---|
| **Data Processing & Storage** | `pandas`, `numpy`, `pyarrow` | High-performance data manipulation and Parquet I/O. |
| **Machine Learning** | `lightgbm` | Fast, distributed gradient boosting framework. |
| **Model Evaluation** | `scikit-learn` | Regression metrics (RMSE, MAE, MAPE). |
| **Explainable AI (xAI)** | `lime`, `shap`, `joblib` | Surrogate models, Shapley value estimation, and multi-threading. |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` | Statistical and interactive plotting. |
| **Orchestration / IDE** | `jupyter`, `notebook`, `ipython` | Notebook execution interface and kernel management. |

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/lessing-anderson/demand-forecast-xai.git
cd demand-forecast-xai
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade Package Manager and Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Register Jupyter Kernel (Optional)

If running notebooks from an external IDE (such as VS Code or JupyterLab):

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=demand-forecast-xai --display-name "Python (demand-forecast-xai)"
```

## Verification

To verify that all required packages are installed correctly, run:

```bash
python -c "import pandas, numpy, lightgbm, lime, shap; print('Environment ready!')"
```

## Related Documentation

- [Runbook](04_runbook.md)
- [Reproducibility Guide](02_reproducibility.md)
