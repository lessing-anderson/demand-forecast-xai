# Demand Forecasting with xAI - M5 Dataset

A modular framework for demand forecasting with explainable AI (xAI) using the M5 competition dataset.

## 🎯 Project Structure

```
src/
├── data/              # Data loading & preprocessing
│   ├── loader.py      # Raw M5 data loading
│   ├── preprocessing.py  # Feature engineering
│   └── splitting.py   # Train/test splits
│
├── models/            # Model implementations
│   ├── base_model.py  # Abstract base class
│   └── lightgbm_model.py  # LightGBM implementation
│
├── xai/               # Explainability modules
│   ├── shap_explainer.py        # SHAP explanations
│   ├── lime_explainer.py        # LIME explanations
│   ├── feature_importance.py    # Importance analysis
│   └── visualization.py         # Plotting utilities
│
├── utils/             # Shared utilities
│   ├── metrics.py     # Evaluation metrics
│   ├── logger.py      # Logging setup
│   └── helpers.py     # Helper functions
│
└── pipeline.py        # End-to-end orchestration

experiments/          # Experiment scripts
├── baseline.py       # Baseline model training
├── feature_selection.py  # Feature importance analysis
└── xai_analysis.py   # xAI explanations

scripts/              # CLI entry points
├── run_pipeline.py   # Training pipeline
└── evaluate_model.py # Evaluation & reporting

config/              # Configuration files
├── config.yaml      # Default experiment config
└── experiment_config.py  # Config loader

tests/               # Unit tests
├── test_data.py
├── test_models.py
└── test_xai.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python scripts/run_pipeline.py --store CA_1
```

Options:
- `--store CA_1` - Filter by store (optional)
- `--split-date YYYY-MM-DD` - Train/test split date
- `--optimize-memory` - Enable memory optimization
- `--num-rounds N` - LightGBM boosting rounds
- `--early-stopping N` - Early stopping rounds
- `--save-dir PATH` - Output directory

### 3. Run Experiments

**Baseline training:**
```bash
python experiments/baseline.py
```

### 4. Evaluate Model

```bash
python scripts/evaluate_model.py --store CA_1 --plot
```

## 📊 Key Features

### Data Layer
- ✅ M5 dataset loading with memory optimization
- ✅ Temporal features (day, week, month, calendar events)
- ✅ Lagged features (7-day, 28-day lags)
- ✅ Rolling averages
- ✅ Train/test splitting with no data leakage

### Model Layer
- ✅ LightGBM baseline
- ✅ Extensible base model class for new models
- ✅ Hyperparameter configuration
- ✅ Feature importance extraction

### xAI Layer
- ✅ **SHAP** - Tree-based SHAP explanations
- ✅ **LIME** - Local interpretable model-agnostic explanations
- ✅ **Permutation Importance** - Feature importance via permutation
- ✅ **Gain Importance** - Model's built-in importance
- ✅ **Visualizations** - Feature importance plots, prediction scatter plots

## 📈 Module Dependencies

```
config/ → experiment_config.py (no deps)

src/data/
  ├─ loader.py (no deps)
  ├─ preprocessing.py (→ loader)
  └─ splitting.py (→ preprocessing)

src/models/
  ├─ base_model.py (no deps)
  └─ lightgbm_model.py (→ base_model, utils.metrics)

src/xai/
  ├─ shap_explainer.py (→ models.lightgbm_model)
  ├─ lime_explainer.py (→ models.lightgbm_model)
  ├─ feature_importance.py (→ models.lightgbm_model, utils.metrics)
  └─ visualization.py (no internal deps)

src/utils/
  ├─ logger.py (no deps)
  ├─ metrics.py (no deps)
  └─ helpers.py (no deps)

src/pipeline.py (→ data.*, models.*, utils.*)

experiments/ (→ pipeline, models, xai)
scripts/ (→ pipeline, xai)
```

**No circular dependencies** - clean, testable architecture.

## 🧪 Running Tests

```bash
# All tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_data.py -v

# Single test
python -m pytest tests/test_models.py::TestLightGBMModel::test_model_training -v
```

## 📝 Example Usage in Python

```python
from src.pipeline import DemandForecastPipeline
from src.xai.shap_explainer import SHAPExplainer
from src.xai.feature_importance import FeatureImportanceAnalyzer

# Run full pipeline
pipeline = DemandForecastPipeline('./data/raw', store_filter='CA_1')
metrics, results = pipeline.run_full_pipeline(optimize_memory=True)

# Get predictions and evaluate
preds = pipeline.predict()
print(f"RMSE: {metrics['rmse']:.4f}")

# SHAP explanations
shap_explainer = SHAPExplainer(pipeline.model)
shap_explainer.fit(pipeline.X_train)
shap_values = shap_explainer.explain_batch(pipeline.X_test.head(100))

# Feature importance comparison
analyzer = FeatureImportanceAnalyzer(pipeline.model)
comparison = analyzer.compare_importance_methods(
    pipeline.X_test, 
    pipeline.test_df['sales'],
    top_k=20
)
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize experiments:

```yaml
data:
  raw_path: ./data/raw
  store_filter: null  # null for all, 'CA_1' for single store
  train_split_date: '2016-04-24'

model:
  type: lightgbm
  params:
    num_leaves: 31
    learning_rate: 0.05
    num_rounds: 1000

xai:
  shap:
    enabled: true
  lime:
    enabled: true
  feature_importance:
    methods: ['gain', 'permutation']
    top_k: 20
```

## 🔄 Workflow

1. **Data Preparation** → Load M5 → Engineer features → Split train/test
2. **Model Training** → Train LightGBM → Evaluate metrics
3. **Feature Analysis** → Gain importance, Permutation importance
4. **xAI Analysis** → SHAP values, LIME explanations
5. **Visualization** → Feature importance plots, residual plots

## 🎓 Extending the Framework

### Adding a New Model

1. Create class inheriting from `BaseModel` in `src/models/`
2. Implement `train()`, `predict()`, `get_feature_importance()`
3. Use in experiments: `pipeline.train_model('new_model_name')`

### Adding a New Explainer

1. Create new class in `src/xai/`
2. Implement explanation methods
3. Use in experiments and notebooks

### Adding Tests

Add test files to `tests/` following naming convention `test_*.py`

## 📖 References

- [M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
