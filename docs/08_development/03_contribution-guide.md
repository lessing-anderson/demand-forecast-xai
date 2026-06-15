# Contribution Guide

## Overview

Welcome to the **demand-forecast-xai** project. This guide outlines the development workflow, coding standards, branch conventions, and Pull Request (PR) requirements for contributors.

## Git Branching Strategy

Follow a feature-branch workflow:

- **`main`**: Production-ready, stable codebase and documentation.
- **`feature/<feature-name>`**: New features, data transformations, or model enhancements.
- **`fix/<bug-name>`**: Bug fixes and patch releases.
- **`exp/<experiment-name>`**: Experimental pipelines or model exploration.

## Development Workflow

### 1. Create a Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/add-new-feature
```

### 2. Environment Setup

Ensure virtual environment is active and up to date:

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### 3. Implement Changes

- Place reusable logic in `src/`.
- Ensure paths use `pathlib.Path`.
- Keep notebooks in `notebooks/` clean and clear all outputs before committing if requested by team policy.

### 4. Verify Notebook Execution

Before submitting a PR, ensure all notebooks execute cleanly top-to-bottom without errors:

```bash
jupyter nbconvert --execute --to notebook notebooks/02_data_processing.ipynb
```

## Pull Request Requirements

When submitting a Pull Request:

1. **Clear Summary**: Provide a description of added features, modified modules, or bug fixes.
2. **Preserve Baseline**: Do not overwrite `experiments/exp_001_baseline_lgbm/artifacts/` unless updating the reference experiment deliberately.
3. **Update Documentation**: Update relevant markdown documents under `docs/` if function signatures, data schemas, or notebook execution steps are altered.
4. **Code Quality**: Ensure code complies with PEP 8 and contains clear docstrings.

## Related Documentation

- [Code Organization](01_code-organization.md)
- [Testing Strategy](02_testing-strategy.md)
- [Operational Runbook](../07_operations/04_runbook.md)
