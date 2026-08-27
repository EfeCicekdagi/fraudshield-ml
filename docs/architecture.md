# FraudShield ML Architecture

This document describes the updated architecture of the FraudShield ML project.

## Core Design Principles
1. **Modularity**: The project is structured as a standard Python package (`src/fraudshield`), allowing modules to be imported cleanly anywhere in the codebase.
2. **Configuration-Driven**: Hardcoded paths and parameters are moved to `configs/data.yaml`. This ensures the code is environment-agnostic and easier to experiment with.
3. **Command-Line Interface (CLI)**: Complex pipelines (like data validation, feature engineering, and splitting) are triggered via Typer CLI commands (`fraudshield validate-data`, etc.).
4. **Notebooks for Exploration**: Jupyter notebooks are strictly used for Exploratory Data Analysis (EDA), visualizing findings, and presenting results. Business logic resides in the `fraudshield` package, avoiding notebook code duplication.
5. **Artifact Management**: Large datasets, trained models, and intermediate caches are explicitly ignored via `.gitignore` to prevent repository bloat.

## System Layers

### 1. Data Layer (`fraudshield.data`)
Handles loading (`loader.py`), robust validation (`validation.py`), and temporally consistent train/test splits (`splitting.py`). 

### 2. Features Layer (`fraudshield.features`)
Contains all transformations and feature engineering logic (`builder.py`). This is separated from the data loading to ensure features can be generated on the fly during inference in the future.

### 3. Models Layer (`fraudshield.models`)
Responsible for model training (`train_baseline.py`), evaluating metrics (`evaluate.py`), and calculating optimal operational thresholds (`threshold_analysis.py`).

### 4. Schemas Layer (`fraudshield.schemas`)
Uses `pydantic` to parse and validate incoming configurations (e.g., `configs/data.yaml`).

### 5. CLI Layer (`fraudshield.cli`)
Exposes the underlying modules to the user via terminal commands, providing immediate feedback with standard logging.

## Pipeline Workflow (Current)
1. **Raw Data** -> `validate-data` -> Validation Report
2. **Raw Data** -> `build-features` -> `data/processed/features.parquet`
3. **Features Data** -> `split-data` -> `train.parquet`, `val.parquet`, `test.parquet`

## Future Extensions
- **Training Pipeline**: A `train` command to train deep learning or XGBoost models.
- **Inference/API**: A FastAPI application to expose the trained model as a REST API.
- **Dashboard**: A Streamlit interface for live monitoring and interactive prediction.
