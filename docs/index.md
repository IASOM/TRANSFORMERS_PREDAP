# Predap — Healthcare Demand Forecasting

**Predap** is a deep learning ecosystem for forecasting diagnostic visit demand in healthcare systems. It uses a **three-phase Transformer-based residual correction pipeline** with production-ready inference capabilities via FastAPI.

---

## Value Proposition

- **Accurate multi-horizon forecasts** of healthcare diagnostic visits (1 to 365 days ahead)
- **Three-phase residual correction** progressively refines predictions from a univariate baseline through diagnostic and seasonal enrichment
- **Production-ready**: quantized models, FastAPI endpoints, automated data ingestion
- **Experiment tracking** with MLflow and Hydra-based grid search
- **COVID-aware**: built-in pandemic wave handling with Catalan public health calendar

---

## Architecture Overview

```mermaid
graph TD
    subgraph Data Ingestion
        A[Raw Parquet Data] --> B[data_preparation.py]
        B --> C[Temporal Feature Engineering]
        C --> D[Cyclical Encodings + Holiday/Vacation Flags]
    end

    subgraph Phase 1 — Univariate Transformer
        D --> E[RevIN Normalization]
        E --> F[Positional Encoding]
        F --> G[Multi-Head Self-Attention × N Blocks]
        G --> H[AveragePooling1D + MLP Head]
        H --> I[Baseline Forecast ŷ₁]
    end

    subgraph Phase 2 — Diagnostic Residual Transformer
        I --> J[Compute Residuals: r₁ = y − ŷ₁]
        K[Diagnostic Covariates LMLR + GCausal] --> L
        J --> L[Hybrid LSTM-Transformer]
        L --> M[Predicted Residuals r̂₁]
        M --> N[Corrected Forecast ŷ₂ = ŷ₁ + r̂₁]
    end

    subgraph Phase 3 — Seasonal Residual Transformer
        N --> O[Compute Residuals: r₂ = y − ŷ₂]
        P[Seasonal Covariates: DoW / Month / Holidays] --> Q
        O --> Q[Hybrid LSTM-Transformer]
        Q --> R[Predicted Residuals r̂₂]
        R --> S[Final Forecast ŷ₃ = ŷ₂ + r̂₂]
    end

    subgraph Production
        S --> T[Model Quantization float16]
        T --> U[FastAPI Inference Endpoint]
        U --> V[Client Applications]
    end

    subgraph Experiment Tracking
        W[MLflow] --> X[Metrics + Artifacts + Models]
        Y[Hydra Grid Search] --> W
    end
```

---

## Three-Phase Pipeline Summary

| Phase | Pipeline Class | Input | Output |
|-------|---------------|-------|--------|
| **1. Univariate** | `UnivariateTransformerPipeline` | Target time series + temporal features | Baseline forecast $\hat{y}_1$ |
| **2. Diagnostic Residual** | `DiagnosticResidualTransformerPipeline` | Residuals $r_1$ + diagnostic covariates | Corrected forecast $\hat{y}_2 = \hat{y}_1 + \hat{r}_1$ |
| **3. Seasonal Residual** | `SeasonalResidualTransformerPipeline` | Residuals $r_2$ + seasonal covariates | Final forecast $\hat{y}_3 = \hat{y}_2 + \hat{r}_2$ |

---

## Quick Links

[Installation](getting-started/installation.md){ .md-button .md-button--primary }
[5-Minute Quickstart](getting-started/quickstart.md){ .md-button }
[API Reference](api-reference/rest-api.md){ .md-button }

---

## Project Structure

```
TRANSFORMERS_PREDAP/
├── main.py                          # Main training (3-phase + MLflow)
├── main_experiments_hydra.py        # Hydra-based grid search runner
├── main_grid_search_hyperparameters.py
├── api/                             # FastAPI REST API
│   ├── main.py                      # App entry point
│   ├── routers/production.py        # /production endpoints
│   └── schemas/                     # Pydantic request models
├── conf/                            # Hydra YAML configurations
├── production/                      # Production deployment pipelines
│   ├── add_new_data_pipeline.py     # AddNewDataPipeline
│   ├── model_reconstruction_pipeline.py  # ModelPredictionPipeline
│   └── model_quantization_pipeline.py    # ModelQuantizationPipeline
├── src/                             # Core library
│   ├── config/                      # BaseTransformerConfig dataclass
│   ├── utils/                       # Data prep, evaluation, experiments
│   ├── univariate_transformer/      # Phase 1 model + training + eval
│   ├── residual_multivariate_transformers/  # Phases 2 & 3
│   ├── main_train_univ_transformer_class.py
│   ├── main_train_diagnostic_residual_transformer_class.py
│   └── main_train_seasonal_residual_transformer_class.py
├── models/                          # Saved .keras model weights
├── mlruns/                          # MLflow experiment data
├── plots/                           # Generated visualizations
└── notebooks/                       # Jupyter analysis notebooks
```

---

## Supported Model Architectures

Predap ships with four Transformer variants selectable at build time:

| Architecture | Description | Key Innovation |
|-------------|-------------|----------------|
| **Base Transformer** | Standard multi-head attention encoder | RevIN + cosine LR schedule |
| **Informer** | Distilling layers with ProbSparse attention | Reduced quadratic complexity |
| **LogSparse Transformer** | Logarithmic sparse attention masking | Local + powers-of-2 connectivity |
| **LSTNet** | CNN → GRU + Skip-GRU + linear AR | Hybrid sequential architecture |
