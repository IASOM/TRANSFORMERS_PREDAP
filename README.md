# PREDAP TRANSFORMERS PREDICTION 

## MODULES 
### 1- AQUAS DATA RETRIEVAL
### 2- CCLR_PREDAP
### 3- TRANSFORMERS_PREDAP (src)
## EXECUTION FILES 
1- Inference file
## PREDAP Inference & Data Pipeline Runner

This repository contains the inference execution pipeline for the PREDAP platform. It integrates a Hydra-managed Transformer model inference script with an optimized backend data retrieval pipeline (`run_pipeline_optimized.py`) that leverages Parquet storage.

## System Architecture Overview

When you execute `main_inference.py`, the runtime orchestrates the execution in two distinct phases:

1. **Phase 0 (Data Layer Initialization):** The script parses your configuration using Hydra, extracts the relevant runtime boundaries (`training.cutoff_date` and `inference.max_date`), mocks the system CLI arguments (`sys.argv`), and invokes the optimized pipeline runner.
2. **Phase 1 (Deep Learning Inference):** The script initializes a `ModelPredictionPipeline` using a `BaseTransformerConfig` built directly from your configuration, maps sequence matrices using your target code, performs reconstruction forecasting, saves outputs, and clears out stale evaluation files.

---

## 🛠 Prerequisites & Setup

Ensure your local configuration directory structure matches the following minimum baseline:
```text
├── main_inference.py
├── AQUAS_DATA_RETRIEVAL/
│   └── run_pipeline_optimized.py
└── conf/
    └── config_inference.yaml
```

## EXECUTION

**1. Default Run**
To execute the pipeline using the default parameters defined in conf/config_inference.yaml:
```python main_inference.py```

**2. Common Runtime Modifications**
To change the target diagnosis/demand code, modify data timeline matrices, or change processing boundaries on the fly, pass them as arguments:

```python main_inference.py \
  model.target_code="Z51" \
  training.cutoff_date="2025-01-01" \
  inference.max_date="2026-06-15"
```

**3. Adjusting Transformer Horizons & Architecture**
To tune the network architecture or alter your slicing window configurations (e.g., expanding lookback intervals or changing depth metrics):

```python main_inference.py \
  model.lookback=180 \
  model.forecast=30 \
  model.num_transformer_blocks=6 \
  model.dropout=0.25 \
  model.learning_rate=0.0005
```
**4. Modifying Data Engine Directories**
To redirect data paths to specific historical weights, alternative feature files, or change evaluation report targets:
```python main_inference.py \
  data.data_path="data/custom_input.parquet" \
  data.models_folder="weights/production_v1/" \
  output.output_path="outputs/predictions.parquet"
```

**5. Multirun Sweeps**
If you wish to sweep across multiple parameters sequentially (e.g., testing multiple target codes back-to-back), use Hydra's multi-run flag -m:
```python main_inference.py -m model.target_code="Z51","M54","U07.1"```

## CLI Help System Extension

This extension provides a dedicated, lightweight inspection mechanism inside the production pipeline. It allows engineers to query all acceptable Hydra paths and variables directly via the command console without parsing long config files or initializing heavy tracking dependencies.

---

## 🚀 Usage Guide

To invoke the help interface, append the `help=True` utility flag to your execution command. 

### 1. Basic Help Invocation
To display all configuration groups, active paths, and current defaults, run:
```bash
python run_quantization_pipeline.py help=True
```

# PREDAP Train Quantized Production Pipeline

This repository hosts the specialized training, residual error correction, and post-training weight quantization pipeline for the PREDAP forecasting framework. The application orchestrates an additive multi-stage architecture leveraging **Hydra** for parameter abstraction and **MLflow** for full lineage tracking.

---

## 🏗 Multi-Stage Architecture Layout

The pipeline addresses sequential forecast error by stacking deep models sequentially to calculate, learn, and eliminate predictive variance:

1. **Phase 2 (Base Model):** Trains a standard core univariate Transformer model on core target historical values.
2. **Phase 3 (Diagnostic Correction):** Extracts the error residuals ($Real - Base\_Prediction$) and trains a second Transformer utilizing external target diagnostic covariates to correct information gaps.
3. **Phase 4 (Seasonal Correction):** Computes secondary error residuals from Phase 3 and trains a final structural Transformer utilizing rolling time-series features to capture lingering temporal variance.
4. **Quantization Engine:** Quantizes all structural model layers via targeted manual post-training operations to decrease downstream runtime hardware footprint.

---

## 🚀 Execution & CLI Parameter Modifications

The orchestrator utilizes **Hydra**, meaning you can modify any variable found inside your local `conf/config_production_quantization.yaml` right from your terminal without opening the source code.

### 1. Default Pipeline Trigger
Executes the pipeline utilizing the base values recorded inside your workspace config files:
```bash
python run_quantization_pipeline.py
```

### 2. Tailoring Pipeline Target Codes & Inputs
To isolate a different metrics code slice or supply a specific data file path:
```bash
python run_quantization_pipeline.py model.target_code="M54" data.data_path="data/production_v3.parquet"
```
### 3. Adjusting Architectural Capacities & Learning
To modulate network scale, learning paths, or attention configurations across your models:
```bash
python run_quantization_pipeline.py \
  model.lookback=120 \
  model.forecast=30 \
  model.num_transformer_blocks=4 \
  model.learning_rate=0.00025

```

### 4. Overriding Data Split Boundaries & Timeline Parameters
To change data timeline metrics or shift your training cutoff points:
```bash
python run_quantization_pipeline.py \
  training.cutoff_date="2024-12-31" \
  model.activation="gelu"
```



## CLI Help System Extension

This extension provides a dedicated, lightweight inspection mechanism inside the production pipeline. It allows engineers to query all acceptable Hydra paths and variables directly via the command console without parsing long config files or initializing heavy tracking dependencies.

---

## 🚀 Usage Guide

To invoke the help interface, append the `help=True` utility flag to your execution command. 

### 1. Basic Help Invocation
To display all configuration groups, active paths, and current defaults, run:
```bash
python run_quantization_pipeline.py help=True
```