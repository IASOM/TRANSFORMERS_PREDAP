# REST API (FastAPI)

Predap exposes a production-ready REST API built with **FastAPI** for model inference, data management, and pipeline orchestration.

---

## Starting the Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## App Configuration

```python
# api/main.py
from fastapi import FastAPI
from api.routers import production

app = FastAPI(title="Predap API")
app.include_router(production.router)
```

---

## Endpoints

All production endpoints are mounted under the `/production` prefix.

### `POST /production/add_new_data`

Appends a new row of data to the dataset. If manual data is not provided, values are imputed using a **3-year seasonal mean** (same day/month from the 3 most recent years).

**Request Body** (`AddNewDataRequest`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `new_data_path` | `str?` | Config default | Path to the source `.parquet` file |
| `cutoff_date` | `str?` | `"2008-01-01"` | Start boundary for data processing |
| `max_date` | `str?` | `"2025-09-30"` | Upper boundary for data timeline |
| `eliminate_covid_data` | `bool?` | `false` | Exclude COVID years from mean calculations |
| `covid_token` | `bool?` | `true` | Add/maintain COVID period flag |
| `provided_data` | `list[float]?` | `null` | Manual values for the new row (must match feature count) |
| `save_path` | `str?` | `"../data/FINAL_DB1"` | Output directory for updated file |
| `delete_old` | `bool?` | `true` | Delete existing file before saving |

**Response:**

```json
{
  "status": "success",
  "message": "New data added successfully!",
  "saved_path": "/path/to/saved/file.parquet",
  "total_rows": 6205,
  "new_row": { "timestamp": "2025-10-01", "J00": 45.0, ... }
}
```

---

### `GET /production/model_reconstruction_pipeline`

Triggers the full model reconstruction pipeline: loads quantized model weights, reconstructs the architecture, generates predictions, and saves partitioned Parquet output.

**Request Body** (`ModelReconstructionRequest`):

| Field | Type | Description |
|-------|------|-------------|
| `code` | `str` | Target diagnostic code (e.g., `"J00"`) |
| `forecast_horizon` | `int` | Forecast horizon in days |
| `head_size` | `int` | Attention head dimensionality |
| `num_heads` | `int` | Number of attention heads |
| `ff_dim` | `int` | Feed-forward hidden dimension |
| `num_transformer_blocks` | `int` | Number of Transformer blocks |
| `mlp_units` | `int` | Units per MLP layer |
| `activation_function` | `str` | Activation function name |
| `dropout` | `float` | Dropout rate (default: `0.0`) |
| `learning_rate` | `float` | Learning rate (default: `0.001`) |
| `epochs` | `int` | Number of training epochs (default: `50`) |
| `batch_size` | `int` | Batch size (default: `32`) |
| `cutoff_date` | `str` | Training data cutoff date |
| `covid_token` | `bool` | COVID token flag (default: `true`) |
| `positional_encoding` | `bool` | Positional encoding flag (default: `true`) |
| `evaluate_model` | `bool` | Run evaluation after reconstruction (default: `true`) |
| `data_path` | `str` | Path to input data directory |
| `save_path` | `str` | Path to save results |

**Response:**

```json
{
  "status": "success",
  "message": "Model reconstruction pipeline triggered successfully!",
  "final_output_df": [100, 8]
}
```

---

### `DELETE /production/delete_old_data`

Removes expired forecast rows from the production predictions dataset. A row is deleted when the difference between `target_date` and `forecast_date` equals the `forecast` horizon value.

**No request body required.** Uses hardcoded paths:

- Predictions: `../production_predictions/final_output_predictions.parquet`
- Metrics: `../production_predictions/production_evaluation_metrics.parquet`
- Real data: `../data/FINAL_DB/full_CAT1.parquet`

**Response:**

```json
{
  "status": "success",
  "message": "Old data deleted successfully from: /path/to/updated.parquet",
  "updated_dataset_path": "/path/to/updated.parquet"
}
```

---

## Production Pipeline Classes

The API endpoints delegate to these pipeline classes:

| Class | Module | Responsibility |
|-------|--------|---------------|
| `AddNewDataPipeline` | `production/add_new_data_pipeline.py` | Append data using seasonal mean imputation |
| `ModelPredictionPipeline` | `production/model_reconstruction_pipeline.py` | Reconstruct model → predict → save Parquet |
| `DataPreparationInProduction` | `production/data_preparation_in_poduction.py` | Base class for production data prep |

---

## Error Handling

All endpoints return standard HTTP error codes:

| Code | Condition |
|------|-----------|
| `200` | Success |
| `404` | Data file not found |
| `500` | Internal processing error (details in response body) |
