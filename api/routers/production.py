from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.base_transformer_config import BaseTransformerConfig

from production.add_new_data_pipeline import AddNewDataPipeline
from production.model_reconstruction_pipeline import ModelPredictionPipeline
from api.schemas.production_schemas import AddNewDataRequest
from api.schemas.production_schemas import ModelReconstructionRequest



router = APIRouter(prefix="/production", tags=["production"])


@router.post("/add_new_data")
def add_new_data(request: AddNewDataRequest = None):
    """
    Appends a new row of data to the dataset. If manual data isn't provided,
    it imputes values using a 3-year seasonal mean (same day/month).
    """
    try:
        config = BaseTransformerConfig()
        pipeline = AddNewDataPipeline(config)

        # Use request values or fall back to config defaults
        new_data_path = request.new_data_path if request and request.new_data_path else config.data_path
        cutoff_date = request.cutoff_date if request and request.cutoff_date else config.cutoff_date
        max_date = request.max_date if request and request.max_date else config.final_cutoff_date
        eliminate_covid_data = request.eliminate_covid_data if request and request.eliminate_covid_data is not None else config.eliminate_covid_data
        covid_token = request.covid_token if request and request.covid_token is not None else config.covid_token
        provided_data = request.provided_data if request else None
        save_path = request.save_path if request and request.save_path else "../data/FINAL_DB1"
        delete_old = request.delete_old if request and request.delete_old is not None else True

        # Run the pipeline to add new data
        updated_df = pipeline.add_new_data(
            new_data_path=new_data_path,
            cutoff_date=cutoff_date,
            max_date=max_date,
            eliminate_covid_data=eliminate_covid_data,
            covid_token=covid_token,
            provided_data=provided_data
        )

        # Generate save name from the data path
        save_name = new_data_path.split('/')[-1].replace('.parquet', '')

        # Save the updated data
        saved_path = pipeline.save_updated_data(
            df=updated_df,
            save_path=save_path,
            save_name=save_name,
            delete_old=delete_old
        )

        # Get summary info from the updated dataframe
        last_row = updated_df.iloc[-1].to_dict()
        last_row['timestamp'] = str(last_row['timestamp'])

        return JSONResponse(content={
            "status": "success",
            "message": "New data added successfully!",
            "saved_path": saved_path,
            "total_rows": len(updated_df),
            "new_row": last_row
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding new data: {str(e)}")
    

@router.get("/model_reconstruction_pipeline")
def model_reconstruction_pipeline(request: ModelReconstructionRequest):
    """
    Endpoint to trigger the model reconstruction pipeline. This will retrain the model using the updated dataset.
    """

    try:
        # Placeholder for actual model reconstruction logic
        # You would call your model training functions here, passing in the updated dataset path if needed
        config = BaseTransformerConfig()
        data_preparation = ModelPredictionPipeline(config)

        final_output_df = data_preparation.run_reconstruct_save_results_pipeline(
            code=request.code,
            forecast_horizon_list=request.forecast_horizon_list,
            lookback_list=request.lookback_list,
            data_path=request.data_path,
            save_path=request.save_path
        )

        data_preparation.save_final_output_predictions(final_output_df, save_path=request.save_path)


        return JSONResponse(content={
            "status": "success",
            "message": "Model reconstruction pipeline triggered successfully!",
            "final_output_df": final_output_df.shape if final_output_df is not None else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering model reconstruction: {str(e)}")

@router.delete("/delete_old_data")
def delete_old_data():
    """
    Deletes old data from the production predictions dataset based on the logic:
    If the difference in days between target_date and forecast_date equals the forecast value, delete that row.
    
    Args:
        dataset_path (str): The path to the dataset from which old data should be deleted.
    
    Returns:
        str: The path to the updated dataset after deletion.
    """
    dataset_path = "../production_predictions/final_output_predictions.parquet"
    metrics_df_path = "../production_predictions/production_evaluation_metrics.parquet"
    input_directory = '../data/FINAL_DB/full_CAT1.parquet'


    if not os.path.exists(dataset_path):
        print(f"No data file found at: {dataset_path}")
        return None

    try:    
        config = BaseTransformerConfig()
        pipeline = ModelPredictionPipeline(config)
        updated_path = pipeline.delete_old_data(predictions_dataset_path=dataset_path, real_data_dataset_path=input_directory, metrics_df_path=metrics_df_path)
        return JSONResponse(content={
            "status": "success",
            "message": f"Old data deleted successfully from: {updated_path}",
            "updated_dataset_path": updated_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting old data: {str(e)}")
    




