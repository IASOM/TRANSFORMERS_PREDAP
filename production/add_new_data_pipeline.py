
import keras
import mlflow
from sklearn.pipeline import FunctionTransformer
import tensorflow as tf
import numpy as np
import os
import argparse
import pandas as pd
from datetime import timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.base_transformer_config import BaseTransformerConfig

class AddNewDataPipeline:
    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.config.print_config()

    def add_new_data(self, new_data_path: str, cutoff_date: str, max_date: str, 
                 eliminate_covid_data: bool, covid_token: bool, 
                 provided_data: list = None):
        
        """
        Appends a new row of data to the dataset. If manual data isn't provided, 
        it imputes values using a 3-year seasonal mean (same day/month).

        Args:
            new_data_path (str): Path to the .parquet file containing the source data.
            cutoff_date (str): The starting boundary for data processing.
            max_date (str): The upper boundary for the dataset timeline.
            eliminate_covid_data (bool): If True, excludes 2020-2022 from mean calculations.
            covid_token (bool): If True, adds/maintains a boolean flag for COVID periods.
            provided_data (list, optional): An array of values to use for the new row. 
                                            Must match the number of feature columns.

        Returns:
            pd.DataFrame: The original dataset plus the new row for the 'next day'.
        """
        # Load Data
        df = pd.read_parquet(new_data_path)
        df = df[(df['timestamp'] >= cutoff_date) & (df['timestamp'] <= max_date)].reset_index(drop=True) 
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. Determine the "Next Day"
        last_date = df['timestamp'].max()
        next_day = last_date + timedelta(days=1)
        
        # 2. Prepare the pool for mean calculation (Imputation Pool)
        # We keep the original 'df' intact for the final return
        mean_pool_df = df.copy()
        if eliminate_covid_data:
            # Exclude common COVID impact years (2020-2022) from the MEAN only
            mean_pool_df = mean_pool_df[~mean_pool_df['timestamp'].dt.year.isin([2020, 2021, 2022])]

        # 3. Handle New Row Generation
        data_cols = [col for col in df.columns if col != 'timestamp']
        new_row = {'timestamp': next_day}

        if provided_data is not None:
            # Use provided array (ensure length matches column count)
            for i, col in enumerate(data_cols):
                new_row[col] = provided_data[i]
        else:
            # Perform Imputation: Mean of same day/month for last 3 years
            for col in data_cols:
                target_years = [next_day.year - 1, next_day.year - 2, next_day.year - 3]
                
                seasonal_matches = mean_pool_df[
                    (mean_pool_df['timestamp'].dt.month == next_day.month) &
                    (mean_pool_df['timestamp'].dt.day == next_day.day) &
                    (mean_pool_df['timestamp'].dt.year.isin(target_years))
                ][col]
                
                # If no historical matches found for that specific day, 
                # fallback to the overall column mean
                new_row[col] = seasonal_matches.mean() if not seasonal_matches.empty else mean_pool_df[col].mean()

        # 4. Handle covid_token column if it exists/is requested
        if 'covid_token' in df.columns:
            new_row['covid_token'] = 0 # Assume the 'next day' is not a COVID peak

        # 5. Combine and Return
        new_row_df = pd.DataFrame([new_row])
        result_df = pd.concat([df, new_row_df], ignore_index=True)
        
        return result_df

    def save_updated_data(self, df: pd.DataFrame, save_path: str, save_name: str, delete_old: bool = False):
        """
        Saves the updated DataFrame to a .parquet file.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            save_path (str): The directory path where the file will be saved.
            save_name (str): The name of the saved file (without extension).
            delete_old (bool): If True, deletes the existing file with the same name before saving.
        """
        save_name = save_name.split('_')[0] + "_" + save_name.split('_')[1]
         # Remove existing file if it exists
        old_path = os.path.join(save_path, f"{save_name}.parquet")
        if os.path.exists(old_path) and delete_old:
            os.remove(old_path)
            print(f"Deleted existing file: {old_path}")
        else:
            print(f"No existing file to delete at: {old_path}")

        last_date = df['timestamp'].max()

        full_save_path = os.path.join(save_path, f"{save_name}_{last_date.strftime('%Y-%m-%d')}.parquet")
        df.to_parquet(full_save_path, index=False)
        print(f"Updated dataset saved to {full_save_path}")

        return full_save_path


    

if __name__ == "__main__":
    config = BaseTransformerConfig()
    pipeline = AddNewDataPipeline(config)
    updated_df = pipeline.add_new_data(
        new_data_path=config.data_path, 
        cutoff_date=config.cutoff_date, 
        max_date=config.final_cutoff_date, 
        eliminate_covid_data=config.eliminate_covid_data, 
        covid_token=config.covid_token
    )
    print(updated_df.tail())
    save_path = '../data/FINAL_DB1'
    save_name = config.data_path.split('/')[-1].replace('.parquet', '') 
    pipeline.save_updated_data(
        df=updated_df, 
        save_path=save_path, 
        save_name=save_name,
        delete_old=True
    )