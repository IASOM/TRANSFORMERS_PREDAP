"""
Main Training and Evaluation Script for Univariate Transformer - Class-Based
============================================================================
This script provides a class-based approach for orchestrating the entire training and evaluation pipeline.
"""

import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import pickle
import mlflow

# Import data preparation module
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'src' else current_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config.base_transformer_config import BaseTransformerConfig
import data_preparation
from causal_transformer.main_architechture_univ_causal1 import build_transformer_forecaster
from causal_transformer.evaluate_univ_causal import evaluate_univ_causal_transformer
from causal_transformer.training_evaluation_univ_causal import train_given_causal_model_and_data


from univariate_transformer import (
    build_model, CustomCosineDecay,
    setup_gpu_memory, create_model_directories, 
    create_pandemic_waves_df, load_and_preprocess_data,
    plt_model,
    plot_predictions_with_waves,

)



@dataclass
class TransformerUnivConfig(BaseTransformerConfig):
    """Configuration class for transformer training parameters"""
    
    # Configuration object reference
    config_object: Optional[Any] = field(default=None)
    
    # Encoder-Decoder Architecture Parameters
    use_encoder_decoder: bool = field(default=True)
    num_encoder_blocks: int = field(default=4)
    num_decoder_blocks: int = field(default=2)
    decoder_target_length: int = field(default=1)
    #decoder_strategy: str = field(default='learned')  # 'zeros', 'learned', 'random'
    decoder_strategy: str = field(default='autoregressive')  # 'zeros', 'learned', 'random'
    
    # Teacher Forcing Parameters
    use_teacher_forcing: bool = field(default=False)
    start_token_value: float = field(default=0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "lookback": self.lookback,
            "forecast": self.forecast,
            "code": self.code,
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_units": self.mlp_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "cutoff_date": self.cutoff_date,
            "covid_token": self.covid_token,
            "positional_encoding": self.positional_encoding,
            "activation_function": self.activation_function,
            "evaluate_model": self.evaluate_model, 
            "data_path": self.data_path,
        }
    
    def __post_init__(self):
        """Initialize configuration after dataclass creation"""
        # Call parent's __post_init__ first
        super().__post_init__()
        
        # Set diagnostic_covariates_path using the code from parent
        if self.diagnostic_covariates_path is None:
            self.diagnostic_covariates_path = f'../data/best_features/BEST_features_NOSMOOTH_{self.code}.xlsx'
        else:
            self.diagnostic_covariates_path =  self.diagnostic_covariates_path + f'{self.code}.xlsx'
    
    def print_config(self):
        """Print configuration in a readable format"""
        print("\n" + "="*60)
        print("TRANSFORMER TRAINING CONFIGURATION")
        print("="*60)
        for key, value in self.to_dict().items():
            print(f"{key:25}: {value}")
        print("="*60)
    
    def get_lr_schedule_params(self, learning_rate=None):
        """Get learning rate schedule parameters."""
        if learning_rate is not None:
            lr_init = learning_rate
        else:
            lr_init = self.learning_rate
        lr_max = lr_init * self.lr_max_multiplier
        lr_min = lr_init * self.lr_min_multiplier
        warmup_steps = int(self.epochs * self.lr_warmup_ratio)
        
        return {
            'initial_lr': lr_init,
            'max_lr': lr_max,
            'min_lr': lr_min,
            'warmup_steps': warmup_steps,
            'total_steps': self.epochs
        }


class UnivariateTransformerPipeline:
    """
    Main class for managing the complete univariate transformer training and evaluation pipeline.
    """
    
    def __init__(self, config: TransformerUnivConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: TransformerUnivConfig object containing all parameters
        """
        self.config = config
        self.model = None
        self.model_name = None
        self.training_history = None
        self.evaluation_results = None
        self.data_prep_time = 0
        
        # Initialize paths
        self.data_path =self.config.data_path
        self.plots_dir = self.config.plots_dir
        
    def setup_environment(self):
        """Setup GPU memory and create necessary directories"""
        print("Setting up environment...")
        setup_gpu_memory()
        create_model_directories()
        print("Environment setup complete!")

    def prepare_time_features(self, df: pd.DataFrame, generate_y: bool = False) -> np.ndarray:
        """
        Prepare time-based features for the dataset.
        
        Args:
            df: Input DataFrame with datetime index
        Returns:"""

        # Prepare seasonal features for training data
        print("Preparing seasonal features for training data...")
        df_train_processed = data_preparation.prepare_time_series_features(
            df, 
            self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS, 
            cutoff_date=self.config.cutoff_date,
            scaler = self.config.scaler,
        )
        
        # Generate rolling sequences with covariates for training
        print("Generating rolling sequences with seasonal covariates for training...")
        self.X_dates = data_preparation.generate_rolling_sequences_covariates(
            df_train_processed, 
            lookback= self.config.lookback, 
            forecast = self.config.forecast,
            generate_y=True
        )

        self.Y_dates = data_preparation.generate_rolling_sequences_covariates(
            df_train_processed, 
            lookback= self.config.lookback, 
            forecast = self.config.forecast,
            
        )


        
        print(f"Data preparation completed!")

        return self.X_dates, self.Y_dates
    
    def load_diagnostic_covariates(self):
        diagnostic_covariates_df = pd.read_excel(self.config.diagnostic_covariates_path, engine='openpyxl')
        self.diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df['LAG'] == self.config.forecast]['predictors'])[0].split(',')
    
        return self.diagnostic_covariates_list
    
    def prepare_causal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data using the configuration parameters.
        
        Returns:
            Tuple of (X, Y) training data arrays
        """
        print("\n" + "="*50)
        print("DATA PREPARATION PHASE")
        print("="*50)
        
        start_time = time.perf_counter()

        relevant_feature_cols = self.load_diagnostic_covariates()
        self.config.relevant_feature_cols = relevant_feature_cols    


        X_train, Y_train = data_preparation.prepare_data(
            self.data_path, 
            self.config.code, 
            self.config.lookback, 
            self.config.forecast,
            covid_token=self.config.covid_token, 
            cutoff_date=self.config.cutoff_date,
            train=True, 
            debug=True, 
            univariate=False,
            relevant_feature_cols = self.config.relevant_feature_cols,
            scaler = self.config.scaler,
        )

        X_test, Y_test = data_preparation.prepare_data(
            self.data_path, 
            self.config.code, 
            self.config.lookback, 
            self.config.forecast,
            covid_token=self.config.covid_token, 
            cutoff_date=self.config.cutoff_date,
            train=False, 
            debug=True, 
            univariate=False, 
            relevant_feature_cols = self.config.relevant_feature_cols,
            scaler = self.config.scaler,
            
        )

        dataframe = pd.read_csv(self.data_path)
        dataframe_cut = data_preparation.cut_dataframe(dataframe, date_cutoff=self.config.cutoff_date)
        # Load and split the original data for covariate extraction
        train_split, test_split = data_preparation.split_train_test(
            dataframe_cut, 
            split_ratio=0.8, 
        )

        X_time_train, Y_time_train = self.prepare_time_features(df=train_split)
        X_time_train = X_time_train.astype(np.float32)
        Y_time_train = Y_time_train.astype(np.float32)
        X_time_test, Y_time_test = self.prepare_time_features(df=test_split, generate_y=True)
        X_time_test = X_time_test.astype(np.float32)
        Y_time_test = Y_time_test.astype(np.float32)
        
        #train_dataset = self.build_dataset(X_train, X_time_train, Y_train, batch_size=self.config.batch_size, shuffle=False)
        #test_dataset = self.build_dataset(X_test, X_time_test, Y_test, batch_size=self.config.batch_size, shuffle=False)
        
        return X_train, Y_train, X_test, Y_test, X_time_train, Y_time_train, X_time_test, Y_time_test
    
    def build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """
        Build the transformer model based on configuration.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras model
        """
        print("\n" + "="*50)
        print("MODEL BUILDING PHASE")
        # Build training model
        model = build_transformer_forecaster(
            enc_len=self.config.lookback, target_len=self.config.forecast,
            in_dim=12, out_dim=1, time_dim=12,
            time2vec_dim=16, d_model=self.config.mlp_units, num_heads=self.config.num_heads, d_ff=self.config.ff_dim,
            enc_layers=self.config.num_transformer_blocks, dec_layers=self.config.num_transformer_blocks, dropout=self.config.dropout,
        )
        model.summary()
        
        return model
    
    def setup_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Setup training callbacks including learning rate scheduler and early stopping.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Learning rate scheduler
        lr_params = self.config.get_lr_schedule_params(learning_rate=self.config.learning_rate)
        scheduler = CustomCosineDecay(**lr_params)
        
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks.append(lr_callback)
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=self.config.early_stop_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        return callbacks
    
    def compile_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Compile the model with appropriate optimizer and loss.
        
        Args:
            model: The model to compile
            
        Returns:
            Compiled model
        """
        model.compile(
            loss='MAE', 
            metrics=['mae', 'mse'], 
            optimizer=Adam(
                clipnorm=2.0, 
                learning_rate=self.config.learning_rate, 

            )
        )
        return model
    
    def train_model(self, X:np.ndarray, Y: np.ndarray, save_history: bool = True, save_model: bool = True) -> Dict[str, Any]:
        """
        Train the transformer model.
        
        Args:
            X: Input training data
            Y: Target training data
            
        Returns:
            Dictionary containing training results
        """
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model((self.config.lookback, X[0].shape[-1]))
            self.model = self.compile_model(self.model)
        
        # Generate model name
        self.model_name = self.config.get_model_name()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        model_hist = self.model.fit(
            x=X,
            y=Y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=0.3,
            shuffle=False,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = model_hist.history

        if save_history:  # save training history
            raw_model_name = self.model_name.replace('.keras', '')
            os.makedirs('../history', exist_ok=True)
            with open(f'../history/{raw_model_name}_history.pkl', 'wb') as file_pi:
                pickle.dump(self.training_history, file_pi)
    
        if save_model and self.config.epochs > 1:  # save model
            os.makedirs(self.config.model_folder, exist_ok=True)
            self.model.save(self.config.model_folder +"/" + self.model_name)



        predictions_train = self.model.predict(X)

        print("Training completed successfully!")
        return model_hist, predictions_train
    
    
    def evaluate_model(self, X_test: np.ndarray, Y_test: np.ndarray, X_time_test: np.ndarray, Y_time_test: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Evaluate the trained model if evaluation is enabled.
        
        Returns:
            Tuple of (loss, mae, mse) if evaluation is performed, None otherwise
        """
        
        if not self.config.evaluate_model or self.model_name is None:
            return None
            
        print("\n" + "="*50)
        print("EVALUATION PHASE")
        print("="*50)
        
        print("Files in model folder:", os.listdir(self.config.model_folder))
        
        trained_models = [f for f in os.listdir(self.config.model_folder) if f.endswith('.keras')]
        #print("Trained models:", trained_models)
        
        # Create pandemic waves DataFrame
        df_waves = create_pandemic_waves_df()

        start_token = np.zeros((Y_test.shape[0], Y_test.shape[1], 1)) 
        decoder_input = start_token
        

        # Initialize decoder time features
        decoder_time_input = Y_time_test.copy() #np.zeros((Y_time_test.shape[0], Y_time_test.shape[1], Y_time_test.shape[-1]))
        predictions = []
        loss = 0 
        loss_fn = tf.keras.losses.MeanAbsoluteError()


        for step in range(self.config.forecast):

            # 1. model takes: 
            # (encoder_inputs, encoder_time_feats, decoder_inputs)
            pred = self.model([X_test, X_time_test, decoder_input, decoder_time_input], training = False)

            # pred has shape: (1, 1, num_outputs)
            next_value = pred[:, 0:1, :]       # last predicted step

            predictions.append(next_value)
            loss += loss_fn(Y_test[:, step:step+1], next_value)
            # 2. append predicted value as next decoder token
            decoder_input[:,step:step+1,:] = next_value

            # Update decoder time features (use last encoder time feature as approximation)
            next_time_feature = Y_time_test[:, 0:1, :]  # Take last encoder time feature
            decoder_time_input[:,step:step+1,:] = next_time_feature

        predictions = tf.concat(predictions, axis=1).numpy()

        original_scale_df = pd.read_csv(self.config.data_path)
        predictions_to_plot = data_preparation.inverse_causal_transform_predictions(
                                                predictions, original_scale_df, 
                                                self.config.code, 
                                                forecast=self.config.forecast, 
                                                lookback=self.config.lookback, 
                                                cutoff_date=self.config.cutoff_date,
                                                scaler = self.config.scaler,

                                                )
        
        # Evaluate model
        X_test_orig, Y_test_orig = data_preparation.prepare_data_not_normalized(
            self.config.data_path, self.config.code, self.config.lookback, self.config.forecast,covid_token=self.config.covid_token,  cutoff_date=self.config.cutoff_date, train=False, debug=True, univariate=True
            )
        

        original_mae = mean_absolute_error(Y_test_orig, predictions_to_plot)
        original_mse = mean_squared_error(Y_test_orig, predictions_to_plot)
        original_rmse = np.sqrt(original_mse)
        print(f"Test Results - Loss: {loss:.4f}, MAE: {original_mae:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}")
        

        mlflow.log_metrics({
            "eval/univ_transformer_loss": loss,
            "eval/univ_transformer_mae": original_mae,
            "eval/univ_transformer_mse": original_mse,
            "eval/univ_transformer_rmse": original_rmse
        })


        # Generate plots
        model_display_name = self.model_name.replace('.keras', '')

        date_list = data_preparation.extract_dates(self.config.data_path, self.config.code, self.config.lookback, self.config.forecast, train=False, cutoff_date=self.config.cutoff_date)
        plt_model(Y_test_orig, predictions_to_plot, date_list, model_name=model_display_name, show_plt=False)
        # Plot predictions with pandemic waves
        plot_predictions_with_waves(Y_test_orig, predictions_to_plot, date_list, df_waves, model_display_name)


        # Sliding window evaluation (optional)
        # evaluate_model_sliding_window(model, model_display_name, X_test, Y_test, date_list, df_waves, sliding_window=forecast)
        return loss, original_mae, original_mse, predictions

    
    
    def build_dataset(X_enc, X_time, Y, batch_size = 32, shuffle = False):
        ds = tf.data.Dataset.from_tensor_slices(((X_enc, X_time), Y))
        if shuffle:
            ds = ds.shuffle(1000)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    def run_complete_pipeline(self) -> Tuple[tf.keras.Model, str, Optional[float], Optional[float], Optional[float]]:
        """
        Run the complete training and evaluation pipeline.
        
        Returns:
            Tuple of (model, model_name, loss, mae, mse)
        """
        # Print configuration
        self.config.print_config()
        
        # Setup environment
        self.setup_environment()
        
        # Prepare data
        X_train, Y_train, X_test, Y_test, X_time_train, Y_time_train, X_time_test, Y_time_test = self.prepare_causal_data()

        
        Y_train_reshape = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)
        start = np.zeros((X_train.shape[0],1,Y_train_reshape.shape[-1]))
        dec_input = np.concatenate([start, Y_train_reshape[:, :-1, :]], axis=1)
        # Train model
        history, predictions_train = self.train_model([X_train, X_time_train, dec_input, Y_time_train], Y_train)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(X_test, Y_test, X_time_test, Y_time_test)

        
        # Extract evaluation metrics
        loss, mae, mse = None, None, None
        if evaluation_results is not None:
            loss, mae, mse, predictions_test = evaluation_results
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model name: {self.model_name}")
        print(f"Data preparation time: {self.data_prep_time:.2f} seconds")
        if evaluation_results:
            print(f"Final metrics - Loss: {loss:.6f}, MAE: {mae:.6f}, MSE: {mse:.6f}")
        
        return self.model, self.model_name, loss, mae, mse, predictions_train, predictions_test
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of pipeline results.
        
        Returns:
            Dictionary containing all results and configuration
        """
        return {
            "config": self.config.to_dict(),
            "model_name": self.model_name,
            "training_history": self.training_history,
            "evaluation_results": self.evaluation_results,
            "data_preparation_time": self.data_prep_time,
            "model_summary": self.model.summary() if self.model else None
        }


# Convenience function for backward compatibility
def main_univ_transformer(lookback: int, forecast: int, code: str, **kwargs) -> Tuple[tf.keras.Model, str, Optional[float], Optional[float], Optional[float]]:
    """
    Main function that orchestrates the training and evaluation pipeline using the class-based approach.
    This function maintains backward compatibility while using the new OOP structure internally.
    
    Args:
        lookback: Number of past time steps to use for prediction
        forecast: Number of future time steps to predict
        code: Target code for data filtering
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (model, model_name, loss, mae, mse)
    """
    
    # Create configuration object
    config = TransformerUnivConfig(
        lookback=lookback,
        forecast=forecast,
        code=code,
        decoder_target_length=1,
        **kwargs
    )
    
    # Create and run pipeline
    pipeline = UnivariateTransformerPipeline(config)
    return pipeline.run_complete_pipeline()




if __name__ == "__main__":
    print("Choose training mode:")
    print("1. Standard encoder-only transformer")
    print("2. Encoder-decoder transformer")
    print("3. Teacher forcing encoder-decoder transformer")
    print("4. Original example")
    
    choice = input("Enter choice (1, 2, 3, or 4): ").strip()
    
    
    # Original Example: Using the class-based approach directly
    transformer_config = TransformerUnivConfig(
                            lookback=7,
                            forecast=7,
                            code="T14",
                            activation_function='gelu',
                            cutoff_date="2008-01-01",
                            evaluate_model=True,
                            positional_encoding=False,
                        )

    pipeline = UnivariateTransformerPipeline(transformer_config)
    model, model_name, loss, mae, mse = pipeline.run_complete_pipeline()
    print(f"Original example completed! Model: {model_name}")
else:
    print("Invalid choice. Running teacher forcing example by default.")
