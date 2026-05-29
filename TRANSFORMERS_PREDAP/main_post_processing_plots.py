"""
Interactive 3D Visualization for Transformer Model Performance Analysis

This module creates interactive 3D plots showing the relationship between:
- Mean Absolute Error (X-axis)
- Forecast Horizon (Y-axis) 
- Lookback Window (Z-axis)

Separate plots are generated for:
- Different target codes (T14, T15, etc.)
- Different model types (DIAGNOSTIC vs SEASONAL residual models)

Data is extracted from JSON files in the results folder, with model type 
determined from filename and all other information from JSON content.
"""

import json
import glob
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime

class Interactive3DPlotter:
    """
    Class for creating interactive 3D plots of transformer model performance
    """
    
    def __init__(self, results_directory: str = "results"):
        """
        Initialize the plotter with results directory
        
        Args:
            results_directory: Path to the results folder containing JSON files
        """
        self.results_dir = results_directory
        self.data = None
        
    def extract_model_type_from_filename(self, filename: str) -> str:
        """
        Extract only model type from filename (DIAGNOSTIC or SEASONAL)
        
        Args:
            filename: JSON filename
            
        Returns:
            Model type string
        """
        if 'DIAGNOSTIC' in filename:
            return 'DIAGNOSTIC'
        elif 'SEASONAL' in filename:
            return 'SEASONAL'
        else:
            return 'UNKNOWN'
    
    def load_results_data(self) -> pd.DataFrame:
        """
        Load and parse all JSON result files
        
        Returns:
            DataFrame with all model results
        """
        # Find all JSON files in results directory
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.results_dir}")
        
        data_list = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    result_data = json.load(f)
                
                # Extract model type from filename (only info we get from filename)
                filename = os.path.basename(json_file)
                model_type = self.extract_model_type_from_filename(filename)
                
                # Extract all other information from JSON data
                model_info = result_data.get('model_info', {})
                
                # Get target code from JSON data
                target_code = model_info.get('target_code', 'Unknown')
                
                # Get model parameters from JSON data
                forecast_horizon = model_info.get('forecast_horizon', 0)
                lookback_window = model_info.get('lookback_window', 0)
                
                # Extract additional model parameters from model name if available
                model_name = model_info.get('residual_model_name', '')
                ff_dim = 0
                learning_rate = 0.0
                
                # Parse ff_dim and learning_rate from model name
                if model_name:
                    ff_match = re.search(r'(\d+)ff', model_name)
                    ff_dim = int(ff_match.group(1)) if ff_match else 0
                    
                    lr_match = re.search(r'(0\.\d+)initlr', model_name)
                    learning_rate = float(lr_match.group(1)) if lr_match else 0.0
                
                # Extract performance metrics
                if 'corrected_model_performance' in result_data:
                    performance = result_data['corrected_model_performance']
                elif 'original_model_performance' in result_data:
                    performance = result_data['original_model_performance']
                else:
                    continue
                
                # Create data point with information from JSON
                data_point = {
                    'target_code': target_code,
                    'model_type': model_type,
                    'forecast_horizon': forecast_horizon,
                    'lookback_window': lookback_window,
                    'ff_dim': ff_dim,
                    'learning_rate': learning_rate,
                    'filename': filename,
                    'MAE': performance.get('MAE', 0),
                    'MSE': performance.get('MSE', 0),
                    'RMSE': performance.get('RMSE', 0),
                    'timestamp': model_info.get('evaluation_timestamp', ''),
                    'mae_improvement': result_data.get('improvements', {}).get('MAE_improvement_percent', 0),
                    'mse_improvement': result_data.get('improvements', {}).get('MSE_improvement_percent', 0),
                    'rmse_improvement': result_data.get('improvements', {}).get('RMSE_improvement_percent', 0),
                    'overall_assessment': result_data.get('improvements', {}).get('overall_assessment', 'neutral')
                }
                
                data_list.append(data_point)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        if not data_list:
            raise ValueError("No valid data found in JSON files")
        
        self.data = pd.DataFrame(data_list)
        print(f"✅ Loaded {len(self.data)} model results")
        print(f"📊 Target codes found: {sorted(self.data['target_code'].unique())}")
        print(f"🔧 Model types found: {sorted(self.data['model_type'].unique())}")
        
        return self.data
    
    def create_3d_plot(self, 
                      target_code: str, 
                      model_type: str, 
                      metric: str = 'MAE') -> go.Figure:
        """
        Create 3D scatter plot for specific target code and model type
        
        Args:
            target_code: Target code (e.g., 'T14')
            model_type: Model type ('DIAGNOSTIC' or 'SEASONAL')
            metric: Metric to plot on Z-axis ('MAE', 'MSE', 'RMSE')
            
        Returns:
            Plotly figure object
        """
        # Filter data
        filtered_data = self.data[
            (self.data['target_code'] == target_code) & 
            (self.data['model_type'] == model_type)
        ]
        
        if filtered_data.empty:
            print(f"⚠️ No data found for {target_code} - {model_type}")
            return None
        
        # Create color mapping based on performance improvement
        colors = []
        for _, row in filtered_data.iterrows():
            if row['overall_assessment'] == 'positive':
                colors.append('green')
            elif row['overall_assessment'] == 'negative':
                colors.append('red')
            else:
                colors.append('orange')
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add scatter plot with corrected axis assignment: MAE (X), Forecast (Y), Lookback (Z)
        scatter = go.Scatter3d(
            x=filtered_data[metric],  # MAE on X-axis
            y=filtered_data['forecast_horizon'],  # Forecast on Y-axis
            z=filtered_data['lookback_window'],  # Lookback on Z-axis
            mode='markers',
            marker=dict(
                size=8,
                color=filtered_data[f'{metric.lower()}_improvement'],
                colorscale='RdYlGn',
                colorbar=dict(
                    title=f"{metric} Improvement (%)",
                    titleside="right"
                ),
                showscale=True,
                opacity=0.8,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[
                f"Code: {row['target_code']}<br>"
                f"Type: {row['model_type']}<br>"
                f"{metric}: {row[metric]:.6f}<br>"
                f"FH: {row['forecast_horizon']}<br>"
                f"LB: {row['lookback_window']}<br>"
                f"FF Dim: {row['ff_dim']}<br>"
                f"LR: {row['learning_rate']}<br>"
                f"Improvement: {row[f'{metric.lower()}_improvement']:.2f}%<br>"
                f"Assessment: {row['overall_assessment']}"
                for _, row in filtered_data.iterrows()
            ],
            hovertemplate='%{text}<extra></extra>',
            name=f"{target_code} - {model_type}"
        )
        
        fig.add_trace(scatter)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"3D Performance Analysis: {target_code} - {model_type} Models<br>"
                     f"<span style='font-size:12px'>{metric} vs Forecast Horizon vs Lookback Window</span>",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(
                    title=f"{metric} Value",
                    titlefont=dict(size=14),
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title="Forecast Horizon (days)",
                    titlefont=dict(size=14),
                    tickfont=dict(size=12)
                ),
                zaxis=dict(
                    title="Lookback Window (days)",
                    titlefont=dict(size=14),
                    tickfont=dict(size=12)
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig
    
    def create_comparative_plot(self, 
                               target_code: str, 
                               metric: str = 'MAE') -> go.Figure:
        """
        Create comparative 3D plot showing both DIAGNOSTIC and SEASONAL models
        
        Args:
            target_code: Target code to analyze
            metric: Metric to plot on Z-axis
            
        Returns:
            Plotly figure with subplots
        """
        # Filter data for target code
        target_data = self.data[self.data['target_code'] == target_code]
        
        if target_data.empty:
            print(f"⚠️ No data found for {target_code}")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(f'{target_code} - DIAGNOSTIC Models', 
                           f'{target_code} - SEASONAL Models'),
            horizontal_spacing=0.1
        )
        
        # Add DIAGNOSTIC models
        diagnostic_data = target_data[target_data['model_type'] == 'DIAGNOSTIC']
        if not diagnostic_data.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=diagnostic_data[metric],  # MAE on X-axis
                    y=diagnostic_data['forecast_horizon'],  # Forecast on Y-axis
                    z=diagnostic_data['lookback_window'],  # Lookback on Z-axis
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=diagnostic_data[f'{metric.lower()}_improvement'],
                        colorscale='RdYlBu',
                        showscale=True,
                        opacity=0.8,
                        colorbar=dict(x=0.45, title="Improvement (%)")
                    ),
                    text=[
                        f"{metric}: {row[metric]:.6f}<br>"
                        f"FH: {row['forecast_horizon']}<br>"
                        f"LB: {row['lookback_window']}<br>"
                        f"Improvement: {row[f'{metric.lower()}_improvement']:.2f}%"
                        for _, row in diagnostic_data.iterrows()
                    ],
                    hovertemplate='%{text}<extra></extra>',
                    name='DIAGNOSTIC'
                ),
                row=1, col=1
            )
        
        # Add SEASONAL models
        seasonal_data = target_data[target_data['model_type'] == 'SEASONAL']
        if not seasonal_data.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=seasonal_data[metric],  # MAE on X-axis
                    y=seasonal_data['forecast_horizon'],  # Forecast on Y-axis
                    z=seasonal_data['lookback_window'],  # Lookback on Z-axis
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=seasonal_data[f'{metric.lower()}_improvement'],
                        colorscale='RdYlGn',
                        showscale=True,
                        opacity=0.8,
                        colorbar=dict(x=1.05, title="Improvement (%)")
                    ),
                    text=[
                        f"{metric}: {row[metric]:.6f}<br>"
                        f"FH: {row['forecast_horizon']}<br>"
                        f"LB: {row['lookback_window']}<br>"
                        f"Improvement: {row[f'{metric.lower()}_improvement']:.2f}%"
                        for _, row in seasonal_data.iterrows()
                    ],
                    hovertemplate='%{text}<extra></extra>',
                    name='SEASONAL'
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Comparative 3D Analysis: {target_code} Models<br>"
                     f"<span style='font-size:12px'>DIAGNOSTIC vs SEASONAL Performance</span>",
                x=0.5,
                font=dict(size=16)
            ),
            width=1400,
            height=700,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        # Update scene properties for both subplots
        for i in [1, 2]:
            fig.update_scenes(
                xaxis_title=f"{metric} Value",
                yaxis_title="Forecast Horizon", 
                zaxis_title="Lookback Window",
                row=1, col=i
            )
        
        return fig
    
    def generate_all_plots(self, show_plots: bool = True):
        """
        Generate all possible 3D plots for available data
        
        Args:
            show_plots: Whether to display plots in browser
        """
        if self.data is None:
            self.load_results_data()
        
        # Get unique combinations
        target_codes = sorted(self.data['target_code'].unique())
        model_types = sorted(self.data['model_type'].unique())
        metrics = ['MAE', 'MSE', 'RMSE']
        
        print("🎯 Generating 3D visualization plots...")
        print(f"📊 Target codes: {target_codes}")
        print(f"🔧 Model types: {model_types}")
        print(f"📈 Metrics: {metrics}")
        
        plots_generated = 0
        
        # Generate individual plots for each combination
        for target_code in target_codes:
            for model_type in model_types:
                # Check if data exists for this combination
                subset = self.data[
                    (self.data['target_code'] == target_code) & 
                    (self.data['model_type'] == model_type)
                ]
                
                if subset.empty:
                    continue
                
                for metric in metrics:
                    try:
                        fig = self.create_3d_plot(target_code, model_type, metric)
                        if fig:
                            plots_generated += 1
                            if show_plots:
                                fig.show()
                            
                            # Save plot as HTML
                            filename = f"3D_plot_{target_code}_{model_type}_{metric}.html"
                            fig.write_html(filename)
                            print(f"💾 Saved: {filename}")
                    
                    except Exception as e:
                        print(f"❌ Error creating plot for {target_code}-{model_type}-{metric}: {e}")
        
        # Generate comparative plots for each target code
        for target_code in target_codes:
            for metric in metrics:
                try:
                    fig = self.create_comparative_plot(target_code, metric)
                    if fig:
                        plots_generated += 1
                        if show_plots:
                            fig.show()
                        
                        # Save comparative plot
                        filename = f"3D_comparative_{target_code}_{metric}.html"
                        fig.write_html(filename)
                        print(f"💾 Saved: {filename}")
                
                except Exception as e:
                    print(f"❌ Error creating comparative plot for {target_code}-{metric}: {e}")
        
        print(f"✅ Generated {plots_generated} interactive 3D plots")
        return plots_generated
    
    def create_overview_dashboard(self) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple metrics
        
        Returns:
            Plotly figure with dashboard layout
        """
        if self.data is None:
            self.load_results_data()
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('MAE Distribution', 'MSE Distribution', 
                           'RMSE Distribution', 'Performance Improvement'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        metrics = ['MAE', 'MSE', 'RMSE']
        positions = [(1,1), (1,2), (2,1)]
        
        # Add plots for each metric
        for i, metric in enumerate(metrics):
            row, col = positions[i]
            
            for model_type in ['DIAGNOSTIC', 'SEASONAL']:
                subset = self.data[self.data['model_type'] == model_type]
                if subset.empty:
                    continue
                
                color_map = 'Viridis' if model_type == 'DIAGNOSTIC' else 'Plasma'
                
                fig.add_trace(
                    go.Scatter3d(
                        x=subset[metric],  # MAE on X-axis
                        y=subset['forecast_horizon'],  # Forecast on Y-axis
                        z=subset['lookback_window'],  # Lookback on Z-axis
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=subset[metric],
                            colorscale=color_map,
                            opacity=0.7,
                            showscale=False
                        ),
                        name=f'{model_type}',
                        legendgroup=model_type,
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        # Add improvement plot
        fig.add_trace(
            go.Scatter3d(
                x=self.data['mae_improvement'],  # MAE improvement on X-axis
                y=self.data['forecast_horizon'],  # Forecast on Y-axis
                z=self.data['lookback_window'],  # Lookback on Z-axis
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.data['mae_improvement'],
                    colorscale='RdYlGn',
                    colorbar=dict(x=1.1, title="Improvement (%)"),
                    opacity=0.8,
                    showscale=True
                ),
                text=[f"Code: {row['target_code']}<br>Type: {row['model_type']}<br>Improvement: {row['mae_improvement']:.2f}%" 
                     for _, row in self.data.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name='All Models',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Transformer Models Performance Dashboard",
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig


def main():
    """
    Main function to run the 3D plotting analysis
    """
    print("🚀 Starting Interactive 3D Model Performance Analysis")
    print("=" * 60)
    
    # Initialize plotter
    plotter = Interactive3DPlotter("c:/Users/Sira/Escritorio/predapProject/results")
    
    try:
        # Load data
        print("📥 Loading results data...")
        plotter.load_results_data()
        
        # Generate all plots
        print("\n🎨 Generating interactive 3D plots...")
        plots_count = plotter.generate_all_plots(show_plots=True)
        
        # Create overview dashboard
        print("\n📊 Creating performance dashboard...")
        dashboard = plotter.create_overview_dashboard()
        dashboard.show()
        dashboard.write_html("3D_performance_dashboard.html")
        print("💾 Saved: 3D_performance_dashboard.html")
        
        print(f"\n✅ Analysis complete! Generated {plots_count} plots + 1 dashboard")
        print("🌐 All plots are interactive - you can zoom, rotate, and hover for details")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
