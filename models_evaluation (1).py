"""
Model Evaluation Module for Day-Ahead Electricity Price Forecasting

This module provides functions to evaluate the performance of multiple machine learning models
for electricity price forecasting using various metrics and visualization tools.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import warnings
from pathlib import Path

# Metrics
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape

# Graphics
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from jupyterthemes import jtplot
except ImportError:
    jtplot = None
import plotly.express as px
import six

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def evaluation_project(zone, table_pred, column_pred, training_version, 
                      start_date, days_back=10, create_plots=False, data_dir="data"):
    """
    Evaluate model performance for a specific project.
    
    Parameters:
    -----------
    zone : str
        Bidding zone (e.g., 'DE')
    table_pred : str
        Table name for predictions (e.g., 'Prices')
    column_pred : str
        Column name for prediction target (e.g., 'DE_LU')
    training_version : str
        Version of the training (e.g., 'v1')
    start_date : str
        Start date for evaluation
    days_back : int
        Number of days back to evaluate
    create_plots : bool
        Whether to create evaluation plots
    data_dir : str
        Directory containing CSV files
        
    Returns:
    --------
    pd.DataFrame
        Evaluation results sorted by performance metrics
    """
    
    # Set pandas display options
    pd.set_option('display.max_columns', 100)
    
    # Create output directories
    models_path = Path("models")
    evaluation_path = Path("evaluation") / zone / table_pred / training_version
    evaluation_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Define time period
    today = pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S')
    start = (today - timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M:%S')
    end = today.strftime('%Y-%m-%d %H:%M:%S')
    
    # Define project name
    project = f"{zone}_{table_pred}_{training_version}_"
    print(f'Current project name to evaluate: {project}')
    
    try:
        # Load historical forecasts data
        historical_forecasts = pd.read_csv(Path(data_dir) / "historical_predictions.csv")
        historical_forecasts['Datetime'] = pd.to_datetime(
            historical_forecasts['Datetime'], format='%Y-%m-%d %H:%M:%S'
        )
        
        # Filter by date range
        historical_forecasts_filtered = historical_forecasts[
            (historical_forecasts['Datetime'] > start) & 
            (historical_forecasts['Datetime'] < end)
        ].copy()
        
        if historical_forecasts_filtered.empty:
            print('No forecast data for the selected dates')
            return pd.DataFrame()
            
        # Keep forecasts of the same project
        forecasts = historical_forecasts_filtered[historical_forecasts_filtered['project'] == project]
        
        if forecasts.empty:
            print(f'No forecast data for project: {project}')
            return pd.DataFrame()
        
        # Load actual data
        actual = pd.read_csv(Path(data_dir) / "actual_data.csv", sep=';')
        actual['Datetime'] = pd.to_datetime(actual['Datetime'], format='%Y-%m-%d %H:%M:%S')
        actual = actual[
            (actual['Datetime'] > start) & 
            (actual['Datetime'] < end)
        ].copy()
        
        # Load naive model data (previous day prices)
        naive_data = pd.read_csv(Path(data_dir) / "DE_LU_DAM_prices.csv")
        start_naive = pd.to_datetime(start) - timedelta(days=1)
        end_naive = pd.to_datetime(end) - timedelta(days=1)
        
        naive_data['Datetime'] = pd.to_datetime(naive_data['Datetime'], format='%Y-%m-%d %H:%M:%S')
        naive = naive_data[
            (naive_data['Datetime'] > start_naive) & 
            (naive_data['Datetime'] < end_naive)
        ].copy()
        
        # Preprocess data
        forecasts = forecasts.T.drop_duplicates().T  # Remove duplicate time columns
        forecasts.drop(['project'], axis=1, inplace=True, errors='ignore')
        
        # Shift naive model datetime one day ahead
        naive['Datetime'] = naive['Datetime'] + timedelta(days=1)
        naive.rename(columns={column_pred: 'Naive'}, inplace=True)
        
        # Merge datasets
        df = pd.merge(actual, forecasts, on='Datetime', how='inner')
        df = pd.merge(df, naive, on='Datetime', how='inner')
        
        print(f'Forecast missing values:\n{df.isnull().sum()}\nForecast table size: {df.shape}')
        
        # Remove columns with missing values
        df = df.loc[:, df.notna().all(axis=0)]
        
        print(f'Evaluation days: {df.shape[0]/24:.1f}')
        
        # Set datetime as index
        df.set_index('Datetime', inplace=True)
        df.replace(['None'], np.nan, inplace=True)
        df = df.astype(float)
        
        # Calculate metrics
        metrics_list = ['Model', 'MAE', 'RMSE', 'MAPE', 'sMAPE', 'NegPos']
        metrics = pd.DataFrame(columns=metrics_list)
        results_sub = pd.DataFrame(df.iloc[:, 0])
        
        # Calculate evaluation metrics for each model
        for column in df.columns.difference([column_pred]):
            df_clean = df[df[column].notna()]
            
            if df_clean.empty:
                continue
                
            # Calculate metrics
            mae_val = mae(df_clean[column_pred], df_clean[column]).round(2)
            rmse_val = rmse(df_clean[column_pred], df_clean[column], squared=False).round(2)
            mape_val = calculate_mape(df_clean[column_pred], df_clean[column]).round(2)
            smape_val = calculate_smape(df_clean[column_pred], df_clean[column]).round(2)
            neg_pos = round(len(df_clean.loc[df_clean[column_pred].mul(df_clean[column]).ge(0)]) / len(df_clean), 2)
            
            # Store metrics
            metric_series = pd.Series([column, mae_val, rmse_val, mape_val, smape_val, neg_pos], 
                                    index=metrics_list)
            metrics = pd.concat([metrics, metric_series.to_frame().T], ignore_index=True)
            
            # Store predictions and deviations
            results_sub = pd.concat([results_sub, df_clean[column]], axis=1)
            deviation_column = f"Deviation-{column}"
            deviation = pd.DataFrame(df_clean[column].subtract(df_clean[column_pred]), 
                                   columns=[deviation_column])
            results_sub = pd.concat([results_sub, deviation], axis=1)
        
        # Sort results by performance
        results_eval = metrics.dropna(axis=1, how='all').sort_values(
            by=['MAE', 'RMSE', 'MAPE', 'sMAPE']
        )
        results_eval.dropna(axis=0, how='any', inplace=True)
        
        # Save results to Excel
        excel_file_path = evaluation_path / 'Evaluation_Results.xlsx'
        try:
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode="a", 
                              if_sheet_exists='replace') as writer:
                results_eval.to_excel(writer, sheet_name="Evaluation Results", index=False)
        except FileNotFoundError:
            with pd.ExcelWriter(excel_file_path) as writer:
                results_eval.to_excel(writer, sheet_name="Evaluation Results", index=False)
        
        # Create best hourly models table
        best_model_per_hour = create_best_hourly_table(results_sub, column_pred)
        save_best_hourly_table(best_model_per_hour, zone, table_pred, training_version, models_path)
        
        # Create plots if requested
        if create_plots:
            create_evaluation_plots(results_eval, results_sub, column_pred, evaluation_path, df)
        
        # Create summary
        create_summary(results_eval, df, column_pred, evaluation_path)
        
        print('Evaluation completed')
        return results_eval.reset_index(drop=True)
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return pd.DataFrame()


def calculate_mape(actual, forecast):
    """Calculate Mean Absolute Percentage Error."""
    mape_values = []
    length = 0
    
    for i in range(len(actual)):
        if abs(actual.iloc[i]) > 10:
            per_err = abs(actual.iloc[i] - forecast.iloc[i]) / abs(actual.iloc[i])
            mape_values.append(per_err)
            length += 1
    
    return (sum(mape_values) / length * 100) if length > 0 else 0


def calculate_smape(actual, forecast):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    smape_values = []
    
    for i in range(len(actual)):
        denominator = abs(actual.iloc[i]) + abs(forecast.iloc[i])
        if denominator > 0:
            per_err = 2 * abs(actual.iloc[i] - forecast.iloc[i]) / denominator
            smape_values.append(per_err)
    
    return (sum(smape_values) / len(smape_values) * 100) if smape_values else 0


def create_best_hourly_table(results_sub, column_pred):
    """Create table of best models per hour."""
    # Create timestamp chart
    timestamp = pd.to_datetime(results_sub.index).hour
    timestamp_df = pd.DataFrame(timestamp, columns=['Datetime'])
    
    # Get deviation columns (excluding best models)
    deviation_columns = [col for col in results_sub.columns if 'Deviation' in col]
    best_columns = [col for col in deviation_columns if 'Best' in col]
    deviations_without_best = [col for col in deviation_columns if col not in best_columns]
    
    deviations = results_sub[deviations_without_best].abs()
    deviations.reset_index(drop=True, inplace=True)
    
    abs_deviations = pd.concat([timestamp_df, deviations], axis=1)
    
    # Group by hour and find best model for each hour
    group_by = abs_deviations.groupby(by="Datetime").mean()
    
    best_model_per_hour = group_by.apply(
        lambda x: pd.Series(x.nsmallest(len(group_by.columns)).index.values), axis=1
    )
    
    best_model_per_hour.reset_index(level=0, inplace=True)
    
    # Clean model names
    for m in range(1, len(best_model_per_hour.columns)):
        if best_model_per_hour.iloc[:, m].dtype == 'object':
            best_model_per_hour.iloc[:, m] = best_model_per_hour.iloc[:, m].str.replace(
                "Deviation-", "", regex=False
            )
    
    return best_model_per_hour


def save_best_hourly_table(best_model_per_hour, zone, table_pred, training_version, models_path):
    """Save best hourly table to pickle file."""
    table_file_path = models_path / 'Best_Hourly_Table.pkl'
    
    # Load existing dictionary or create new one
    if table_file_path.exists():
        with open(table_file_path, 'rb') as f:
            best_hourly_dict = pickle.load(f)
    else:
        best_hourly_dict = {}
    
    # Update dictionary
    if zone not in best_hourly_dict:
        best_hourly_dict[zone] = {}
    if table_pred not in best_hourly_dict[zone]:
        best_hourly_dict[zone][table_pred] = {}
    
    best_hourly_dict[zone][table_pred][training_version] = best_model_per_hour
    
    # Save updated dictionary
    with open(table_file_path, 'wb') as f:
        pickle.dump(best_hourly_dict, f)


def create_evaluation_plots(results_eval, results_sub, column_pred, evaluation_path, df):
    """Create evaluation plots and save them."""
    try:
        # Set plot style
        if jtplot:
            jtplot.style(theme='chesterish', context='paper', fscale=1.6, 
                        ticks=True, grid=False, figsize=(10, 10))
        
        # 1. Hourly deviations plot
        create_hourly_deviations_plot(results_sub, column_pred, evaluation_path)
        
        # 2. Best model per hour table
        create_best_hourly_table_plot(results_sub, column_pred, evaluation_path)
        
        # 3. Model evaluation bar chart
        create_model_evaluation_plot(results_eval, evaluation_path)
        
        # 4. Percentage evaluation plot
        create_percentage_evaluation_plot(results_eval, evaluation_path)
        
        # 5. Residuals boxplot
        create_residuals_boxplot(results_sub, evaluation_path)
        
        # 6. Residuals histogram
        create_residuals_histogram(results_sub, evaluation_path)
        
        # 7. Top models vs actual prices
        create_top_models_plots(results_eval, df, column_pred, evaluation_path)
        
    except Exception as e:
        print(f"Error creating plots: {e}")


def create_hourly_deviations_plot(results_sub, column_pred, evaluation_path):
    """Create hourly deviations plot."""
    timestamp = pd.to_datetime(results_sub.index).hour
    timestamp_df = pd.DataFrame(timestamp, columns=['Datetime'])
    
    deviation_columns = [col for col in results_sub.columns if 'Deviation' in col and 'Best' not in col]
    deviations = results_sub[deviation_columns].abs()
    deviations.reset_index(drop=True, inplace=True)
    
    abs_deviations = pd.concat([timestamp_df, deviations], axis=1)
    group_by = abs_deviations.groupby(by="Datetime").mean()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title('Deviations from actual price')
    group_by.plot(ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
    ax.set_ylabel('Mean absolute error')
    ax.set_xlabel('Hour')
    
    plt.savefig(evaluation_path / 'Hourly_Deviations.png', bbox_inches='tight', transparent=True)
    plt.close()


def create_best_hourly_table_plot(results_sub, column_pred, evaluation_path):
    """Create best model per hour table plot."""
    best_model_per_hour = create_best_hourly_table(results_sub, column_pred)
    
    def render_mpl_table(data, col_width=3.0, row_height=0.6, font_size=14,
                        header_color='#40466e', row_colors=['#f1f1f2', 'w'],
                        edge_color='w', bbox=[0, 0, 1, 1], ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)
        
        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        
        return ax
    
    render_mpl_table(best_model_per_hour.iloc[:, :6], header_columns=0, col_width=3.5)
    plt.savefig(evaluation_path / 'Best_model_per_hour.png', transparent=True)
    plt.close()


def create_model_evaluation_plot(results_eval, evaluation_path):
    """Create model evaluation bar chart."""
    fig = plt.figure(figsize=(25, 25))
    df = results_eval.reset_index()
    x = np.arange(len(df))
    w = 0.4
    
    ax1 = plt.subplot(1, 1, 1)
    plt.xticks(x + w/2, df['Model'], rotation='vertical')
    
    mae_bars = ax1.bar(x=x, height=df['MAE'], width=w, color='b', align='center')
    ax2 = ax1.twinx()
    rmse_bars = ax2.bar(x=x + w, height=df['RMSE'], width=w, color='g', align='center')
    
    ax1.set_ylabel('MAE')
    ax2.set_ylabel('RMSE')
    plt.xlabel('Model')
    plt.title('Model evaluation')
    plt.legend([mae_bars, rmse_bars], ['MAE', 'RMSE'], loc="upper left", prop=dict(size=30))
    
    plt.savefig(evaluation_path / 'Model_evaluation.png', bbox_inches='tight', dpi=100, transparent=True)
    plt.close()


def create_percentage_evaluation_plot(results_eval, evaluation_path):
    """Create percentage evaluation plot."""
    df = results_eval.reset_index()
    fig = plt.figure(figsize=(25, 25))
    ax1 = plt.subplot(1, 1, 1)
    
    x = np.arange(len(df))
    w = 0.4
    plt.xticks(x + w/2, df['Model'], rotation='vertical')
    
    mape_bars = ax1.bar(x=x, height=df['MAPE'], width=w, color='b', align='center')
    ax2 = ax1.twinx()
    smape_bars = ax2.bar(x=x + w, height=df['sMAPE'], width=w, color='g', align='center')
    
    ax1.set_ylabel('MAPE')
    ax2.set_ylabel('sMAPE')
    plt.xlabel('Model')
    plt.title('Model evaluation')
    plt.legend([mape_bars, smape_bars], ['MAPE', 'sMAPE'], loc="upper left", prop=dict(size=30))
    
    plt.savefig(evaluation_path / 'Model_percentage_evaluation.png', bbox_inches='tight', dpi=100, transparent=True)
    plt.close()


def create_residuals_boxplot(results_sub, evaluation_path):
    """Create residuals boxplot."""
    deviation_columns = [col for col in results_sub.columns if 'Deviation' in col]
    residuals = results_sub[deviation_columns]
    
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    residuals.boxplot(vert=False, grid=False, fontsize=16)
    plt.title('Residuals distribution - boxplot', fontsize=20)
    ax.set_xlabel('Residuals', fontsize=18)
    
    plt.savefig(evaluation_path / 'Residuals_evaluation.png', bbox_inches='tight', 
               dpi=100, facecolor='#e6f2ff')
    plt.close()


def create_residuals_histogram(results_sub, evaluation_path):
    """Create residuals histogram."""
    deviation_columns = [col for col in results_sub.columns if 'Deviation' in col]
    residuals = results_sub[deviation_columns]
    
    fig = plt.figure(figsize=(26, 26))
    ax = fig.gca()
    residuals.hist(ax=ax, bins=20)
    
    plt.savefig(evaluation_path / 'Residuals_histogram.png', bbox_inches='tight', 
               dpi=100, facecolor='#e6f2ff')
    plt.close()


def create_top_models_plots(results_eval, df, column_pred, evaluation_path):
    """Create plots for top 3 models vs actual prices."""
    df_reset = df.reset_index()
    df_reset["Datetime"] = df_reset["Datetime"].dt.strftime('%d-%m-%Y %H:%M')
    
    for i in range(min(3, len(results_eval))):
        best_model = str(results_eval.iloc[i, 0])
        
        if best_model in df_reset.columns:
            fig, ax = plt.subplots(figsize=(20, 15))
            
            plt.plot(df_reset["Datetime"], df_reset[column_pred], 
                    label='Actual prices', color='#2A3D54')
            plt.plot(df_reset["Datetime"], df_reset[best_model], 
                    label=f'{best_model} model', color="#8A1108")
            
            ax.set_ylabel('Price [Euro/MWh]', fontsize=18)
            ax.set_xlabel('Datetime', fontsize=16)
            ax.tick_params(labelsize=16, size=4)
            ax.set_xticks(ax.get_xticks()[::24])
            plt.xticks(rotation=65)
            plt.legend(prop={"size": 18}, loc="upper left")
            
            plt.savefig(evaluation_path / f'Top_{i+1}_model_vs_Actual_prices.png', 
                       bbox_inches='tight', dpi=100, facecolor='#FFFFFF')
            plt.close()


def create_summary(results_eval, df, column_pred, evaluation_path):
    """Create summary text file."""
    if results_eval.empty:
        return
        
    accuracy_metric = ((np.mean(df[column_pred]) - results_eval.iloc[0, 1]) * 100 / 
                      np.mean(df[column_pred])).round(2)
    
    df_reset = df.reset_index()
    
    notepad = [
        f'Evaluation period: {df_reset.iloc[0, 0]} until {df_reset.iloc[-1, 0]}',
        f'Days in the dataset: {df.shape[0]/24:.1f}',
        f'Top model: {results_eval.iloc[0, 0]}',
        f'with MAE in euro/MWh: {results_eval.iloc[0, 1]}',
        f'Average Market price in euro/MWh: {np.mean(df[column_pred]):.2f}',
        f'Top model calculated accuracy %: {accuracy_metric}'
    ]
    
    with open(evaluation_path / 'Summary_results.txt', 'w') as f:
        for line in notepad:
            f.write(line + '\n')


if __name__ == "__main__":
    # Example usage
    results = evaluation_project(
        zone="DE",
        table_pred="Prices", 
        column_pred="DE_LU",
        training_version="v1",
        start_date="2024-02-04 00:00:00",
        days_back=10,
        create_plots=True
    )
    print("Evaluation completed!")
