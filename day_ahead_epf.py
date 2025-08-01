"""
Day-Ahead Electricity Price Forecasting (EPF) System

This module implements a comprehensive system for day-ahead electricity price forecasting
using ensemble methods combining machine learning and time series models.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from functools import reduce
import pickle
import json
import requests
import collections

# Time series and ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Deep learning
try:
    from tensorflow import keras
    from keras.models import model_from_json
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Keras/TensorFlow not available. Neural network models will be skipped.")

# Time series models
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import pmdarima as pm
    from scalecast.Forecaster import Forecaster
    from tbats import TBATS
    from ThymeBoost import ThymeBoost as tb
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    print("Time series libraries not available. Time series models will be skipped.")

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from jupyterthemes import jtplot
except ImportError:
    jtplot = None

import plotly.express as px
import plotly.graph_objects as go

# Local imports
try:
    import Models_Evaluation
except ImportError:
    print("Models_Evaluation module not found. Evaluation functionality will be limited.")

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DayAheadEPF:
    """Day-Ahead Electricity Price Forecasting System."""
    
    def __init__(self, data_dir="data", models_dir="models", evaluation_dir="evaluation"):
        """
        Initialize the EPF system.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing input CSV files
        models_dir : str
            Directory containing trained models
        evaluation_dir : str
            Directory for evaluation outputs
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.evaluation_dir = Path(evaluation_dir)
        
        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.countries_dict = {
            'v1': {
                'DE': {
                    'Fundamentals_forecast': ['Load_DE', 'Wind_DE', 'Solar_DE', 'Hydro_DE', 'Nuc_DE'],
                    'GasSpot': ['THE'],
                    'EUA': ['EUA_DecLastPrice'],
                    'Prices': ['DE_LU']
                }
            }
        }
        
        self.daily_granularity = ['GasSpot', 'EUA', 'GasFutures']
        
        # ML Model names
        self.ml_model_names = [
            'LinearRegression', 'Lasso', 'Ridge', 'kNN', 'XGB', 'RandomForest', 
            'GAM', 'SVM', 'GBM', 'AdaBoost', 'CatBoost', 'EXT', 'MLP'
        ]
        
        self.nn_model_names = [
            'LSTM', 'BILSTM', 'GRU', 'LSTM1', 'LSTM2', 'LSTM3', 'LSTM4',
            'GRU1', 'GRU2', 'GRU3', 'BILSTM1', 'BILSTM2', 'BILSTM3'
        ] if KERAS_AVAILABLE else []
        
        self.model_names = self.ml_model_names + self.nn_model_names
        
        # Initialize prediction storage
        self.sugg_predictions = collections.defaultdict(lambda: collections.defaultdict(dict))
    
    def load_csv_data(self, filename, separator=','):
        """Load CSV data with error handling."""
        try:
            file_path = self.data_dir / filename
            if separator == ';':
                df = pd.read_csv(file_path, sep=';')
            else:
                df = pd.read_csv(file_path)
            
            # Convert Datetime column
            df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S')
            return df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def prepare_forecast_data(self, zone, training_version, start_datetime, end_datetime):
        """
        Prepare data for forecasting by loading and merging different data sources.
        
        Parameters:
        -----------
        zone : str
            Bidding zone (e.g., 'DE')
        training_version : str
            Training version (e.g., 'v1')
        start_datetime : str
            Start datetime for forecast period
        end_datetime : str
            End datetime for forecast period
            
        Returns:
        --------
        pd.DataFrame
            Merged data ready for forecasting
        """
        list_of_dfs = []
        
        # Load evaluation data (contains fundamentals forecast)
        eval_data = self.load_csv_data("evaluation_data.csv", separator=';')
        if not eval_data.empty:
            # Filter for forecast period
            forecast_data = eval_data[
                (eval_data['Datetime'] >= start_datetime) & 
                (eval_data['Datetime'] <= end_datetime)
            ].copy()
            
            if not forecast_data.empty:
                list_of_dfs.append(forecast_data)
        
        if not list_of_dfs:
            print("No forecast data available for the specified period")
            return pd.DataFrame()
        
        # Merge all dataframes
        if len(list_of_dfs) == 1:
            data = list_of_dfs[0]
        else:
            data = reduce(lambda x, y: pd.merge(x, y, on='Datetime', how='outer'), list_of_dfs)
        
        print(f'\nMissing values:\n{data.isnull().sum()}\nTable size: {data.shape}')
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        print('\nAll NaN values were imputed')
        
        # Set datetime as index and convert to float
        data.set_index('Datetime', inplace=True)
        data = data.apply(pd.to_numeric, errors='coerce', downcast='float')
        
        return data
    
    def load_trained_model(self, model_name, project_name):
        """
        Load a pre-trained model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
        project_name : str
            Project identifier
            
        Returns:
        --------
        tuple
            (model, scaler) if successful, (None, None) otherwise
        """
        try:
            if model_name in self.nn_model_names and KERAS_AVAILABLE:
                # Load neural network model
                json_path = self.models_dir / f"{project_name}{model_name}.json"
                h5_path = self.models_dir / f"{project_name}{model_name}.h5"
                
                if json_path.exists() and h5_path.exists():
                    with open(json_path, 'r') as json_file:
                        loaded_model_json = json_file.read()
                    model = model_from_json(loaded_model_json)
                    model.load_weights(str(h5_path))
                else:
                    print(f"Model files not found for {model_name}")
                    return None, None
            else:
                # Load sklearn/other models
                model_path = self.models_dir / f"{project_name}{model_name}.sav"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    print(f"Model file not found for {model_name}")
                    return None, None
            
            # Load scaler
            scaler_path = self.models_dir / f"{project_name}X_scaler.sav"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                print(f"Scaler not found for {project_name}")
                return model, None
            
            return model, scaler
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None, None
    
    def make_model_forecast(self, model_name, x_forecasts, project_name):
        """
        Make forecast using a trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        x_forecasts : pd.DataFrame
            Features for forecasting
        project_name : str
            Project identifier
            
        Returns:
        --------
        np.array
            Forecast values
        """
        model, scaler = self.load_trained_model(model_name, project_name)
        
        if model is None:
            print(f"Could not load model {model_name}")
            return np.full(len(x_forecasts), np.nan)
        
        try:
            # Scale features if scaler is available
            if scaler is not None:
                x_forecasts.columns = scaler.get_feature_names_out()
                x_forecasts_scaled = scaler.transform(x_forecasts)
            else:
                x_forecasts_scaled = x_forecasts.values
            
            # Prepare data for neural networks
            if model_name in self.nn_model_names and KERAS_AVAILABLE:
                time_steps = 1
                xs = []
                for i in range(len(x_forecasts_scaled) - time_steps + 1):
                    v = x_forecasts_scaled[i:i + time_steps, :]
                    xs.append(v)
                
                if xs:
                    x_forecasts_final = np.array(xs)
                    y_forecasts_model = model.predict(x_forecasts_final)
                    
                    # Load Y scaler and inverse transform
                    y_scaler_path = self.models_dir / f"{project_name}Y_scaler.sav"
                    if y_scaler_path.exists():
                        with open(y_scaler_path, 'rb') as f:
                            scaler_y = pickle.load(f)
                        y_forecasts = scaler_y.inverse_transform(y_forecasts_model.reshape(-1, 1)).flatten()
                        # Handle the last prediction
                        y_forecasts = np.append(y_forecasts, y_forecasts[-1])
                    else:
                        y_forecasts = y_forecasts_model.flatten()
                else:
                    y_forecasts = np.full(len(x_forecasts), np.nan)
            else:
                # Regular ML models
                y_forecasts_model = model.predict(x_forecasts_scaled)
                if model_name == 'MLP':
                    y_forecasts = y_forecasts_model.flatten()
                else:
                    y_forecasts = y_forecasts_model
            
            return np.round(y_forecasts.astype(float), 2)
            
        except Exception as e:
            print(f"Error making forecast with {model_name}: {e}")
            return np.full(len(x_forecasts), np.nan)
    
    def run_timeseries_models(self, prices_data, forecasts_df, column_pred):
        """
        Run time series models for forecasting.
        
        Parameters:
        -----------
        prices_data : pd.DataFrame
            Historical price data
        forecasts_df : pd.DataFrame
            Forecasts dataframe to update
        column_pred : str
            Target column name
            
        Returns:
        --------
        pd.DataFrame
            Updated forecasts dataframe
        """
        if not TIMESERIES_AVAILABLE:
            print("Time series libraries not available. Skipping time series models.")
            return forecasts_df
        
        try:
            # SARIMA model
            sarima_model = SARIMAX(
                endog=prices_data[column_pred],
                order=(3, 0, 3),
                seasonal_order=(2, 1, 0, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = sarima_model.fit(disp=False)
            predictions = results.get_forecast(24).conf_int(alpha=0.05)
            forecasts_df['SARIMA_TS'] = predictions.mean(axis=1).reset_index(drop=True)
            
        except Exception as e:
            print(f"Error with SARIMA model: {e}")
        
        try:
            # ETS model
            ets_model = ExponentialSmoothing(
                endog=prices_data[column_pred],
                trend='additive',
                seasonal='additive',
                seasonal_periods=12,
                damped=True
            )
            ets_fitted = ets_model.fit(optimized=True, remove_bias=False)
            predictions = ets_fitted.forecast(24)
            forecasts_df['ETS_TS'] = predictions.reset_index(drop=True)
            
        except Exception as e:
            print(f"Error with ETS model: {e}")
        
        try:
            # ThymeBoost model
            boosted_model = tb.ThymeBoost(verbose=1)
            output = boosted_model.fit(
                prices_data[column_pred],
                trend_estimator=['linear', 'arima'],
                arima_order='auto',
                global_cost='mse'
            )
            predictions = boosted_model.predict(output, 24)
            forecasts_df['GBLA_TS'] = predictions.reset_index(drop=True)['predictions']
            
        except Exception as e:
            print(f"Error with GBLA model: {e}")
        
        # Additional scalecast models
        if TIMESERIES_AVAILABLE:
            self._run_scalecast_models(prices_data, forecasts_df, column_pred)
        
        return forecasts_df
    
    def _run_scalecast_models(self, prices_data, forecasts_df, column_pred):
        """Run scalecast-based time series models."""
        df = prices_data.set_index('Datetime').asfreq('h')
        
        models_config = [
            ('LSTM_TS', 'lstm', {
                'lags': 24, 'batch_size': 32, 'epochs': 20, 'validation_split': 0.2,
                'shuffle': True, 'activation': 'tanh', 'optimizer': 'Adam',
                'learning_rate': 0.001, 'lstm_layer_sizes': (72,)*4,
                'dropout': (0,)*4, 'plot_loss': False
            }),
            ('Ridge_TS', 'ridge', {'alpha': 0.11}),
            ('Lasso_TS', 'lasso', {'alpha': 0.01}),
            ('GBT_TS', 'gbt', {'max_depth': 3, 'max_features': None}),
            ('kNN_TS', 'knn', {'n_neighbors': 61}),
            ('MLP_TS', 'mlp', {
                'activation': 'relu', 'hidden_layer_sizes': (25,), 'solver': 'lbfgs'
            }),
            ('MLR_TS', 'mlr', {}),
            ('RF_TS', 'rf', {
                'max_depth': 5, 'n_estimators': 100, 'max_features': 'sqrt', 'max_samples': 0.75
            }),
            ('SVR_TS', 'svr', {'kernel': 'linear', 'C': 3, 'epsilon': 0.01}),
            ('XBD_TS', 'xgboost', {
                'n_estimators': 150, 'scale_pos_weight': 5, 'learning_rate': 0.1,
                'gamma': 5, 'subsample': 0.8
            }),
            ('LXBD_TS', 'lightgbm', {
                'n_estimators': 150, 'boosting_type': 'goss', 'max_depth': 2, 'learning_rate': 0.1
            })
        ]
        
        for model_name, estimator, params in models_config:
            try:
                f = Forecaster(y=df[column_pred], current_dates=df.index)
                f.set_test_length(24)
                f.generate_future_dates(24)
                f.add_ar_terms(1)
                f.add_AR_terms((3, 3))
                f.add_seasonal_regressors('day', raw=False, sincos=True)
                f.add_time_trend()
                f.set_estimator(estimator)
                f.manual_forecast(**params)
                forecasts_df[model_name] = f.forecast
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
    
    def create_custom_models(self, forecasts_df, zone, table_pred, training_version, 
                           column_pred, today, days_back):
        """
        Create custom ensemble models based on historical performance.
        
        Parameters:
        -----------
        forecasts_df : pd.DataFrame
            Current forecasts
        zone : str
            Bidding zone
        table_pred : str
            Prediction table name
        training_version : str
            Training version
        column_pred : str
            Target column
        today : datetime
            Current date
        days_back : int
            Days to look back for evaluation
            
        Returns:
        --------
        tuple
            (updated_forecasts_df, suggested_model)
        """
        try:
            # Load historical forecasts for best model selection
            best_models = self._find_best_historical_models(forecasts_df, zone, table_pred, 
                                                          training_version, column_pred, today)
            
            if best_models:
                # Create ensemble models
                forecasts_df['Best_Daily'] = forecasts_df.get(best_models.get('3'), np.nan)
                
                if len(best_models) >= 3:
                    model_3 = forecasts_df.get(best_models.get('3'), 0)
                    model_2 = forecasts_df.get(best_models.get('2'), 0)
                    model_1 = forecasts_df.get(best_models.get('1'), 0)
                    
                    forecasts_df['Best_811'] = model_3 * 0.8 + model_2 * 0.1 + model_1 * 0.1
                    forecasts_df['Best_721'] = model_3 * 0.7 + model_2 * 0.2 + model_1 * 0.1
                    forecasts_df['Best_631'] = model_3 * 0.6 + model_2 * 0.3 + model_1 * 0.1
                    forecasts_df['Best_532'] = model_3 * 0.5 + model_2 * 0.3 + model_1 * 0.2
            
            # Create best hourly model
            best_hourly = self._create_best_hourly_model(forecasts_df, zone, table_pred, 
                                                       training_version, today, days_back)
            if best_hourly is not None:
                forecasts_df['Best_Hourly'] = best_hourly
            
            # Create average of best models
            best_columns = ['Best_Hourly', 'Best_Daily', 'Best_811', 'Best_721', 'Best_631', 'Best_532']
            available_best = [col for col in best_columns if col in forecasts_df.columns]
            if available_best:
                forecasts_df['Best_Average'] = forecasts_df[available_best].mean(axis=1)
            
            # Find suggested model
            suggested_model = self._find_suggested_model(zone, table_pred, training_version, 
                                                       today, days_back, forecasts_df.columns)
            
            return forecasts_df, suggested_model
            
        except Exception as e:
            print(f"Error creating custom models: {e}")
            return forecasts_df, None
    
    def _find_best_historical_models(self, forecasts_df, zone, table_pred, training_version, 
                                   column_pred, today):
        """Find best performing models from historical data."""
        best_models = {}
        days = 3
        search_ending = 60
        i = 0
        
        # Load historical predictions
        historical_data = self.load_csv_data("historical_predictions.csv")
        if historical_data.empty:
            return best_models
        
        # Load actual data for comparison
        actual_data = self.load_csv_data("actual_data.csv", separator=';')
        if actual_data.empty:
            return best_models
        
        project = f"{zone}_{table_pred}_{training_version}_"
        
        while days > 0 and i < search_ending:
            # Define date range for each past day
            day_from = pd.to_datetime(today) - timedelta(days=i)
            day_until = day_from + timedelta(hours=23)
            
            # Get historical forecasts for this day
            day_forecasts = historical_data[
                (historical_data['Datetime'] >= day_from) &
                (historical_data['Datetime'] <= day_until) &
                (historical_data['project'] == project)
            ].copy()
            
            if not day_forecasts.empty:
                # Get actual prices for comparison
                actual_prices = actual_data[
                    (actual_data['Datetime'] >= day_from) &
                    (actual_data['Datetime'] <= day_until)
                ].copy()
                
                if not actual_prices.empty and column_pred in actual_prices.columns:
                    # Calculate metrics for each model
                    metrics = []
                    exclude_cols = ['Datetime', 'project']
                    model_cols = [col for col in day_forecasts.columns if col not in exclude_cols]
                    
                    for model_col in model_cols:
                        if model_col in forecasts_df.columns:  # Only consider available models
                            try:
                                mae_val = mean_absolute_error(
                                    actual_prices[column_pred].values,
                                    day_forecasts[model_col].values
                                )
                                metrics.append({'model': model_col, 'mae': mae_val})
                            except:
                                continue
                    
                    if metrics:
                        # Sort by MAE and select best model
                        sorted_metrics = sorted(metrics, key=lambda d: d['mae'])
                        best_model = sorted_metrics[0]['model']
                        best_models[str(days)] = best_model
                        print(f'For day {days}, best model found: {best_model}')
                        days -= 1
            
            i += 1
        
        return best_models
    
    def _create_best_hourly_model(self, forecasts_df, zone, table_pred, training_version, 
                                today, days_back):
        """Create best hourly model based on historical performance."""
        try:
            # Run evaluation to get best hourly table
            evaluation_df = Models_Evaluation.evaluation_project(
                zone=zone,
                table_pred=table_pred,
                column_pred=self.countries_dict[training_version][zone]['Prices'][0],
                training_version=training_version,
                start_date=today.strftime('%Y-%m-%d %H:%M:%S'),
                days_back=days_back,
                create_plots=False,
                data_dir=str(self.data_dir)
            )
            
            # Load best hourly ranking table
            ranking_file = self.models_dir / 'Best_Hourly_Table.pkl'
            if ranking_file.exists():
                with open(ranking_file, 'rb') as f:
                    ranking_table = pickle.load(f)
                
                if (zone in ranking_table and 
                    table_pred in ranking_table[zone] and 
                    training_version in ranking_table[zone][table_pred]):
                    
                    table = ranking_table[zone][table_pred][training_version]
                    best_model_result = []
                    
                    for hour in range(24):
                        best_prediction = np.nan
                        # Try successive models based on ranking
                        for col in range(1, len(table.columns)):
                            model_name = table.iloc[hour, col]
                            if model_name in forecasts_df.columns:
                                best_prediction = forecasts_df.loc[hour, model_name]
                                break
                        best_model_result.append(best_prediction)
                    
                    return pd.Series(best_model_result)
            
        except Exception as e:
            print(f"Error creating best hourly model: {e}")
        
        return None
    
    def _find_suggested_model(self, zone, table_pred, training_version, today, 
                            days_back, available_columns):
        """Find the suggested model based on recent evaluation."""
        try:
            evaluation_df = Models_Evaluation.evaluation_project(
                zone=zone,
                table_pred=table_pred,
                column_pred=self.countries_dict[training_version][zone]['Prices'][0],
                training_version=training_version,
                start_date=today.strftime('%Y-%m-%d %H:%M:%S'),
                days_back=days_back,
                create_plots=False,
                data_dir=str(self.data_dir)
            )
            
            # Find first available model from evaluation results
            for i in range(len(evaluation_df)):
                model_name = evaluation_df.iloc[i, 0]
                if model_name in available_columns:
                    return model_name
            
        except Exception as e:
            print(f"Error finding suggested model: {e}")
        
        return None
    
    def run_forecasting(self, zones=['DE'], starting_date=None, days_ahead=1, 
                       training_version='v1', table_pred='Prices',
                       start_date_ts='2022-06-09', days_back=20,
                       naive_model=False, time_series=True, create_plots=False):
        """
        Run the complete forecasting pipeline.
        
        Parameters:
        -----------
        zones : list
            List of bidding zones to forecast
        starting_date : str, optional
            Starting date in 'dd/mm/yyyy' format
        days_ahead : int
            Number of days ahead to forecast
        training_version : str
            Version of trained models
        table_pred : str
            Prediction table name
        start_date_ts : str
            Start date for time series models
        days_back : int
            Days back for evaluation
        naive_model : bool
            Whether to include naive model
        time_series : bool
            Whether to include time series models
        create_plots : bool
            Whether to create evaluation plots
            
        Returns:
        --------
        dict
            Dictionary containing forecasts for each zone
        """
        print("Starting Day-Ahead Electricity Price Forecasting...")
        
        all_results = {}
        
        for zone in zones:
            print(f"\n{'='*50}")
            print(f"Processing zone: {zone}")
            print(f"{'='*50}")
            
            # Define project name
            project = f"{zone}_{table_pred}_{training_version}_"
            print(f'Current project name: {project}')
            
            # Get target column
            column_pred = self.countries_dict[training_version][zone]['Prices'][0]
            
            zone_results = {}
            
            for day_loop in range(days_ahead):
                print(f"\nForecasting day {day_loop + 1} of {days_ahead}")
                
                # Define dates
                if starting_date:
                    start_time = pd.to_datetime(starting_date, format='%d/%m/%Y') + timedelta(days=day_loop)
                else:
                    start_time = datetime.now() + timedelta(days=day_loop)
                
                today = pd.to_datetime(start_time.strftime('%Y-%m-%d'))
                
                # Define forecast period (next day)
                forecast_start = (today + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                forecast_end = (today + timedelta(days=1, hours=23)).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f'Prediction period: {forecast_start} to {forecast_end}')
                
                # Initialize forecasts dataframe
                forecasts = pd.DataFrame({
                    'Datetime': pd.date_range(forecast_start, periods=24, freq='H'),
                    'project': project
                })
                
                # Prepare forecast data
                forecast_data = self.prepare_forecast_data(
                    zone, training_version, forecast_start, forecast_end
                )
                
                if forecast_data.empty:
                    print("No forecast data available. Skipping this period.")
                    continue
                
                print(forecast_data.head())
                
                # Run ML models
                print("\nRunning ML models...")
                for model_name in self.model_names:
                    print(f'Model running: {model_name}')
                    try:
                        predictions = self.make_model_forecast(model_name, forecast_data, project)
                        if len(predictions) == 24:
                            forecasts[model_name] = predictions
                        else:
                            print(f"Warning: {model_name} returned {len(predictions)} predictions instead of 24")
                    except Exception as e:
                        print(f"Error with {model_name}: {e}")
                
                # Load historical prices for time series models
                if time_series:
                    print("\nLoading historical prices for time series models...")
                    prices_data = self.load_csv_data("DE_LU_DAM_prices.csv")
                    if not prices_data.empty:
                        # Filter prices up to today
                        prices_data = prices_data[
                            (prices_data['Datetime'] >= start_date_ts) &
                            (prices_data['Datetime'] <= today.strftime('%Y-%m-%d %H:%M:%S'))
                        ].copy()
                        
                        prices_data = prices_data.fillna(method='ffill')
                        
                        if not prices_data.empty:
                            print("Running time series models...")
                            forecasts = self.run_timeseries_models(prices_data, forecasts, column_pred)
                
                # Add naive model if requested
                if naive_model and not prices_data.empty:
                    try:
                        forecasts['Naive'] = prices_data[column_pred].iloc[-24:].values
                    except:
                        print("Could not create naive model")
                
                # Create custom ensemble models
                forecasts, suggested_model = self.create_custom_models(
                    forecasts, zone, table_pred, training_version, column_pred, today, days_back
                )
                
                # Store results
                if suggested_model:
                    suggested_df = forecasts[['Datetime', suggested_model]].copy()
                    suggested_df.rename(columns={suggested_model: 'Prices €/MWh'}, inplace=True)
                    
                    self.sugg_predictions[zone][forecast_start]['model'] = suggested_model
                    self.sugg_predictions[zone][forecast_start]['preds'] = suggested_df
                    
                    print(f'Suggested model: {suggested_model}')
                    print(f'Predictions for {(today + timedelta(days=1)).strftime("%d/%m")}:')
                    print(suggested_df.head())
                
                # Clean and save forecasts
                forecasts_clean = forecasts.set_index('Datetime')
                numeric_cols = forecasts_clean.select_dtypes(include=np.number).columns
                forecasts_clean[numeric_cols] = forecasts_clean[numeric_cols].round(2)
                
                # Save to CSV
                output_file = self.evaluation_dir / zone / f"forecasts_{today.strftime('%Y%m%d')}.csv"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                forecasts_clean.to_csv(output_file)
                
                zone_results[forecast_start] = {
                    'forecasts': forecasts_clean,
                    'suggested_model': suggested_model,
                    'suggested_predictions': suggested_df if suggested_model else None
                }
            
            all_results[zone] = zone_results
        
        print("\n" + "="*50)
        print("Forecasting completed!")
        print("="*50)
        
        return all_results
    
    def send_teams_notification(self, zones, days_ahead, days_back, webhook_url=None):
        """
        Send Teams notification with forecast results.
        
        Parameters:
        -----------
        zones : list
            List of zones processed
        days_ahead : int
            Number of days forecasted
        days_back : int
            Days used for evaluation
        webhook_url : str, optional
            Teams webhook URL
        """
        if not webhook_url:
            print("No Teams webhook URL provided. Skipping notification.")
            return
        
        try:
            # Consolidate predictions
            consolidated_df = self._consolidate_predictions(zones, days_ahead)
            
            if consolidated_df.empty:
                print("No predictions to send.")
                return
            
            # Create Teams message
            day_ref = 'day' if days_ahead == 1 else 'days'
            title = f"Day-ahead price predictions ready! ({days_ahead} {day_ref})"
            
            # Convert DataFrame to HTML
            html_table = consolidated_df.to_html(index=False, table_id="predictions-table")
            
            content = f"""
            <br>
            {html_table}
            <br>
            <i>Evaluated over the last {days_back} days</i>
            """
            
            # Send to Teams
            response = self._send_teams_message(webhook_url, content, title)
            print(f"Teams notification: {response}")
            
        except Exception as e:
            print(f"Error sending Teams notification: {e}")
    
    def _consolidate_predictions(self, zones, days_ahead):
        """Consolidate predictions from all zones."""
        start_date = datetime.now() + timedelta(days=1)
        teams_df = pd.DataFrame({
            'Datetime': pd.date_range(start_date, periods=24*days_ahead, freq='H')
        })
        
        for zone in zones:
            zone_data = []
            for date_key in self.sugg_predictions[zone]:
                if 'preds' in self.sugg_predictions[zone][date_key]:
                    zone_data.append(self.sugg_predictions[zone][date_key]['preds'])
            
            if zone_data:
                zone_df = pd.concat(zone_data, axis=0).reset_index(drop=True)
                zone_df['Datetime'] = pd.to_datetime(zone_df['Datetime'])
                zone_df.rename(columns={'Prices €/MWh': zone}, inplace=True)
                teams_df = pd.merge(teams_df, zone_df, on='Datetime', how='left')
        
        # Add average row
        numeric_cols = teams_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            avg_row = teams_df[numeric_cols].mean().round(2)
            avg_row['Datetime'] = 'Average prices'
            teams_df = pd.concat([teams_df, avg_row.to_frame().T], ignore_index=True)
        
        return teams_df
    
    def _send_teams_message(self, webhook_url, content, title, color="#bfbf9f"):
        """Send message to Microsoft Teams."""
        response = requests.post(
            url=webhook_url,
            headers={"Content-Type": "application/json"},
            json={
                "themeColor": color,
                "summary": title,
                "sections": [{
                    "activityTitle": title,
                    "activitySubtitle": content
                }],
            },
        )
        
        if response.status_code == 200:
            return 'Posted to Teams successfully'
        else:
            return f'Error posting to Teams: {response.status_code}'


def main():
    """Main function to run the forecasting system."""
    # Configuration
    config = {
        'zones': ['DE'],
        'starting_date': None,  # Use None for current date, or 'dd/mm/yyyy'
        'days_ahead': 1,
        'training_version': 'v1',
        'table_pred': 'Prices',
        'start_date_ts': '2022-06-09',
        'days_back': 20,
        'naive_model': False,
        'time_series': True,
        'create_plots': False,
        'teams_notification': False,
        'teams_webhook_url': None  # Add your Teams webhook URL here
    }
    
    # Initialize system
    epf_system = DayAheadEPF()
    
    # Run forecasting
    results = epf_system.run_forecasting(
        zones=config['zones'],
        starting_date=config['starting_date'],
        days_ahead=config['days_ahead'],
        training_version=config['training_version'],
        table_pred=config['table_pred'],
        start_date_ts=config['start_date_ts'],
        days_back=config['days_back'],
        naive_model=config['naive_model'],
        time_series=config['time_series'],
        create_plots=config['create_plots']
    )
    
    # Send Teams notification if configured
    if config['teams_notification'] and config['teams_webhook_url']:
        epf_system.send_teams_notification(
            zones=config['zones'],
            days_ahead=config['days_ahead'],
            days_back=config['days_back'],
            webhook_url=config['teams_webhook_url']
        )
    
    return results


if __name__ == "__main__":
    results = main()
    print("Forecasting pipeline completed!")
