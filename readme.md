# Day-Ahead Electricity Price Forecasting (EPF) System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

## Overview

This repository contains an advanced **Day-Ahead Electricity Price Forecasting System** that implements ensemble approaches for enhanced price prediction in electricity markets. The system combines multiple machine learning algorithms, deep learning models, and time series forecasting methods to provide accurate day-ahead price predictions.

**Associated Research**: This codebase is associated with the manuscript titled *"An Ensemble Approach for Enhanced Day Ahead Price Forecasting in Electricity Markets"* by Alkis Kitsatoglou et al., submitted to Expert Systems With Applications (currently under review).

## Key Features

- **Ensemble Modeling**: Combines 25+ different forecasting models including:
  - Traditional ML models (Linear Regression, Random Forest, XGBoost, etc.)
  - Deep Learning models (LSTM, GRU, Bidirectional LSTM variants)
  - Time Series models (SARIMA, ETS, ThymeBoost, etc.)
  - Custom ensemble strategies (Best Hourly, Best Daily, weighted combinations)

- **Automated Model Selection**: Dynamic model selection based on historical performance evaluation

- **Comprehensive Evaluation**: Multiple evaluation metrics (MAE, RMSE, MAPE, sMAPE) with visualization tools

- **Flexible Data Input**: CSV-based data input system (no database dependencies)

- **Real-time Integration**: Teams notification system for operational deployment

- **Visualization Suite**: Comprehensive plotting and analysis tools for model performance

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/day-ahead-epf.git
   cd day-ahead-epf
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv epf_env
   source epf_env/bin/activate  # On Windows: epf_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies

Some models require additional libraries. You can install them selectively:

```bash
# For deep learning models
pip install tensorflow keras

# For advanced time series models
pip install scalecast ThymeBoost tbats

# For enhanced visualizations
pip install pandasgui pygwalker

# For GUI components (if using interactive features)
pip install PySimpleGUI
```

## Data Requirements

The system expects the following CSV files in the `data/` directory:

### Required Data Files

1. **`DE_LU_DAM_prices.csv`**
   - Columns: `Datetime`, `DE_LU`
   - Historical day-ahead market prices

2. **`actual_data.csv`**
   - Columns: `Datetime;Load_DE;Wind_DE;Solar_DE;Hydro_DE;Nuc_DE;THE;EUA_DecLastPrice`
   - Actual market data for evaluation

3. **`evaluation_data.csv`**
   - Columns: `Datetime;Load_DE;Wind_DE;Solar_DE;Hydro_DE;Nuc_DE;THE;EUA_DecLastPrice`
   - Data for model input features

4. **`historical_predictions.csv`**
   - Columns: `Datetime`, `project`, `[model_names...]`
   - Historical forecasts from all models for evaluation

### Data Format Requirements

- **Datetime format**: `YYYY-MM-DD HH:MM:SS`
- **Frequency**: Hourly data
- **Separator**: Comma (`,`) for most files, semicolon (`;`) for some specified files
- **Missing values**: Will be handled automatically using forward/backward fill

## Quick Start

### Basic Usage

```python
from day_ahead_EPF import DayAheadEPF

# Initialize the system
epf_system = DayAheadEPF(data_dir="data", models_dir="models")

# Run forecasting for German market
results = epf_system.run_forecasting(
    zones=['DE'],
    days_ahead=1,
    time_series=True,
    create_plots=True
)
```

### Configuration Options

```python
# Advanced configuration
results = epf_system.run_forecasting(
    zones=['DE'],                    # Bidding zones to forecast
    starting_date='04/02/2024',      # Starting date (dd/mm/yyyy) or None for today
    days_ahead=1,                    # Number of days to forecast
    training_version='v1',           # Model version
    days_back=20,                    # Days for evaluation lookback
    naive_model=False,              # Include naive forecasting
    time_series=True,               # Include time series models
    create_plots=True               # Generate evaluation plots
)
```

### Command Line Usage

```bash
python day_ahead_EPF.py
```

## Project Structure

```
day-ahead-epf/
├── data/                          # CSV data files
│   ├── DE_LU_DAM_prices.csv
│   ├── actual_data.csv
│   ├── evaluation_data.csv
│   └── historical_predictions.csv
├── models/                        # Trained model files
│   ├── *.sav                     # Sklearn models
│   ├── *.json/*.h5               # Neural network models
│   └── Best_Hourly_Table.pkl     # Best model rankings
├── evaluation/                    # Evaluation results
│   └── [zone]/[table]/[version]/ # Organized by zone/table/version
├── day_ahead_EPF.py              # Main forecasting script
├── Models_Evaluation.py          # Model evaluation utilities
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Model Types

### Machine Learning Models
- **Linear Models**: Linear Regression, Lasso, Ridge
- **Tree-based**: Random Forest, XGBoost, CatBoost, Extra Trees
- **Other**: k-NN, SVM, GAM, Gradient Boosting, AdaBoost, MLP

### Deep Learning Models
- **LSTM**: Standard and variants (LSTM1-4)
- **GRU**: Standard and variants (GRU1-3)
- **Bidirectional LSTM**: Standard and variants (BILSTM1-3)

### Time Series Models
- **Statistical**: SARIMA, ETS (Exponential Smoothing)
- **Advanced**: ThymeBoost, TBATS
- **ML-based**: LSTM-TS, Ridge-TS, Lasso-TS, XGBoost-TS, etc.

### Ensemble Models
- **Best Hourly**: Hour-specific best performing models
- **Best Daily**: Best performing model from previous day
- **Weighted Ensembles**: Various weighted combinations (811, 721, 631, 532)
- **Best Average**: Average of all best models

## Evaluation Metrics

The system evaluates models using multiple metrics:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)  
- **MAPE** (Mean Absolute Percentage Error)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)
- **NegPos** (Directional accuracy)

## Configuration

### Model Configuration

Edit the `countries_dict` in `day_ahead_EPF.py` to configure:
- Feature columns for each country/zone
- Data sources and table mappings
- Model versions

```python
countries_dict = {
    'v1': {
        'DE': {
            'Fundamentals_forecast': ['Load_DE', 'Wind_DE', 'Solar_DE', 'Hydro_DE', 'Nuc_DE'],
            'GasSpot': ['THE'],
            'EUA': ['EUA_DecLastPrice'],
            'Prices': ['DE_LU']
        }
    }
}
```

### Teams Integration

To enable Teams notifications:

1. Get a Teams webhook URL from your Teams channel
2. Set the webhook URL in the configuration:

```python
config = {
    'teams_notification': True,
    'teams_webhook_url': 'your_webhook_url_here'
}
```

## Advanced Usage

### Custom Model Training

To add new models:

1. Train your model using the same feature set
2. Save the model using pickle (`.sav`) or Keras format (`.json` + `.h5`)
3. Save the feature scaler as `[project_name]X_scaler.sav`
4. Add model name to the appropriate list in the configuration

### Evaluation Only

To run evaluation without forecasting:

```python
from Models_Evaluation import evaluation_project

results = evaluation_project(
    zone="DE",
    table_pred="Prices",
    column_pred="DE_LU", 
    training_version="v1",
    start_date="2024-02-04 00:00:00",
    days_back=10,
    create_plots=True
)
```

## Output Files

### Forecasts
- `evaluation/[zone]/forecasts_[date].csv`: Daily forecast results
- `evaluation/[zone]/[table]/[version]/Evaluation_Results.xlsx`: Model performance metrics

### Visualizations (if `create_plots=True`)
- `Hourly_Deviations.png`: Hourly model performance
- `Best_model_per_hour.png`: Best model selection by hour
- `Model_evaluation.png`: Overall model comparison
- `Residuals_evaluation.png`: Residual analysis
- `Top_[N]_model_vs_Actual_prices.png`: Best model performance visualization

### Reports
- `Summary_results.txt`: Evaluation summary with key metrics

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install optional dependencies based on which models you want to use
2. **Data Format**: Ensure CSV files have correct datetime format and column names
3. **Model Files**: Trained models must be present in the `models/` directory
4. **Memory Issues**: For large datasets, consider processing smaller time windows

### Error Handling

The system includes comprehensive error handling:
- Missing models are skipped with warnings
- Data loading errors are reported
- Model prediction failures are handled gracefully

## Performance Tips

1. **Model Selection**: Disable unused model types to improve performance
2. **Parallel Processing**: The system can be extended for parallel model training
3. **Memory Management**: Use smaller evaluation windows for large datasets
4. **GPU Support**: Neural network models will automatically use GPU if available

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run code formatting
black day_ahead_EPF.py Models_Evaluation.py

# Run linting
flake8 day_ahead_EPF.py Models_Evaluation.py
```

## Research Citation

If you use this code in your research, please cite:

```bibtex
@article{kitsatoglou2024ensemble,
  title={An Ensemble Approach for Enhanced Day Ahead Price Forecasting in Electricity Markets},
  author={Kitsatoglou, Alkis and others},
  journal={Expert Systems With Applications},
  year={2024},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research conducted in collaboration with energy market analysis teams
- Built upon established machine learning and time series forecasting methodologies
- Inspired by the need for accurate electricity price forecasting in modern energy markets

## Contact

For questions, issues, or collaboration opportunities:
- Create an issue in this repository
- Contact the research team through the associated academic publication

## Version History

- **v1.0.0**: Initial release with ensemble forecasting capabilities
- **v1.1.0**: Added CSV-based data loading and improved modularity
- **v1.2.0**: Enhanced error handling and visualization tools

---

**Note**: This system is designed for research and educational purposes. For production deployment in electricity markets, additional validation and compliance checks may be required.
