# Data Directory

This directory contains the CSV data files required for the Day-Ahead Electricity Price Forecasting system.

## Required Data Files

### 1. `DE_LU_DAM_prices.csv`
**Description**: Historical day-ahead market prices for the German-Luxembourg bidding zone.

**Format**:
```
Datetime,DE_LU
2022-06-09 00:00:00,45.23
2022-06-09 01:00:00,42.15
...
```

**Columns**:
- `Datetime`: Timestamp in format `YYYY-MM-DD HH:MM:SS`
- `DE_LU`: Day-ahead market price in EUR/MWh

**Frequency**: Hourly data
**Usage**: Used for time series models and as historical price reference

---

### 2. `actual_data.csv`
**Description**: Actual market data used for model evaluation and validation.

**Format**:
```
Datetime;Load_DE;Wind_DE;Solar_DE;Hydro_DE;Nuc_DE;THE;EUA_DecLastPrice
2022-06-09 00:00:00;45234.5;12345.2;0.0;2134.1;0.0;28.45;62.30
...
```

**Columns**:
- `Datetime`: Timestamp in format `YYYY-MM-DD HH:MM:SS`
- `Load_DE`: Electricity demand/load in MW
- `Wind_DE`: Wind power generation in MW
- `Solar_DE`: Solar power generation in MW
- `Hydro_DE`: Hydro power generation in MW
- `Nuc_DE`: Nuclear power generation in MW
- `THE`: Natural gas spot price (THE Hub) in EUR/MWh
- `EUA_DecLastPrice`: EU Allowances (carbon) price in EUR/tonne

**Separator**: Semicolon (`;`)
**Frequency**: Hourly data
**Usage**: Ground truth data for model evaluation

---

### 3. `evaluation_data.csv`
**Description**: Input features for model forecasting (fundamentals and commodity prices).

**Format**: Same as `actual_data.csv`
```
Datetime;Load_DE;Wind_DE;Solar_DE;Hydro_DE;Nuc_DE;THE;EUA_DecLastPrice
2024-02-04 00:00:00;48787.4;37744.5;0.0;2083.6;0.0;28.93;62.30
...
```

**Columns**: Same as `actual_data.csv`
**Separator**: Semicolon (`;`)
**Frequency**: Hourly data
**Usage**: Input features for generating forecasts

---

### 4. `historical_predictions.csv`
**Description**: Historical forecasts from all models for performance evaluation.

**Format**:
```
Datetime,project,LinearRegression,Lasso,Ridge,kNN,GAM,XGB,RandomForest,SVM,GBM,AdaBoost,CatBoost,EXT,MLP,LSTM,LSTM1,LSTM2,LSTM3,LSTM4,GRU,GRU1,GRU2,GRU3,BILSTM,BILSTM1,BILSTM2,BILSTM3,SARIMA_TS,ETS_TS,GBLA_TS,LSTM_TS,Ridge_TS,Lasso_TS,GBT_TS,kNN_TS,MLP_TS,MLR_TS,RF_TS,SVR_TS,XBD_TS,LXBD_TS,Best_Hourly,Best_Daily,Best_811,Best_721,Best_631,Best_532,Best_Average
2024-02-03 00:00:00,DE_Prices_v1_,45.23,44.89,45.12,46.01,44.78,45.45,45.33,44.92,45.21,45.08,45.18,45.25,45.02,45.11,45.15,45.09,45.13,45.06,45.17,45.12,45.14,45.10,45.16,45.18,45.13,45.11,45.20,45.08,45.22,45.09,45.14,45.12,45.19,45.07,45.15,45.11,45.16,45.13,45.18,45.10,45.14,45.12,45.13,45.11,45.12,45.10,45.13
...
```

**Columns**:
- `Datetime`: Timestamp in format `YYYY-MM-DD HH:MM:SS`
- `project`: Project identifier (e.g., `DE_Prices_v1_`)
- `[Model Names]`: Forecast values from each model in EUR/MWh

**Separator**: Comma (`,`)
**Frequency**: Hourly data
**Usage**: Historical model performance analysis and ensemble model creation

## Data Requirements

### Data Quality
- **No missing timestamps**: Data should be continuous with hourly frequency
- **Realistic ranges**: 
  - Prices: -500 to 3000 EUR/MWh
  - Load: 20,000 to 80,000 MW (for Germany)
  - Generation: 0 to capacity limits
- **Consistent format**: All datetime columns must use `YYYY-MM-DD HH:MM:SS` format

### Data Completeness
- Minimum 30 days of historical data recommended
- For time series models: At least 1 year of historical price data
- For evaluation: At least 10 days of overlapping actual and predicted data

### Time Alignment
- All data should be in the same timezone (typically UTC)
- Daylight saving time transitions should be handled consistently
- Missing hours during time changes should be interpolated

## Data Processing

The system automatically handles:
1. **Missing value imputation** using forward/backward fill
2. **Data type conversion** to appropriate numeric types
3. **Time zone alignment** and datetime parsing
4. **Outlier detection** (optional, can be enabled in config)

## Sample Data

Sample data files with the correct format are provided:
- `sample_prices.csv`: Small sample of price data
- `sample_fundamentals.csv`: Sample fundamental data
- `sample_predictions.csv`: Sample historical predictions

## Data Sources

Typical data sources for electricity market forecasting:
- **Market prices**: Power exchanges (EPEX SPOT, Nord Pool, etc.)
- **Load forecasts**: Transmission system operators (TSOs)
- **Generation forecasts**: Weather services, TSOs
- **Commodity prices**: Trading platforms, financial data providers
- **Carbon prices**: EU ETS registry, trading platforms

## Data Privacy and Security

- **No personal data**: This system uses only aggregated market data
- **Public data sources**: Most data is publicly available
- **API rate limits**: Be mindful of data provider rate limits
- **Data licensing**: Ensure compliance with data provider licenses

## Troubleshooting

### Common Data Issues

1. **Date format errors**: Ensure consistent `YYYY-MM-DD HH:MM:SS` format
2. **Separator mismatch**: Check if semicolon vs comma separators are correct
3. **Missing columns**: Verify all required columns are present
4. **Encoding issues**: Use UTF-8 encoding for all CSV files
5. **Large file sizes**: Consider data compression for very large files

### Data Validation

Run the data validation in the Jupyter notebook to check:
- File existence and accessibility
- Column names and data types
- Data completeness and quality
- Temporal alignment

### Performance Considerations

- **File sizes**: Large CSV files (>1GB) may require chunked processing
- **Memory usage**: Monitor RAM usage with large datasets
- **Processing time**: More historical data = longer processing time

## Data Updates

For operational use:
1. **Daily updates**: Add new actual data and evaluation data
2. **Model retraining**: Update historical predictions periodically
3. **Data quality checks**: Implement automated data validation
4. **Backup strategy**: Maintain backups of critical data files

## Contact

For data-related questions or issues:
- Check the main README.md for general support
- Create an issue in the GitHub repository
- Ensure data privacy when sharing sample files
