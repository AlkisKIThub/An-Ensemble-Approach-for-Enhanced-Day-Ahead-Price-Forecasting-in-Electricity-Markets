"""
Example Usage Script for Day-Ahead Electricity Price Forecasting System

This script demonstrates how to use the EPF system for different scenarios.
"""

from day_ahead_EPF import DayAheadEPF
from Models_Evaluation import evaluation_project
import pandas as pd
from datetime import datetime, timedelta
import os

def example_basic_forecasting():
    """Example 1: Basic forecasting for tomorrow's prices."""
    print("="*60)
    print("EXAMPLE 1: Basic Day-Ahead Forecasting")
    print("="*60)
    
    # Initialize system
    epf = DayAheadEPF()
    
    # Run basic forecasting
    results = epf.run_forecasting(
        zones=['DE'],
        days_ahead=1,
        time_series=True,
        create_plots=False
    )
    
    if results:
        print("✓ Forecasting completed successfully!")
        for zone, zone_results in results.items():
            print(f"Zone {zone}: {len(zone_results)} forecast periods generated")
    else:
        print("✗ Forecasting failed. Check data files and configuration.")
    
    return results

def example_multi_day_forecasting():
    """Example 2: Multi-day forecasting with plots."""
    print("="*60)
    print("EXAMPLE 2: Multi-Day Forecasting with Visualization")
    print("="*60)
    
    epf = DayAheadEPF()
    
    # Forecast for next 3 days with visualization
    results = epf.run_forecasting(
        zones=['DE'],
        days_ahead=3,
        time_series=True,
        create_plots=True,  # Generate evaluation plots
        naive_model=True    # Include naive model for comparison
    )
    
    if results:
        print("✓ Multi-day forecasting completed!")
        # Show summary
        for zone, zone_results in results.items():
            total_periods = sum(1 for period_data in zone_results.values() 
                              if period_data['suggested_model'] is not None)
            print(f"Zone {zone}: {total_periods} successful forecast periods")
    
    return results

def example_model_evaluation():
    """Example 3: Standalone model evaluation."""
    print("="*60)
    print("EXAMPLE 3: Standalone Model Evaluation")
    print("="*60)
    
    # Run evaluation for last 14 days
    eval_results = evaluation_project(
        zone="DE",
        table_pred="Prices",
        column_pred="DE_LU",
        training_version="v1",
        start_date="2024-02-04 00:00:00",
        days_back=14,
        create_plots=True
    )
    
    if not eval_results.empty:
        print("✓ Model evaluation completed!")
        print("\nTop 5 Models:")
        print("-" * 40)
        top_models = eval_results.head()
        for i, row in top_models.iterrows():
            print(f"{i+1}. {row['Model']:<15} MAE: {row['MAE']:.2f} RMSE: {row['RMSE']:.2f}")
        
        # Show best model details
        best_model = eval_results.iloc[0]
        print(f"\n🏆 Best Model: {best_model['Model']}")
        print(f"   MAE: {best_model['MAE']:.2f} €/MWh")
        print(f"   RMSE: {best_model['RMSE']:.2f} €/MWh") 
        print(f"   MAPE: {best_model['MAPE']:.2f}%")
    else:
        print("✗ Model evaluation failed. Check data files.")
    
    return eval_results

def example_custom_configuration():
    """Example 4: Custom configuration and specific date forecasting."""
    print("="*60)
    print("EXAMPLE 4: Custom Configuration")
    print("="*60)
    
    epf = DayAheadEPF(
        data_dir="data",
        models_dir="models",
        evaluation_dir="custom_evaluation"
    )
    
    # Forecast for a specific date
    specific_date = "04/02/2024"  # dd/mm/yyyy format
    
    results = epf.run_forecasting(
        zones=['DE'],
        starting_date=specific_date,
        days_ahead=1,
        training_version='v1',
        days_back=30,  # Use 30 days for evaluation
        time_series=True,
        create_plots=True
    )
    
    if results:
        print(f"✓ Forecasting for {specific_date} completed!")
        # Export results to custom location
        os.makedirs("custom_output", exist_ok=True)
        
        for zone, zone_results in results.items():
            for date_key, result in zone_results.items():
                if result['forecasts'] is not None:
                    # Save detailed forecasts
                    date_str = pd.to_datetime(date_key).strftime('%Y%m%d')
                    filename = f"custom_output/detailed_forecasts_{zone}_{date_str}.csv"
                    result['forecasts'].to_csv(filename)
                    print(f"   Detailed forecasts saved: {filename}")
    
    return results

def example_teams_integration():
    """Example 5: Teams integration setup."""
    print("="*60)
    print("EXAMPLE 5: Teams Integration Setup")
    print("="*60)
    
    # Note: This is a demonstration - replace with actual webhook URL
    webhook_url = "https://your-organization.webhook.office.com/..."
    
    epf = DayAheadEPF()
    
    # Run forecasting (without actual Teams notification for demo)
    results = epf.run_forecasting(
        zones=['DE'],
        days_ahead=1,
        time_series=True
    )
    
    if results:
        print("✓ Forecasting completed!")
        print("\nTo enable Teams notifications:")
        print("1. Get a webhook URL from your Teams channel")
        print("2. Replace the webhook_url variable above")
        print("3. Uncomment the notification code below")
        
        # Uncomment to actually send notification:
        # epf.send_teams_notification(
        #     zones=['DE'],
        #     days_ahead=1,
        #     days_back=20,
        #     webhook_url=webhook_url
        # )
        
        print("\nExample Teams message content would include:")
        print("- Forecast date and time")
        print("- Suggested model name")
        print("- Average predicted price")
        print("- Price range (min/max)")
    
    return results

def example_error_handling():
    """Example 6: Error handling and data validation."""
    print("="*60)
    print("EXAMPLE 6: Error Handling and Data Validation")
    print("="*60)
    
    # Check data files
    required_files = [
        'DE_LU_DAM_prices.csv',
        'actual_data.csv',
        'evaluation_data.csv', 
        'historical_predictions.csv'
    ]
    
    data_dir = "data"
    missing_files = []
    
    for filename in required_files:
        if not os.path.exists(os.path.join(data_dir, filename)):
            missing_files.append(filename)
    
    if missing_files:
        print("✗ Missing required data files:")
        for filename in missing_files:
            print(f"   - {filename}")
        print("\nPlease ensure all required data files are present.")
        return None
    
    print("✓ All required data files found!")
    
    # Initialize with error handling
    try:
        epf = DayAheadEPF()
        print("✓ System initialized successfully!")
        
        # Test with minimal configuration
        results = epf.run_forecasting(
            zones=['DE'],
            days_ahead=1,
            time_series=False,  # Disable to reduce dependencies
            create_plots=False  # Disable to speed up
        )
        
        if results:
            print("✓ Basic forecasting test passed!")
        else:
            print("⚠ Forecasting returned no results - check data quality")
            
    except Exception as e:
        print(f"✗ Error during initialization or forecasting: {e}")
        print("Common solutions:")
        print("1. Check data file formats and column names")
        print("2. Verify model files are present in models/ directory")
        print("3. Install missing dependencies from requirements.txt")
    
    return results

def main():
    """Run all examples."""
    print("Day-Ahead EPF System - Example Usage")
    print("="*60)
    print("This script demonstrates various usage patterns of the EPF system.")
    print("Make sure you have the required data files in the data/ directory.")
    print("")
    
    examples = [
        ("Basic Forecasting", example_basic_forecasting),
        ("Multi-Day Forecasting", example_multi_day_forecasting), 
        ("Model Evaluation", example_model_evaluation),
        ("Custom Configuration", example_custom_configuration),
        ("Teams Integration", example_teams_integration),
        ("Error Handling", example_error_handling)
    ]
    
    results = {}
    
    for name, func in examples:
        try:
            print(f"\n\nRunning: {name}")
            result = func()
            results[name] = result
            print(f"✓ {name} completed successfully!")
            
        except Exception as e:
            print(f"✗ {name} failed with error: {e}")
            results[name] = None
        
        print("\n" + "-"*60)
    
    # Summary
    print("\n\nEXAMPLE SUMMARY")
    print("="*40)
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    print(f"Successful examples: {successful}/{total}")
    
    for name, result in results.items():
        status = "✓" if result is not None else "✗"
        print(f"{status} {name}")
    
    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("The EPF system is ready for use.")
    else:
        print(f"\n⚠ {total - successful} examples failed.")
        print("Check the error messages above and verify your setup.")

if __name__ == "__main__":
    main()
