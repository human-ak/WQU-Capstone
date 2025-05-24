import pandas as pd
import numpy as np
import random
import os
import pickle
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_pca_pairs(pickle_path):
    """
    Load PCA selected pairs from pickle file
    """
    with open(pickle_path, 'rb') as f:
        pca_pairs = pickle.load(f)
    return pca_pairs

def load_and_prepare_data(file_path):
    """
    Load and prepare the currency pair data
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def train_test_split(data, train_size=0.8):
    """
    Split the data into training and testing sets
    """
    train_size = int(len(data) * train_size)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def fit_arima(train_data, order=(1,1,1)):
    """
    Fit ARIMA model
    """
    model = ARIMA(train_data, order=order)
    results = model.fit()
    return results

def forecast_arima(model, n_periods):
    """
    Generate forecasts using fitted ARIMA model
    """
    forecast = model.forecast(steps=n_periods)
    return forecast

def evaluate_model(actual, predicted):
    """
    Calculate performance metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

def extract_pair_names(pair_tuple):
    """
    Extract pair names from the PCA tuple format
    """
    if isinstance(pair_tuple, tuple):
        return pair_tuple[0], pair_tuple[1]
    return pair_tuple

def plot_forecasts(results_dict, save_dir='plots'):
    """
    Create and save plots for each currency pair's forecasts
    """
    # Create plots directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set style for better visualization
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    for pair, results in results_dict.items():
        plt.figure(figsize=(12, 6))
        
        # Extract pair names if in tuple format
        if isinstance(pair, tuple):
            pair_name = f"{pair[0]}_{pair[1]}"
        else:
            pair_name = pair.replace("/", "")
        
        # Plot actual values
        actual = results['Actual']
        predictions = results['Predictions']
        
        plt.plot(actual.index, actual.values, label='Actual', color='blue')
        plt.plot(actual.index, predictions, label='Forecast', color='red', linestyle='--')
        
        plt.title(f'ARIMA Forecast vs Actual for {pair_name}\nOrder {results["Best Order"]}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'forecast_{pair_name}.png'))
        plt.close()

def plot_metrics_comparison(results_dict, save_dir='plots'):
    """
    Create and save comparison plots for metrics across all pairs
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Extract metrics for all pairs
    metrics_data = {}
    for pair, results in results_dict.items():
        if isinstance(pair, tuple):
            pair_name = f"{pair[0]}_{pair[1]}"
        else:
            pair_name = pair.replace("/", "")
        metrics_data[pair_name] = results['Metrics']
    
    metrics_df = pd.DataFrame(metrics_data).T
    
    # Create bar plots for each metric
    plt.figure(figsize=(12, 6))
    metrics_df[['RMSE', 'MAE']].plot(kind='bar')
    plt.title('RMSE and MAE Comparison Across Pairs')
    plt.xlabel('Pair')
    plt.ylabel('Error Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'))
    plt.close()

def main():
    # Load PCA selected pairs
    pca_pairs = load_pca_pairs(r'D:\WQU\CAPSTONE\WQU-Capstone\notebooks\PCA\PCA_selected_pairs.pickle')
    
    results_dict = {}
    
    for pair in pca_pairs:
        print(f"\nProcessing {pair}")
        
        try:
            # Extract pair names if in tuple format
            if isinstance(pair, tuple):
                pair_name = f"{pair[0]}_{pair[1]}"
                file_path = f'data/{pair_name}.csv'
            else:
                file_path = f'data/{pair.replace("/", "")}.csv'
            
            # Load data for the current pair
            data = load_and_prepare_data(file_path)
            
            # Split into train and test sets
            train, test = train_test_split(data['Close'])
            
            # Try different ARIMA orders
            orders = [(1,1,1), (1,1,2), (2,1,2)]
            best_metrics = {'RMSE': float('inf')}
            best_order = None
            best_predictions = None
            
            for order in orders:
                try:
                    # Fit model
                    model = fit_arima(train, order=order)
                    
                    # Make predictions
                    predictions = forecast_arima(model, len(test))
                    
                    # Evaluate
                    metrics = evaluate_model(test, predictions)
                    
                    if metrics['RMSE'] < best_metrics['RMSE']:
                        best_metrics = metrics
                        best_order = order
                        best_predictions = predictions
                
                except Exception as e:
                    print(f"Error with order {order}: {str(e)}")
                    continue
            
            if best_order:
                results_dict[pair] = {
                    'Best Order': best_order,
                    'Metrics': best_metrics,
                    'Predictions': best_predictions,
                    'Actual': test
                }
                print(f"Best ARIMA order for {pair}: {best_order}")
                print(f"RMSE: {best_metrics['RMSE']:.4f}")
                print(f"MAE: {best_metrics['MAE']:.4f}")
        
        except Exception as e:
            print(f"Error processing {pair}: {str(e)}")
            continue
        
        # Clear memory
        gc.collect()
    
    # After saving results, create plots
    plot_forecasts(results_dict)
    plot_metrics_comparison(results_dict)
    
    # Save results to CSV and pickle
    results_df = pd.DataFrame.from_dict(
        {(pair[0] + "_" + pair[1] if isinstance(pair, tuple) else pair.replace("/", "")): 
         {'Best Order': v['Best Order'], **v['Metrics']} 
         for pair, v in results_dict.items()}, 
        orient='index'
    )
    results_df.to_csv('arima_results.csv')
    
    # Save full results including predictions
    with open('arima_full_results.pickle', 'wb') as f:
        pickle.dump(results_dict, f)

if __name__ == "__main__":
    main() 