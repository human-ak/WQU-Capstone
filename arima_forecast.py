import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

def train_test_split(data, train_size=0.8):
    """Split the data into training and testing sets"""
    train_size = int(len(data) * train_size)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def fit_arima(train_data, order=(1,1,1)):
    """Fit ARIMA model"""
    model = ARIMA(train_data, order=order)
    results = model.fit()
    return results

def forecast_arima(model, n_periods):
    """Generate forecasts using fitted ARIMA model"""
    forecast = model.forecast(steps=n_periods)
    return forecast

def evaluate_model(actual, predicted):
    """Calculate performance metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

def plot_forecasts(results_dict, save_dir='plots'):
    """Create and save plots for each pair's forecasts"""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('default')
    
    for pair, results in results_dict.items():
        plt.figure(figsize=(12, 6))
        pair_name = f"{pair[0]}_{pair[1]}"
        
        actual = results['Actual']
        predictions = results['Predictions']
        
        plt.plot(actual.index, actual.values, label='Actual', color='blue')
        plt.plot(actual.index, predictions, label='Forecast', color='red', linestyle='--')
        plt.title(f'ARIMA Forecast vs Actual for {pair_name}\nOrder {results["Best Order"]}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/forecast_{pair_name}.png')
        plt.close()

def plot_metrics_comparison(results_dict, save_dir='plots'):
    """Create and save comparison plots for metrics"""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('default')
    
    metrics_data = {
        f"{pair[0]}_{pair[1]}": results['Metrics']
        for pair, results in results_dict.items()
    }
    
    metrics_df = pd.DataFrame(metrics_data).T
    plt.figure(figsize=(12, 6))
    metrics_df[['RMSE', 'MAE']].plot(kind='bar')
    plt.title('RMSE and MAE Comparison Across Pairs')
    plt.xlabel('Pair')
    plt.ylabel('Error Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison.png')
    plt.close()

def main():
    # Load PCA selected pairs
    with open(r'D:\WQU\CAPSTONE\WQU-Capstone\notebooks\PCA\PCA_selected_pairs.pickle', 'rb') as f:
        pca_pairs = pickle.load(f)
    
    results_dict = {}
    
    for pair_data in pca_pairs:
        pair = (pair_data[0], pair_data[1])
        print(f"\nProcessing {pair[0]}_{pair[1]}")
        
        # Extract training and test data from the pair data
        train_y = pair_data[2]['Y_train']
        test_y = pair_data[2]['Y_test']
        
        # Try different ARIMA orders
        orders = [(1,1,1), (1,1,2), (2,1,2)]
        best_metrics = {'RMSE': float('inf')}
        
        for order in orders:
            model = fit_arima(train_y, order=order)
            predictions = forecast_arima(model, len(test_y))
            metrics = evaluate_model(test_y, predictions)
            
            if metrics['RMSE'] < best_metrics['RMSE']:
                best_metrics = metrics
                best_order = order
                best_predictions = predictions
        
        results_dict[pair] = {
            'Best Order': best_order,
            'Metrics': best_metrics,
            'Predictions': best_predictions,
            'Actual': test_y
        }
        print(f"Best ARIMA order: {best_order}")
        print(f"RMSE: {best_metrics['RMSE']:.4f}")
        print(f"MAE: {best_metrics['MAE']:.4f}")
    
    # Create plots and save results
    plot_forecasts(results_dict)
    plot_metrics_comparison(results_dict)
    
    # Save results
    results_df = pd.DataFrame.from_dict(
        {f"{pair[0]}_{pair[1]}": 
         {'Best Order': v['Best Order'], **v['Metrics']} 
         for pair, v in results_dict.items()}, 
        orient='index'
    )
    results_df.to_csv('arima_results.csv')
    
    with open('arima_full_results.pickle', 'wb') as f:
        pickle.dump(results_dict, f)

if __name__ == "__main__":
    main() 