import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Wrapper
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from datetime import timedelta
import json
import os

# ... (Keep all the previous function definitions and parameter setup)

# Create a directory to store individual parameter files
os.makedirs('attack_parameters', exist_ok=True)

for attack_type in tqdm(attacks, desc="Processing attack types"):
    print(f"\nProcessing {attack_type}")
    
    attack_data = data[attack_type].values.reshape(-1, 1)
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(attack_data)

    train_size = len(scaled_data) - 36
    train, test = scaled_data[:train_size], scaled_data[train_size:]

    best_smape = float('inf')
    best_params = {}
    best_model = None

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(alphas, betas, lags, epochs_list, rdos, layers))

    # Iterate through all combinations
    for alpha, beta, n_input, n_epochs, rdo, layer in tqdm(param_combinations, desc="Evaluating Models", leave=False):
        if alpha != 1 and beta != 1:
            smoothed_data = double_exponential_smoothing(attack_data.flatten(), alpha, beta).reshape(-1, 1)
        elif alpha != 1 and beta == 1:
            smoothed_data = exponential_smoothing(attack_data.flatten(), alpha).reshape(-1, 1)
        else:
            smoothed_data = attack_data
        
        scaled_data = scaler.fit_transform(smoothed_data)
        
        train_size = len(scaled_data) - 36
        train, test = scaled_data[:train_size], scaled_data[train_size:]
        
        for units in units_options[layer]:
            try:
                smape_value, mae, rmse, predictions_mean, accuracy, y_test_inv, lower_bound, upper_bound = evaluate_model(train, test, n_input, layer, units, rdo, scaler, n_epochs)
                
                if smape_value < best_smape:
                    best_smape = smape_value
                    best_params = {
                        'alpha': alpha, 'beta': beta, 'n_input': n_input, 'n_epochs': n_epochs,
                        'layer': layer, 'units': units, 'rdo': rdo, 'smape': smape_value,
                        'mae': mae, 'rmse': rmse, 'accuracy': accuracy
                    }
                    best_model = build_model(n_input, layer, units, rdo)
            except Exception as e:
                print(f"Error with parameters: {alpha}, {beta}, {n_input}, {n_epochs}, {layer}, {units}, {rdo}")
                print(f"Error message: {str(e)}")
                continue

    # Save best parameters for this attack type
    filename = f'attack_parameters/{attack_type.replace("-", "_").replace(" ", "_")}_params.json'
    with open(filename, 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"Best parameters for {attack_type} saved to {filename}")

    # Forecast for the next 3 years (36 months)
    future_predictions, lower_bound_future, upper_bound_future = forecast_next_years(best_model, scaled_data, scaler, best_params['n_input'], n_months=36)
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=30), periods=36, freq='ME')

    # Plot future forecast with 95% Confidence Interval
    plt.figure(figsize=(15, 8))
    plt.plot(data.index, attack_data.flatten(), label='Actual Data', color='blue')
    plt.plot(future_dates, future_predictions, label='Forecast', color='green', linestyle='--')
    plt.fill_between(future_dates, lower_bound_future, upper_bound_future, color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.title(f'{attack_type} Forecasting')
    plt.xlabel('Date')
    plt.ylabel(f'Number of {attack_type} Attacks')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{attack_type.replace("-", "_").replace(" ", "_")}_forecast.png')
    plt.close()

print("All processing complete. Best parameters saved in individual files in the 'attack_parameters' directory.")
