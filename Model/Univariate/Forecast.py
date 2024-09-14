import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Wrapper, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load the saved parameters
def load_best_params(attack, param_dir='best_params_files_uni'):
    file_path = os.path.join(param_dir, f'{attack}_best_params.json')
    with open(file_path, 'r') as f:
        best_params = json.load(f)
    return best_params

# Define the KeepDropout Wrapper class for Monte Carlo Dropout
class KeepDropout(Wrapper):
    def __init__(self, layer, **kwargs):
        super(KeepDropout, self).__init__(layer, **kwargs)
        self.layer = layer

    def call(self, inputs, training=None):
        return self.layer.call(inputs, training=True)

# Build the model using the best hyperparameters from the file
def build_model(n_input, layer, units, rdo):
    model = Sequential()
    model.add(Input(shape=(n_input, 1)))  # Correct input shape initialization
    model.add(LSTM(units[0], recurrent_dropout=rdo, activation='relu', return_sequences=(layer > 1)))
    
    for i in range(1, layer):
        model.add(KeepDropout(LSTM(units[min(i, len(units)-1)], activation='relu', return_sequences=(i < layer-1), recurrent_dropout=rdo)))
    
    model.add(RepeatVector(1))
    
    for i in range(layer):
        model.add(KeepDropout(LSTM(units[min(layer-i-1, len(units)-1)], activation='relu', return_sequences=True, recurrent_dropout=rdo)))
    
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model

# Exponential Smoothing
def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)

# Double Exponential Smoothing
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    level, trend = series[0], series[1] - series[0]
    for n in range(1, len(series)):
        value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return np.array(result)

# Prepare data for LSTM input
def prepare_data(data, n_input):
    X, y = [], []
    for i in range(len(data) - n_input):
        X.append(data[i:(i + n_input)])  # Create input sequence
        y.append(data[i + n_input])      # Create corresponding output
    return np.array(X), np.array(y)

# Detect emerging index where values are consistently above zero
def detect_emerging_index(scaled_data):
    e_index = 0
    for i in range(len(scaled_data) - 2):
        if scaled_data[i, 0] > 0 and scaled_data[i + 1, 0] > 0 and scaled_data[i + 2, 0] > 0:
            e_index = i
            break
    return e_index

# Forecast using Monte Carlo Dropout for the next 36 months
def forecast_next_years(model, scaled_data, scaler, n_input, n_months=36, n_iterations=20, seed_value=42):
    # Set seed for consistent predictions
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Start with the last known n_input data points
    history = scaled_data[-n_input:].reshape(1, n_input, 1)  # Reshape for model input
    
    predictions_mean = []
    predictions_std = []

    for i in range(n_months):  # Loop to predict month-by-month
        # Monte Carlo Dropout for step-by-step forecasting
        mc_predictions = np.array([model(history, training=True) for _ in range(n_iterations)])
        mean_prediction = np.mean(mc_predictions, axis=0).flatten()  # Take mean prediction
        std_prediction = np.std(mc_predictions, axis=0).flatten()    # Get standard deviation

        # Store the mean and std predictions
        predictions_mean.append(mean_prediction[0])
        predictions_std.append(std_prediction[0])

        # Update the sliding window: Remove the oldest value and add the new predicted value
        new_prediction = mean_prediction[0].reshape(1, 1, 1)
        history = np.append(history[:, 1:, :], new_prediction, axis=1)  # Sliding window update

    predictions_mean = np.array(predictions_mean).reshape(-1, 1)
    predictions_std = np.array(predictions_std).reshape(-1, 1)

    # Inverse transform the predictions and confidence intervals to the original scale
    return (
        scaler.inverse_transform(predictions_mean).flatten(),
        scaler.inverse_transform(predictions_mean - 1.96 * predictions_std).flatten(),
        scaler.inverse_transform(predictions_mean + 1.96 * predictions_std).flatten()
    )

# Perform regular forecasting using the same logic as seed-based forecasting
def main_forecast_with_smooth_transition(model, scaled_data, scaler, n_input, ddos_data, data, attack, folder_name):
    # Perform regular forecasting
    future_predictions, lower_bound, upper_bound = forecast_next_years(model, scaled_data, scaler, n_input)

    # Flatten the last actual data point to match the dimensions of future_predictions
    last_actual_value = ddos_data[-1].flatten()  # Ensure it is 1D
    future_predictions_combined = np.concatenate([last_actual_value, future_predictions])  # Concatenate the actual and forecast values

    # Generate future dates for plotting
    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=len(future_predictions), freq='ME')

    # Plot regular forecast
    plt.figure(figsize=(15, 8))
    plt.plot(data.index, ddos_data, label=f'Actual {attack}', color='blue', linewidth=2)
    
    # Concatenate actual data and forecast for smooth plotting
    forecast_dates = pd.date_range(start=data.index[-1], periods=len(future_predictions_combined), freq='ME')
    plt.plot(forecast_dates, future_predictions_combined, label=f'Forecast {attack}', color='red', linewidth=2)
    
    plt.fill_between(future_dates, lower_bound, upper_bound, color='green', alpha=0.3, label='95% Confidence Interval')
    plt.title(f'Forecast for {attack}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the regular forecast plot to the folder attack_forecasts
    plot_filename = os.path.join(folder_name, f"{attack}_forecast.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Regular forecast for {attack} saved in {folder_name}/{attack}_forecast.png")


folder_name = "attack_forecasts"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
def main():
    # Create folder for saving plots
    folder_name = "attack_forecasts"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Load dataset and best parameters
    data = pd.read_csv('/Users/ashit/Downloads/UOL Library/FinalDataset.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data.set_index('Date', inplace=True)
    attack = 'DDoS-ALL'  # Example attack type
    ddos_data = data[attack].values.reshape(-1, 1)

    # Load best hyperparameters for this attack
    best_params = load_best_params(attack)

    # Apply exponential smoothing based on the best hyperparameters
    alpha = best_params['Best Parameters']['alpha']
    beta = best_params['Best Parameters']['beta']
    if alpha != 1 and beta != 1:
        smoothed_data = double_exponential_smoothing(ddos_data.flatten(), alpha, beta).reshape(-1, 1)
    elif alpha != 1 and beta == 1:
        smoothed_data = exponential_smoothing(ddos_data.flatten(), alpha).reshape(-1, 1)
    else:
        smoothed_data = ddos_data  # No smoothing if alpha == 1 and beta == 1

    # Scale the smoothed data
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(smoothed_data)

    # Extract best parameters
    n_input = best_params['Best Parameters']['n_input']
    layer = best_params['Best Parameters']['layer']
    units = best_params['Best Parameters']['units']
    rdo = best_params['Best Parameters']['rdo']
    n_epochs = best_params['Best Parameters']['n_epochs']

    # Build the model
    model = build_model(n_input, layer, units, rdo)

    # Prepare the data for training
    train_x, train_y = prepare_data(scaled_data, n_input)
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

    # Define early stopping
    es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

    # Train the model
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=32, callbacks=[es], verbose=0)
    main_forecast_with_smooth_transition(model, scaled_data, scaler, n_input, ddos_data, data, attack, folder_name)

    # Perform forecasting for different seeds
    for seed in range(1, 4):  # Seeds 1, 2, 3
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        pred_mean, pred_lower, pred_upper = forecast_next_years(model, scaled_data, scaler, n_input, seed_value=seed)
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=36, freq='ME')

        plt.figure(figsize=(15, 8))
        plt.plot(data.index, ddos_data, label='Actual Data', color='blue', linewidth=2)
        
        # Plot the forecast line starting from the last actual data point
        forecast_dates = pd.date_range(start=last_date, periods=37, freq='ME')
        forecast_values = np.concatenate([[ddos_data[-1][0]], pred_mean])
        plt.plot(forecast_dates, forecast_values, label=f'Forecast (Seed {seed})', color='red', linewidth=2)
        
        plt.fill_between(future_dates, pred_lower, pred_upper, color='lightgreen', alpha=0.3, label='95% Confidence Interval')
        plt.title(f'{attack}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Add vertical line to separate historical data from forecasts
        plt.axvline(x=last_date, color='gray', linestyle='--')
        plt.text(last_date, plt.ylim()[1], '', horizontalalignment='center')

        # Save each seed forecast plot to the folder attack_forecasts
        plot_filename = os.path.join(folder_name, f"{attack}_seed_{seed}_forecast.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Forecast for seed {seed} saved in {folder_name}/{attack}_seed_{seed}_forecast.png")

if __name__ == "__main__":
    main()