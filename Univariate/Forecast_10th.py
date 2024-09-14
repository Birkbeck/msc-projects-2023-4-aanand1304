import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Wrapper
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping

# Load the saved parameters
def load_best_params(attack, param_dir='best_params_files'):
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
    model.add(Input(shape=(n_input, 1)))
    model.add(KeepDropout(LSTM(units[0], recurrent_dropout=rdo, activation='relu', return_sequences=(layer > 1))))
    
    for i in range(1, layer):
        model.add(KeepDropout(LSTM(units[min(i, len(units)-1)], activation='relu', return_sequences=(i < layer-1), recurrent_dropout=rdo)))
    
    model.add(RepeatVector(1))
    
    for i in range(layer):
        model.add(KeepDropout(LSTM(units[min(layer-i-1, len(units)-1)], activation='relu', return_sequences=True, recurrent_dropout=rdo)))
    
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
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

# Prepare data for LSTM input (sliding window)
def prepare_data(data, n_input):
    X, y = [], []
    for i in range(len(data) - n_input):
        X.append(data[i:(i + n_input)])
        y.append(data[i + n_input])
    return np.array(X), np.array(y)

# Forecast using sliding window and dynamic updating of n_input for the next 36 months
def forecast_next_years(model, scaled_data, scaler, n_input, n_months=36, n_iterations=20):
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

        # Log the prediction for debugging purposes
        print(f"Month {i+1} - Mean Prediction: {mean_prediction[0]}")

    # Convert predictions back to the original scale using the scaler
    predictions_mean = np.array(predictions_mean).reshape(-1, 1)
    predictions_std = np.array(predictions_std).reshape(-1, 1)

    # Inverse transform the predictions and confidence intervals to the original scale
    return (
        scaler.inverse_transform(predictions_mean).flatten(),
        scaler.inverse_transform(predictions_mean - 1.96 * predictions_std).flatten(),
        scaler.inverse_transform(predictions_mean + 1.96 * predictions_std).flatten()
    )

# Load the dataset
def load_data():
    data = pd.read_csv('FinalDataset.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data.set_index('Date', inplace=True)
    return data

# Main function to perform forecasting for the next 3 years
def main():
    # Directory containing the best parameters and plots
    param_dir = 'best_params_files'
    output_plot_dir = 'output_plot'
    
    # Load the dataset
    data = load_data()

    # Ensure the output directory exists
    os.makedirs(output_plot_dir, exist_ok=True)
    
    # Forecast for a specific attack type (can loop over multiple types if needed)
    attack = 'DDoS-ALL'  # Example attack type
    ddos_data = data[attack].values.reshape(-1, 1)

    # Load best hyperparameters for this attack
    best_params = load_best_params(attack, param_dir)
    
    # Apply smoothing based on the best hyperparameters
    alpha = best_params['Best Parameters']['alpha']
    beta = best_params['Best Parameters']['beta']
    
    if alpha != 1 and beta != 1:
        smoothed_data = double_exponential_smoothing(ddos_data.flatten(), alpha, beta).reshape(-1, 1)
    elif alpha != 1 and beta == 1:
        smoothed_data = exponential_smoothing(ddos_data.flatten(), alpha).reshape(-1, 1)
    else:
        smoothed_data = ddos_data  # No smoothing if alpha == 1 and beta == 1
    
    # Scale the smoothed data using RobustScaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(smoothed_data)

    # Extract the best parameters
    n_input = best_params['Best Parameters']['n_input']
    layer = best_params['Best Parameters']['layer']
    units = best_params['Best Parameters']['units']
    rdo = best_params['Best Parameters']['rdo']
    n_epochs = best_params['Best Parameters']['n_epochs']

    # Build the model using the best hyperparameters
    model = build_model(n_input, layer, units, rdo)

    # Prepare the data for training
    train_x, train_y = prepare_data(scaled_data, n_input)
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))

    # Define early stopping
    es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

    # Train the model
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=32, callbacks=[es], verbose=0)

    # Forecast the next 3 years (36 months)
    n_months = 36
    pred_mean, pred_lower, pred_upper = forecast_next_years(model, scaled_data, scaler, n_input, n_months)

    # Generate future dates for plotting
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='M')

    # Plot actual data and forecast
    plt.figure(figsize=(10, 6))

    # Plot the actual data
    plt.plot(data.index, ddos_data, label='Actual Data', color='blue', linewidth=2)

    # Plot the forecasted data
    plt.plot(future_dates, pred_mean, label='Predicted Mean', color='red', linewidth=2)
    plt.fill_between(
        future_dates,
        pred_lower,  # Lower bound of the 95% confidence interval
        pred_upper,  # Upper bound of the 95% confidence interval
        color='green', alpha=0.3, label='95% Confidence Interval'
    )
    
    # Add titles, labels, and legend
    plt.title(f'Forecast for the Next 3 Years for {attack}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    # Save the forecast plot
    forecast_plot_path = os.path.join(output_plot_dir, f'{attack}_3_year_forecast.png')
    plt.savefig(forecast_plot_path)
    plt.close()

    print(f"3-year forecast plot for {attack} saved to {forecast_plot_path}.")

if __name__ == "__main__":
    main()
