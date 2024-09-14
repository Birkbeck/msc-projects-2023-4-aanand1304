import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Wrapper, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Custom Layer: KeepDropout
class KeepDropout(Wrapper):
    def __init__(self, layer, **kwargs):
        super(KeepDropout, self).__init__(layer, **kwargs)
        self.layer = layer

    def call(self, inputs, training=None):
        return self.layer.call(inputs, training=True)

# Function to create Bayesian LSTM layers
def BayesianLSTM(units, **kwargs):
    return KeepDropout(LSTM(units, **kwargs))

# Load best hyperparameters from the JSON file
with open('best_hyperparameters.json', 'r') as file:
    best_params = json.load(file)

# Smoothing Functions
def exponential_smoothing(series, alpha):
    result = [series[0]]  # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    level, trend = series[0], series[1] - series[0]
    for n in range(1, len(series)):
        value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return np.array(result)

# Build the Bayesian LSTM model
def build_model(n_input, layer, units, rdo, lr):
    model = Sequential()

    if not isinstance(units, list):
        units = [units]

    # Explicit Input Layer to avoid warnings
    model.add(Input(shape=(n_input, 1)))

    # Add Bayesian LSTM Layers with 'tanh' activation
    model.add(BayesianLSTM(units[0], recurrent_dropout=rdo, activation='tanh', return_sequences=(layer > 1)))
    
    for i in range(1, layer):
        model.add(BayesianLSTM(units[min(i, len(units)-1)], activation='tanh', return_sequences=(i < layer-1), recurrent_dropout=rdo))
    
    model.add(RepeatVector(1))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    
    # Compile the model with the best learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
    return model

# Load the dataset
data = pd.read_csv('FinalDataset.csv')  # Replace with your actual file path

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Ensure proper datetime conversion

# Assuming your target column is 'DDoS-ALL', change it if necessary
ddos_data = data['DDoS-ALL'].values.reshape(-1, 1)

# Apply smoothing before scaling the data
alpha = best_params['alpha']
beta = best_params['beta']

# Apply either exponential smoothing or double exponential smoothing
if alpha != 1 and beta != 1:
    smoothed_data = double_exponential_smoothing(ddos_data.flatten(), alpha, beta).reshape(-1, 1)
elif alpha != 1 and beta == 1:
    smoothed_data = exponential_smoothing(ddos_data.flatten(), alpha).reshape(-1, 1)
else:
    smoothed_data = ddos_data  # If alpha and beta are both 1, skip smoothing

# Visualize smoothing
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], ddos_data.flatten(), label='Original Data', color='blue')
plt.plot(data['Date'], smoothed_data.flatten(), label='Smoothed Data', color='orange')
plt.legend()
plt.show()

# Now, scale the smoothed data
scaler = MinMaxScaler()  # Or try StandardScaler()
scaled_data = scaler.fit_transform(smoothed_data)

# Optionally, visualize scaled data
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], scaled_data.flatten(), label='Scaled Data', color='green')
plt.legend()
plt.show()

# Load the trained model
try:
    best_model = load_model('best_model.h5', custom_objects={'KeepDropout': KeepDropout, 'BayesianLSTM': BayesianLSTM})
    print("Loaded 'best_model.h5' successfully.")
except Exception as e:
    print(f"Error loading 'best_model.h5': {e}")
    # If loading fails, build the model architecture and load weights
    best_model = build_model(
        best_params['n_input'],
        best_params['layer'],
        best_params['units'],
        best_params['rdo'],
        best_params['lr']
    )
    try:
        best_model.load_weights('best_hyperparameters.json')  # Ensure 'best_weights.h5' exists
        print("Loaded 'best_weights.h5' successfully.")
    except Exception as e:
        print(f"Error loading 'best_weights.h5': {e}")
        # If you cannot load the model or weights, you need to retrain your model.

# Function to forecast future values using the trained model
# Forecasting function to predict future values using the best model
# Forecasting function to predict future values using the best model
def forecast_next_years(model, scaled_data, n_input, n_years=3):
    history = scaled_data[-n_input:]  # Start with the last n_input values
    predictions = []

    for _ in range(n_years * 12):  # Forecasting for 3 years (36 months)
        # Reshape history to the required shape (batch_size, time_steps, features)
        pred = model.predict(history.reshape(1, history.shape[0], history.shape[1]), verbose=0)
        
        # Extract the prediction (flatten the 3D prediction array)
        pred_value = pred[0][0]  # Extract the first value from the first prediction
        
        # Append the prediction to the list
        predictions.append(pred_value)
        
        # Reshape pred_value to be 2D before appending to history (ensure dimensions match)
        pred_value = np.array(pred_value).reshape(1, 1)
        
        # Append pred_value to history and slide the window
        history = np.append(history, pred_value, axis=0)[1:]  # Move the window by appending the prediction
    
    # Convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, 1)

    # **Check predictions before inverse scaling**
    print("Predictions before inverse transform:", predictions)
    
    # Inverse transform the forecasted values to the original scale
    return scaler.inverse_transform(predictions).flatten()


# Optional: Function to perform Monte Carlo Dropout for uncertainty estimation
def forecast_next_years_mc(model, scaled_data, n_input, n_years=3, n_iterations=100):
    history = scaled_data[-n_input:]
    predictions = []

    for _ in range(n_years * 12):
        mc_preds = []
        for _ in range(n_iterations):
            pred = model.predict(history.reshape(1, history.shape[0], history.shape[1]), verbose=0)
            mc_preds.append(pred[0][0])
        pred_mean = np.mean(mc_preds)
        predictions.append(pred_mean)
        history = np.append(history, [[pred_mean]], axis=0)[1:]
    
    predictions = np.array(predictions).reshape(-1, 1)
    print("Predictions before inverse transform:", predictions)
    return scaler.inverse_transform(predictions).flatten()

# Forecast future for 3 years using standard forecasting
future_predictions = forecast_next_years(best_model, scaled_data, best_params['n_input'])

# Alternatively, use Monte Carlo forecasting to capture uncertainty
# future_predictions = forecast_next_years_mc(best_model, scaled_data, best_params['n_input'])

# Function to adjust the forecast to follow historical trends
def adjust_forecast_trend(original_data, forecasted_data):
    """
    Adjust forecasted data to follow the trend of the original data more closely.
    This can involve techniques like trend correction, quantile adjustments, etc.
    """
    # Calculate the slope of the historical data
    x = np.arange(len(original_data))
    slope, intercept = np.polyfit(x, original_data, 1)
    
    # Apply the trend to the forecasted data
    forecast_length = len(forecasted_data)
    trend = slope * np.arange(len(original_data), len(original_data) + forecast_length) + intercept
    
    # Adjust the forecasted data based on the trend
    # This is a simple trend addition; you might want to use more sophisticated methods
    forecast_trend_adjusted = forecasted_data + (trend - forecasted_data[0])
    
    return forecast_trend_adjusted

# Adjust the forecast with the observed trend in the original data
adjusted_forecast = adjust_forecast_trend(ddos_data.flatten(), future_predictions)

# If you used Monte Carlo, you might skip trend adjustment or handle it differently
# future_predictions_original_scale = adjusted_forecast.flatten()

# Here, assuming you used standard forecasting
future_predictions_original_scale = adjusted_forecast.flatten()

# Generate future dates for plotting (assuming monthly data)
last_date = pd.to_datetime(data['Date'].iloc[-1])  # Replace 'Date' with your actual date column name
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=36, freq='MS')

# Plot the forecast results
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], ddos_data.flatten(), label='Actual Data', color='blue')  # Plot actual data
plt.plot(future_dates, future_predictions_original_scale, label='Adjusted Forecast', color='green', linestyle='--')  # Plot forecasted data

plt.title('Forecast for DDoS Attacks (Next 3 Years)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('DDoS Attacks', fontsize=12)
plt.legend()

# Format the x-axis for dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
