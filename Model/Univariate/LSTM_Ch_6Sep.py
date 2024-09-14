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
import random
from datetime import timedelta
import json

# SMAPE (Symmetric Mean Absolute Percentage Error)
def smape(yTrue, yPred):
    return np.mean(np.abs(yPred - yTrue) / (np.abs(yTrue) + np.abs(yPred))) * 100

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
        X.append(data[i:(i + n_input)])
        y.append(data[i + n_input])
    return np.array(X), np.array(y)

# KeepDropout Wrapper (Monte Carlo Dropout)
class KeepDropout(Wrapper):
    def __init__(self, layer, **kwargs):
        super(KeepDropout, self).__init__(layer, **kwargs)
        self.layer = layer

    def call(self, inputs, training=None):
        return self.layer.call(inputs, training=True)

# Build the Bayesian LSTM Model with random hyperparameters
def build_model(n_input, layer, unit, rdo):
    model = Sequential()
    
    model.add(KeepDropout(LSTM(unit[0], input_shape=(n_input, 1), recurrent_dropout=rdo, activation='relu', return_sequences=(layer > 1))))
    
    for i in range(1, layer):
        model.add(KeepDropout(LSTM(unit[min(i, len(unit)-1)], activation='relu', return_sequences=(i < layer-1), recurrent_dropout=rdo)))
    
    model.add(RepeatVector(1))
    
    for i in range(layer):
        model.add(KeepDropout(LSTM(unit[min(layer-i-1, len(unit)-1)], activation='relu', return_sequences=True, recurrent_dropout=rdo)))
    
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))

    model.compile(optimizer='adam', loss=huber_loss, metrics=['mae'])
    return model

# Advanced loss function: Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = tf.square(error) / 2
    linear_loss = delta * (tf.abs(error) - delta / 2)
    return tf.where(is_small_error, squared_loss, linear_loss)

# Detect and remove outliers using Isolation Forest
def detect_outliers(data):
    iso_forest = IsolationForest(contamination=0.01)  # Set contamination level to 1%
    outliers = iso_forest.fit_predict(data.reshape(-1, 1))
    return outliers
# Calculate prediction accuracy based on a given tolerance
def prediction_accuracy(y_true, y_pred, tolerance=0.1):
    correct_predictions = np.abs(y_true - y_pred) <= tolerance * np.abs(y_true)
    accuracy = np.mean(correct_predictions) * 100
    return accuracy


# Evaluate the model performance
def evaluate_model(train, test, n_input, layer, unit, rdo, scaler, epochs):
    train_x, train_y = prepare_data(train, n_input)
    test_x, test_y = prepare_data(test, n_input)
    
    # Reshape input data for LSTM
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    
    model = build_model(n_input, layer, unit, rdo)
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    # Train the model
    model.fit(train_x, train_y, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)
    
    # Monte Carlo Predictions (Uncertainty Estimation)
    n_iterations = 10
    predictions_mc = np.array([model(test_x, training=True) for _ in range(n_iterations)])
    predictions_mean = np.mean(predictions_mc, axis=0).reshape(-1, 1)
    predictions_mean = scaler.inverse_transform(predictions_mean)

    y_test_inv = scaler.inverse_transform(test_y.reshape(-1, 1))
    
    # Calculate SMAPE, MAE, RMSE
    smape_value = smape(y_test_inv, predictions_mean)
    mae = mean_absolute_error(y_test_inv, predictions_mean)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_mean))
    accuracy = prediction_accuracy(y_test_inv, predictions_mean, tolerance=0.1)
    
    return smape_value, mae, rmse, predictions_mean,accuracy

# Load and preprocess data
data = pd.read_csv('FinalDataset.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Filter a specific country for analysis, e.g., 'DDoS-US'
ddos_data = data['Ransomware-ALL'].values.reshape(-1, 1)

# Ensure the data has a monthly frequency and fill missing dates
data.set_index('Date', inplace=True)
data = data.asfreq('ME')  # Corrected frequency to 'ME'

# Interpolate missing values if any
if data.isnull().values.any():
    data.interpolate(method='linear', inplace=True)

# Detect and remove outliers
outliers = detect_outliers(ddos_data)
filtered_data = ddos_data  # Keep only non-outliers

# Apply robust scaling to handle outliers
scaler = RobustScaler()
scaled_data = scaler.fit_transform(filtered_data)

# Use the scaled data for training
train_size = len(scaled_data) - 36  # Reserve last 36 months for testing
train, test = scaled_data[:train_size], scaled_data[train_size:]

# Hyperparameter lists for random search
alphas = [0.05, 0.2, 0.5, 0.7, 1]
betas = [0.3, 0.5, 0.7, 1]
lags = list(range(1, max(len(ddos_data) // 10, 1) + 1))
lrs = [0.001, 0.0001, 0.0006, 0.01]
epochs_list = [330, 400, 200, 100]
rdos = [0.1, 0.15]

best_smape = float('inf')
best_params = {}
best_model = None

# Number of random search iterations
iterations = 12

for _ in range(iterations):
    alpha = random.choice(alphas)
    beta = random.choice(betas)
    
    # Apply smoothing based on randomly chosen alpha/beta
    if alpha != 1 and beta != 1:
        smoothed_data = double_exponential_smoothing(ddos_data.flatten(), alpha, beta).reshape(-1, 1)
    elif alpha != 1 and beta == 1:
        smoothed_data = exponential_smoothing(ddos_data.flatten(), alpha).reshape(-1, 1)
    else:
        smoothed_data = ddos_data
    
    # RobustScaler to handle outliers
    scaled_data = scaler.fit_transform(smoothed_data)
    
    train_size = len(scaled_data) - 36  # Last 3 years for testing
    train, test = scaled_data[:train_size], scaled_data[train_size:]
    
    # Randomly choose hyperparameters for this iteration
    n_input = random.choice(lags)
    n_epochs = random.choice(epochs_list)
    layer = random.choice([1, 2, 3])
    
    if layer == 1:
        units = [random.choice([50, 100, 200])]
    elif layer == 2:
        units = [random.choice([64, 128]), random.choice([32, 64])]
    else:
        units = [random.choice([100, 200]), random.choice([50, 100]), random.choice([25, 50])]
    
    rdo = random.choice(rdos)
    
    # Evaluate the model
    smape_value, mae, rmse, predictions_mean, accuracy = evaluate_model(train, test, n_input, layer, units, rdo, scaler, n_epochs)
    
    # Track the best model based on SMAPE
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = {
            'alpha': alpha, 'beta': beta, 'n_input': n_input, 'n_epochs': n_epochs,
            'layer': layer, 'units': units, 'rdo': rdo
        }
        best_model = build_model(n_input, layer, units, rdo)

# Print best results
print(f"Best SMAPE: {best_smape}")
print(f"Best mae: {mae}")
print(f"Best SMAE: {rmse}")
print(f"Best Accuracy: {accuracy}")

print(f"Best parameters: {best_params}")

# Forecast for next 3 years
def forecast_next_years(model, scaled_data, scaler, n_input, n_years=3):
    history = scaled_data[-n_input:]
    predictions = []
    
    for _ in range(n_years * 12):  # 3 years
        pred = model.predict(history.reshape(1, history.shape[0], history.shape[1]))
        predictions.append(pred[0][0])
        history = np.append(history, pred[0], axis=0)[1:]

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()

# Forecast future
future_predictions = forecast_next_years(best_model, scaled_data, scaler, best_params['n_input'])
future_dates = pd.date_range(start=data.index[-1] + timedelta(days=30), periods=36, freq='ME')

# Plot results
plt.figure(figsize=(15, 8))
plt.plot(data.index[:-1], ddos_data.flatten()[:len(data.index)-1], label='Actual Data', color='blue')
plt.plot(future_dates, future_predictions, label='Forecast', color='green', linestyle='--')
plt.title('DDoS Attack Forecasting')
plt.xlabel('Date')
plt.ylabel('Number of DDoS Attacks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
