import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random

def exponential_smoothing(series, alpha):
    result = [series[0]]
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

def prepare_data(data, n_input):
    X, y = [], []
    for i in range(len(data) - n_input):
        X.append(data[i:(i + n_input)])
        y.append(data[i + n_input])
    return np.array(X), np.array(y)

def build_model(n_input, layer, unit, rdo):
    model = Sequential()
    if layer == 1:
        model.add(LSTM(unit[0], activation='relu', input_shape=(n_input, 1), return_sequences=False))
        model.add(Dropout(rdo))
        model.add(RepeatVector(1))
        model.add(LSTM(unit[0], activation='relu', return_sequences=True))
        model.add(Dropout(rdo))
    elif layer == 2:
        model.add(LSTM(unit[0], activation='relu', input_shape=(n_input, 1), return_sequences=True))
        model.add(Dropout(rdo))
        model.add(LSTM(unit[1], activation='relu', return_sequences=False))
        model.add(Dropout(rdo))
        model.add(RepeatVector(1))
        model.add(LSTM(unit[1], activation='relu', return_sequences=True))
        model.add(Dropout(rdo))
        model.add(LSTM(unit[0], activation='relu', return_sequences=True))
        model.add(Dropout(rdo))
    elif layer == 3:
        model.add(LSTM(unit[0], activation='relu', input_shape=(n_input, 1), return_sequences=True))
        model.add(Dropout(rdo))
        model.add(LSTM(unit[1], activation='relu', return_sequences=True))
        model.add(Dropout(rdo))
        model.add(LSTM(unit[2], activation='relu', return_sequences=False))
        model.add(Dropout(rdo))
        model.add(RepeatVector(1))
        model.add(LSTM(unit[2], activation='relu', return_sequences=True))
        model.add(Dropout(rdo))
        model.add(LSTM(unit[1], activation='relu', return_sequences=True))
        model.add(Dropout(rdo))
        model.add(LSTM(unit[0], activation='relu', return_sequences=True))
        model.add(Dropout(rdo))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def evaluate_model(data, n_input, n_epochs, layer, unit, lr, rdo, scaler):
    # Split data
    train_size = len(data) - 36  # Last 3 years for testing
    train, test = data[:train_size], data[train_size:]
    
    # Prepare data
    X_train, y_train = prepare_data(train, n_input)
    X_test, y_test = prepare_data(test, n_input)
    
    # Build and train model
    model = build_model(n_input, layer, unit, rdo)
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)
    
    # Monte Carlo predictions
    n_iterations = 100
    predictions_mc = np.array([model.predict(X_test, verbose=0) for _ in range(n_iterations)])
    
    # Reshape predictions
    predictions_mean = np.mean(predictions_mc, axis=0).reshape(-1, 1)
    predictions_std = np.std(predictions_mc, axis=0).reshape(-1, 1)
    
    # Inverse transform predictions and actual values
    predictions_mean = scaler.inverse_transform(predictions_mean)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_inv, predictions_mean)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_mean))
    
    return mae, rmse, predictions_mean, predictions_std, y_test_inv


# Load and preprocess data
data = pd.read_csv('FinalDataset.csv')
ddos_data = data['DDoS-ALL'].values.reshape(-1, 1)

# Apply smoothing
alpha, beta = 0.3, 0.5  # You can adjust these or make them part of the random search
smoothed_data = double_exponential_smoothing(ddos_data.flatten(), alpha, beta).reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
ddos_data_scaled = scaler.fit_transform(smoothed_data)

# Random search
best_mae = float('inf')
best_params = {}

for _ in range(100):  # Number of random searches
    n_input = random.choice([6, 12, 18, 24])
    n_epochs = random.choice([100, 200, 500, 1000])
    layer = random.choice([1, 2, 3])
    if layer == 1:
        units = [random.choice([50, 100, 200])]
    elif layer == 2:
        units = [random.choice([64, 128]), random.choice([32, 64])]
    else:
        units = [random.choice([100, 200]), random.choice([50, 100]), random.choice([25, 50])]
    lr = random.choice([0.001, 0.0001, 0.01])
    rdo = random.choice([0.2, 0.3, 0.4])

    mae, rmse, predictions, predictions_std, y_test = evaluate_model(ddos_data_scaled, n_input, n_epochs, layer, units, lr, rdo, scaler)
    
    if mae < best_mae:
        best_mae = mae
        best_params = {'n_input': n_input, 'n_epochs': n_epochs, 'layer': layer, 'units': units, 'lr': lr, 'rdo': rdo}
        best_predictions = predictions
        best_predictions_std = predictions_std
        best_y_test = y_test

print(f"Best Mean Absolute Error (MAE): {best_mae}")
print(f"Best parameters: {best_params}")

# Plot results
plt.figure(figsize=(12,6))
plt.plot(best_y_test, label='Actual', color='blue')
plt.plot(best_predictions, label='Predicted', color='red')
plt.fill_between(range(len(best_predictions)), 
                 best_predictions.flatten() - 1.96 * best_predictions_std.flatten(), 
                 best_predictions.flatten() + 1.96 * best_predictions_std.flatten(), 
                 color='red', alpha=0.2)
plt.legend()
plt.title('DDoS Attack Prediction with Uncertainty')
plt.xlabel('Time')
plt.ylabel('Number of DDoS Attacks')
plt.show()