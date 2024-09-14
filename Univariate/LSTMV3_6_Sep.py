import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Wrapper
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Evaluation metric (symmetric mean absolute percentage error)
def smape(yTrue, yPred):
    return np.mean(np.abs(yPred - yTrue) / (np.abs(yTrue) + np.abs(yPred))) * 100

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

class KeepDropout(Wrapper):
    def __init__(self, layer, **kwargs):
        super(KeepDropout, self).__init__(layer, **kwargs)
        self.layer = layer

    def call(self, inputs, training=None):
        return self.layer.call(inputs, training=True)

def BayesianLSTM(units, **kwargs):
    return KeepDropout(LSTM(units, **kwargs))

def build_model(n_input, layer, unit, rdo):
    model = Sequential()
    
    # Ensure unit is a list and has at least one element
    if not isinstance(unit, list):
        unit = [unit]
    
    # Add input layer
    model.add(BayesianLSTM(unit[0], input_shape=(n_input, 1), recurrent_dropout=rdo, activation='relu', return_sequences=(layer > 1)))
    
    # Add hidden layers
    for i in range(1, layer):
        model.add(BayesianLSTM(unit[min(i, len(unit)-1)], activation='relu', return_sequences=(i < layer-1), recurrent_dropout=rdo))
    
    # Add RepeatVector
    model.add(RepeatVector(1))
    
    # Add decoder layers
    for i in range(layer):
        model.add(BayesianLSTM(unit[min(layer-i-1, len(unit)-1)], activation='relu', return_sequences=True, recurrent_dropout=rdo))
    
    # Add output layers
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def evaluate_model(train, test, n_input, alpha, beta, lag, layer, unit, lr, epoch, rdo, scaler):
    # Prepare data
    train_x, train_y = prepare_data(train, n_input)
    test_x, test_y = prepare_data(test, n_input)
    
    # Build and train model
    model = build_model(n_input, layer, unit, rdo)
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    model.fit(train_x, train_y, epochs=epoch, batch_size=32, 
              validation_split=0.2, callbacks=[es], verbose=0)
    
    # Monte Carlo predictions
    n_iterations = 100
    predictions_mc = np.array([model(test_x, training=True) for _ in range(n_iterations)])
    predictions_mean = np.mean(predictions_mc, axis=0).reshape(-1, 1)
    predictions_std = np.std(predictions_mc, axis=0).reshape(-1, 1)
    
    # Inverse transform predictions and actual values
    predictions_mean = scaler.inverse_transform(predictions_mean)
    y_test_inv = scaler.inverse_transform(test_y.reshape(-1, 1))
    
    # Calculate metrics
    smape_value = smape(y_test_inv, predictions_mean)
    mae = mean_absolute_error(y_test_inv, predictions_mean)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_mean))
    
    return smape_value, mae, rmse, predictions_mean, predictions_std, y_test_inv

# Load data
data = pd.read_csv('FinalDataset.csv')
ddos_data = data['DDoS-ALL'].values.reshape(-1, 1)

# Hyperparameters for random search
alphas = [0.05, 0.2, 0.5, 0.7, 1]
betas = [0.3, 0.5, 0.7, 1]
lags = list(range(1, max(len(ddos_data)//10, 1)+1))
lrs = [0.001, 0.0001, 0.0006, 0.01]
epochs = [300, 500, 600, 700]
units = [[50], [25], [100], [200], [64, 28], [40, 20], [100, 50]]
rdos = [0.2, 0.3, 0.4, 0.5]

# Random search
best_smape = float('inf')
best_params = {}
iterations = 10

for _ in range(iterations):
    alpha = random.choice(alphas)
    beta = random.choice(betas)
    
    # Apply smoothing
    if alpha != 1 and beta != 1:
        smoothed_data = double_exponential_smoothing(ddos_data.flatten(), alpha, beta).reshape(-1, 1)
    elif alpha != 1 and beta == 1:
        smoothed_data = exponential_smoothing(ddos_data.flatten(), alpha).reshape(-1, 1)
    else:
        smoothed_data = ddos_data
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(smoothed_data)
    
    # Split data
    train_size = len(scaled_data) - 36  # Last 3 years for testing
    train, test = scaled_data[:train_size], scaled_data[train_size:]
    
    # Model parameters
    n_input = random.choice(lags)
    n_epochs = random.choice(epochs)
    layer = random.choice([1, 2, 3])
    
    # Generate appropriate number of units based on number of layers
    if layer == 1:
        units = [random.choice([50, 100, 200])]
    elif layer == 2:
        units = [random.choice([64, 128]), random.choice([32, 64])]
    else:  # layer == 3
        units = [random.choice([100, 200]), random.choice([50, 100]), random.choice([25, 50])]
    
    lr = random.choice(lrs)
    rdo = random.choice(rdos)
    
    smape_value, mae, rmse, predictions, predictions_std, y_test = evaluate_model(
        train, test, n_input, alpha, beta, n_input, layer, units, lr, n_epochs, rdo, scaler)
    
    if smape_value < best_smape:
        best_smape = smape_value
        best_params = {
            'alpha': alpha, 'beta': beta, 'n_input': n_input, 'n_epochs': n_epochs,
            'layer': layer, 'units': units, 'lr': lr, 'rdo': rdo
        }
        best_predictions = predictions
        best_predictions_std = predictions_std
        best_y_test = y_test

print(f"Best SMAPE: {best_smape}")
print(f"Best parameters: {best_params}")
test_dates = data['Date'].iloc[-len(best_y_test):]

plt.figure(figsize=(15, 8))

# Plot actual values
plt.plot(test_dates, best_y_test, label='Actual', color='blue')

# Plot predicted values
plt.plot(test_dates, best_predictions, label='Predicted', color='red')

# Plot confidence interval
plt.fill_between(test_dates, 
                 best_predictions.flatten() - 1.96 * best_predictions_std.flatten(), 
                 best_predictions.flatten() + 1.96 * best_predictions_std.flatten(), 
                 color='red', alpha=0.2)

# Customize the plot
plt.title('DDoS Attack Prediction with Uncertainty', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of DDoS Attacks', fontsize=12)
plt.legend(fontsize=10)

# Format x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()  # Rotation

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
