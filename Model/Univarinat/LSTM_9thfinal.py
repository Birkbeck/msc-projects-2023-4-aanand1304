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
import os
from tensorflow.keras.layers import Input

def smape(yTrue, yPred):
    denominator = (np.abs(yTrue) + np.abs(yPred))
    smape_value = np.mean(2 * np.abs(yPred - yTrue) / np.where(denominator == 0, 1, denominator)) * 100
    return smape_value



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
    model.add(Input(shape=(n_input, 1)))
    model.add(KeepDropout(LSTM(unit[0], recurrent_dropout=rdo, activation='relu', return_sequences=(layer > 1))))
    
    for i in range(1, layer):
        model.add(KeepDropout(LSTM(unit[min(i, len(unit)-1)], activation='relu', return_sequences=(i < layer-1), recurrent_dropout=rdo)))
    
    model.add(RepeatVector(1))
    
    for i in range(layer):
        model.add(KeepDropout(LSTM(unit[min(layer-i-1, len(unit)-1)], activation='relu', return_sequences=True, recurrent_dropout=rdo)))
    
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
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
    iso_forest = IsolationForest(contamination=0.01)
    outliers = iso_forest.fit_predict(data.reshape(-1, 1))
    return outliers

# Calculate prediction accuracy based on a given tolerance
def prediction_accuracy(y_true, y_pred, tolerance=0.1):
    correct_predictions = np.abs(y_true - y_pred) <= tolerance * np.abs(y_true)
    accuracy = np.mean(correct_predictions) * 100
    return accuracy

# Evaluate the model performance with confidence intervals
def evaluate_model(train, test, n_input, layer, unit, rdo, scaler, epochs):
    train_x, train_y = prepare_data(train, n_input)
    test_x, test_y = prepare_data(test, n_input)
    
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    
    model = build_model(n_input, layer, unit, rdo)
    es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    model.fit(train_x, train_y, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)
    
    n_iterations = 20
    predictions_mc = np.array([model(test_x, training=True) for _ in range(n_iterations)])
    
    predictions_mean = np.mean(predictions_mc, axis=0).reshape(-1, 1)
    predictions_std = np.std(predictions_mc, axis=0).reshape(-1, 1)
    
    lower_bound = predictions_mean - 1.96 * predictions_std
    upper_bound = predictions_mean + 1.96 * predictions_std
    
    predictions_mean = scaler.inverse_transform(predictions_mean)
    lower_bound = scaler.inverse_transform(lower_bound)
    upper_bound = scaler.inverse_transform(upper_bound)
    y_test_inv = scaler.inverse_transform(test_y.reshape(-1, 1))
    
    smape_value = smape(y_test_inv, predictions_mean)
    mae = mean_absolute_error(y_test_inv, predictions_mean)
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_mean))
    accuracy = prediction_accuracy(y_test_inv, predictions_mean, tolerance=0.1)
    
    return smape_value, mae, rmse, predictions_mean, accuracy, y_test_inv, lower_bound, upper_bound

# # Forecast for the next 3 years with monthly predictions (36 months)
# def forecast_next_years(model, scaled_data, scaler, n_input, n_months=36):
#     history = scaled_data[-n_input:]  # Start with the last known input
#     predictions = []
#     predictions_std = []

#     # Loop over the number of future months we want to predict
#     for _ in range(n_months):  # Predicting monthly for n_months
#         # Reshape the current history to match LSTM input format
#         history_reshaped = history.reshape(1, history.shape[0], 1)
        
#         # Generate a Monte Carlo ensemble of predictions (dropout enabled during testing)
#         pred_mc = np.array([model.predict(history_reshaped) for _ in range(20)])
#         pred_mean = np.mean(pred_mc, axis=0)  # Take the mean prediction
#         pred_std = np.std(pred_mc, axis=0)  # Standard deviation for confidence intervals
        
#         # Add the mean prediction to the history
#         predictions.append(pred_mean[0][0])
#         predictions_std.append(pred_std[0][0])
        
#         # Now roll the history window to include the new prediction and drop the oldest value
#         history = np.append(history, pred_mean[0], axis=0)[1:]

#     # Transform the predictions back to the original scale
#     predictions = np.array(predictions).reshape(-1, 1)
#     predictions_std = np.array(predictions_std).reshape(-1, 1)
    
#     return (
#         scaler.inverse_transform(predictions).flatten(),
#         scaler.inverse_transform(predictions - 1.96 * predictions_std).flatten(),
#         scaler.inverse_transform(predictions + 1.96 * predictions_std).flatten()
#     )

# Load and preprocess data
data = pd.read_csv('FinalDataset.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data.set_index('Date', inplace=True)


alphas = [0.05, 0.2, 0.5, 0.7, 1]
betas = [0.3, 0.5, 0.7, 1]
lags = [list(range(1, max(151 // 10, 1) + 1))]
epochs_list = [500, 330, 400, 200, 100,1000, 600]
rdos = [0.1, 0.15]
attacks = ['DDoS-ALL', 'Phishing-ALL', 'Ransomware-ALL', 'Password Attack-ALL', 'SQL Injection-ALL', 'Account Hijacking-ALL', 
           'Defacement-ALL', 'Trojan-ALL', 'Vulnerability-ALL', 'Zero-day-ALL', 'Malware-ALL', 'Advanced persistent threat-ALL', 
           'XSS-ALL', 'Data Breach-ALL', 'Disinformation/Misinformation-ALL', 'Targeted Attack-ALL','Adware-ALL',
           'Brute Force Attack-ALL', 'Malvertising-ALL', 'Backdoor-ALL', 'Botnet-ALL', 'Cryptojacking-ALL',
           'Worms-ALL', 'Spyware-ALL', 'Mentions-MITM', 'Mentions-DNS Spoofing', 
           'Mentions-WannaCry Ransomware','Mentions-Dropper', 'Mentions-Wiper', 'Mentions-Pharming', 'Mentions-Insider Threat',
           'Mentions-Drive-by', 'Mentions-Rootkit', 'Mentions-Adversarial Attack', 'Mentions-Data Poisoning', 'Mentions-Deepfake',
           'Mentions-Supply Chain', 'Mentions-IoT Device Attack', 'Mentions-Keylogger', 'Mentions-DNS Tunneling', 'Mentions-Session Hijacking',
           'Mentions-URL manipulation']


# Create directories for saving best parameters and plots
output_dir = 'best_params_files'
output_plot_dir = 'output_plot'  # Directory for saving plots

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_plot_dir, exist_ok=True)  # Ensure the output_plot directory is created

# Loop through each attack type
for attack in attacks:
    print(f"Processing attack: {attack}")
    
    # Select the data for the current attack
    ddos_data = data[attack].values.reshape(-1, 1)
    
    # Detect and remove outliers (optional, can remove if not needed)
    # outliers = detect_outliers(ddos_data)
    filtered_data = ddos_data
    
    # Scale the data
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(filtered_data)
    
    train_size = len(scaled_data) - 36
    train, test = scaled_data[:train_size], scaled_data[train_size:]
    test_dates = data.index[train_size:]  # Use actual dates from the dataset
    
    best_smape = float('inf')
    best_params = {}
    best_model = None

    iterations = 100

    # Iterate through random hyperparameters to find the best configuration
    for iteration in range(iterations):
        print(f"Iteration {iteration+1} out of {iterations}")
        alpha = random.choice(alphas)
        beta = random.choice(betas)
        
        if alpha != 1 and beta != 1:
            smoothed_data = double_exponential_smoothing(ddos_data.flatten(), alpha, beta).reshape(-1, 1)
        elif alpha != 1 and beta == 1:
            smoothed_data = exponential_smoothing(ddos_data.flatten(), alpha).reshape(-1, 1)
        else:
            smoothed_data = ddos_data
        
        # Scale the smoothed data
        scaled_data = scaler.fit_transform(smoothed_data)
        train_size = len(scaled_data) - 36
        train, test = scaled_data[:train_size], scaled_data[train_size:]
        
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
        
        smape_value, mae, rmse, predictions_mean, accuracy, y_test_inv, lower_bound, upper_bound = evaluate_model(
            train, test, n_input, layer, units, rdo, scaler, n_epochs
        )
        
        if smape_value < best_smape:
            best_smape = smape_value
            best_params = {
                'alpha': alpha, 'beta': beta, 'n_input': n_input, 'n_epochs': n_epochs,
                'layer': layer, 'units': units, 'rdo': rdo
            }
            best_model = build_model(n_input, layer, units, rdo)
            best_predictions_mean = predictions_mean
            best_y_test_inv = y_test_inv
            best_lower_bound = lower_bound
            best_upper_bound = upper_bound

    # Save the best parameters for the current attack in a separate JSON file
    file_path = os.path.join(output_dir, f'{attack}_best_params.json')
    data_to_save = {
        "SMAPE": best_smape,
        "Best Parameters": best_params
    }
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f)
    
    # Adjust test_dates to match the length of best_y_test_inv
    test_dates = test_dates[-len(best_y_test_inv):]

    print(f"Best parameters for {attack} saved to {file_path}.")
    print(f"Best SMAPE for {attack}: {best_smape}")
    print(f"Best MAE: {mae}")
    print(f"Best RMSE: {rmse}")
    print(f"Model Accuracy: {accuracy}%")
    print(f"Best parameters: {best_params}")

    # Plot the actual vs predicted values and save the plot
    plt.figure(figsize=(10, 6))

    # Plot actual values predicted mean values
    plt.plot(test_dates, best_y_test_inv, label='Actual', color='blue', linewidth=2)
    plt.plot(test_dates, best_predictions_mean, label='Predicted Mean', color='red', linewidth=2)

    # Plot the confidence intervals (upper and lower bounds)
    plt.fill_between(
        test_dates,
        best_lower_bound.flatten(),   # Lower bound of the 95% confidence interval
        best_upper_bound.flatten(),   # Upper bound of the 95% confidence interval
        color='green', alpha=0.3, label='95% Confidence Interval'
    )

    # Add titles, labels, and legend
    plt.title(f'Actual vs Predicted for {attack} (Best SMAPE: {best_smape:.2f})')
    plt.xlabel('Date')  # Use dates from the dataset
    plt.ylabel('Value')
    plt.legend()

    # Save the plot
    plot_path = os.path.join(output_plot_dir, f'{attack}_actual_vs_predicted.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot for {attack} saved to {plot_path}.")

print("Best parameters for all attacks have been saved individually.")
