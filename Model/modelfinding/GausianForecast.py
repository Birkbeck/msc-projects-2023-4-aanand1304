import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel as C,
    ExpSineSquared,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Function to add lag features
def add_lag_features(data, target_variable, n_lags=1):
    for lag in range(1, n_lags + 1):
        data[f'{target_variable}_lag{lag}'] = data[target_variable].shift(lag)
    return data.dropna()

# Build the Gaussian Process model
def build_gaussian_process_model():
    # Simplified kernel
    kernel = C(1.0, (1e-3, 1e3)) * (
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) +
        ExpSineSquared(length_scale=1.0, periodicity=12.0, periodicity_bounds=(6.0, 18.0))
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

    gp_model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=True,
        alpha=1e-5,
        random_state=42,
        optimizer='fmin_l_bfgs_b'  # Removed 'max_iter_predict' parameter
    )
    return gp_model

# Train the Gaussian Process model
def train_gaussian_process(train_x, train_y):
    gp_model = build_gaussian_process_model()
    gp_model.fit(train_x, train_y)
    return gp_model

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv('FinalDataset.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data.set_index('Date', inplace=True)

    target_variable = 'DDoS-ALL'
    n_lags = 12  # Number of lag features

    # Add lag features for temporal dependency
    data = add_lag_features(data, target_variable, n_lags=n_lags)

    # Prepare the dataset
    features = [f'{target_variable}_lag{i}' for i in range(1, n_lags + 1)]
    data_features = data[features]
    data_target = data[target_variable]

    # Scale the features and target separately using StandardScaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaled_features = scaler_x.fit_transform(data_features)
    scaled_target = scaler_y.fit_transform(data_target.values.reshape(-1, 1)).flatten()

    # Train Gaussian Process model
    train_x = scaled_features
    train_y = scaled_target
    gp_model = train_gaussian_process(train_x, train_y)

    # Get the initial lag values from the last row of the training data
    initial_lag_values = train_x[-1, :]

    # Forecast for the next 36 months
    n_forecast = 36
    predictions = []
    lag_values = initial_lag_values.copy()

    for i in range(n_forecast):
        # The input vector is the lag values
        input_vector = lag_values.reshape(1, -1)

        # Make a prediction with the Gaussian Process
        prediction_mean, prediction_std = gp_model.predict(input_vector, return_std=True)
        predictions.append(prediction_mean[0])

        # Update lag_values
        lag_values = np.roll(lag_values, -1)
        lag_values[-1] = prediction_mean[0]

    # Inverse transform the predictions
    inv_predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Generate future dates for plotting
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_forecast, freq='MS')

    # Plot actual vs forecast data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[target_variable], label='Actual Data (2011-2024)', color='blue')
    plt.plot(future_dates, inv_predictions, label='Forecast (2024-2027)', color='red')
    plt.title('3-Year Forecast for DDoS-ALL with Enhanced Gaussian Process Regression')
    plt.xlabel('Year')
    plt.ylabel('Incident Count')
    plt.legend()
    plt.show()
