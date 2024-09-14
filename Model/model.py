import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Step 1: Load the data
print("Loading data...")
data = pd.read_csv('/Users/ashitanand/Documents/Final Project/Dataset/FinalDataset.csv', parse_dates=['Date'], index_col='Date')

# Step 2: Data Preprocessing
print("Preprocessing data...")
data = data.fillna(0)  # Handling missing values by replacing NaNs with 0

# Define the number of future steps to forecast (e.g., 3 years, assuming monthly data)
future_steps = 36  # Forecasting for 3 years (May 2024 to April 2027)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
print("Data scaling complete.")

# Step 3: Use the entire dataset as input for the model
X = scaled_data[:-1, :]  # All data points except the last one (features)
y = scaled_data[1:, :]   # All data points except the first one (targets)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Reshape input for LSTM [samples, time steps, features]
X_train_LSTM = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshape to (samples, time steps, features)
X_test_LSTM = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Step 4: Build and Train the LSTM Model
print("Building and training LSTM model...")
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=False, input_shape=(1, X_train_LSTM.shape[2])))
lstm_model.add(Dense(y_train.shape[1]))  # Adjust output layer for multivariate
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

# Learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,  # Reduce learning rate by a factor of 0.5
    patience=5,  # After 5 epochs with no improvement
    min_lr=1e-5  # Minimum learning rate
)

# Train the LSTM model with increased epochs
lstm_model.fit(
    X_train_LSTM, 
    y_train, 
    validation_split=0.2,  # Use 20% of the training data for validation
    epochs=200,            # Increase the number of epochs
    batch_size=16, 
    verbose=1, 
    callbacks=[early_stopping_callback, lr_scheduler]
)
print("LSTM model training complete.")

# Step 5: Predictions with LSTM on Training Data
print("Making predictions with LSTM model on training data...")
lstm_train_predict = lstm_model.predict(X_train_LSTM)
lstm_test_predict = lstm_model.predict(X_test_LSTM)

# Inverse transform predictions
lstm_train_predict = scaler.inverse_transform(lstm_train_predict)
lstm_test_predict = scaler.inverse_transform(lstm_test_predict)

# Step 6: Train a Random Forest Model
print("Training Random Forest model...")
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

random_forest_train_predict = random_forest_model.predict(X_train)
random_forest_test_predict = random_forest_model.predict(X_test)

# Inverse transform predictions
random_forest_train_predict = scaler.inverse_transform(random_forest_train_predict)
random_forest_test_predict = scaler.inverse_transform(random_forest_test_predict)
print("Random Forest predictions complete.")

# Step 7: Train an XGBoost Model
print("Training XGBoost model...")
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

xgb_train_predict = xgb_model.predict(X_train)
xgb_test_predict = xgb_model.predict(X_test)

# Inverse transform predictions
xgb_train_predict = scaler.inverse_transform(xgb_train_predict)
xgb_test_predict = scaler.inverse_transform(xgb_test_predict)
print("XGBoost predictions complete.")

# Step 8: Ensemble Predictions
print("Creating ensemble predictions...")
ensemble_train_predict = (lstm_train_predict + random_forest_train_predict + xgb_train_predict) / 3
ensemble_test_predict = (lstm_test_predict + random_forest_test_predict + xgb_test_predict) / 3
print("Ensemble predictions complete.")

# Step 9: Forecast the Next 3 Years
print("Forecasting the next 3 years...")
last_input = X_test[-1].reshape(1, -1)
predictions = []

for i in range(future_steps):
    # LSTM Prediction
    lstm_pred = lstm_model.predict(last_input.reshape(1, 1, -1)).flatten()
    # Random Forest Prediction
    rf_pred = random_forest_model.predict(last_input)
    # XGBoost Prediction
    xgb_pred = xgb_model.predict(last_input)
    
    # Ensemble Prediction (Weighted Average)
    ensemble_pred = (lstm_pred + rf_pred + xgb_pred) / 3
    predictions.append(ensemble_pred)
    
    # Update last_input for the next step
    last_input = np.roll(last_input, -1)
    last_input[-1] = ensemble_pred[-1]

# Convert predictions to numpy array
predictions = np.array(predictions)

# Reshape predictions to 2D for inverse_transform
n_samples, n_features = predictions.shape[0], predictions.shape[1]
predictions_reshaped = predictions.reshape(n_samples, n_features)

# Apply inverse_transform
predictions_inverse = scaler.inverse_transform(predictions_reshaped)

# Reshape back to 3D if necessary
predictions_final = predictions_inverse.reshape(n_samples, 1, n_features)

print("Forecasting complete.")

# Step 10: Calculate Accuracy Metrics for Test Data
print("Calculating accuracy metrics...")
def calculate_accuracy(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

train_rmse, train_r2 = calculate_accuracy(y_train, ensemble_train_predict)
test_rmse, test_r2 = calculate_accuracy(y_test, ensemble_test_predict)

print(f'Ensemble Model Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}')
print(f'Ensemble Model Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}')

# Step 11: Plot Results
print("Plotting results...")
def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_results(y_train, ensemble_train_predict, 'Train Data - Actual vs Predicted')
plot_results(y_test, ensemble_test_predict, 'Test Data - Actual vs Predicted')

# Plot the forecasted values for the next 3 years
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(data)), scaler.inverse_transform(scaled_data), label='Actual Data')
plt.plot(np.arange(len(data), len(data) + future_steps), predictions_final, label='Forecast for May 2024 to April 2027')
plt.title('3-Year Forecast (May 2024 to April 2027)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Display Accuracy Percentage for Test Data
accuracy_percentage = test_r2 * 100
print(f'Accuracy of the model on test data: {accuracy_percentage:.2f}%')

print("Script complete.")
