import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('FinalDataset.csv')

# Convert 'Date' to datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Prepare the dataframe for Prophet
df = data[['Date', 'DDoS-ALL', 'Phishing-ALL', 'Ransomware-ALL', 'GDP-USA', 'WAR/CONFLICT ALL', 'Internet Users (Millions)']]

# Rename columns to 'ds' and 'y' for Prophet compatibility
df = df.rename(columns={'Date': 'ds', 'DDoS-ALL': 'y'})

# Split the data into training and testing sets (train until 2023, test from 2024 onwards)
train_df = df[df['ds'] < '2024-01-01']  # Training set
test_df = df[df['ds'] >= '2024-01-01']  # Test set

# Initialize the Prophet model
m = Prophet()

# Adding external regressors
m.add_regressor('Phishing-ALL')
m.add_regressor('Ransomware-ALL')
m.add_regressor('GDP-USA')
m.add_regressor('WAR/CONFLICT ALL')
m.add_regressor('Internet Users (Millions)')

# Fit the model on the training set
m.fit(train_df)

# Make a DataFrame with future dates, including the test set
future = m.make_future_dataframe(periods=len(test_df), freq='M')

# Add the regressors for the future dates from the test set
future['Phishing-ALL'] = pd.concat([train_df['Phishing-ALL'], test_df['Phishing-ALL']])
future['Ransomware-ALL'] = pd.concat([train_df['Ransomware-ALL'], test_df['Ransomware-ALL']])
future['GDP-USA'] = pd.concat([train_df['GDP-USA'], test_df['GDP-USA']])
future['WAR/CONFLICT ALL'] = pd.concat([train_df['WAR/CONFLICT ALL'], test_df['WAR/CONFLICT ALL']])
future['Internet Users (Millions)'] = pd.concat([train_df['Internet Users (Millions)'], test_df['Internet Users (Millions)']])

# Make predictions
forecast = m.predict(future)

# Extract the predictions for the test period
test_forecast = forecast[forecast['ds'] >= '2024-01-01']

# Calculate performance metrics (MAE, RMSE, SMAPE)
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

y_true = test_df['y'].values  # Actual values
y_pred = test_forecast['yhat'].values  # Predicted values

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
smape_value = smape(y_true, y_pred)

# Print metrics
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'SMAPE: {smape_value:.2f}%')

# Plot the forecast with actual values for the test set
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(test_df['ds'], y_true, label='Actual', color='blue')
plt.plot(test_df['ds'], y_pred, label='Forecast', color='red')
plt.fill_between(test_df['ds'], test_forecast['yhat_lower'], test_forecast['yhat_upper'], color='green', alpha=0.3, label='95% Confidence Interval')
plt.title("Prophet Forecast vs Actuals (Test Period)")
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
