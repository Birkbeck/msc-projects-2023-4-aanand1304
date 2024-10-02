import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, WhiteKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv('FinalDataset.csv', parse_dates=True, index_col='Date')

# Ensure the date index has a proper frequency
if df.index.freq is None:
    df = df.asfreq('MS')

# Fill any NaNs in the dataset (forward fill + mean for remaining)
df = df.ffill().fillna(0)

# Define the target variable and features
target_variable = 'DDoS-ALL'
features = ['Mentions-MITM', 'WAR/CONFLICT ALL', 'Internet Users (Millions)']

# Define features (X) and target variable (y)
X = df[features]
y = df[target_variable]

# Check for NaNs after preprocessing
print("\nNaN check after preprocessing:", X.isna().sum())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Adjusted kernel for better flexibility
kernel = (1.0 * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2)) +  # Smooth trend
          RationalQuadratic(length_scale=1.0, alpha=0.1, length_scale_bounds=(1e-2, 1e2)) +  # Flexibility
          ExpSineSquared(periodicity=12.0, length_scale=1.0, periodicity_bounds=(1, 24)) +  # Seasonal component
          WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-2)))  # Adjusted noise kernel

# Initialize Gaussian Process Regressor with optimized restarts
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=20)

# Fit the model
gpr.fit(X_scaled, y)

# Optional: Evaluate performance using cross-validation
scores = cross_val_score(gpr, X_scaled, y, scoring='neg_mean_absolute_error', cv=5)
print("\nCross-validation MAE scores:", -scores)
print("Mean MAE:", np.mean(-scores))

# Generate future dates (forecasting for 3 years)
future_dates = [df.index[-1] + DateOffset(months=i) for i in range(1, 37)]
future_df = pd.DataFrame(index=future_dates, columns=features)

# Fill future features with reasonable estimates
future_df['Mentions-MITM'] = X['Mentions-MITM'].iloc[-1]  # Using last known value
future_df['WAR/CONFLICT ALL'] = X['WAR/CONFLICT ALL'].iloc[-1]  # Using last known value
future_df['Internet Users (Millions)'] = X['Internet Users (Millions)'].iloc[-1]  # Using last known value

# Scale the future data
future_df_scaled = scaler.transform(future_df)

# Predict the future using Gaussian Process
y_future_pred, y_future_pred_std = gpr.predict(future_df_scaled, return_std=True)

# Plotting the true data and future forecast with confidence intervals
plt.figure(figsize=(12, 8))
plt.plot(df.index, df[target_variable], label='True Data', color='blue')
plt.plot(future_dates, y_future_pred, label='Forecast', color='red')
plt.fill_between(future_dates, y_future_pred - 1.96 * y_future_pred_std, y_future_pred + 1.96 * y_future_pred_std, color='green', alpha=0.2, label='95% CI')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Incident Count')
plt.title('DDoS-ALL Forecast (2024-2027)')
plt.show()
