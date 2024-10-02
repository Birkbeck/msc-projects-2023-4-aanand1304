import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def smape(yTrue, yPred):
    denominator = (np.abs(yTrue) + np.abs(yPred))
    return np.mean(2 * np.abs(yPred - yTrue) / np.where(denominator == 0, 1, denominator)) * 100

def prepare_data_lstm(data, n_input):
    X, y = [], []
    for i in range(len(data) - n_input):
        X.append(data[i:i + n_input])
        y.append(data[i + n_input])
    return np.array(X), np.array(y)

def build_lstm_model(n_input, layer, unit, dropout_rate):
    model = Sequential()
    model.add(Input(shape=(n_input, 1)))
    
    for i in range(layer):
        return_sequences = i < layer - 1
        model.add(LSTM(unit, activation='relu', return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_arima_model(train, order):
    model = ARIMA(train, order=order)
    return model.fit()

def build_prophet_model(train_df):
    model = Prophet()
    model.fit(train_df)
    return model

def forecast_lstm(model, train, test, n_input, scaler):
    X_train, y_train = prepare_data_lstm(train, n_input)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)
    
    X_test, y_test = prepare_data_lstm(test, n_input)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test)
    
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return predictions_inv, y_test_inv

def forecast_arima(model, steps):
    return model.forecast(steps)

def forecast_prophet(model, future_df):
    forecast = model.predict(future_df)
    return forecast['yhat'].values

def ensemble_forecast(lstm_preds, arima_preds, prophet_preds):
    return (lstm_preds + arima_preds + prophet_preds) / 3

if __name__ == "__main__":
    data = pd.read_csv('FinalDataset.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%b-%y')
    data.set_index('Date', inplace=True)
    
    attack = 'DDoS-ALL'
    attack_data = data[attack].values.reshape(-1, 1)

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(attack_data)

    train_size = len(scaled_data) - 36
    train, test = scaled_data[:train_size], scaled_data[train_size:]
    test_dates = data.index[train_size:]

    n_input = 12
    lstm_model = build_lstm_model(n_input=n_input, layer=2, unit=100, dropout_rate=0.2)
    arima_model = build_arima_model(train.flatten(), order=(5,1,0))

    train_df = pd.DataFrame({'ds': data.index[:train_size], 'y': train.flatten()})
    future_df = pd.DataFrame({'ds': test_dates})
    prophet_model = build_prophet_model(train_df)

    lstm_preds, y_test_inv = forecast_lstm(lstm_model, train, test, n_input, scaler)
    arima_preds = forecast_arima(arima_model, len(test))
    prophet_preds = forecast_prophet(prophet_model, future_df)

    ensemble_preds = ensemble_forecast(lstm_preds.flatten(), arima_preds, prophet_preds)
    
    smape_value = smape(y_test_inv, ensemble_preds)
    mae_value = mean_absolute_error(y_test_inv, ensemble_preds)
    rmse_value = np.sqrt(mean_squared_error(y_test_inv, ensemble_preds))
    
    print(f'SMAPE: {smape_value:.2f}')
    print(f'MAE: {mae_value:.2f}')
    print(f'RMSE: {rmse_value:.2f}')
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, y_test_inv, label='Actual', color='blue')
    plt.plot(test_dates, ensemble_preds, label='Ensemble Prediction', color='red')
    plt.title(f'{attack} - Actual vs Ensemble Prediction')
    plt.xlabel('Time')
    plt.ylabel('Incident Count')
    plt.legend()
    plt.show()
