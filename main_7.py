import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.svm import LinearSVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Passengers.csv')
print(data.columns)

data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)

# Extract the number of passengers
time_series = data['#Passengers'].values

# Split data into 70-15-15% for training, validation, and testing
train_size = int(0.7 * len(time_series))
val_size = int(0.15 * len(time_series))

train = time_series[:train_size]
val = time_series[train_size:train_size + val_size]
test = time_series[train_size + val_size:]

from sklearn.metrics import mean_squared_error, mean_absolute_error
# Function to create sequences for LSTM/GRU training
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Linear model
linear_model = LinearRegression()
X_train = np.arange(train_size).reshape(-1, 1)
y_train = train
linear_model.fit(X_train, y_train)

# Forecast using linear model
linear_forecast = linear_model.predict(np.arange(len(time_series)).reshape(-1, 1))

# Calculate residuals
residuals = time_series - linear_forecast

# Nonlinear model (LSTM/GRU) for residuals
seq_length = 12  # Length of input sequences for LSTM/GRU
X_train_residual, y_train_residual = create_sequences(residuals[:train_size], seq_length)

# LSTM model for residuals
model = Sequential([
    LSTM(64, input_shape=(seq_length, 1), return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Reshape data for LSTM input
X_train_residual = X_train_residual.reshape((X_train_residual.shape[0], X_train_residual.shape[1], 1))

# Train the LSTM model
model.fit(X_train_residual, y_train_residual, epochs=20, batch_size=16, validation_split=0.2)

# Make predictions on residuals
predicted_residuals = model.predict(X_train_residual).flatten()

# Final forecast = Linear forecast + predicted residuals
final_forecast = linear_forecast[:train_size - seq_length] + predicted_residuals



# Plot results
plt.plot(time_series, label='Original')
plt.plot(final_forecast, label='Additive Hybrid Forecast')
plt.legend()
plt.show()


# Calculate residuals for multiplicative model
residuals_multiplicative = time_series / linear_forecast

# Create sequences for LSTM/GRU training
X_train_residual_mult, y_train_residual_mult = create_sequences(residuals_multiplicative[:train_size], seq_length)

# LSTM model for residuals
model_mult = Sequential([
    LSTM(64, input_shape=(seq_length, 1), return_sequences=False),
    Dense(1)
])
model_mult.compile(optimizer='adam', loss='mse')

# Reshape data for LSTM input
X_train_residual_mult = X_train_residual_mult.reshape((X_train_residual_mult.shape[0], X_train_residual_mult.shape[1], 1))

# Train the LSTM model
model_mult.fit(X_train_residual_mult, y_train_residual_mult, epochs=20, batch_size=16, validation_split=0.2)

# Make predictions on residuals
predicted_residuals_mult = model_mult.predict(X_train_residual_mult).flatten()

# Final forecast = Linear forecast * predicted residuals
final_forecast_mult = linear_forecast[:train_size - seq_length] * predicted_residuals_mult

# Plot results
plt.plot(time_series, label='Original')
plt.plot(final_forecast_mult, label='Multiplicative Hybrid Forecast')
plt.legend()
plt.show()


# STL decomposition with specified period
stl = STL(time_series, seasonal=13, period=12)  # period=12 for monthly data
result = stl.fit()
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# 1. Model the trend component using linear regression
linear_model = LinearRegression()
X_trend = np.arange(len(trend)).reshape(-1, 1)
linear_model.fit(X_trend[:train_size], trend[:train_size])
trend_forecast = linear_model.predict(X_trend)

# 2. Model the seasonal component using LSTM
seq_length = 12
X_seasonal, y_seasonal = create_sequences(seasonal[:train_size], seq_length)

# LSTM model for seasonal component
seasonal_model = Sequential([
    LSTM(64, input_shape=(seq_length, 1), return_sequences=False),
    Dense(1)
])
seasonal_model.compile(optimizer='adam', loss='mse')

X_seasonal = X_seasonal.reshape((X_seasonal.shape[0], X_seasonal.shape[1], 1))
seasonal_model.fit(X_seasonal, y_seasonal, epochs=20, batch_size=16, validation_split=0.2)
seasonal_forecast = seasonal_model.predict(X_seasonal).flatten()

# 3. Model the residual component using GRU
X_residual, y_residual = create_sequences(residual[:train_size], seq_length)

residual_model = Sequential([
    GRU(64, input_shape=(seq_length, 1), return_sequences=False),
    Dense(1)
])
residual_model.compile(optimizer='adam', loss='mse')

X_residual = X_residual.reshape((X_residual.shape[0], X_residual.shape[1], 1))
residual_model.fit(X_residual, y_residual, epochs=20, batch_size=16, validation_split=0.2)
residual_forecast = residual_model.predict(X_residual).flatten()

# Combine the forecasts
final_forecast_additive = (trend_forecast[:train_size - seq_length] +
                           seasonal_forecast +
                           residual_forecast)

# Plot the results
plt.plot(time_series, label='Original')
plt.plot(np.arange(seq_length, seq_length + len(final_forecast_additive)), final_forecast_additive, label='Additive STL Hybrid Forecast')
plt.legend()
plt.show()



from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler


# Apply logarithmic transformation for multiplicative decomposition
log_data = np.log(data['#Passengers'])

# Perform STL decomposition
stl = STL(log_data, seasonal=13)
result = stl.fit()

# Extract components
trend_log = result.trend
seasonal_log = result.seasonal
residual_log = result.resid

# Plot STL decomposition (log scale)
result.plot()
plt.show()

### Trend Component Modeling using Linear Regression ###
# Prepare the trend data for linear regression
trend_non_nan = trend_log.dropna()
X = np.arange(len(trend_non_nan)).reshape(-1, 1)
y = trend_non_nan.values

# Train-test split (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=False)

# Linear Regression Model for Trend
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict the trend component
trend_predictions_train = linear_model.predict(X_train)
trend_predictions_val = linear_model.predict(X_val)
trend_predictions_test = linear_model.predict(X_test)

### Seasonal Component Modeling using LSTM ###
# Normalize the seasonal component
scaler_seasonal = MinMaxScaler()
seasonal_scaled = scaler_seasonal.fit_transform(seasonal_log.dropna().values.reshape(-1, 1))

# Prepare data for LSTM (create sequences)
def create_sequences(data, sequence_length=12):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

sequence_length = 12  # One year of data for sequence length
X_seasonal = create_sequences(seasonal_scaled, sequence_length)
y_seasonal = seasonal_scaled[sequence_length:]

# Train-test split for seasonal component
X_train_seasonal, X_temp_seasonal, y_train_seasonal, y_temp_seasonal = train_test_split(X_seasonal, y_seasonal, test_size=0.30, shuffle=False)
X_val_seasonal, X_test_seasonal, y_val_seasonal, y_test_seasonal = train_test_split(X_temp_seasonal, y_temp_seasonal, test_size=0.50, shuffle=False)

# LSTM Model for Seasonal Component
lstm_model = Sequential()
lstm_model.add(LSTM(20, activation='relu', input_shape=(sequence_length, 1)))  # Fewer neurons to prevent overfitting
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Train the LSTM model with fewer epochs
lstm_model.fit(X_train_seasonal, y_train_seasonal, epochs=50, verbose=0, validation_data=(X_val_seasonal, y_val_seasonal))

# Predict the seasonal component
seasonal_predictions_train = lstm_model.predict(X_train_seasonal)
seasonal_predictions_val = lstm_model.predict(X_val_seasonal)
seasonal_predictions_test = lstm_model.predict(X_test_seasonal)

# Inverse transform to original scale
seasonal_predictions_train = scaler_seasonal.inverse_transform(seasonal_predictions_train)
seasonal_predictions_val = scaler_seasonal.inverse_transform(seasonal_predictions_val)
seasonal_predictions_test = scaler_seasonal.inverse_transform(seasonal_predictions_test)

### Residual Component Modeling using GRU ###
# Normalize the residual component
scaler_residual = MinMaxScaler()
residual_scaled = scaler_residual.fit_transform(residual_log.dropna().values.reshape(-1, 1))

# Prepare data for GRU (create sequences)
X_residual = create_sequences(residual_scaled, sequence_length)
y_residual = residual_scaled[sequence_length:]

# Train-test split for residual component
X_train_residual, X_temp_residual, y_train_residual, y_temp_residual = train_test_split(X_residual, y_residual, test_size=0.30, shuffle=False)
X_val_residual, X_test_residual, y_val_residual, y_test_residual = train_test_split(X_temp_residual, y_temp_residual, test_size=0.50, shuffle=False)

# GRU Model for Residual Component
gru_model = Sequential()
gru_model.add(GRU(20, activation='relu', input_shape=(sequence_length, 1)))  # Fewer neurons to prevent overfitting
gru_model.add(Dense(1))
gru_model.compile(optimizer='adam', loss='mse')

# Train the GRU model with fewer epochs
gru_model.fit(X_train_residual, y_train_residual, epochs=50, verbose=0, validation_data=(X_val_residual, y_val_residual))

# Predict the residual component
residual_predictions_train = gru_model.predict(X_train_residual)
residual_predictions_val = gru_model.predict(X_val_residual)
residual_predictions_test = gru_model.predict(X_test_residual)

# Inverse transform to original scale
residual_predictions_train = scaler_residual.inverse_transform(residual_predictions_train)
residual_predictions_val = scaler_residual.inverse_transform(residual_predictions_val)
residual_predictions_test = scaler_residual.inverse_transform(residual_predictions_test)

### Combine the Components ###
# Align the lengths of the predictions (trim based on sequence length)
n_lost = sequence_length  # Data lost during sequence creation
min_length_train = min(len(trend_predictions_train[n_lost:]), len(seasonal_predictions_train), len(residual_predictions_train))
min_length_val = min(len(trend_predictions_val[n_lost:]), len(seasonal_predictions_val), len(residual_predictions_val))
min_length_test = min(len(trend_predictions_test[n_lost:]), len(seasonal_predictions_test), len(residual_predictions_test))

# Trim all components to minimum length
trend_train = trend_predictions_train[n_lost:][:min_length_train]
seasonal_train = seasonal_predictions_train[:min_length_train]
residual_train = residual_predictions_train[:min_length_train]

trend_val = trend_predictions_val[n_lost:][:min_length_val]
seasonal_val = seasonal_predictions_val[:min_length_val]
residual_val = residual_predictions_val[:min_length_val]

trend_test = trend_predictions_test[n_lost:][:min_length_test]
seasonal_test = seasonal_predictions_test[:min_length_test]
residual_test = residual_predictions_test[:min_length_test]

# Final predictions (exponentiating to reverse the log transformation)
final_train_predictions = np.exp(trend_train + seasonal_train.flatten() + residual_train.flatten())
final_val_predictions = np.exp(trend_val + seasonal_val.flatten() + residual_val.flatten())
final_test_predictions = np.exp(trend_test + seasonal_test.flatten() + residual_test.flatten())

### Plot Final Predictions ###
train_index = trend_non_nan.index[n_lost:n_lost + len(final_train_predictions)]
val_index = trend_non_nan.index[n_lost + len(final_train_predictions):n_lost + len(final_train_predictions) + len(final_val_predictions)]
test_index = trend_non_nan.index[n_lost + len(final_train_predictions) + len(final_val_predictions):n_lost + len(final_train_predictions) + len(final_val_predictions) + len(final_test_predictions)]

# Plot final predictions
plt.plot(data.index, data['#Passengers'], label='Actual Passengers')
plt.plot(train_index, final_train_predictions, label='Predicted Passengers (Train)', linestyle='--')
plt.plot(val_index, final_val_predictions, label='Predicted Passengers (Validation)', linestyle='--')
plt.plot(test_index, final_test_predictions, label='Predicted Passengers (Test)', linestyle='--')
plt.legend()
plt.title("Final Passenger Forecast (Multiplicative Decomposition)")
plt.show()


