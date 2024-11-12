import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('Passengers.csv')

# Step 1: Create the Universe of Discourse (UOD)
# Extract passenger counts
passenger_counts = data['#Passengers'].values


# UOD calculation with 10% increment and decrement
min_val = np.min(passenger_counts)
max_val = np.max(passenger_counts)
uod_min = min_val - 0.1 * min_val
uod_max = max_val + 0.1 * max_val

# Partition UOD into intervals of length 10
interval_length = 10
intervals = np.arange(uod_min, uod_max + interval_length, interval_length)


# Compute midpoints of each interval for defuzzification
midpoints = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]



# Fuzzification function
def fuzzify_observation(value, intervals):
    for i in range(len(intervals) - 1):
        if intervals[i] <= value < intervals[i + 1]:
            return i + 1  # Interval index (1-based)
    return len(intervals) - 1  # If value equals the max

# Apply fuzzification
fuzzified_data = [fuzzify_observation(val, intervals) for val in data['#Passengers'].values]
data['Fuzzified'] = fuzzified_data

# Step 2: Normalize the fuzzified data
scaler = MinMaxScaler()
data['Normalized'] = scaler.fit_transform(data[['Fuzzified']])


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 3: Create sequences of patterns
def get_patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

lag_length = 12  # Using a lag length of 12 months
horizon = 1      # Predicting 1 step ahead
X, y = get_patterns(data[['Normalized']].values, lag_length, horizon)

# Split into train, validation, and test sets
lenTrain = int(round(len(X) * 0.7))
lenVal = int(round(len(X) * 0.15))  # Validation set size (15%)
X_train, X_val, X_test = X[:lenTrain], X[lenTrain:lenTrain + lenVal], X[lenTrain + lenVal:]
y_train, y_val, y_test = y[:lenTrain], y[lenTrain:lenTrain + lenVal], y[lenTrain + lenVal:]

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future values
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Denormalize the predictions
y_pred_train_denorm = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
y_pred_val_denorm = scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
y_pred_test_denorm = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

# Step 5: Defuzzification of predictions
def defuzzify_fuzzy_value(fuzzy_value, midpoints):
    rounded_index = int(round(fuzzy_value)) - 1
    rounded_index = min(max(0, rounded_index), len(midpoints) - 1)
    return midpoints[rounded_index]

# Apply defuzzification
y_pred_train_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_train_denorm]
y_pred_val_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_val_denorm]
y_pred_test_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_test_denorm]

# Step 6: Calculate MAE and MSE for train, validation, and test
train_mae = mean_absolute_error(data['#Passengers'].values[lag_length:lag_length + lenTrain], y_pred_train_with_fuzzy)
train_mse = mean_squared_error(data['#Passengers'].values[lag_length:lag_length + lenTrain], y_pred_train_with_fuzzy)
val_mae = mean_absolute_error(data['#Passengers'].values[lag_length + lenTrain:lag_length + lenTrain + lenVal], y_pred_val_with_fuzzy)
val_mse = mean_squared_error(data['#Passengers'].values[lag_length + lenTrain:lag_length + lenTrain + lenVal], y_pred_val_with_fuzzy)
test_mae = mean_absolute_error(data['#Passengers'].values[lag_length + lenTrain + lenVal:], y_pred_test_with_fuzzy)
test_mse = mean_squared_error(data['#Passengers'].values[lag_length + lenTrain + lenVal:], y_pred_test_with_fuzzy)

# Print MAE and MSE
print(f'Train MAE: {train_mae}, Train MSE: {train_mse}')
print(f'Validation MAE: {val_mae}, Validation MSE: {val_mse}')
print(f'Test MAE: {test_mae}, Test MSE: {test_mse}')


# Step 7: Plotting the results
plt.figure(figsize=(14, 8))

# Plot Original Data
plt.plot(data['#Passengers'].values, label='Original Series', color='blue')

# Plot Train Predictions
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_fuzzy, label='Train Predictions', color='orange')

# Plot Validation Predictions
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + lenVal), y_pred_val_with_fuzzy, label='Validation Predictions', color='green')

# Plot Test Predictions
plt.plot(np.arange(lag_length + lenTrain + lenVal, lag_length + lenTrain + lenVal + len(y_pred_test)), y_pred_test_with_fuzzy, label='Test Predictions', color='red')

# Labels and Legend
plt.xlabel('Time (Months)')
plt.ylabel('Number of Passengers')
plt.title('Passengers Time Series Prediction using Linear Regression with Fuzzy Defuzzification')
plt.legend()
plt.grid()
plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# Step 1: Create UOD and calculate intervals
min_val = data['#Passengers'].min()
max_val = data['#Passengers'].max()
uod_min = min_val - 0.1 * min_val
uod_max = max_val + 0.1 * max_val

# Partition UOD into intervals of length 10
interval_length = 10
intervals = np.arange(uod_min, uod_max + interval_length, interval_length)

# Compute midpoints of each interval for defuzzification
midpoints = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]

# Fuzzification function
def fuzzify_observation(value, intervals):
    for i in range(len(intervals) - 1):
        if intervals[i] <= value < intervals[i + 1]:
            return i + 1  # Interval index (1-based)
    return len(intervals) - 1  # If value equals the max

# Apply fuzzification
fuzzified_data = [fuzzify_observation(val, intervals) for val in data['#Passengers'].values]
data['Fuzzified'] = fuzzified_data

# Step 2: Normalize the fuzzified data
scaler = MinMaxScaler()
data['Normalized'] = scaler.fit_transform(data[['Fuzzified']])

# Step 3: Create sequences of patterns
def get_patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

lag_length = 12  # Using a lag length of 12 months
horizon = 1      # Predicting 1 step ahead
X, y = get_patterns(data[['Normalized']].values, lag_length, horizon)

# Split into train, validation, and test sets
lenTrain = int(round(len(X) * 0.7))
lenVal = int(round(len(X) * 0.15))  # Validation set size (15%)
X_train, X_val, X_test = X[:lenTrain], X[lenTrain:lenTrain + lenVal], X[lenTrain + lenVal:]
y_train, y_val, y_test = y[:lenTrain], y[lenTrain:lenTrain + lenVal], y[lenTrain + lenVal:]

# Reshape for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 4: Build the LSTM model with 64, 32, and 1 layers
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(lag_length, 1), return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model
model = create_lstm_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

# Predict future values
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Denormalize the predictions
y_pred_train_denorm = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
y_pred_val_denorm = scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
y_pred_test_denorm = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

# Step 5: Defuzzification of predictions
def defuzzify_fuzzy_value(fuzzy_value, midpoints):
    rounded_index = int(round(fuzzy_value)) - 1
    rounded_index = min(max(0, rounded_index), len(midpoints) - 1)
    return midpoints[rounded_index]

# Apply defuzzification
y_pred_train_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_train_denorm]
y_pred_val_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_val_denorm]
y_pred_test_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_test_denorm]

# Step 6: Calculate MAE and MSE for train, validation, and test
train_mae = mean_absolute_error(data['#Passengers'].values[lag_length:lag_length + lenTrain], y_pred_train_with_fuzzy)
train_mse = mean_squared_error(data['#Passengers'].values[lag_length:lag_length + lenTrain], y_pred_train_with_fuzzy)
val_mae = mean_absolute_error(data['#Passengers'].values[lag_length + lenTrain:lag_length + lenTrain + lenVal], y_pred_val_with_fuzzy)
val_mse = mean_squared_error(data['#Passengers'].values[lag_length + lenTrain:lag_length + lenTrain + lenVal], y_pred_val_with_fuzzy)
test_mae = mean_absolute_error(data['#Passengers'].values[lag_length + lenTrain + lenVal:], y_pred_test_with_fuzzy)
test_mse = mean_squared_error(data['#Passengers'].values[lag_length + lenTrain + lenVal:], y_pred_test_with_fuzzy)

# Print MAE and MSE
print(f'Train MAE: {train_mae}, Train MSE: {train_mse}')
print(f'Validation MAE: {val_mae}, Validation MSE: {val_mse}')
print(f'Test MAE: {test_mae}, Test MSE: {test_mse}')

# Step 7: Plotting the results
plt.figure(figsize=(14, 8))

# Plot Original Data
plt.plot(data['#Passengers'].values, label='Original Series', color='blue')

# Plot Train Predictions
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_fuzzy, label='Train Predictions (LSTM)', color='orange')

# Plot Validation Predictions
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + lenVal), y_pred_val_with_fuzzy, label='Validation Predictions (LSTM)', color='green')

# Plot Test Predictions
plt.plot(np.arange(lag_length + lenTrain + lenVal, lag_length + lenTrain + lenVal + len(y_pred_test)), y_pred_test_with_fuzzy, label='Test Predictions (LSTM)', color='red')

# Labels and Legend
plt.xlabel('Time (Months)')
plt.ylabel('Number of Passengers')
plt.title('Passengers Time Series Prediction using LSTM')
plt.legend()
plt.grid()
plt.show()


from tensorflow.keras.layers import GRU

# Step 1: Create UOD and calculate intervals
min_val = data['#Passengers'].min()
max_val = data['#Passengers'].max()
uod_min = min_val - 0.1 * min_val
uod_max = max_val + 0.1 * max_val

# Partition UOD into intervals of length 10
interval_length = 10
intervals = np.arange(uod_min, uod_max + interval_length, interval_length)

# Compute midpoints of each interval for defuzzification
midpoints = [(intervals[i] + intervals[i + 1]) / 2 for i in range(len(intervals) - 1)]

# Fuzzification function
def fuzzify_observation(value, intervals):
    for i in range(len(intervals) - 1):
        if intervals[i] <= value < intervals[i + 1]:
            return i + 1  # Interval index (1-based)
    return len(intervals) - 1  # If value equals the max

# Apply fuzzification
fuzzified_data = [fuzzify_observation(val, intervals) for val in data['#Passengers'].values]
data['Fuzzified'] = fuzzified_data

# Step 2: Normalize the fuzzified data
scaler = MinMaxScaler()
data['Normalized'] = scaler.fit_transform(data[['Fuzzified']])

# Step 3: Create sequences of patterns
def get_patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

lag_length = 12  # Using a lag length of 12 months
horizon = 1      # Predicting 1 step ahead
X, y = get_patterns(data[['Normalized']].values, lag_length, horizon)

# Split into train, validation, and test sets
lenTrain = int(round(len(X) * 0.7))
lenVal = int(round(len(X) * 0.15))  # Validation set size (15%)
X_train, X_val, X_test = X[:lenTrain], X[lenTrain:lenTrain + lenVal], X[lenTrain + lenVal:]
y_train, y_val, y_test = y[:lenTrain], y[lenTrain:lenTrain + lenVal], y[lenTrain + lenVal:]

# Reshape for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 4: Build the GRU model
def create_gru_model():
    model = Sequential()
    model.add(GRU(64, activation='relu', input_shape=(lag_length, 1)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the LSTM model for 10 epochs
model = create_gru_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

# Predict future values
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Denormalize the predictions
y_pred_train_denorm = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
y_pred_val_denorm = scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten()
y_pred_test_denorm = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

# Step 5: Defuzzification of predictions using midpoints
def defuzzify_fuzzy_value(fuzzy_value, midpoints):
    # Round the fuzzy index to the nearest integer
    fuzzy_index = int(round(fuzzy_value)) - 1
    # Ensure the index is within valid range
    fuzzy_index = min(max(0, fuzzy_index), len(midpoints) - 1)
    return midpoints[fuzzy_index]

# Apply defuzzification using midpoints
y_pred_train_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_train_denorm]
y_pred_val_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_val_denorm]
y_pred_test_with_fuzzy = [defuzzify_fuzzy_value(val, midpoints) for val in y_pred_test_denorm]

# Step 6: Calculate MAE and MSE for train, validation, and test
train_mae = mean_absolute_error(data['#Passengers'].values[lag_length:lag_length + lenTrain], y_pred_train_with_fuzzy)
train_mse = mean_squared_error(data['#Passengers'].values[lag_length:lag_length + lenTrain], y_pred_train_with_fuzzy)
val_mae = mean_absolute_error(data['#Passengers'].values[lag_length + lenTrain:lag_length + lenTrain + lenVal], y_pred_val_with_fuzzy)
val_mse = mean_squared_error(data['#Passengers'].values[lag_length + lenTrain:lag_length + lenTrain + lenVal], y_pred_val_with_fuzzy)
test_mae = mean_absolute_error(data['#Passengers'].values[lag_length + lenTrain + lenVal:], y_pred_test_with_fuzzy)
test_mse = mean_squared_error(data['#Passengers'].values[lag_length + lenTrain + lenVal:], y_pred_test_with_fuzzy)

# Print MAE and MSE
print(f'Train MAE: {train_mae}, Train MSE: {train_mse}')
print(f'Validation MAE: {val_mae}, Validation MSE: {val_mse}')
print(f'Test MAE: {test_mae}, Test MSE: {test_mse}')

# Step 7: Plotting the results
plt.figure(figsize=(14, 8))

# Plot Original Data
plt.plot(data['#Passengers'].values, label='Original Series', color='blue')

# Plot Train Predictions
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_fuzzy, label='Train Predictions (LSTM)', color='orange')

# Plot Validation Predictions
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + lenVal), y_pred_val_with_fuzzy, label='Validation Predictions (LSTM)', color='green')

# Plot Test Predictions
plt.plot(np.arange(lag_length + lenTrain + lenVal, lag_length + lenTrain + lenVal + len(y_pred_test)), y_pred_test_with_fuzzy, label='Test Predictions (LSTM)', color='red')

# Labels and Legend
plt.xlabel('Time (Months)')
plt.ylabel('Number of Passengers')
plt.title('Passengers Time Series Prediction using LSTM with Fuzzy Defuzzification')
plt.legend()
plt.grid()
plt.show()
