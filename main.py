# Import required packages
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import os
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
warnings.filterwarnings("ignore")

# split a univariate time series into patterns
def get_Patterns(TSeries, n_inputs,h):
    X,y,z = pd.DataFrame(np.zeros((len(TSeries)-n_inputs-h+1,n_inputs))), pd.DataFrame(), pd.DataFrame()
    for i in range(len(TSeries)):
        # find the end of this pattern
        end_ix = i + n_inputs + h - 1
        # check if we are beyond the time series
        if end_ix > len(TSeries)-1:
            break
        # gather input and output parts of the pattern
        for j in range(n_inputs):
            X.loc[i,j]=TSeries.iloc[i+j,0]
        i=i+n_inputs
        #y=y.append(TSeries.iloc[end_ix], ignore_index = True)
        y=pd.concat([y, TSeries.iloc[end_ix]], ignore_index=True)
        sinX=pd.DataFrame(np.sin(X))
        cosX=pd.DataFrame(np.cos(X))
        squareX=pd.DataFrame(np.power(X,2))
        #X1=pd.concat([X,sinX,cosX,squareX], axis=1)
        X1=X
    return pd.DataFrame(X),pd.DataFrame(y)

# originalData should be a Column Vectored DataFrame
def minmaxNorm(originalData, lenTrainValidation):
    max2norm=max(originalData.iloc[0:lenTrainValidation,0])
    min2norm=min(originalData.iloc[0:lenTrainValidation,0])
    lenOriginal=len(originalData)
    normalizedData=np.zeros(lenOriginal)   
    normalizedData = []
    for i in range (lenOriginal):
        normalizedData.append((originalData.iloc[i]-min2norm)/(max2norm-min2norm))    
    return pd.DataFrame(normalizedData)
# originalData and forecastedData should be Column Vectored DataFrames
def minmaxDeNorm( originalData, forecastedData, lenTrainValidation):
    # Maximum Value
    max2norm=max(originalData.iloc[0:lenTrainValidation,0])
    # Minimum Value
    min2norm=min(originalData.iloc[0:lenTrainValidation,0])
    lenOriginal=len(originalData)
    denormalizedData=[]   
    #De-Normalize using Min-Max Normalization
    for i in range (lenOriginal):
        denormalizedData.append((forecastedData.iloc[i]*(max2norm-min2norm))+min2norm)  
    return pd.DataFrame(denormalizedData)
# Timeseries_Data and forecasted_value should be Column Vectored DataFrames
def findRMSE( Timeseries_Data, forecasted_value,lenTrainValidation):
    l=Timeseries_Data.shape[0]
    lenTest=l-lenTrainValidation
    # RMSE on Train & Validation Set
    trainRMSE=0;
    for i in range (lenTrainValidation):
        trainRMSE=trainRMSE+np.power((forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0]),2) 
    trainRMSE=np.sqrt(trainRMSE/lenTrainValidation)
    # RMSE on Test Set
    testRMSE=0;
    for i in range (lenTrainValidation,l,1):
        testRMSE=testRMSE+np.power((forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0]),2)
    testRMSE=np.sqrt(testRMSE/lenTest)
    return trainRMSE, testRMSE 

# Timeseries_Data and forecasted_value should be Column Vectored DataFrames
def findMAE(Timeseries_Data, forecasted_value,lenTrainValidation):
    l=Timeseries_Data.shape[0]
    lenTest=l-lenTrainValidation
    # MAE on Train & Validation Set
    trainMAE=0;
    for i in range (lenTrainValidation):
        trainMAE=trainMAE+np.abs(forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0]) 
    trainMAE=(trainMAE/(lenTrainValidation));
    # MAE on Test Set
    testMAE=0;
    for i in range (lenTrainValidation,l,1):
        testMAE=testMAE+np.abs(forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0])
    testMAE=(testMAE/lenTest);
    return trainMAE, testMAE

def Find_Fitness(x,y,lenValid,lenTest,model):
    NOP=y.shape[0]
    lenTrain=NOP-lenValid-lenTest
    xTrain=x.iloc[0:lenTrain,:]
    xValid=x.iloc[lenTrain:(lenTrain+lenValid),:]
    xTest=x.iloc[(lenTrain+lenValid):NOP,:]
    yTrain=y.iloc[0:lenTrain,0]
    yValid=y.iloc[lenTrain:(lenTrain+lenValid),0]
    yTest=y.iloc[(lenTrain+lenValid):NOP,0]
    model.fit(xTrain, yTrain)
    yhatNorm=model.predict(x).flatten().reshape(x.shape[0],1)
    return pd.DataFrame(yhatNorm)

#Read the Time Series Dataset
Timeseries_Data=pd.read_csv('Lynx.csv',header=None)
Timeseries_Data.describe()

plt.title("Autocorrelation Plot") 
# Providing x-axis name.
plt.xlabel("Lags") 
# Plotting the Autocorrelation plot.
plt.acorr(np.array(Timeseries_Data.iloc[:,0], dtype=float), maxlags = 20) 
# Displaying the plot.
print("The Autocorrelation plot for the data is:")
plt.grid(True)
plt.show() 

#4. Rug plot — sns.rugplot()
sns.rugplot(data=Timeseries_Data, height=.03, color='darkblue')
sns.histplot(data=Timeseries_Data, kde=True)

LagLength=10
h=1
lt=Timeseries_Data.shape[0]
lenTrain=int(round(lt*0.7))
lenValidation=int(round(lt*0.15))
lenTest=int(lt-lenTrain-lenValidation)
# NORMALIZE THE DATA
print(Timeseries_Data.shape)
normalizedData=minmaxNorm(Timeseries_Data,lenTrain+lenValidation);
# Transform the Time Series into Patterns Using Sliding Window
print(normalizedData.shape)
X, y = get_Patterns(normalizedData, LagLength, h)
model=LinearRegression()
name='LinearRegression'
file1='./'+str(name)+"_Accuracy.xlsx"
file2='./'+str(name)+"_Forecasts.xlsx"
Forecasts=pd.DataFrame()
Accuracy=pd.DataFrame()

ynorm1=Find_Fitness(X,y,lenValidation,lenTest,model)
ynorm=pd.DataFrame(normalizedData.iloc[0:(LagLength+h-1),0])
ynorm=pd.concat([ynorm, ynorm1], ignore_index=True)
# print(ynorm.shape)
yhat=minmaxDeNorm(Timeseries_Data, ynorm, lenTrain+lenValidation)

Accuracy.loc[0,0],Accuracy.loc[0,1]=findRMSE( Timeseries_Data,yhat,lenTrain+lenValidation)
Accuracy.loc[0,2],Accuracy.loc[0,3]=findMAE( Timeseries_Data,yhat,lenTrain+lenValidation)
Forecasts=pd.concat([Forecasts, yhat.T], ignore_index=True)
Accuracy.to_excel(file1,sheet_name='Accuracy',index=False)
Forecasts.to_excel(file2,sheet_name='Forecasts',index=False)
print(Accuracy)

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

LagLength = 10
h = 1
lt = Timeseries_Data.shape[0]
lenTrain = int(round(lt * 0.7))
lenValidation = int(round(lt * 0.15))
lenTest = int(lt - lenTrain - lenValidation)

# Normalize the data
normalizedData = minmaxNorm(Timeseries_Data, lenTrain + lenValidation)

# Transform the Time Series into Patterns Using Sliding Window
X, y = get_Patterns(normalizedData, LagLength, h)

# Define the MLP model function
def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create the MLP model once outside the loop
input_shape = X.shape[1]
model = create_mlp_model(input_shape)

# Run 10 independent simulations
name = 'MLP'
file1 = './' + str(name) + "_Accuracy.xlsx"
file2 = './' + str(name) + "_Forecasts.xlsx"
Forecasts = pd.DataFrame()
Accuracy = pd.DataFrame()
train_rmse_list, test_rmse_list, train_mae_list, test_mae_list = [], [], [], []

for i in range(10):
    # Reinitialize weights for the model
    model.set_weights([np.random.normal(size=w.shape) for w in model.get_weights()])
    
    # Train the model
    model.fit(X, y, epochs=100, batch_size=16, verbose=0)

    # Predict using the trained model
    ynorm1 = Find_Fitness(X, y, lenValidation, lenTest, model)
    
    # Concatenate initial part and the predictions
    ynorm = pd.DataFrame(normalizedData.iloc[0:(LagLength + h - 1), 0])
    ynorm = pd.concat([ynorm, ynorm1], ignore_index=True)
    
    # Denormalize the predictions
    yhat = minmaxDeNorm(Timeseries_Data, ynorm, lenTrain + lenValidation)
    
    # Calculate and store accuracy metrics
    rmse_train, rmse_test = findRMSE(Timeseries_Data, yhat, lenTrain + lenValidation)
    mae_train, mae_test = findMAE(Timeseries_Data, yhat, lenTrain + lenValidation)
    
    Accuracy.loc[i, 0] = rmse_train
    Accuracy.loc[i, 1] = rmse_test
    Accuracy.loc[i, 2] = mae_train
    Accuracy.loc[i, 3] = mae_test
    
    Forecasts = pd.concat([Forecasts, yhat.T], ignore_index=True)
    
    # Store results for mean calculation
    train_rmse_list.append(rmse_train)
    test_rmse_list.append(rmse_test)
    train_mae_list.append(mae_train)
    test_mae_list.append(mae_test)

# Save results to Excel
Accuracy.to_excel(file1, sheet_name='Accuracy', index=False)
Forecasts.to_excel(file2, sheet_name='Forecasts', index=False)

# Print the mean RMSE and MAE
print(f'Mean Train RMSE: {np.mean(train_rmse_list)}, Mean Test RMSE: {np.mean(test_rmse_list)}')
print(f'Mean Train MAE: {np.mean(train_mae_list)}, Mean Test MAE: {np.mean(test_mae_list)}')
print(Accuracy)


import numpy as np
import pandas as pd

# Assume 'Timeseries_Data' is a DataFrame with the original time series
cyclicity_length = 10

# Calculate the cyclic average
def calculate_cyclic_average(series, cyclicity_length):
    cyclic_averages = []
    for i in range(cyclicity_length):
        # Get all elements in the same cycle
        cycle_values = series[i::cyclicity_length]
        # Calculate the mean for the current cycle position
        cyclic_averages.append(np.mean(cycle_values))
    return np.array(cyclic_averages)

# Apply cyclic average treatment
def subtract_cyclic_average(series, cyclic_averages, cyclicity_length):
    adjusted_series = []
    for i in range(len(series)):
        cycle_position = i % cyclicity_length
        adjusted_value = series[i] - cyclic_averages[cycle_position]
        adjusted_series.append(adjusted_value)
    return np.array(adjusted_series)

# Original series (assuming it's in the first column of 'Timeseries_Data')
original_series = normalizedData.iloc[:, 0].values

# Calculate cyclic averages
cyclic_averages = calculate_cyclic_average(original_series, cyclicity_length)

# Subtract the cyclic average from the original series
adjusted_series = subtract_cyclic_average(original_series, cyclic_averages, cyclicity_length)

# Convert the adjusted series into a DataFrame for further processing
adjusted_series_df = pd.DataFrame(adjusted_series, columns=['Adjusted'])

plt.title("Autocorrelation Plot") 
# Providing x-axis name.
plt.xlabel("Lags") 
# Plotting the Autocorrelation plot.
plt.acorr(np.array(adjusted_series_df.iloc[:,0], dtype=float), maxlags = 20) 
# Displaying the plot.
print("The Autocorrelation plot for the data is:")
plt.grid(True)
plt.show() 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the Lynx data
df = pd.read_csv('Lynx.csv')  # Replace with the path to the Lynx dataset
df.columns = ['Lynx']

# Define the cyclicity length
cyclicity_length = 10

# Calculate cyclic averages based on cyclicity
cyclic_means = df.groupby(df.index % cyclicity_length).mean()

# Repeat cyclic_means to match the length of the original data
cyclic_means_repeated = np.tile(cyclic_means.values.flatten(), len(df) // cyclicity_length + 1)[:len(df)]

# Subtract cyclic average from the original series (detrend the data)
detrended_data = df['Lynx'].values - cyclic_means_repeated

# Normalize the detrended data
scaler = MinMaxScaler()
normalizedData = scaler.fit_transform(detrended_data.reshape(-1, 1))
normalizedData = pd.DataFrame(normalizedData, columns=['Normalized'])

# Transform the Time Series into Patterns Using Sliding Window
def get_Patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

# Set parameters for time series transformation
lag_length = 10
horizon = 1

# Get patterns
X, y = get_Patterns(normalizedData.values, lag_length, horizon)

# Split into train and test sets
lenTrain = int(round(len(X) * 0.7))
X_train, X_test = X[:lenTrain], X[lenTrain:]
y_train, y_test = y[:lenTrain], y[lenTrain:]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future values
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Denormalize the predictions
y_pred_train_denorm = scaler.inverse_transform(y_pred_train)
y_pred_test_denorm = scaler.inverse_transform(y_pred_test)

# Correct the slicing here to match the length of y_pred_test
y_pred_train_with_seasonal = y_pred_train_denorm.flatten() + cyclic_means_repeated[lag_length:lag_length + lenTrain]
y_pred_test_with_seasonal = y_pred_test_denorm.flatten() + cyclic_means_repeated[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)]

# Calculate MAE and MSE for train and test
train_mae = mean_absolute_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
train_mse = mean_squared_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
test_mae = mean_absolute_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)
test_mse = mean_squared_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)

# Print MAE and MSE
print(f'Train MAE: {train_mae}, Train MSE: {train_mse}')
print(f'Test MAE: {test_mae}, Test MSE: {test_mse}')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df.values, label='Original Series')
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_seasonal, label='Train Predictions with Seasonal Trend')
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + len(y_pred_test)), y_pred_test_with_seasonal, label='Test Predictions with Seasonal Trend')
plt.legend()
plt.title('Lynx Time Series Prediction using Linear Regression with Seasonal Trend')
plt.show()


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the Lynx data
df = pd.read_csv('Lynx.csv')  # Replace with the path to the Lynx dataset
df.columns = ['Lynx']

# Define the cyclicity length
cyclicity_length = 10

# Calculate cyclic averages based on cyclicity
cyclic_means = df.groupby(df.index % cyclicity_length).mean()

# Repeat cyclic_means to match the length of the original data
cyclic_means_repeated = np.tile(cyclic_means.values.flatten(), len(df) // cyclicity_length + 1)[:len(df)]

# Subtract cyclic average from the original series (detrend the data)
detrended_data = df['Lynx'].values - cyclic_means_repeated

# Normalize the detrended data
scaler = MinMaxScaler()
normalizedData = scaler.fit_transform(detrended_data.reshape(-1, 1))
normalizedData = pd.DataFrame(normalizedData, columns=['Normalized'])

# Transform the Time Series into Patterns Using Sliding Window
def get_Patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

# Set parameters for time series transformation
lag_length = 10
horizon = 1

# Get patterns
X, y = get_Patterns(normalizedData.values, lag_length, horizon)

# Split into train and test sets
lenTrain = int(round(len(X) * 0.7))
X_train, X_test = X[:lenTrain], X[lenTrain:]
y_train, y_test = y[:lenTrain], y[lenTrain:]

# Lists to store MAE and MSE for 10 runs
train_mae_list, train_mse_list = [], []
test_mae_list, test_mse_list = [], []

# Run the model 10 times
n_runs = 10
for i in range(n_runs):
    # Train the MLP model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=None)
    mlp_model.fit(X_train, y_train.ravel())

    # Predict future values
    y_pred_train = mlp_model.predict(X_train)
    y_pred_test = mlp_model.predict(X_test)

    # Denormalize the predictions
    y_pred_train_denorm = scaler.inverse_transform(y_pred_train.reshape(-1, 1))
    y_pred_test_denorm = scaler.inverse_transform(y_pred_test.reshape(-1, 1))

    # Add the seasonal trend back to the denormalized predictions
    y_pred_train_with_seasonal = y_pred_train_denorm.flatten() + cyclic_means_repeated[lag_length:lag_length + lenTrain]
    y_pred_test_with_seasonal = y_pred_test_denorm.flatten() + cyclic_means_repeated[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)]

    # Calculate MAE and MSE for train and test
    train_mae = mean_absolute_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
    train_mse = mean_squared_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
    test_mae = mean_absolute_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)
    test_mse = mean_squared_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)

    # Store the metrics for each run
    train_mae_list.append(train_mae)
    train_mse_list.append(train_mse)
    test_mae_list.append(test_mae)
    test_mse_list.append(test_mse)

    print(f'Run {i+1}: Train MAE: {train_mae}, Train MSE: {train_mse}, Test MAE: {test_mae}, Test MSE: {test_mse}')

# Calculate mean MAE and MSE over 10 runs
mean_train_mae = np.mean(train_mae_list)
mean_train_mse = np.mean(train_mse_list)
mean_test_mae = np.mean(test_mae_list)
mean_test_mse = np.mean(test_mse_list)

print(f'\nMean Train MAE over {n_runs} runs: {mean_train_mae}')
print(f'Mean Train MSE over {n_runs} runs: {mean_train_mse}')
print(f'Mean Test MAE over {n_runs} runs: {mean_test_mae}')
print(f'Mean Test MSE over {n_runs} runs: {mean_test_mse}')

# Plotting the results for the last run
plt.figure(figsize=(12, 6))
plt.plot(df.values, label='Original Series')
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_seasonal, label='Train Predictions with Seasonal Trend (Last Run)')
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + len(y_pred_test)), y_pred_test_with_seasonal, label='Test Predictions with Seasonal Trend (Last Run)')
plt.legend()
plt.title('Lynx Time Series Prediction using MLP with Seasonal Trend')
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the Lynx data
df = pd.read_csv('Lynx.csv')  # Replace with the path to the Lynx dataset
df.columns = ['Lynx']

# Define the cyclicity length
cyclicity_length = 10

# Calculate differencing to remove cyclic effects
df['Lynx_diff'] = df['Lynx'] - df['Lynx'].shift(cyclicity_length)
df.dropna(inplace=True)  # Drop rows with NaN values created by differencing

# Normalize the differenced data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['Lynx_diff']])
normalized_data = pd.DataFrame(normalized_data, columns=['Normalized'])

# Transform the Time Series into Patterns Using Sliding Window
def get_Patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

# Set parameters for time series transformation
lag_length = 10
horizon = 1

# Get patterns
X, y = get_Patterns(normalized_data.values, lag_length, horizon)

# Split into train and test sets
lenTrain = int(round(len(X) * 0.7))
X_train, X_test = X[:lenTrain], X[lenTrain:]
y_train, y_test = y[:lenTrain], y[lenTrain:]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future values
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Denormalize the predictions
y_pred_train_denorm = scaler.inverse_transform(y_pred_train.reshape(-1, 1))
y_pred_test_denorm = scaler.inverse_transform(y_pred_test.reshape(-1, 1))

# Reconstruct the predictions by adding back the differenced part
y_pred_train_with_seasonal = y_pred_train_denorm.flatten() + df['Lynx'].shift(cyclicity_length).dropna().values[:lenTrain]
y_pred_test_with_seasonal = y_pred_test_denorm.flatten() + df['Lynx'].shift(cyclicity_length).dropna().values[lenTrain:lenTrain + len(y_pred_test)]

# Calculate MAE and MSE for train and test
train_mae = mean_absolute_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
train_mse = mean_squared_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
test_mae = mean_absolute_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)
test_mse = mean_squared_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)

# Print MAE and MSE
print(f'Train MAE: {train_mae}, Train MSE: {train_mse}')
print(f'Test MAE: {test_mae}, Test MSE: {test_mse}')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df['Lynx'].values, label='Original Series')
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_seasonal, label='Train Predictions with Seasonal Trend')
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + len(y_pred_test)), y_pred_test_with_seasonal, label='Test Predictions with Seasonal Trend')
plt.legend()
plt.title('Lynx Time Series Prediction using Linear Regression with Seasonal Trend Added Back')
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the Lynx data
df = pd.read_csv('Lynx.csv')  # Replace with the path to the Lynx dataset
df.columns = ['Lynx']

# Define the cyclicity length
cyclicity_length = 10

# Calculate differencing to remove cyclic effects
df['Lynx_diff'] = df['Lynx'] - df['Lynx'].shift(cyclicity_length)
df.dropna(inplace=True)  # Drop rows with NaN values created by differencing

# Normalize the differenced data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['Lynx_diff']])
normalized_data = pd.DataFrame(normalized_data, columns=['Normalized'])

# Transform the Time Series into Patterns Using Sliding Window
def get_Patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

# Set parameters for time series transformation
lag_length = 10
horizon = 1

# Get patterns
X, y = get_Patterns(normalized_data.values, lag_length, horizon)

# Split into train and test sets
lenTrain = int(round(len(X) * 0.7))
X_train, X_test = X[:lenTrain], X[lenTrain:]
y_train, y_test = y[:lenTrain], y[lenTrain:]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future values
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Denormalize the predictions
y_pred_train_denorm = scaler.inverse_transform(y_pred_train.reshape(-1, 1))
y_pred_test_denorm = scaler.inverse_transform(y_pred_test.reshape(-1, 1))

# Reconstruct the predictions by adding back the differenced part
y_pred_train_with_seasonal = y_pred_train_denorm.flatten() + df['Lynx'].shift(cyclicity_length).dropna().values[:lenTrain]
y_pred_test_with_seasonal = y_pred_test_denorm.flatten() + df['Lynx'].shift(cyclicity_length).dropna().values[lenTrain:lenTrain + len(y_pred_test)]

# Calculate MAE and MSE for train and test
train_mae = mean_absolute_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
train_mse = mean_squared_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
test_mae = mean_absolute_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)
test_mse = mean_squared_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)

# Print MAE and MSE
print(f'Train MAE: {train_mae}, Train MSE: {train_mse}')
print(f'Test MAE: {test_mae}, Test MSE: {test_mse}')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df['Lynx'].values, label='Original Series')
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_seasonal, label='Train Predictions with Seasonal Trend')
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + len(y_pred_test)), y_pred_test_with_seasonal, label='Test Predictions with Seasonal Trend')
plt.legend()
plt.title('Lynx Time Series Prediction using Linear Regression with Seasonal Trend Added Back')
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the Lynx data
df = pd.read_csv('Lynx.csv')  # Replace with the path to the Lynx dataset
df.columns = ['Lynx']

# Define the cyclicity length
cyclicity_length = 10

# Apply differencing to remove cyclic effects
df['Lynx_diff'] = df['Lynx'] - df['Lynx'].shift(cyclicity_length)
df.dropna(inplace=True)  # Drop rows with NaN values created by differencing

# Normalize the differenced data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['Lynx_diff']])
normalized_data = pd.DataFrame(normalized_data, columns=['Normalized'])

# Transform the Time Series into Patterns Using Sliding Window
def get_Patterns(data, lag_length, horizon):
    X, y = [], []
    for i in range(len(data) - lag_length - horizon + 1):
        X.append(data[i:(i + lag_length), 0])
        y.append(data[i + lag_length:i + lag_length + horizon, 0])
    return np.array(X), np.array(y)

# Set parameters for time series transformation
lag_length = 10
horizon = 1

# Get patterns
X, y = get_Patterns(normalized_data.values, lag_length, horizon)

# Split into train and test sets
lenTrain = int(round(len(X) * 0.7))
X_train, X_test = X[:lenTrain], X[lenTrain:]
y_train, y_test = y[:lenTrain], y[lenTrain:]

# Lists to store MAE and MSE for 10 runs
train_mae_list, train_mse_list = [], []
test_mae_list, test_mse_list = [], []

# Run the model 10 times
n_runs = 10
for i in range(n_runs):
    # Train the MLP model
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=None)
    mlp_model.fit(X_train, y_train.ravel())

    # Predict future values
    y_pred_train = mlp_model.predict(X_train)
    y_pred_test = mlp_model.predict(X_test)

    # Denormalize the predictions
    y_pred_train_denorm = scaler.inverse_transform(y_pred_train.reshape(-1, 1))
    y_pred_test_denorm = scaler.inverse_transform(y_pred_test.reshape(-1, 1))

    # Add the seasonal trend back to the denormalized predictions
    y_pred_train_with_seasonal = y_pred_train_denorm.flatten() + df['Lynx'].shift(cyclicity_length).dropna().values[:lenTrain]
    y_pred_test_with_seasonal = y_pred_test_denorm.flatten() + df['Lynx'].shift(cyclicity_length).dropna().values[lenTrain:lenTrain + len(y_pred_test)]

    # Calculate MAE and MSE for train and test
    train_mae = mean_absolute_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
    train_mse = mean_squared_error(df['Lynx'].values[lag_length:lag_length + lenTrain], y_pred_train_with_seasonal)
    test_mae = mean_absolute_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)
    test_mse = mean_squared_error(df['Lynx'].values[lag_length + lenTrain:lag_length + lenTrain + len(y_pred_test)], y_pred_test_with_seasonal)

    # Store the metrics for each run
    train_mae_list.append(train_mae)
    train_mse_list.append(train_mse)
    test_mae_list.append(test_mae)
    test_mse_list.append(test_mse)

    print(f'Run {i+1}: Train MAE: {train_mae}, Train MSE: {train_mse}, Test MAE: {test_mae}, Test MSE: {test_mse}')

# Calculate mean MAE and MSE over 10 runs
mean_train_mae = np.mean(train_mae_list)
mean_train_mse = np.mean(train_mse_list)
mean_test_mae = np.mean(test_mae_list)
mean_test_mse = np.mean(test_mse_list)

print(f'\nMean Train MAE over {n_runs} runs: {mean_train_mae}')
print(f'Mean Train MSE over {n_runs} runs: {mean_train_mse}')
print(f'Mean Test MAE over {n_runs} runs: {mean_test_mae}')
print(f'Mean Test MSE over {n_runs} runs: {mean_test_mse}')

# Plotting the results for the last run
plt.figure(figsize=(12, 6))
plt.plot(df['Lynx'].values, label='Original Series')
plt.plot(np.arange(lag_length, lag_length + lenTrain), y_pred_train_with_seasonal, label='Train Predictions with Seasonal Trend (Last Run)')
plt.plot(np.arange(lag_length + lenTrain, lag_length + lenTrain + len(y_pred_test)), y_pred_test_with_seasonal, label='Test Predictions with Seasonal Trend (Last Run)')
plt.legend()
plt.title('Lynx Time Series Prediction using MLP with Seasonal Trend')
plt.show()
