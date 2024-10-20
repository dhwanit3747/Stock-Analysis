# Install required libraries
#pip install pandas numpy scikit-learn matplotlib yfinance

# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Data Collection
# Download historical data for a specific stock
ticker = 'XOM'  # Apple Inc.
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Step 2: Data Preprocessing
# Selecting relevant features
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Check for missing values
data.isnull().sum()

# Handle missing values by forward filling
data = data.fillna(method='ffill')

# Step 3: Feature Engineering
# Creating features and labels
data['Next_Close'] = data['Close'].shift(-1)
data = data.dropna()

# Features and labels
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]                        
y = data['Next_Close']

# Step 4: Train-Test Split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Data Normalization
# Normalize the features for better performance of the neural network
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Model Selection and Training
# Define the model
model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500)

# Train the model
model.fit(X_train_scaled, y_train)

# Step 7: Model Evaluation
# Predicting the closing prices
y_pred = model.predict(X_test_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the predicted vs actual prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Prices')
plt.plot(y_test.index, y_pred, label='Predicted Prices')
plt.legend()
plt.show()

# Step 8: Deployment
def predict_next_close(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    return model.predict(new_data_scaled)

# Example usage
open_price = 170.00  # Example open price
high_price = 172.50  # Example high price
low_price = 169.00  # Example low price
close_price = 171.00  # Example close price
volume = 1000000  # Example volume

new_data = np.array([[open_price, high_price, low_price, close_price, volume]])
predicted_close = predict_next_close(model, scaler, new_data)
print(f'Predicted Close Price: {predicted_close[0]}')
