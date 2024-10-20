README: Stock Price Prediction using MLP Regressor

Overview

This project implements a Multi-Layer Perceptron (MLP) Regressor to predict the next day's closing price of a stock based on historical data. The model is trained using Exxon Mobil Corp. (XOM) data, collected from Yahoo Finance. This project demonstrates how machine learning can be applied to financial forecasting through data preprocessing, feature engineering, and model evaluation.


---

Prerequisites

Before running the code, ensure you have the following Python libraries installed:

pip install pandas numpy scikit-learn matplotlib yfinance


---

Project Structure

1. Data Collection:

Downloads historical stock data using yfinance.



2. Data Preprocessing:

Selects relevant features and handles missing values.



3. Feature Engineering:

Creates input features and labels.



4. Train-Test Split:

Divides the data into training (80%) and testing (20%) sets.



5. Data Normalization:

Scales data using Min-Max normalization.



6. Model Training:

Trains an MLP Regressor with two hidden layers.



7. Model Evaluation:

Evaluates the model using mean squared error (MSE) and plots predictions.



8. Deployment:

Provides a function to predict future closing prices with new data inputs.





---

How to Use

1. Download the Stock Data:

Modify the ticker variable if you wish to use data from a different stock.


ticker = 'XOM'  # Modify this to any other stock ticker if needed.


2. Run the Script:

Execute the code from your Python environment. The model will train and display the MSE and a plot comparing actual vs. predicted prices.



3. Predict New Prices:

Use the predict_next_close() function to predict future closing prices with custom input data.


open_price = 170.00  
high_price = 172.50  
low_price = 169.00  
close_price = 171.00  
volume = 1000000  

new_data = np.array([[open_price, high_price, low_price, close_price, volume]])
predicted_close = predict_next_close(model, scaler, new_data)
print(f'Predicted Close Price: {predicted_close[0]}')




---

File Breakdown

Script: Contains the full pipeline from data collection to deployment.

Plot: A graph will be generated comparing actual vs predicted prices during the test period.

Prediction Function: Allows forecasting future prices with new inputs.



---

Example Output

1. Mean Squared Error (MSE):

Mean Squared Error: 3.25  # Example Output


2. Prediction Example:

Predicted Close Price: 171.23


3. Plot Example: The plot shows how closely the predicted values match the actual prices over the test period.




---

Customization

Change Stock Ticker: Modify the ticker variable to collect data for a different stock.

Tuning Model Parameters: Experiment with different hidden layer sizes, activation functions, and optimizers for better performance.

Add Features: Include technical indicators like moving averages to enhance prediction accuracy.



---

Dependencies

Python 3.x

pandas - Data manipulation

numpy - Numerical operations

scikit-learn - Machine learning algorithms

yfinance - Stock data collection

matplotlib - Visualization



---

Conclusion

This project provides a basic template for building stock price prediction models using an MLP Regressor. You can extend this by experimenting with advanced models (e.g., LSTM or RNN) or integrating real-time data for live predictions.


---

Contact

Feel free to contribute or ask questions by submitting issues or pull requests.

