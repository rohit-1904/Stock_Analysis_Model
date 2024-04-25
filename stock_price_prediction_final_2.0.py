import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import streamlit as st

# Function to download data from Yahoo Finance
def download_data(stock, start, end):
    data = yf.download(stock, start, end)
    return data

# Function to prepare data for training
def prepare_data(data):
    df = data.reset_index()
    X = df[['Open','High','Low','Close']]
    y = df['Adj Close']
    return X.values, y.values

# Function to train the model
def train_model(X_train, y_train):
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    return regression_model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_predict = model.predict(X_test)
    regression_model_mse = mean_squared_error(y_predict, y_test)
    return regression_model_mse

# Function to predict next day's price
def predict_next_day_price(model, latest_data):
    return model.predict(latest_data)

# Main function
def main():
    st.title('Stock Price Prediction')
    st.sidebar.title('Options')

    stock = st.sidebar.text_input("Enter stock symbol (e.g., SBIN.NS)", 'SBIN.NS')
    start = st.sidebar.text_input("Enter start date (YYYY-MM-DD)", '2016-01-01')
    end = st.sidebar.text_input("Enter end date (YYYY-MM-DD)", '2018-01-01')

    data = download_data(stock, start, end)
    st.subheader('Data')
    st.write(data)

    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    regression_model = train_model(X_train, y_train)
    mse = evaluate_model(regression_model, X_test, y_test)
    st.write("Mean Squared Error:", mse)

    latest_open = st.sidebar.number_input("Latest Open Price", value=167.81)
    latest_high = st.sidebar.number_input("Latest High Price", value=171.75)
    latest_low = st.sidebar.number_input("Latest Low Price", value=165.19)
    latest_close = st.sidebar.number_input("Latest Close Price", value=310.89)

    latest_data = np.array([[latest_open, latest_high, latest_low, latest_close]])
    next_day_price = predict_next_day_price(regression_model, latest_data)
    st.write("Predicted next day price:", next_day_price[0])

if __name__ == "__main__":
    main()
