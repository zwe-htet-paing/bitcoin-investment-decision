## Bitcoin Investment Decision

DatatalksClub ML Zoomcamp Midterm Project

### Objectives of Midterm Project:

- Problem description
- Find a dataset
- EDA
- Exporting notebook to script
- Deploy model as a web service
- Containerization


### Problem Description: Bitcoin Investment Decision

I am considering investing in Bitcoin (BTC) and need to make an informed decision based on historical price data. I have access to daily price information, including open, close, low, and high values. However, I am uncertain about the profitability and risks associated with investing in BTC. I need to develop a strategy to determine whether it's a suitable time to buy BTC or if I should wait for more favorable market conditions.

### Dataset

For this project I used the Bitcoin USD (BTC-USD) dataset which is a collection of historical price and trading data for Bitcoin, the most well-known and widely used cryptocurrency, denominated in US dollars (USD). This dataset provides a detailed record of Bitcoin's market performance over a specific period, allowing analysts, traders, and researchers to study its price movements, trading volume, and other related metrics.

Typically, a BTC-USD dataset includes various columns or features, each providing specific information about Bitcoin's market behavior. Here are some common features found in a BTC-USD dataset and their explanations:

1. **Date:** The date and time of the recorded data point.
2. **Open Price:** The price of Bitcoin at the beginning of a specific time period (e.g., an hour, day, or minute).
3. **High Price:** The highest price of Bitcoin observed during the time period.
4. **Low Price:** The lowest price of Bitcoin observed during the time period.
5. **Close Price:** The price of Bitcoin at the end of the time period.
6. **Volume:** The total trading volume of Bitcoin (measured in BTC) during the time period.

Dataset Link : https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD&guccounter=1

### Data Analysis

Constructing a dataset that contains the predictors which will be used to make the predictions, and the output variable.

The current Data of the bicoin consists of date, open, high, low, close and volume. Using this data we calculate the following technical indicators:

- Moving Average : A moving average provides an indication of the trend of the price movement by cut down the amount of "noise" on a price chart.
- Stochastic Oscillator %K and %D : A stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. %K and %D are slow and fast indicators.
- Relative Strength Index(RSI) :It is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.
- Rate Of Change(ROC): It is a momentum oscillator, which measures the percentage change between the current price and the n period past price.
  Momentum (MOM) : It is the rate of acceleration of a security's price or volume – that is, the speed at which the price is changing.


### Prepare data for classification

Calculates short and long Simple Moving Averages for the 'close' price and generates trading signals based on the relationship between these averages. When the short-term average is above the long-term average, it implies a bullish trend, and a buy signal is generated; otherwise, a sell signal is generated, indicating a potential bearish trend.

Attach a label to each movement and use for classification task:

- 1 if the signal is that short term price will go up as compared to the long term.
- 0 if the signal is that short term price will go down as compared to the long term.


### Dependency and environment management

- `pip install pipenv`
- `pipenv install`

### Run Locally

- Run web server using `python predict.py`
- Open a new terminal and run  `python test_predict.py`

### Run on Docker

- `docker build -t midterm_project .`
- `docker run -it --rm -p 9696:9696 midterm_project`
- Open a new terminal and run `python test_predict.py`

### Requirements

- Python:3.9
- Scikit-learn:1.3.2
- xgboost:2.0.0
- flask
- numpy
- pandas
- gunicorn
