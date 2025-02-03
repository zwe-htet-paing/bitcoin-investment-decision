# Bitcoin Investment Decision

## Objectives
- Problem Description
- Find a Dataset
- Exploratory Data Analysis (EDA)
- Export Notebook to Script
- Deploy Model as a Web Service
- Containerization

## Problem Description
Bitcoin (BTC) is a highly volatile asset, making investment decisions challenging. This project aims to analyze historical BTC price data to develop a strategy for determining whether to buy BTC or wait for more favorable market conditions. The analysis includes studying trends, risk factors, and key technical indicators to inform investment decisions.

## Dataset
The dataset used is the **Bitcoin USD (BTC-USD)** dataset, which contains historical price and trading data for BTC denominated in US dollars. This dataset allows for an in-depth analysis of Bitcoin's market performance over time.

### Features:
1. **Date:** Timestamp of the recorded data.
2. **Open Price:** BTC price at the start of the period.
3. **High Price:** The highest BTC price during the period.
4. **Low Price:** The lowest BTC price during the period.
5. **Close Price:** BTC price at the end of the period.
6. **Volume:** Total BTC traded during the period.

**Dataset Link:** [Yahoo Finance BTC-USD](https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD&guccounter=1)

## Data Analysis
The analysis focuses on calculating key technical indicators to construct a dataset with predictors and an output variable.

### Technical Indicators:
- **Moving Average (MA):** Reduces noise and identifies price trends.
- **Stochastic Oscillator (%K and %D):** Measures price momentum and helps identify overbought/oversold conditions.
- **Relative Strength Index (RSI):** Evaluates the magnitude of price changes to determine market strength.
- **Rate of Change (ROC):** Measures percentage price change over time.
- **Momentum (MOM):** Assesses the speed of price movements.

## Data Preparation for Classification
The dataset is prepared by computing short-term and long-term Simple Moving Averages (SMA) of the closing price. Trading signals are generated as follows:
- **Buy Signal (1):** Short-term SMA is above long-term SMA, indicating a bullish trend.
- **Sell Signal (0):** Short-term SMA is below long-term SMA, indicating a bearish trend.

## Dependency and Environment Management
To set up the environment, install dependencies using `pipenv`:
```sh
pip install pipenv
pipenv install
```

## Running the Model Locally
1. Start the web server:
```sh
python predict.py
```
2. Open a new terminal and run the test script:
```sh
python test_predict.py
```

## Running the Model in Docker
1. Build the Docker image:
```sh
docker build -t bitcoin-investment-decision .
```
2. Run the container:
```sh
docker run -it --rm -p 9696:9696 bitcoin-investment-decision
```
3. In a new terminal, test the model:
```sh
python test_predict.py
```

## Requirements
- **Python:** 3.12
- **Libraries:**
  - yfinance
  - scikit-learn
  - xgboost
  - Flask
  - NumPy
  - Pandas
  - Gunicorn

