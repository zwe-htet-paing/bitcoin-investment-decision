# 🚀 Bitcoin Investment Decision

## 🌟 Overview
Bitcoin (BTC) is known for its high volatility, making investment decisions both exciting and challenging. This project leverages historical BTC price data to develop a strategy that helps determine whether to **buy BTC** or **wait** for more favorable market conditions. 

By analyzing trends, risk factors, and key technical indicators, we aim to create a data-driven approach to optimize BTC investment decisions.

---

## 🔍 Objectives
- **Analyze** historical Bitcoin price trends.
- **Develop** predictive models using technical indicators.
- **Generate** trading signals (Buy or Sell).
- **Deploy** the model as a web service for easy access.
- **Containerize** the project using Docker for scalability.

---

## 📆 Dataset
We utilize the **Bitcoin USD (BTC-USD)** dataset from [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/history/?p=BTC-USD), containing:

- **Date:** Timestamp of the recorded data
- **Open Price:** BTC price at the start of the period
- **High Price:** Highest BTC price during the period
- **Low Price:** Lowest BTC price during the period
- **Close Price:** BTC price at the end of the period
- **Volume:** Total BTC traded during the period

### 📥 Download the Dataset
You can download the dataset using the following code:

```python
import yfinance as yf

# Fetch BTC historical data
btc = yf.Ticker("BTC-USD")
df = btc.history(period="max")  # Fetch all available data

# Save as CSV
df.to_csv("dataset/btc_historical_data.csv")

print("BTC historical data downloaded successfully!")
```

---

## 📊 Data Analysis & Technical Indicators
To enhance predictive capabilities, we calculate key technical indicators:

- **Moving Average (MA):** Smooths price data to identify trends.
- **Stochastic Oscillator (%K, %D):** Highlights overbought/oversold conditions.
- **Relative Strength Index (RSI):** Measures price momentum.
- **Rate of Change (ROC):** Assesses percentage price changes over time.
- **Momentum (MOM):** Evaluates the speed of price movements.

### 🚀 Trading Signal Generation
- **Buy Signal (1):** When short-term SMA > long-term SMA (Bullish trend).
- **Sell Signal (0):** When short-term SMA < long-term SMA (Bearish trend).

---

## 🤖 Model Training & Fine-Tuning
We trained and fine-tuned multiple models, including:

- **Linear Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost** (chosen for deployment due to its superior performance)

For detailed model training, hyperparameter tuning, and evaluation, refer to the [notebook.ipynb](notebook.ipynb).

---

## 🌍 Deployment

### 🛠️ Clone the Repository
```bash
git clone https://github.com/zwe-htet-paing/bitcoin-investment-decision.git
cd bitcoin-investment-decision
```

### 🎓 Local Setup
1. **Install Dependencies:**  
   ```bash
   pip install pipenv
   pipenv install
   ```

2. **Run the Web Server:**  
   ```bash
   python predict.py
   ```

3. **Test the Model:**  
   Open a new terminal:
   ```bash
   python test_predict.py
   ```

### 🏗️ Docker Deployment
1. **Build the Docker Image:**  
   ```bash
   docker build -t bitcoin-investment-decision .
   ```

2. **Run the Container:**  
   ```bash
   docker run -it --rm -p 9696:9696 bitcoin-investment-decision
   ```

3. **Test the Model:**  
   In a new terminal:
   ```bash
   python test_predict.py
   ```

---

## 🚀 Tech Stack
- **Python:** 3.12
- **Libraries:**
  - `yfinance`
  - `scikit-learn`
  - `xgboost`
  - `Flask`
  - `NumPy`
  - `Pandas`
  - `Gunicorn`

---

*Happy Investing!* 📈🚀

