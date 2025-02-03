import re
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')


# Calculation of exponential moving average
def EMA(df, n):
    return pd.Series(df['close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))

# Calculation of rate of change
def ROC(df, n):
    return pd.Series(((df.diff(n - 1) / df.shift(n - 1)) * 100), name='ROC_' + str(n))

# Calculation of price momentum
def MOM(df, n):
    return pd.Series(df.diff(n), name='Momentum_' + str(n))

# Calculation of stochastic oscillator
def STOK(close, low, high, n):
    return ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100

def STOD(close, low, high, n):
    STOK_value = STOK(close, low, high, n)
    return STOK_value.rolling(3).mean()

# Calculation of relative strength index
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    
    u[u.index[period-1]] = np.mean(u[:period])
    d[d.index[period-1]] = np.mean(d[:period])
    
    u = u.drop(u.index[:(period-1)])
    d = d.drop(d.index[:(period-1)])
    
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

# Load and process data
def load_data():
    data = pd.read_csv('dataset/BTC-USD.csv')
    data.columns = data.columns.str.replace(' ', '_').str.lower()
    data = data.drop(['adj_close'], axis=1)
    
    df = data.copy()

    # Moving averages
    df['short_mavg'] = df['close'].rolling(window=10, min_periods=1).mean()
    df['long_mavg'] = df['close'].rolling(window=60, min_periods=1).mean()

    # Create signals based on moving averages
    df['signals'] = np.where(df['short_mavg'] > df['long_mavg'], 1, 0)
    
    # Add technical indicators
    df['EMA7'] = EMA(df, 7)
    df['EMA14'] = EMA(df, 14)
    df['EMA30'] = EMA(df, 30)
    
    df['ROC7'] = ROC(df['close'], 7)
    df['ROC14'] = ROC(df['close'], 14)
    df['ROC30'] = ROC(df['close'], 30)
    
    df['MOM7'] = MOM(df['close'], 7)
    df['MOM14'] = MOM(df['close'], 14)
    df['MOM30'] = MOM(df['close'], 30)
    
    df['RSI7'] = RSI(df['close'], 7)
    df['RSI14'] = RSI(df['close'], 14)
    df['RSI30'] = RSI(df['close'], 30)
    
    df['%K7'] = STOK(df['close'], df['low'], df['high'], 7)
    df['%D7'] = STOD(df['close'], df['low'], df['high'], 7)

    df['%K14'] = STOK(df['close'], df['low'], df['high'], 14)
    df['%D14'] = STOD(df['close'], df['low'], df['high'], 14)

    df['%K30'] = STOK(df['close'], df['low'], df['high'], 30)
    df['%D30'] = STOD(df['close'], df['low'], df['high'], 30)
    
    df = df.dropna(axis=0).reset_index(drop=True)
    
    return df

# Train Random Forest model
def train_rf_model(df):
    dv = DictVectorizer(sparse=False)
    rf = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)
    
    features = ['EMA7', 'EMA14', 'EMA30', 'ROC7', 'ROC14', 'ROC30', 'MOM7', 'MOM14', 'MOM30', 'RSI7', 'RSI14', 'RSI30', '%K7', '%D7', '%K14', '%D14', '%K30', '%D30']
    
    # Split data
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_train_full = df_train_full['signals'].values
    y_test = df_test['signals'].values
    
    # Transform data with DictVectorizer
    train_dict_full = df_train_full[features].to_dict(orient="records")
    dv.fit(train_dict_full)
    X_train_full = dv.transform(train_dict_full)
    
    test_dict = df_test[features].to_dict(orient="records")
    X_test = dv.transform(test_dict)
    
    # Train model
    rf.fit(X_train_full, y_train_full)
    
    # Predict and calculate ROC AUC
    y_pred = rf.predict(X_test)
    score = roc_auc_score(y_test, y_pred)
    print("Random Forest ROC AUC Score:", score)
    
    # Save models
    joblib.dump(dv, 'dv_rf.pkl')
    joblib.dump(rf, 'rf_model.pkl')

# Train XGBoost classifier
def train_xgb_classifier(df):
    features = ['EMA7', 'EMA14', 'EMA30', 'ROC7', 'ROC14', 'ROC30', 'MOM7', 'MOM14', 'MOM30', 'RSI7', 'RSI14', 'RSI30', '%K7', '%D7', '%K14', '%D14', '%K30', '%D30']
    
    # Split data
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_train_full = df_train_full['signals'].values
    y_test = df_test['signals'].values
    
    # Transform data with DictVectorizer
    dv = DictVectorizer(sparse=False)
    train_dict = df_train_full[features].to_dict(orient="records")
    dv.fit(train_dict)
    X_train = dv.transform(train_dict)
    
    test_dict = df_test[features].to_dict(orient='records')
    X_test = dv.transform(test_dict)
    
    # Convert features to xgb.DMatrix
    features = dv.feature_names_
    regex = re.compile(r"<", re.IGNORECASE)
    features = [regex.sub("_", col) for col in features]

    dtrain = xgb.DMatrix(X_train, label=y_train_full, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    
    # XGBoost parameters
    xgb_params = {
        'eta': 0.3, 
        'max_depth': 10,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    
    # Predict and calculate ROC AUC
    y_pred = model.predict(dtest)
    score = roc_auc_score(y_test, y_pred)
    print("XGBoost ROC AUC Score:", score)
    
    # Save models
    joblib.dump(dv, 'dv_xgb.pkl')
    joblib.dump(model, 'xgb_model.pkl')


if __name__ == '__main__':
    df = load_data()
    
    # Train Random Forest model
    # train_rf_model(df)
    
    # Train XGBoost model
    train_xgb_classifier(df)
