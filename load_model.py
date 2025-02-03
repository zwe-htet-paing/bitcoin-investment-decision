import joblib
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# Load the vectorizer and model
dv = joblib.load('dv.pkl')
model = joblib.load('xgb_model.pkl')

# Client data
client = {'EMA7': 20768.484481399686, 'EMA14': 20490.05642871806, 'EMA30': 20146.747057307526, 
          'ROC7': 2.1014717561301666, 'ROC14': 8.17196844974428, 'ROC30': 7.776440929635841, 
          'MOM7': 290.88281199999983, 'MOM14': 1359.4785149999989, 'MOM30': 1379.6367189999983, 
          'RSI7': 60.814408187394406, 'RSI14': 59.67152906192006, 'RSI30': 53.807929837547135, 
          '%K7': 61.75344612013603, '%D7': 81.3790872452849, 
          '%K14': 76.77367219169297, '%D14': 88.90518384754644, 
          '%K30': 83.35818148868839, '%D30': 91.98348494419609}

# Transform the client data
X = dv.transform([client])

# Get the feature names
features = dv.feature_names_

# Predict using xgboost DMatrix (probability output)
dmat = xgb.DMatrix(X, feature_names=features)
y_pred = model.predict(dmat)

# Print predicted probability for the positive class
print(y_pred[0])
