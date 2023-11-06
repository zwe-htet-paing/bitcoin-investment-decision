import joblib
import xgboost as xgb
from flask import Flask, request, jsonify

 
dv = joblib.load('dv.pkl')
model = joblib.load('xgb_model.pkl')

app = Flask('BTC coin signals')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    # y_pred = model.predict_proba(X)[0, 1]
    
    features = dv.feature_names_
    y_pred = model.predict(xgb.DMatrix(X, feature_names=features))
    
    score = y_pred >= 0.5

    result = {
        'probability': float(y_pred),
        'signals': bool(score)
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)