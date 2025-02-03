import joblib
import xgboost as xgb
from flask import Flask, request, jsonify

import warnings
warnings.filterwarnings('ignore')

# Load the vectorizer and model
dv = joblib.load('dv.pkl')
model = joblib.load('xgb_model.pkl')

app = Flask('BTC coin signals')

@app.route('/predict', methods=['POST'])
def predict():
    # Get client input from the request
    client = request.get_json()

    # Transform the input data using the vectorizer
    X = dv.transform([client])

    # Get the feature names
    features = dv.feature_names_

    # Make the prediction using the trained model
    y_pred = model.predict(xgb.DMatrix(X, feature_names=features))

    # Convert the prediction to a boolean signal based on a threshold (e.g., 0.5 for binary classification)
    score = y_pred[0] >= 0.5  # Check if the prediction is above the threshold (assuming a binary classifier)

    # Create the response with prediction probability and signal
    result = {
        'probability': float(y_pred[0]),  # Return the prediction as a float
        'signals': bool(score)  # Return True/False based on the threshold
    }

    # Return the result as a JSON response
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
