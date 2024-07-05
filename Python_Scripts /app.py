from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models and scaler
with open('linear_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Fantasy Football Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    # Extract the features from the request data
    features = np.array([data['FantPt_rolling_avg'], data['exceptional_performance'], data['normalized_team_performance'], data['Age']])
    features = features.reshape(1, -1)
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict using both models
    prediction_linear = linear_model.predict(features_scaled)[0]
    prediction_rf = rf_model.predict(features_scaled)[0]
    
    # Return the predictions as a JSON response
    return jsonify({
        'Linear Regression Prediction': prediction_linear,
        'Random Forest Prediction': prediction_rf
    })

if __name__ == '__main__':
    app.run(debug=True)
