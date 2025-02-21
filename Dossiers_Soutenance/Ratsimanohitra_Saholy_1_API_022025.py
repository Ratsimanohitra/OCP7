print("Starting Flask API...")
from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

app = Flask(__name__)

# Load the MinMaxScaler and LightGBM model from pickle files
with open('./models/minmax_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('./models/lightgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Transform JSON data to a format suitable for MinMaxScaler
    data = np.array(data['data'])
    # Apply MinMaxScaler
    scaled_data = scaler.transform(data)
    
    # Predict using LightGBM model
    prediction = model.predict_proba(scaled_data)[:,1]
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=False)