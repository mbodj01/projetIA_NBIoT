from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load("random_forest_model.pkl")

@app.route('/')
def home():
    return "API de classification N-BaIoT"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

import sys
print(sys.version)
