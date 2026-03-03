import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Model is running successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    
    # Example: if your model takes 2 inputs
    features = np.array([data["feature1"], data["feature2"]]).reshape(1, -1)
    
    prediction = model.predict(features)
    
    return jsonify({
        "prediction": prediction.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
