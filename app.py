import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route("/")
def home():
    return "Model is running successfully!"

@app.route("/predict")
def predict():
    try:
        feature1 = float(request.args.get("feature1"))
        feature2 = float(request.args.get("feature2"))

        features = np.array([[feature1, feature2]])
        prediction = model.predict(features)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
