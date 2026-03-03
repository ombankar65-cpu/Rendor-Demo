import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = pickle.load(open(model_path, "rb"))


# Home Route
@app.route("/")
def home():
    return """
    <h2>Model is running successfully!</h2>
    <p>To test prediction:</p>
    <p>/predict?feature1=10&feature2=20</p>
    """


# Prediction Route (Browser Friendly)
@app.route("/predict")
def predict():
    try:
        # Get values from URL
        feature1 = float(request.args.get("feature1"))
        feature2 = float(request.args.get("feature2"))

        # Convert into numpy array
        features = np.array([[feature1, feature2]])

        # Predict
        prediction = model.predict(features)

        return jsonify({
            "feature1": feature1,
            "feature2": feature2,
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
