from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image

app = Flask(__name__)

# ---------- Load Model ----------
with open("cat_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract parameters
w = model["w"]
b = model["b"]

classes = [b"non-cat", b"cat"]  # your classes

# ---------- Sigmoid + Predict ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def custom_predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    return (A > 0.5).astype(int)

# ---------- Health Check ----------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Server is running ✅"})

# ---------- Prediction Route ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Open and preprocess image
        img = Image.open(file).convert("RGB")
        img = img.resize((64, 64))  
        img_array = np.array(img).reshape(-1, 1) / 255.0   # (12288, 1)

        # Predict
        pred = custom_predict(w, b, img_array)  # 0 or 1
        prediction = classes[int(pred[0,0])].decode("utf-8")

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
