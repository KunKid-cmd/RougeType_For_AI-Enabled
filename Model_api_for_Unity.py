from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = pd.DataFrame([{
            "wpm": float(data["wpm"]),
            "combo_length": int(data["combo_length"]),
            "mistake_count": int(data["mistake_count"]),
            "recent_accuracy": float(data["recent_accuracy"]),
            "wave_number": int(data["wave_number"])
        }])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)
