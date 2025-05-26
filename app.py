from flask import Flask, request, jsonify
from train_model import train_and_save_model
import joblib
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf

# Load model & tools saat Flask start
model = tf.keras.models.load_model("./model/resep_model.h5")
vectorizer = joblib.load("./model/vectorizer.pkl")
label_encoder = joblib.load("./model/label_encoder.pkl")

app = Flask(__name__)



@app.route("/api/v1/retrain-model", methods=["POST"])
def retrain_model():
    try:
        message = train_and_save_model()
        return jsonify({
            "code"  : 200,
             "status": "success",
              "message": message
              })
    except Exception as e:
        return jsonify({
            "code"  : 500,
            "status": "error", 
            "message": str(e)
            }), 500

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    data = request.json
    input_ingredients = data.get("ingredients")

    if not input_ingredients:
        return jsonify({
            "code": 400,
            "status": "error",
            "message": "Please enter 'ingredients' in the form of a list of ingredients"}), 400

    try:
        input_vec = vectorizer.transform([" ".join(input_ingredients)])
        pred_probs = model.predict(input_vec.toarray())[0]
        top_indices = pred_probs.argsort()[-5:][::-1]
        results = []
        for idx in top_indices:
            try:
                food_id = label_encoder.inverse_transform([idx])[0]
                food_id_int = int(food_id)
            except (ValueError, TypeError):
                continue
            results.append({
                "food_id": int(food_id_int),
                "score": float(pred_probs[idx])
            })

        return jsonify({
            "code": 200,
            "status": "success",
            "data": results}),200

    except Exception as e:
        return jsonify({
            "code": 500,
            "status": "error",
            "message": str(e)
            }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False)
