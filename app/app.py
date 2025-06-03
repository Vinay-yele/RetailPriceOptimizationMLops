#app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, render_template
from src.predict import predict_sales

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Render the HTML page

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = predict_sales(data)
    return jsonify({"predicted_total_price": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
