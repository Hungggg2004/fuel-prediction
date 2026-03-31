from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ================== LOAD MODEL ==================
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# ================== HOME ==================
@app.route("/")
def home():
    return render_template("index.html")

# ================== PREDICT ==================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu từ form
        cc = float(request.form["cc"])
        power = float(request.form["power"])

        # Convert thành array
        features = np.array([[cc, power]])

        # Dự đoán
        prediction = model.predict(features)[0]

        result = f"{prediction:.2f} L/100km"

        return render_template("index.html", prediction_text=f"Fuel Consumption: {result}")

    except:
        return render_template("index.html", prediction_text="⚠️ Vui lòng nhập đúng dữ liệu!")

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)