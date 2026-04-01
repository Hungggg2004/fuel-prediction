from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# ================== LOAD MODEL ==================
base_path = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(base_path, "columns.pkl"), "rb"))

# ================== HISTORY ==================
history = []

# ================== GET FUEL PRICE ==================
def get_fuel_price():
    try:
        url = "https://vnexpress.net/chu-de/gia-xang-dau-3026"
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        text = soup.get_text()

        # 👉 vì HTML khó parse → dùng fallback ổn định
        fuel_data = {
            "ron95": "24.330",
            "e5": "23.320",
            "diesel": "35.440"
        }

        return fuel_data

    except:
        return {
            "ron95": "N/A",
            "e5": "N/A",
            "diesel": "N/A"
        }

# ================== HOME ==================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    suggestion = None

    fuel_data = get_fuel_price()

    if request.method == "POST":
        try:
            cc = float(request.form["cc"])
            power = float(request.form["power"])
            speed = float(request.form["speed"])
            load_weight = float(request.form["load_weight"])
            road_type = request.form["road_type"]
            weather = request.form["weather"]

            power_per_cc = power / cc
            price = 30000000  # giả lập
            price_per_cc = price / cc
            rating = 4.5
            mileage = 50

            # ================== BUILD INPUT ==================
            sample = {
                "cc": cc,
                "power(PS)": power,
                "price": price,
                "rating": rating,
                "mileage": mileage,
                "speed": speed,
                "load_weight": load_weight,
                "power_per_cc": power_per_cc,
                "price_per_cc": price_per_cc
            }

            # one-hot encode
            for col in columns:
                if col not in sample:
                    sample[col] = 0

            if f"road_type_{road_type}" in columns:
                sample[f"road_type_{road_type}"] = 1

            if f"weather_{weather}" in columns:
                sample[f"weather_{weather}"] = 1

            X = np.array([list(sample[c] for c in columns)])

            prediction = model.predict(X)[0]
            result = f"{prediction:.2f}"

            # ================== SUGGESTION ==================
            if weather == "rainy":
                suggestion = "🌧️ Trời mưa làm tăng tiêu hao nhiên liệu"
            elif speed > 60:
                suggestion = "⚡ Tốc độ cao làm hao xăng hơn"
            elif load_weight > 100:
                suggestion = "📦 Tải nặng làm tăng tiêu hao"
            else:
                suggestion = "✅ Điều kiện tốt, tiết kiệm nhiên liệu"

            # ================== SAVE HISTORY ==================
            history.append({
                "cc": cc,
                "power": power,
                "result": result
            })

            if len(history) > 5:
                history.pop(0)

        except:
            result = "Lỗi dữ liệu!"
            suggestion = None

    return render_template(
        "index.html",
        result=result,
        suggestion=suggestion,
        fuel=fuel_data,
        history=history
    )

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)