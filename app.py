from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# ================== LOAD MODEL ==================
base_path = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(base_path, "columns.pkl"), "rb"))

# ================== HOME ==================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    suggestion = ""

    if request.method == "POST":
        try:
            # ===== Lấy dữ liệu từ form =====
            cc = float(request.form["cc"])
            power = float(request.form["power"])
            speed = float(request.form["speed"])
            load_weight = float(request.form["load_weight"])
            road_type = request.form["road_type"]
            weather = request.form["weather"]

            # ===== Feature Engineering =====
            power_per_cc = power / cc
            price = 30000000   # giả lập
            price_per_cc = price / cc
            rating = 4.5
            mileage = 50

            data = {
                'cc': cc,
                'power(PS)': power,
                'price': price,
                'rating': rating,
                'mileage': mileage,
                'speed': speed,
                'load_weight': load_weight,
                'power_per_cc': power_per_cc,
                'price_per_cc': price_per_cc,
            }

            df = pd.DataFrame([data])

            # ===== One-hot giống lúc train =====
            for col in columns:
                if col not in df.columns:
                    df[col] = 0

            df = df[columns]

            # ===== Predict =====
            pred = model.predict(df)[0]
            result = round(pred, 2)

            # ===== Gợi ý AI =====
            if speed > 60:
                suggestion += "⚡ Giảm tốc độ để tiết kiệm xăng. "

            if load_weight > 100:
                suggestion += "⚖️ Hạn chế chở quá nặng. "

            if road_type == "mountain":
                suggestion += "⛰️ Đường đèo tiêu hao nhiều nhiên liệu. "

            if weather == "rainy":
                suggestion += "🌧️ Trời mưa làm tăng tiêu hao nhiên liệu. "

        except:
            result = "⚠️ Lỗi dữ liệu!"
            suggestion = ""

    return render_template("index.html", result=result, suggestion=suggestion)

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)