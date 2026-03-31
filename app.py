from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Kiểm tra model tồn tại
if not os.path.exists("model.pkl"):
    print("❌ Chưa có model.pkl → hãy chạy train_model.py trước!")
    exit()

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        cc = float(request.form["cc"])
        power = float(request.form["power"])

        data = np.array([[cc, power]])
        prediction = model.predict(data)

        result = round(prediction[0], 2)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)