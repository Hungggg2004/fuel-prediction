import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# 1. Đọc dữ liệu
# =========================
df = pd.read_excel("du_lieu_xe_may.xlsx")

# =========================
# 2. Feature Engineering
# =========================

# Tạo thêm feature mới
df['power_per_cc'] = df['power(PS)'] / df['cc']
df['price_per_cc'] = df['price'] / df['cc']

# =========================
# 3. Encode categorical
# =========================
df = pd.get_dummies(df, columns=['brand', 'road_type', 'weather'], drop_first=True)

# =========================
# 4. Chọn feature và target
# =========================
X = df.drop(columns=['fuel_consumption(L/100km)', 'name'])
y = df['fuel_consumption(L/100km)']

# =========================
# 5. Train test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. Train model
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 7. Predict
# =========================
y_pred = model.predict(X_test)

# =========================
# 8. Đánh giá
# =========================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Đánh giá Random Forest (Nâng cấp)")
print(f"R2   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# =========================
# 9. Feature Importance
# =========================
importances = model.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔥 Top Feature quan trọng:")
print(feat_imp.head(10))

# =========================
# 10. Lưu model + columns
# =========================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("\n✅ Đã lưu model.pkl và columns.pkl")

# =========================
# 11. Test thử
# =========================
sample = pd.DataFrame({
    'cc': [130],
    'power(PS)': [11],
    'price': [30000000],
    'rating': [4.5],
    'mileage': [50],
    'speed': [50],
    'load_weight': [70],
    'power_per_cc': [11/130],
    'price_per_cc': [30000000/130],
})

# ⚠️ thêm các cột còn thiếu (one-hot)
for col in X.columns:
    if col not in sample.columns:
        sample[col] = 0

# đảm bảo đúng thứ tự cột
sample = sample[X.columns]

pred = model.predict(sample)

print(f"\n🚀 Test: {pred[0]:.2f} L/100km")