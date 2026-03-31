import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Đọc dữ liệu
df = pd.read_excel("du_lieu_xe_may.xlsx")

# Feature và target
X = df[['cc', 'power(PS)']]
y = df['fuel_consumption(L/100km)']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Train Random Forest
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Đánh giá
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Đánh giá Random Forest")
print(f"R2   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# Lưu model
pickle.dump(model, open("model.pkl", "wb"))

print("\n✅ Đã lưu model.pkl")

# Test nhanh
sample = pd.DataFrame({'cc': [130], 'power(PS)': [11]})
pred = model.predict(sample)

print(f"\n🚀 Test: {pred[0]:.2f} L/100km")