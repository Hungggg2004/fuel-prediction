import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ================== LOAD DATA ==================
df = pd.read_excel("du_lieu_xe_may.xlsx")

# ================== CLEAN PRICE ==================
def clean_price(x):
    if pd.isna(x):
        return None

    x = str(x).lower()

    x = x.replace("rs.", "")
    x = x.replace("(view on road price)", "")
    x = x.strip()

    if "-" in x:
        x = x.split("-")[0].strip()

    if "lakh" in x:
        try:
            num = float(x.replace("lakh", "").strip())
            return num * 100000
        except:
            return None

    x = x.replace(",", "")

    try:
        return float(x)
    except:
        return None

df['price'] = df['price'].apply(clean_price)

# ================== CLEAN RATING ==================
def clean_rating(x):
    if pd.isna(x):
        return None

    x = str(x)

    if "/" in x:
        x = x.split("/")[0]

    try:
        return float(x)
    except:
        return None

df['rating'] = df['rating'].apply(clean_rating)

# ================== CLEAN NUMERIC ==================
df['cc'] = pd.to_numeric(df['cc'], errors='coerce')
df['power(PS)'] = pd.to_numeric(df['power(PS)'], errors='coerce')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
df['fuel_consumption(L/100km)'] = pd.to_numeric(df['fuel_consumption(L/100km)'], errors='coerce')

# ================== DROP NULL ==================
df = df.dropna()

print("✅ Data sau khi clean:", df.shape)

# ================== ADD SYNTHETIC FEATURES ==================
np.random.seed(42)

df['speed'] = np.random.randint(30, 81, size=len(df))
df['load_weight'] = np.random.randint(50, 121, size=len(df))
df['road_type'] = np.random.choice(['city', 'highway', 'mountain'], size=len(df))
df['weather'] = np.random.choice(['normal', 'rainy'], size=len(df))

# ================== ADJUST TARGET ==================
fuel = df['fuel_consumption(L/100km)'].copy()

fuel += np.where(df['speed'] > 60, 0.3, 0)
fuel += np.where(df['load_weight'] > 100, 0.2, 0)
fuel += np.where(df['road_type'] == 'mountain', 0.4, 0)
fuel += np.where(df['weather'] == 'rainy', 0.1, 0)

df['fuel_consumption(L/100km)'] = fuel

# ================== FEATURE ENGINEERING ==================
df['power_per_cc'] = df['power(PS)'] / df['cc']
df['price_per_cc'] = df['price'] / df['cc']

# ================== ENCODE ==================
df = pd.get_dummies(df, columns=['brand', 'road_type', 'weather'], drop_first=True)

# ================== SELECT FEATURE ==================
X = df.drop(columns=['fuel_consumption(L/100km)', 'name'])
y = df['fuel_consumption(L/100km)']

# ================== SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== TRAIN ==================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ================== PREDICT ==================
y_pred = model.predict(X_test)

# ================== EVALUATE ==================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Đánh giá Random Forest")
print(f"R2   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

# ================== FEATURE IMPORTANCE ==================
importances = model.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n🔥 Top Feature quan trọng:")
print(feat_imp.head(10))

# ================== SAVE MODEL ==================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("\n✅ Đã lưu model.pkl và columns.pkl")

# ================== TEST ==================
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

# fill thiếu cột
for col in X.columns:
    if col not in sample.columns:
        sample[col] = 0

sample = sample[X.columns]

pred = model.predict(sample)

print(f"\n🚀 Test: {pred[0]:.2f} L/100km")