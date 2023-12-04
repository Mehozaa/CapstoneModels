import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

file_path = "PlayersData.csv"
data = pd.read_csv(file_path)

relevant_columns = ["week", "player", "original_rating"] + [
    col for col in data.columns if "pos_role" in col
]
data_ml = data[relevant_columns].copy()
for i in range(1, 35):
    data_ml[f"rating_lag_{i}"] = data_ml.groupby("player")["original_rating"].shift(i)

data_ml.dropna(inplace=True)
data_ml["next_week_rating"] = data_ml.groupby("player")["original_rating"].shift(-1)
data_ml.dropna(inplace=True)

X = data_ml.drop(["week", "player", "original_rating", "next_week_rating"], axis=1)
y = data_ml["next_week_rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")