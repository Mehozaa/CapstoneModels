import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
import numpy as np
import matplotlib.pyplot as plt


file_path = 'PlayersData.csv'
data = pd.read_csv(file_path)
grouped_data = data.groupby(['player', 'week'])['original_rating'].mean().reset_index()
pivot_data = grouped_data.pivot(index='player', columns='week', values='original_rating')
pivot_data_filled = pivot_data.fillna(method='ffill', axis=1)

def check_stationarity(series):
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value <= 0.05

def find_best_arima(series):
    best_aic = np.inf
    best_order = None
    best_model = None

    p = range(0, 3)
    d = range(0, 2)
    q = range(0, 3)

    for i in p:
        for j in d:
            for k in q:
                try:
                    model = ARIMA(series, order=(i, j, k)).fit()
                    aic = model.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (i, j, k)
                        best_model = model
                except:
                    continue

    return best_order, best_model

warnings.filterwarnings("ignore")

def predict_player_rating(player_name, forecast_week):
    if player_name not in pivot_data_filled.index:
        return f"Player '{player_name}' not found in the dataset."

    series = pivot_data_filled.loc[player_name]
    num_existing_points = len(series.dropna())

    if forecast_week <= num_existing_points:
        return f"Forecast week must be greater than the number of existing data points ({num_existing_points})."

    order, model = find_best_arima(series)

    steps_to_forecast = forecast_week - num_existing_points
    forecast = model.get_forecast(steps=steps_to_forecast).predicted_mean
    return forecast, model


def plot_actual_vs_predicted(player_name, forecast_week):
    result = predict_player_rating(player_name, forecast_week)

    if isinstance(result, str):
        print(result)  
        return

    forecast, model = result

    actual_series = pivot_data_filled.loc[player_name].dropna()
    actual_week_numbers = actual_series.index.tolist() 
    last_week_number = actual_week_numbers[-1]
    forecast_week_numbers = list(range(last_week_number + 1, last_week_number + 1 + len(forecast)))

    plt.figure(figsize=(12, 6))
    plt.plot(actual_week_numbers, actual_series, label='Actual Ratings', color='blue')
    plt.plot(forecast_week_numbers, forecast, label='Predicted Ratings', color='red', linestyle='--')
    plt.xlabel('Week')
    plt.ylabel('Rating')
    plt.title(f'Actual vs Predicted Ratings for {player_name}')
    plt.legend()
    plt.show()


player_name_input = "Kevin Kampl"  
forecast_week_input = 43  

plot_actual_vs_predicted(player_name_input, forecast_week_input)