import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'PlayersData.csv'
data = pd.read_csv(file_path)

features_fw = ['assists', 'shots_ontarget', 'keypasses', 'touches', 'drib_success', 'offsides', 'chances2score', 'goal_involvement', 'minutesPlayed']
features_df = ['clearances', 'interceptions', 'tackles', 'aerials_w', 'aerials_l', 'shotsblocked', 'poss_lost', 'dribbled_past', 'defensive_contribution', 'minutesPlayed', 'game_duration']
features_mf = ['passes_acc', 'drib_success', 'keypasses', 'tackles', 'interceptions', 'touches', 'chances2score', 'passes_inacc', 'midfield_dominance', 'minutesPlayed', 'game_duration']
features_gk = ['saves_itb', 'saves_otb', 'saved_pen', 'goals_ag_itb', 'goals_ag_otb', 'clearances', 'stop_shots', 'goalkeeper_impact', 'minutesPlayed', 'game_duration']

target = 'original_rating'

data['goal_involvement'] = data['goals'] + data['assists']
data['defensive_contribution'] = data['clearances'] + data['interceptions']
data['midfield_dominance'] = data['keypasses'] + data['touches']
data['goalkeeper_impact'] = data['saves_itb'] + data['saves_otb']

def train_and_evaluate(data, features, position_flag):

    position_data = data[data[position_flag] == 1][features + [target]].dropna()

    X = position_data[features]
    y = position_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=42)
    parameters = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Ratings', fontsize=14)
    plt.ylabel('Predicted Ratings', fontsize=14)
    plt.title(f'Actual vs Predicted Ratings for {position_flag}', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_model, mse, mae, r2

model_fw, mse_fw, mae_fw, r2_fw = train_and_evaluate(data, features_fw, 'pos_FW')
model_df, mse_df, mae_df, r2_df = train_and_evaluate(data, features_df, 'pos_DF')
model_mf, mse_mf, mae_mf, r2_mf = train_and_evaluate(data, features_mf, 'pos_MF')
model_gk, mse_gk, mae_gk, r2_gk = train_and_evaluate(data, features_gk, 'pos_GK')

print("Forwards Model - MSE:", mse_fw, "MAE:", mae_fw, "R2:", r2_fw)
print("Defenders Model - MSE:", mse_df, "MAE:", mae_df, "R2:", r2_df)
print("Midfielders Model - MSE:", mse_mf, "MAE:", mae_mf, "R2:", r2_mf)
print("Goalkeepers Model - MSE:", mse_gk, "MAE:", mae_gk, "R2:", r2_gk)