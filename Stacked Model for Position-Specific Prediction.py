import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

file_path = 'PlayersData.csv'
data = pd.read_csv(file_path)

features_fw = ['goals', 'assists', 'shots_ontarget', 'keypasses', 'touches', 'drib_success', 'offsides', 'chances2score']
features_df = ['clearances', 'interceptions', 'tackles', 'aerials_w', 'aerials_l', 'shotsblocked', 'poss_lost', 'dribbled_past']
features_mf = ['passes_acc', 'drib_success', 'keypasses', 'tackles', 'interceptions', 'touches', 'chances2score', 'passes_inacc']
features_gk = ['saves_itb', 'saves_otb', 'saved_pen', 'goals_ag_itb', 'goals_ag_otb', 'clearances', 'stop_shots']

target = 'original_rating'

def train_and_evaluate_stacking(data, features_rf, features_gbm, position_flag):

    position_data = data[data[position_flag] == 1].dropna()

    X_rf = position_data[features_rf]
    X_gbm = position_data[features_gbm]
    y = position_data[target]
    X_train_rf, X_test_rf, y_train, y_test = train_test_split(X_rf, y, test_size=0.2, random_state=42)
    X_train_gbm, X_test_gbm = train_test_split(X_gbm, test_size=0.2, random_state=42)

    scaler_rf = StandardScaler()
    scaler_gbm = StandardScaler()
    X_train_scaled_rf = scaler_rf.fit_transform(X_train_rf)
    X_test_scaled_rf = scaler_rf.transform(X_test_rf)
    X_train_scaled_gbm = scaler_gbm.fit_transform(X_train_gbm)
    X_test_scaled_gbm = scaler_gbm.transform(X_test_gbm)

    rf_model = RandomForestRegressor(random_state=42)
    gbm_model = GradientBoostingRegressor(random_state=42)

    stacked_model = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('gbm', gbm_model)
        ],
        final_estimator=LinearRegression()
    )

    stacked_model.fit(np.column_stack((X_train_scaled_rf, X_train_scaled_gbm)), y_train)

    y_pred = stacked_model.predict(np.column_stack((X_test_scaled_rf, X_test_scaled_gbm)))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return stacked_model, y_test, y_pred, mse, mae, r2

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title(title)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.grid(True)

    min_rating = min(y_test.min(), y_pred.min())
    max_rating = max(y_test.max(), y_pred.max())
    plt.plot([min_rating, max_rating], [min_rating, max_rating], 'k--', lw=2)

    plt.show()

models = {}
evaluation_metrics = {}

for features, position_flag in zip([features_fw, features_df, features_mf, features_gk], ['pos_FW', 'pos_DF', 'pos_MF', 'pos_GK']):
    model, y_test, y_pred, mse, mae, r2 = train_and_evaluate_stacking(data, features, features, position_flag)
    models[position_flag] = model
    evaluation_metrics[position_flag] = (mse, mae, r2)
    plot_actual_vs_predicted(y_test, y_pred, f'Actual vs Predicted Ratings for {position_flag}')

for position_flag in ['pos_FW', 'pos_DF', 'pos_MF', 'pos_GK']:
    mse, mae, r2 = evaluation_metrics[position_flag]
    print(f"{position_flag} Model - MSE: {mse}, MAE: {mae}, R2: {r2}")