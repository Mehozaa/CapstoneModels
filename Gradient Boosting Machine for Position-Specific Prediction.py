import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


file_path = 'finally imken2222.csv'
data = pd.read_csv(file_path)

features_fw = ['goals', 'assists', 'shots_ontarget', 'keypasses', 'touches', 'drib_success', 'offsides', 'chances2score']
features_df = ['clearances', 'interceptions', 'tackles', 'aerials_w', 'aerials_l', 'shotsblocked', 'poss_lost', 'dribbled_past']
features_mf = ['passes_acc', 'drib_success', 'keypasses', 'tackles', 'interceptions', 'touches', 'chances2score', 'passes_inacc']
features_gk = ['saves_itb', 'saves_otb', 'saved_pen', 'goals_ag_itb', 'goals_ag_otb', 'clearances', 'stop_shots']

target = 'original_rating'

def train_and_evaluate_gbm(data, features, position_flag):

    position_data = data[data[position_flag] == 1][features + [target]].dropna()

    X = position_data[features]
    y = position_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingRegressor(random_state=42)
    parameters = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model, y_test, y_pred, mse, mae, r2

def plot_actual_vs_predicted(y_test, y_pred, title):
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

model_fw, y_test_fw, y_pred_fw, mse_fw, mae_fw, r2_fw = train_and_evaluate_gbm(data, features_fw, 'pos_FW')
model_df, y_test_df, y_pred_df, mse_df, mae_df, r2_df = train_and_evaluate_gbm(data, features_df, 'pos_DF')
model_mf, y_test_mf, y_pred_mf, mse_mf, mae_mf, r2_mf = train_and_evaluate_gbm(data, features_mf, 'pos_MF')
model_gk, y_test_gk, y_pred_gk, mse_gk, mae_gk, r2_gk = train_and_evaluate_gbm(data, features_gk, 'pos_GK')

plot_actual_vs_predicted(y_test_fw, y_pred_fw, 'Forwards Model - Actual vs Predicted Ratings')
plot_actual_vs_predicted(y_test_df, y_pred_df, 'Defenders Model - Actual vs Predicted Ratings')
plot_actual_vs_predicted(y_test_mf, y_pred_mf, 'Midfielders Model - Actual vs Predicted Ratings')
plot_actual_vs_predicted(y_test_gk, y_pred_gk, 'Goalkeepers Model - Actual vs Predicted Ratings')

print("Forwards Model - MSE:", mse_fw, "MAE:", mae_fw, "R2:", r2_fw)
print("Defenders Model - MSE:", mse_df, "MAE:", mae_df, "R2:", r2_df)
print("Midfielders Model - MSE:", mse_mf, "MAE:", mae_mf, "R2:", r2_mf)
print("Goalkeepers Model - MSE:", mse_gk, "MAE:", mae_gk, "R2:", r2_gk)