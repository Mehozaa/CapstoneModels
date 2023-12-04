import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

def predict_player_rating_up_to_week(file_path, player_name, week, model, scaler, features):

    data = pd.read_csv(file_path)

    if not set(features).issubset(data.columns):
        return "Required features not available in the dataset."

    data_player = data[(data['player'] == player_name) & (data['week'] < week)].copy()

    if data_player.empty:
        return "Data not available for the specified player."

    if data_player[features].isnull().values.any():
        return "Missing data in the required features."

    recent_data = data_player.tail(1)

    X_standardized = scaler.transform(recent_data[features])

    predicted_rating = model.predict(X_standardized)

    return predicted_rating[0]

def apply_positional_weighting(row, features, pos_column, weight_dict):
    for feature in features:
        row[feature] = row[feature] * weight_dict.get(feature, 1) if row[pos_column] == 1 else row[feature]
    return row


file_path = 'PlayersData.csv'  
data = pd.read_csv(file_path)


pos_fw_weights = {
    'original_rating': 1,
    'goals': 1.5,
    'assists': 1.25,
    'shots_ontarget': 1.25,
    'shots_offtarget': 1,
    'shotsblocked': 0.5,
    'chances2score': 1.25,
    'drib_success': 1,
    'drib_unsuccess': 0.5,
    'keypasses': 1,
    'touches': 0.75,
    'passes_acc': 0.75,
    'passes_inacc': 0.5,
    'crosses_acc': 0.75,
    'crosses_inacc': 0.5,
    'lballs_acc': 0.75,
    'lballs_inacc': 0.5,
    'grduels_w': 0.5,
    'grduels_l': 0.5,
    'aerials_w': 0.5,
    'aerials_l': 0.5,
    'poss_lost': 0.75,
    'fouls': 0.5,
    'wasfouled': 0.75,
    'clearances': 0.25,
    'stop_shots': 0.25,
    'interceptions': 0.25,
    'tackles': 0.25,
    'ycards': 0.5,
    'rcards': 0.5,
    'dangmistakes': 1,
    'countattack': 1,
    'offsides': 0.75,
    'goals_ag_otb': 0.25,
    'goals_ag_itb': 0.25,
    'saves_itb': 0,
    'saves_otb': 0,
    'saved_pen': 0,
    'missed_penalties': 1,
    'owngoals': 0.5,
    'win': 1,
    'lost': 1,
    'minutesPlayed': 1,
    'game_duration': 0.5
}



pos_mf_weights = {
    'original_rating': 1,
    'goals': 0.75,
    'assists': 1.25,
    'shots_ontarget': 0.75,
    'shots_offtarget': 0.5,
    'shotsblocked': 0.5,
    'chances2score': 0.75,
    'drib_success': 0.75,
    'drib_unsuccess': 0.5,
    'keypasses': 1.25,
    'touches': 1,
    'passes_acc': 1.25,
    'passes_inacc': 0.75,
    'crosses_acc': 1,
    'crosses_inacc': 0.75,
    'lballs_acc': 1,
    'lballs_inacc': 0.75,
    'grduels_w': 0.75,
    'grduels_l': 0.75,
    'aerials_w': 0.5,
    'aerials_l': 0.5,
    'poss_lost': 1,
    'fouls': 0.75,
    'wasfouled': 0.75,
    'clearances': 0.5,
    'stop_shots': 0.5,
    'interceptions': 1,
    'tackles': 1,
    'ycards': 0.75,
    'rcards': 0.75,
    'dangmistakes': 1,
    'countattack': 0.75,
    'offsides': 0.5,
    'goals_ag_otb': 0.5,
    'goals_ag_itb': 0.25,
    'saves_itb': 0,
    'saves_otb': 0,
    'saved_pen': 0,
    'missed_penalties': 0.75,
    'owngoals': 0.5,
    'win': 1,
    'lost': 1,
    'minutesPlayed': 1,
    'game_duration': 0.5
}

pos_gk_weights = {
    'original_rating': 1,
    'goals': 0,
    'assists': 0,
    'shots_ontarget': 0,
    'shots_offtarget': 0,
    'shotsblocked': 0.75,
    'chances2score': 0,
    'drib_success': 0,
    'drib_unsuccess': 0,
    'keypasses': 0.5,
    'touches': 1,
    'passes_acc': 1,
    'passes_inacc': 0.75,
    'crosses_acc': 0,
    'crosses_inacc': 0,
    'lballs_acc': 1,
    'lballs_inacc': 0.5,
    'grduels_w': 0,
    'grduels_l': 0,
    'aerials_w': 0.75,
    'aerials_l': 0.75,
    'poss_lost': 0.75,
    'fouls': 0.5,
    'wasfouled': 0.5,
    'clearances': 1,
    'stop_shots': 1.5,
    'interceptions': 0.5,
    'tackles': 0,
    'ycards': 0.5,
    'rcards': 0.5,
    'dangmistakes': 1,
    'countattack': 0,
    'offsides': 0,
    'goals_ag_otb': 1.25,
    'goals_ag_itb': 1.5,
    'saves_itb': 1.5,
    'saves_otb': 1.5,
    'saved_pen': 1.5,
    'missed_penalties': 0,
    'owngoals': 0.5,
    'win': 1,
    'lost': 1,
    'minutesPlayed': 1,
    'game_duration': 0.5
}


pos_sub_weights = {
    'original_rating': 1,
    'goals': 1,
    'assists': 1,
    'shots_ontarget': 1,
    'shots_offtarget': 1,
    'shotsblocked': 1,
    'chances2score': 1,
    'drib_success': 1,
    'drib_unsuccess': 1,
    'keypasses': 1,
    'touches': 1,
    'passes_acc': 1,
    'passes_inacc': 1,
    'crosses_acc': 1,
    'crosses_inacc': 1,
    'lballs_acc': 1,
    'lballs_inacc': 1,
    'grduels_w': 1,
    'grduels_l': 1,
    'aerials_w': 1,
    'aerials_l': 1,
    'poss_lost': 1,
    'fouls': 1,
    'wasfouled': 1,
    'clearances': 11,
    'stop_shots': 1,
    'interceptions': 1,
    'tackles': 1,
    'ycards': 1,
    'rcards': 1,
    'dangmistakes': 1,
    'countattack': 1,
    'offsides': 1,
    'goals_ag_otb': 1,
    'goals_ag_itb': 1,
    'saves_itb': 0,
    'saves_otb': 0,
    'saved_pen': 0,
    'missed_penalties': 1,
    'owngoals': 1,
    'win': 1,
    'lost': 1,
    'minutesPlayed': 1,
    'game_duration': 1
}
pos_df_weights = {
    'original_rating': 1,
    'goals': 0.5,
    'assists': 0.5,
    'shots_ontarget': 0.25,
    'shots_offtarget': 0.25,
    'shotsblocked': 1.25,
    'chances2score': 0.25,
    'drib_success': 0.25,
    'drib_unsuccess': 0.25,
    'keypasses': 0.5,
    'touches': 1,
    'passes_acc': 1,
    'passes_inacc': 0.75,
    'crosses_acc': 0.5,
    'crosses_inacc': 0.5,
    'lballs_acc': 1,
    'lballs_inacc': 0.5,
    'grduels_w': 1.25,
    'grduels_l': 1.25,
    'aerials_w': 1.25,
    'aerials_l': 1.25,
    'poss_lost': 1,
    'fouls': 0.75,
    'wasfouled': 0.75,
    'clearances': 1.5,
    'stop_shots': 1,
    'interceptions': 1.25,
    'tackles': 1.25,
    'ycards': 0.75,
    'rcards': 0.75,
    'dangmistakes': 0.75,
    'countattack': 0.75,
    'offsides': 0.25,
    'goals_ag_otb': 0.75,
    'goals_ag_itb': 1,
    'saves_itb': 0,
    'saves_otb': 0,
    'saved_pen': 0,
    'missed_penalties': 0.5,
    'owngoals': 0.75,
    'win': 1,
    'lost': 1,
    'minutesPlayed': 1,
    'game_duration': 0.5
}

data = data.apply(lambda row: apply_positional_weighting(row, pos_fw_weights.keys(), 'pos_FW', pos_fw_weights), axis=1)
data = data.apply(lambda row: apply_positional_weighting(row, pos_df_weights.keys(), 'pos_DF', pos_df_weights), axis=1)
data = data.apply(lambda row: apply_positional_weighting(row, pos_mf_weights.keys(), 'pos_MF', pos_mf_weights), axis=1)
data = data.apply(lambda row: apply_positional_weighting(row, pos_gk_weights.keys(), 'pos_GK', pos_gk_weights), axis=1)
data = data.apply(lambda row: apply_positional_weighting(row, pos_sub_weights.keys(), 'pos_Sub', pos_sub_weights), axis=1)

features = [
    'goals',
    'assists',
    'shots_ontarget',
    'shots_offtarget',
    'shotsblocked',
    'chances2score',
    'drib_success',
    'drib_unsuccess',
    'keypasses',
    'touches',
    'passes_acc',
    'passes_inacc',
    'crosses_acc',
    'crosses_inacc',
    'lballs_acc',
    'lballs_inacc',
    'grduels_w',
    'grduels_l',
    'aerials_w',
    'aerials_l',
    'poss_lost',
    'fouls',
    'wasfouled',
    'clearances',
    'stop_shots',
    'interceptions',
    'tackles',
    'ycards',
    'rcards',
    'dangmistakes',
    'countattack',
    'offsides',
    'goals_ag_otb',
    'goals_ag_itb',
    'saves_itb',
    'saves_otb',
    'saved_pen',
    'missed_penalties',
    'owngoals',
    'win',
    'lost',
    'minutesPlayed',
    'game_duration'
]


target = 'original_rating'

data_revised = data[features + [target]].dropna()

X_revised = data_revised[features]
y_revised = data_revised[target]


scaler_revised = StandardScaler()
X_scaled_revised = scaler_revised.fit_transform(X_revised)


X_train_revised, X_test_revised, y_train_revised, y_test_revised = train_test_split(X_scaled_revised, y_revised, test_size=0.2, random_state=42)


random_forest_revised = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_revised.fit(X_train_revised, y_train_revised)


y_pred_revised = random_forest_revised.predict(X_test_revised)


mse_revised = mean_squared_error(y_test_revised, y_pred_revised)
mae_revised = mean_absolute_error(y_test_revised, y_pred_revised)
r2_revised = r2_score(y_test_revised, y_pred_revised)

print("Mean Squared Error:", mse_revised)
print("Mean Absolute Error:", mae_revised)
print("R-squared Score:", r2_revised)


plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Ratings', fontsize=14)
plt.ylabel('Predicted Ratings', fontsize=14)
plt.title('Actual vs Predicted Ratings', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()


player_name = 'Kevin Kampl'
week = 5

predicted_rating = predict_player_rating_up_to_week(
    file_path, player_name, week, random_forest_revised, scaler_revised, features
)
print(f'Predicted Rating for {player_name} in Week {week}: {predicted_rating}')