import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, \
    Lasso, SGDClassifier, TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from nba_api.stats.static import players


# endpoints: https://github.com/swar/nba_api/tree/master/docs/nba_api/stats/endpoints

header_dict = {
    'User-Agent': 'Mozilla/5.0',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Referer': 'https://stats.nba.com',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'Host': 'stats.nba.com'
}


def get_playerID(name):
    all_players = players.get_players()

    for ind in range(len(all_players)):
        if all_players[ind]['full_name'].upper() == name.upper():
            return all_players[ind]['id']
    return ''


# CHANGE THIS TO ANY PLAYER'S NAME
player_name = "Fred VanVleet"

# CHANGE THIS TO ANY PLAYER'S STATS
betting_stat = "REB"

# CHANGE THIS TO PROJECTED BETTING STAT
projected_betting_stat = 4.5

player_id = get_playerID(player_name)
url = f"https://stats.nba.com/stats/playergamelog?PlayerID={player_id}&Season=2022-23" \
      f"&SeasonType=Regular+Season"
# print(url)

response = requests.request("GET", url, headers=header_dict)
json = response.json()
parameters = json['parameters']

stats_names = json['resultSets'][0]['headers']
game_stats = json['resultSets'][0]['rowSet']

df = pd.DataFrame(game_stats, columns=stats_names)
df = df.drop(['SEASON_ID', 'Player_ID', 'VIDEO_AVAILABLE', "Game_ID", "GAME_DATE"], axis=1)
df.loc[df['WL'] == 'L', 'WL'] = 0  # 0 for loss
df.loc[df['WL'] == 'W', 'WL'] = 1  # 1 for win
df['OVER_UNDER'] = df[betting_stat].gt(projected_betting_stat).astype(int)
print(df.to_string())


stats_values = df[betting_stat].values[::-1]
x = np.arange(0, len(stats_values))
y = np.arange(0, max(stats_values)+1)
mean = np.average(stats_values)


plt.rcParams["figure.figsize"] = [15, 10]
plt.xlabel('Game Number (Index)')
plt.ylabel(betting_stat)
plt.title(f"{betting_stat} per game for {player_name}")
plt.xticks(x)
plt.yticks(y)

coef = np.polyfit(x, stats_values, 1)
poly1d_fn = np.poly1d(coef)

poly1d_prediction = poly1d_fn(max(x)+1)
t = f"Projected: {round(poly1d_prediction, 2)}"
plt.text(max(x)+0.5, poly1d_prediction, t)

t = f"Average: {round(mean, 2)}"
plt.text(max(x)+1.25, mean, t)

plt.plot(x, stats_values, 'o', poly1d_fn(x), '--k')
plt.axhline(y=mean, color="red")
# plt.show()


print(f"\n\nProjecting predictions on over/under {projected_betting_stat} {betting_stat} for {player_name}...")
print(f"{betting_stat} average: {mean}")
print(f"{betting_stat} poly_1d prediction: {poly1d_prediction}")
# print(f"\n{betting_stat} per game for {player_name}: {stats_values}")


# Begin Predictive Modeling (work on hyperparameter tuning)
X = []
for i in range(0, len(df)):
    X.append([i, stats_values[i]])

Y = stats_values
threshold = 0.1
labels = [projected_betting_stat-threshold, projected_betting_stat+threshold]

# Bayesian Ridge Regression
model = BayesianRidge()
model.fit(X, Y)
prediction = model.predict([labels])
score = model.score(X, Y)
print(f"\nBayesian Ridge Regression prediction: {prediction}")
print(f"Omega coefficient for predicting under {projected_betting_stat} {betting_stat}: {model.coef_[0]:.20f}")
print(f"Omega coefficient for predicting over {projected_betting_stat} {betting_stat}: {model.coef_[1]:.20f}")
print(f"Accuracy score: {score}")


# Linear Regression
model = LinearRegression()
model.fit(X, Y)
prediction = model.predict([labels])
score = model.score(X, Y)
print(f"\nLinear Regression predictions: {prediction}")
print(f"Omega coefficient for predicting under {projected_betting_stat} {betting_stat}: {model.coef_[0]:.20f}")
print(f"Omega coefficient for predicting over {projected_betting_stat} {betting_stat}: {model.coef_[1]:.20f}")
print(f"Accuracy score: {score}")


# Lasso
model = Lasso(alpha=0.1)
model.fit(X, Y)
prediction = model.predict([labels])
score = model.score(X, Y)
print(f"\nLasso predictions: {prediction}")
print(f"Accuracy score: {score}")


# Logistic Regression
model = LogisticRegression(penalty="l2", solver="liblinear")
model.fit(X, Y)
prediction = model.predict([labels])
score = model.score(X, Y)
print(f"\nLogistic Regression predictions: {prediction}")
print(f"Logistic Regression Classes: {model.classes_}")
print(f"Accuracy score: {score}")


# Stochastic Gradient Descent
model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
model.fit(X, Y)
prediction = model.predict([labels])
score = model.score(X, Y)
print(f"\nSGD Classifier predictions: {prediction}")
print(f"Accuracy score: {score}")


# Generalized Linear Model (Tweedie Regressor)
model = TweedieRegressor(power=1, alpha=0.5, link='log')
model.fit(X, Y)
prediction = model.predict([labels])
score = model.score(X, Y)
print(f"\nGLM (Tweedie Regressor) Classifier predictions: {prediction}")
print(f"Omega coefficient for predicting under {projected_betting_stat} {betting_stat}: {model.coef_[0]:.20f}")
print(f"Omega coefficient for predicting over {projected_betting_stat} {betting_stat}: {model.coef_[1]:.20f}")
print(f"Accuracy score: {score}")


# K Nearest Neighbors
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, Y)
prediction = model.predict([labels])
score = model.score(X, Y)
print(f"\nkNN predictions: {prediction}")
print(f"kNN Classes: {model.classes_}")
print(f"Accuracy score: {score}")
