import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
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

    for i in range(len(all_players)):
        if all_players[i]['full_name'].upper() == name.upper():
            return all_players[i]['id']
    return ''


# CHANGE THIS TO ANY PLAYER'S NAME
player_name = "Desmond Bane"

# CHANGE THIS TO ANY PLAYER'S STATS
betting_stat = "AST"

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
df = df.drop(['SEASON_ID', 'Player_ID', 'VIDEO_AVAILABLE'], axis=1)
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
plt.show()


print(f"\n\n{betting_stat} average for {player_name}: {mean}")
print(f"{betting_stat} poly_1d prediction for {player_name}: {poly1d_prediction}")
# print(f"\n{betting_stat} per game for {player_name}: {stats_values}")
x = x.reshape(-1, 1)


# work on hyperparameter tuning
model = LinearRegression().fit(x, stats_values)
predictions = model.predict(x)
# print(f"\nMinimal Predictions on {betting_stat} for {player_name}: {predictions}")


d = {f"Actual {betting_stat}": stats_values, f"Predicted {betting_stat}": predictions}
df = pd.DataFrame(d)
print('\n', df.to_string())

























