
# coding: utf-8

# 
# There are several ways to measure football players performance.
# I am trying an approach taking into accounts:
# - the predicted difficulty of the matches they play (according to betting odds)
# - the actual results of the matches they play
# 
# So here is the story. Imagine that, **every time a football player takes part in a match, he bets 100€ on a win of his team**. How much money will he make (or lose) at the end of the season? And thus, who is the best player of his team in terms of winnings?
# 
# For comparison with the overall results of his team, I also calculate the winnings for a 100€ bet on every match of the team.
# 
# NB: I consider the 2015/2016 season in countries with the highest attendance (Germany, England, Spain, Italy and France). The odds used for the calculations are those from bet365.
# 
# Let's begin with the results for the 15 most popular teams (according to number of followers on Twitter).

# In[ ]:


import sqlite3 as lite
database = '../input/database.sqlite'
conn = lite.connect(database)

# Features to load
toload = ['season', 'stage', 'country_id']
toload += [x+'_'+y for x in['home', 'away'] for y in ['team_api_id', 'team_goal']]
toload += [x+'_player_'+str(y) for x in ['home', 'away'] for y in range(1, 12)]
toload += ['B365'+x for x in ['H', 'D', 'A']]

# 2015/2016 season
import pandas as pd
query = 'SELECT '+(', '.join(toload))+' FROM Match'
dfmatch = pd.read_sql(query, conn)
dfmatch = dfmatch[dfmatch['season']=='2015/2016']

# Use team names instead of ids
query = 'SELECT team_api_id, team_long_name FROM Team'
dfteam = pd.read_sql(query, conn)
seteam = pd.Series(data=dfteam['team_long_name'].values, index=dfteam['team_api_id'].values)
dfmatch['home_team_name'] = dfmatch['home_team_api_id'].map(seteam)
dfmatch['away_team_name'] = dfmatch['away_team_api_id'].map(seteam)

# Use country names instead of ids
query = 'SELECT id, name FROM Country'
dfcountry = pd.read_sql(query, conn)
secountry = pd.Series(data=dfcountry['name'].values, index=dfcountry['id'].values)
dfmatch['country'] = dfmatch['country_id'].map(secountry)

# Countries with highest attendance
countries = ['England', 'Spain', 'Germany', 'Italy', 'France']
dfmatch = dfmatch[dfmatch['country'].isin(countries)]

# Use player names instead of ids
query = 'SELECT player_api_id, player_name FROM Player'
dfplayer = pd.read_sql(query, conn)
seplayer = pd.Series(data=dfplayer['player_name'].values, index=dfplayer['player_api_id'].values)
for z in [x+'_player_'+str(y) for x in ['home', 'away'] for y in range(1, 12)]:
    dfmatch[z+'_name'] = dfmatch[z].map(seplayer)

# dict containing all the results
dict_results = dict()
from collections import defaultdict
for t in dfmatch['home_team_name'].unique():
    dict_results[t] = defaultdict(list)

# Winnings calculation
import math
for _, row in dfmatch.iterrows():
    # Some bets are missing, so we skip these matches
    if math.isnan(row['B365D']):
        continue
    home_gain = -100
    away_gain = -100
    if row['home_team_goal']>row['away_team_goal']:
        home_gain += row['B365H']*100
    elif row['home_team_goal']<row['away_team_goal']:
        away_gain += row['B365A']*100
    stage = row['stage']
    home_team = row['home_team_name']
    dict_results[home_team][row['home_team_name']].append([stage, home_gain])
    for z in ['home_player_'+str(y)+'_name' for y in range(1, 12)]:
        dict_results[home_team][row[z]].append([stage, home_gain])
    away_team = row['away_team_name']
    dict_results[away_team][row['away_team_name']].append([stage, away_gain])
    for z in ['away_player_'+str(y)+'_name' for y in range(1, 12)]:
        dict_results[away_team][row[z]].append([stage, away_gain])

# Keeping the results of each team and its player with biggest winnings
res_teams = []
res_bestp = []
relevant_prop_match = 0.5
for t in dfmatch['home_team_name'].unique():
    best = ['', -100000]
    for p in dict_results[t]:
        avg = sum([x[1] for x in dict_results[t][p]])
        if p==t:
            res_teams.append([t, int(avg)])
            avgteam = int(avg)
            continue
        if (avg>best[1]) and (len(dict_results[t][p])>len(dict_results[t][t])*relevant_prop_match):
            best = [p, avg]
    res_bestp.append([best[0], int(best[1]), t, avgteam, len(dict_results[t][best[0]])])

# Popular teams
popular_teams = ['Real Madrid', 'Barcelona', 'Manchester United', 'Arsenal', 'Chelsea', 'Liverpool', 'AC Milan', 'Paris Saint Germain', 'Manchester City', 'Juventus', 'Bayern Munich', 'Atletico Madrid', 'Borussia Dortmund', 'Marseille', 'Tottenham']
res_pop_teams = [x for x in res_teams if x[0] in popular_teams]
res_pop_bestp = [x for x in res_bestp if x[2] in popular_teams]

# Plotting the results
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (6, 24)
ax1 = plt.figure().add_subplot(111)
x_axis = [x[1] for x in res_pop_teams]
ax1.barh(range(45, 0, -3), x_axis, height=1, color='#4163f5')
x_axis = [x[1] for x in res_pop_bestp]
ax1.barh(range(44, 0, -3), x_axis, height=1, color='#49e419')
y_axis_labels = [x[0] for x in res_pop_bestp]+[x[0] for x in res_pop_teams]
y_range = np.concatenate((np.arange(44.5, 0.5, -3), np.arange(45.5, 0.5, -3)))
ax1.set_yticks(y_range)
ax1.set_yticklabels(y_axis_labels)
ax1.tick_params(axis='y', labelsize=18)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(y_range)
def str_euro(n):
    if n>0:
        return '+'+str(n)+'€'
    return str(n)+'€'
y_axis_labels = [str_euro(x[1]) for x in res_pop_bestp]+[str_euro(x[1]) for x in res_pop_teams]
ax2.set_yticklabels(y_axis_labels)
ax2.tick_params(axis='y', labelsize=18)
ax2.set_ylabel('Winnings', fontsize=16, labelpad=30)
ax1.set_ylabel('Team / Player', fontsize=16, labelpad=30)
plt.title('2015/2016\n~ Popular teams ~\nWinnings for a 100€ bet on every match:\n- of a team\n- of the player with best results in the team', fontsize=20)
plt.show()


# NB: I only considered players who took part in at least the half of the matches of their team. Otherwise, we would have seen in some cases better results for players who played only a couple of matches all won by their team, which could not be statistically relevant.
# 
# As we could imagine, this is not necessarily a good idea to bet on the best and famous teams. Their odds are often quite low.
# 
# It is also interesting to see that some players finish with positive winnings whereas their team does not.
# To investigate this a little further, here are the biggest differences observed for a player's winnings compared to his team's winnings.
# 

# In[ ]:


# Keep the 10 biggest differences
diff_bestp_team = sorted(res_bestp, key=lambda h:h[1]-h[3], reverse=True)[:10]

# Plotting the ranking
plt.clf()
plt.rcParams["figure.figsize"] = (6, 21);
ax1 = plt.figure().add_subplot(111)
x_axis = [x[3] for x in diff_bestp_team]
ax1.barh(range(40, 0, -4), x_axis, height=1, color='#4163f5')
x_axis = [x[1] for x in diff_bestp_team]
ax1.barh(range(39, 0, -4), x_axis, height=1, color='#49e419')
y_axis_labels = [x[0] for x in diff_bestp_team]+[x[2] for x in diff_bestp_team]
y_range = np.concatenate((np.arange(39.5, 0.5, -4), np.arange(40.5, 0.5, -4)))
ax1.set_yticks(y_range)
ax1.set_yticklabels(y_axis_labels)
ax1.tick_params(axis='y', labelsize=18)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
y_range = np.concatenate((np.arange(38.5, 0.5, -4), y_range))
ax2.set_yticks(y_range)
y_axis_labels = ['diff = '+str_euro(x[1]-x[3]) for x in diff_bestp_team]+[str_euro(x[1]) for x in diff_bestp_team]+[str_euro(x[3]) for x in diff_bestp_team]
ax2.set_yticklabels(y_axis_labels)
ax2.tick_params(axis='y', labelsize=18)
ax2.set_ylabel('Winnings and differences', fontsize=16, labelpad=30)
ax1.set_ylabel('Team / Player', fontsize=16, labelpad=30)
plt.title('2015/2016\n~ Top 10 differences ~\nTeams with player having the biggest winnings\ncompared to the team\'s winnings', fontsize=20)
plt.show()


# And now maybe the most interesting ranking: the players with the biggest winnings!
# 
# (Note that, if a team would have several players appearing in this ranking, I only keep the player with the biggest winnings in the team.)
# 

# In[ ]:


# Keeping the top 10 winnings
top_bestp = sorted(res_bestp, key=lambda h:h[1], reverse=True)[:10]

# Plotting the ranking
plt.clf()
plt.rcParams["figure.figsize"] = (6, 16);
ax1 = plt.figure().add_subplot(111)
x_axis = [x[3] for x in top_bestp]
ax1.barh(range(30, 0, -3), x_axis, height=1, color='#4163f5')
x_axis = [x[1] for x in top_bestp]
ax1.barh(range(29, 0, -3), x_axis, height=1, color='#49e419')
y_axis_labels = [x[0] for x in top_bestp]+[x[2] for x in top_bestp]
y_range = np.concatenate((np.arange(29.5, 0.5, -3), np.arange(30.5, 0.5, -3)))
ax1.set_yticks(y_range)
ax1.set_yticklabels(y_axis_labels)
ax1.tick_params(axis='y', labelsize=18)
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(y_range)
y_axis_labels = [str_euro(x[1]) for x in top_bestp]+[str_euro(x[3]) for x in top_bestp]
ax2.set_yticklabels(y_axis_labels)
ax2.tick_params(axis='y', labelsize=18)
ax2.set_ylabel('Winnings', fontsize=16, labelpad=30)
ax1.set_ylabel('Team / Player', fontsize=16, labelpad=30)
plt.title('2015/2016\n~ Top 10 winnings ~\nPlayers with the biggest winnings\nalong with their team\'s winnings', fontsize=20)
plt.show()


# You may be surprised by the excellent results of West Ham (well at least I was :) ). But looking more accurately, I have seen that they won difficult away matches like Arsenal (odd: 12.0), Liverpool (9.0) and Man City (11.0).
# 
# Though I really discourage you to bet 100€ on every match of a team or a player, I hope this was a pleasant and enriching reading!
# 
# :)
# 
# 
