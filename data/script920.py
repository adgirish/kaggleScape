
# coding: utf-8

# # Elo ratings based on regular-season games

# This notebook implements Elo ratings for NCAA regular-season games using the same formula as FiveThirtyEight's NBA Elo ratings. My resources for this were:
# 
# - https://en.wikipedia.org/wiki/Elo_rating_system
# - https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
# - https://github.com/fivethirtyeight/nfl-elo-game/blob/master/forecast.py
# 
# (The last link above is for 538's NFL Elos (not NBA), but it was useful for a code example of the approach. )
# 
# The idea here is to get another feature to be plugged in (alongside seeds, etc.) when predicting tournament games.

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


# The following parameter `K` affects how quickly the Elo adjusts to new information. Here I'm just using the value that 538 found most appropriate for the NBA -- I haven't done any analysis around whether this value is also the best in terms of college basketball.
# 
# I also use the same home-court advantage as 538: the host team gets an extra 100 points added to their Elo.

# In[2]:


K = 20.
HOME_ADVANTAGE = 100.


# In[3]:


rs = pd.read_csv("../input/RegularSeasonCompactResults.csv")
rs.head(3)


# In[4]:


team_ids = set(rs.WTeamID).union(set(rs.LTeamID))
len(team_ids)


# I'm going to initialise all teams with a rating of 1500. There are two differences here with the 538 approach:
# 
# - New entrants (when and where there are any) will start at the average 1500 Elo rather than a lower rating probably more appropriate for a new team.
# - There is no reversion to the mean between seasons. Each team's Elo starts exactly where it left off the previous season.  My justification here is that we only care about the end-of-season rating in terms of making predictions on the NCAA tournament, so even if ratings are a little off at first, they have the entire regular season to converge to something more appropriate.

# In[5]:


# This dictionary will be used as a lookup for current
# scores while the algorithm is iterating through each game
elo_dict = dict(zip(list(team_ids), [1500] * len(team_ids)))


# In[6]:


# Elo updates will be scaled based on the margin of victory
rs['margin'] = rs.WScore - rs.LScore


# The three functions below contain the meat of the Elo calculation:

# In[7]:


def elo_pred(elo1, elo2):
    return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

def expected_margin(elo_diff):
    return((7.5 + 0.006 * elo_diff))

def elo_update(w_elo, l_elo, margin):
    elo_diff = w_elo - l_elo
    pred = elo_pred(w_elo, l_elo)
    mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
    update = K * mult * (1 - pred)
    return(pred, update)


# In[ ]:


# I'm going to iterate over the games dataframe using 
# index numbers, so want to check that nothing is out
# of order before I do that.
assert np.all(rs.index.values == np.array(range(rs.shape[0]))), "Index is out of order."


# In[8]:


preds = []
w_elo = []
l_elo = []

# Loop over all rows of the games dataframe
for row in rs.itertuples():
    
    # Get key data from current row
    w = row.WTeamID
    l = row.LTeamID
    margin = row.margin
    wloc = row.WLoc
    
    # Does either team get a home-court advantage?
    w_ad, l_ad, = 0., 0.
    if wloc == "H":
        w_ad += HOME_ADVANTAGE
    elif wloc == "A":
        l_ad += HOME_ADVANTAGE
    
    # Get elo updates as a result of the game
    pred, update = elo_update(elo_dict[w] + w_ad,
                              elo_dict[l] + l_ad, 
                              margin)
    elo_dict[w] += update
    elo_dict[l] -= update
    
    # Save prediction and new Elos for each round
    preds.append(pred)
    w_elo.append(elo_dict[w])
    l_elo.append(elo_dict[l])


# In[9]:


rs['w_elo'] = w_elo
rs['l_elo'] = l_elo


# Let's take a look at the last few games in the games dataframe to check that the Elo ratings look reasonable.

# In[10]:


rs.tail(10)


# Looks OK. How well do they generally predict games? Since all of the Elo predictions calculated above have a true outcome of 1, it's really simple to check what the log loss would be on these 150k games:

# In[11]:


np.mean(-np.log(preds))


# (This is a pretty rough measure, because this is looking only at regular-season games, which is not really what we're ultimately interested in predicting.)

# Final step: for each team, pull out the final Elo rating at the end of each regular season. This is a bit annoying because the team ID could be in either the winner or loser column for their last game of the season..

# In[12]:


def final_elo_per_season(df, team_id):
    d = df.copy()
    d = d.loc[(d.WTeamID == team_id) | (d.LTeamID == team_id), :]
    d.sort_values(['Season', 'DayNum'], inplace=True)
    d.drop_duplicates(['Season'], keep='last', inplace=True)
    w_mask = d.WTeamID == team_id
    l_mask = d.LTeamID == team_id
    d['season_elo'] = None
    d.loc[w_mask, 'season_elo'] = d.loc[w_mask, 'w_elo']
    d.loc[l_mask, 'season_elo'] = d.loc[l_mask, 'l_elo']
    out = pd.DataFrame({
        'team_id': team_id,
        'season': d.Season,
        'season_elo': d.season_elo
    })
    return(out)


# In[13]:


df_list = [final_elo_per_season(rs, id) for id in team_ids]
season_elos = pd.concat(df_list)


# In[14]:


season_elos.sample(10)


# In[15]:


season_elos.to_csv("season_elos.csv", index=None)

