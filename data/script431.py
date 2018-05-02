
# coding: utf-8

# Goes over several of the tables available. Removes some missing hero_id from test_player.csv. 
# 
# Last updated december 2nd 2016

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ###match.csv

# In[ ]:


matches = pd.read_csv('../input/match.csv', index_col=0)
matches.head()


# `match_id`  has been reencoded from steams  `match_id` to save a little but of space.  tower_status and barracks_status are binarymasks indicating whether various structures have been destroyed. See the very bottom of https://wiki.teamfortress.com/wiki/WebAPI/GetMatchDetails#Tower_Status%22tower_status_dire%22:%202047, for details.  **radiant_win** would be good choice for a target of a binary classification task. 

# In[ ]:


# Each file needs to be removed after use.
del matches
gc.collect()


# ###players.csv
# 
# Contains statistics abouts players performance in individual matches. 

# In[ ]:


players = pd.read_csv('../input/players.csv')
players.iloc[:5,:15]


# `account_id` is useful if you want to look at multiple matches for the same player but be aware that many players choose to hide their account_id so it won't be available.

# In[ ]:


players['account_id'].value_counts()


# About 180000 thousand of the records out of a possible 500k are not available. The data was sampled using a time based split so this distribution should be representative of the rest of the available data. 

# In[ ]:


players.iloc[:5,20:30]


# xp_hero, means the amount of experience gained from killing other players.  The three main types of counts included are experience, gold, and user action

# In[ ]:


players.iloc[:5,40:55]


# I am pretty sure these are counts of user issued commands. The only reason they look like floats is because of how pandas handles nan. It is a float so if there is even one in a column all other numbers are converted. There may be a way of approximating actions per minute(not quite clicks per minute) from this. 

# In[ ]:


#cleanup
del players
gc.collect()


# ###player_time.csv
# contains xp, gold, and last hit totals for each player at one minute intervals

# In[ ]:


player_time = pd.read_csv('../input/player_time.csv')
player_time.head()


# In[ ]:


a_match = player_time.query('match_id == 1')


# In[ ]:


a_match.T


# Since each match lasts for a different amount of time storing them with time on the horizontal axis would
# take a lot of space.  The suffix for each variable indicates the value of the player_slot variable allowing this data to be combined with players.csv if desired.  

# In[ ]:


del player_time
gc.collect()


# ###teamfights.csv

# In[ ]:


teamfights = pd.read_csv('../input/teamfights.csv')
teamfights.head()


# `start`, `end` and `last_death` contain the time for those events. Each row contains very basic info about each team fight.  Time is in seconds. I was considering adding a specific column for the count of teamfights in a match. It would make getting the first teamfight for each match easier.

# In[ ]:


del teamfights
gc.collect()


# ###teamfights_players.csv
# More detailed information about each teamfight

# In[ ]:


teamfights_players = pd.read_csv('../input/teamfights_players.csv')
teamfights_players.head()


# Each row in the `teamfights.csv` corrosponds to ten rows in this file. I have marked this file and teamfights to be updated with specific variable indicating which teamfight in the match it belongs to this should make joining and working with these tables easier.

# In[ ]:


del teamfights_players
gc.collect()


# ###chat.csv
# contains chat logs for 50k matches

# In[ ]:


chat = pd.read_csv('../input/chat.csv')
chat.head()


# [Here is a Kernel][1] by mammykins showing how to make a wordcloud from the chat logs
# 
# 
#   [1]: https://www.kaggle.com/mammykins/d/devinanzelmo/dota-2-matches/dota-2-allchat-wordcloud

# ### test_players.csv and hero_names.csv
# removes heros with invalid hero_ids from the test_players.csv file

# In[ ]:


# problem with the hero_ids in test_player brought to my attention by @Dexter, thanks!
# hero_id is 0 in 15 cases. 

test_players = pd.read_csv('../input/test_player.csv')
hero_names = pd.read_csv('../input/hero_names.csv')


# In[ ]:


# As can been seen the number of zeros appearing here are much less then the least popular hero. These are very likely
# caused by processing problems, either in my data generation code, or in the data pulled from steam. 
test_players['hero_id'].value_counts().tail()


# In[ ]:


test_players.query('hero_id == 0')


# No pattern immediately jumps out in relationship to the missing hero IDs. Except maybe they are more common for Dire. The safest way to deal with this is probably to remove the the matches in which any hero_id is 0

# In[ ]:


# remove matches with any invalid hero_ids
# imputing hero_id, is likely possible but the data is not available online in this dataset

matches_with_zero_ids = test_players.query('hero_id == 0')['match_id'].values.tolist()
test_players = test_players.query('match_id != @matches_with_zero_ids')


# In[ ]:


# check that the invalid ids are removed
# This is now on my list of bugs to fix for next release. 
test_players['hero_id'].value_counts().tail()


# ###player_ratings.csv
# A possible way to measure skill rating when we don't have MMR data
# See this kernel for details on how this was calculated https://www.kaggle.com/devinanzelmo/d/devinanzelmo/dota-2-matches/dota-2-skill-rating-with-trueskill

# In[ ]:


# player_ratings.csv contains trueskill ratings for players in the match, and test data.
# True Skill is a rating method somewhat like MMR, and can be used to sort players by skill. 

player_ratings = pd.read_csv('../input/player_ratings.csv')


# In[ ]:


player_ratings.head()


# In addition to trueskill rating total_wins and total_matches have been included to allow for the calculation of winrate. See this kernel for details on how trueskill values were calculated

# In[ ]:


# Now create a list of player rankings by using the formula mu - 3*sigma
# This ranking formula penalizes players with fewer matches because there is more uncertainty

player_ratings['conservative_skill_estimate'] = player_ratings['trueskill_mu'] - 3*player_ratings['trueskill_sigma']


# In[ ]:


player_ratings.head()


# In[ ]:


player_ratings = player_ratings.sort_values(by='conservative_skill_estimate', ascending=False)


# In[ ]:


# negative account ids are players not appearing in other data available in this dataset.

player_ratings.head(10)


# In[ ]:


del player_ratings
gc.collect()


# ###match_outcomes.csv
# Useful for creating custom skill calculations. Contains results with account_ids for 900k matches occuring prior to the other available data. Data is mostly from patch 6.85 and some from 6.84

# In[ ]:


match_outcomes = pd.read_csv('../input/match_outcomes.csv')


# In[ ]:


# each match has data on two rows. the 'rad' tells whether the team is Radiant or not(1 is Radiant 0 is Dire)
# negative account ids are not in the other available data. account_id 0 is for anonymous players.
match_outcomes.head()


# In[ ]:


del match_outcomes
gc.collect()


# ### ability_upgrades.csv and ability_names.csv
# 
# ability_upgrades.csv contains the upgrade performed at each level for each player.
# ability_ids.csv links the ability ids to the english names of abilities.  

# In[ ]:


ability_upgrades = pd.read_csv('../input/ability_upgrades.csv')
ability_ids = pd.read_csv('../input/ability_ids.csv')


# In[ ]:


ability_ids.head()


# In[ ]:


ability_upgrades.head()


# In[ ]:


del ability_upgrades, ability_ids
gc.collect()


# ###purchase_log.csv and item_ids.csv
# 
# purchase_log.csv contains the time for each purchase.
# item_ids.csv contains numeric id's for items and the english names 

# In[ ]:


purchase_log = pd.read_csv('../input/purchase_log.csv')
item_ids = pd.read_csv('../input/item_ids.csv')


# In[ ]:


item_ids.head()


# In[ ]:


purchase_log.head()


# In[ ]:


del purchase_log, item_ids
gc.collect()

