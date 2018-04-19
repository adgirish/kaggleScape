
# coding: utf-8

# # Preprocessing code to join all the tables 

# In[ ]:


#Credits to https://www.kaggle.com/robikscube/eda-of-women-s-ncaa-bracket-data-in-progress by Robert Mulla

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Load all the data as pandas Dataframes

# In[ ]:


cities = pd.read_csv('../input/WCities.csv')
gamecities = pd.read_csv('../input/WGameCities.csv')
tourneycompactresults = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
tourneyseeds = pd.read_csv('../input/WNCAATourneySeeds.csv')
tourneyslots = pd.read_csv('../input/WNCAATourneySlots.csv')
regseasoncompactresults = pd.read_csv('../input/WRegularSeasonCompactResults.csv')
seasons = pd.read_csv('../input/WSeasons.csv')
teamspellings = pd.read_csv('../input/WTeamSpellings.csv', engine='python')
teams = pd.read_csv('../input/WTeams.csv')


# In[ ]:


print(tourneycompactresults.shape)
print(regseasoncompactresults.shape)


# In[ ]:


# Convert Tourney Seed to a Number
tourneyseeds['SeedNumber'] = tourneyseeds['Seed'].apply(lambda x: int(x[-2:]))


# **Merge all the tables:**

# In[ ]:


gamecities = gamecities.merge(cities,how='left',on='CityID')

tourneycompactresults['WSeed'] = tourneycompactresults[['Season','WTeamID']].merge(tourneyseeds,left_on = ['Season','WTeamID'],right_on = ['Season','TeamID'],how='left')[['SeedNumber']]
tourneycompactresults['LSeed'] = tourneycompactresults[['Season','LTeamID']].merge(tourneyseeds,left_on = ['Season','LTeamID'],right_on = ['Season','TeamID'],how='left')[['SeedNumber']]

tourneycompactresults = tourneycompactresults.merge(gamecities,how='left',on=['Season','DayNum','WTeamID','LTeamID'])

regseasoncompactresults['WSeed'] = regseasoncompactresults[['Season','WTeamID']].merge(tourneyseeds,left_on = ['Season','WTeamID'],right_on = ['Season','TeamID'],how='left')[['SeedNumber']]
regseasoncompactresults['LSeed'] = regseasoncompactresults[['Season','LTeamID']].merge(tourneyseeds,left_on = ['Season','LTeamID'],right_on = ['Season','TeamID'],how='left')[['SeedNumber']]

regseasoncompactresults = regseasoncompactresults.merge(gamecities,how='left',on=['Season','DayNum','WTeamID','LTeamID'])


# In[ ]:


regseasoncompactresults = regseasoncompactresults.merge(seasons,how='left',on='Season')
tourneycompactresults = tourneycompactresults.merge(seasons,how='left',on='Season')


# In[ ]:


regseasoncompactresults['WTeamName'] = regseasoncompactresults[['WTeamID']].merge(teams,how='left',left_on='WTeamID',right_on='TeamID')[['TeamName']]
regseasoncompactresults['LTeamName'] = regseasoncompactresults[['LTeamID']].merge(teams,how='left',left_on='LTeamID',right_on='TeamID')[['TeamName']]

tourneycompactresults['WTeamName'] = tourneycompactresults[['WTeamID']].merge(teams,how='left',left_on='WTeamID',right_on='TeamID')[['TeamName']]
tourneycompactresults['LTeamName'] = tourneycompactresults[['LTeamID']].merge(teams,how='left',left_on='LTeamID',right_on='TeamID')[['TeamName']]


# In[ ]:


print(tourneycompactresults.shape)
print(regseasoncompactresults.shape)


# We have merged all the tables now, we can start working on exploring the data without having to worry about joining multiple table.
# 
# The data is present in regseasoncompactresults and tourneycompactresults tables.

# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(12,4))
ax1 = sns.countplot(x=tourneycompactresults['WSeed'],ax=ax1)
ax1.set_title("Seed of Winners - Tourney")
ax2 = sns.countplot(x=tourneycompactresults['LSeed'],ax=ax2);
ax2.set_title("Seed of Losers - Tourney")

plt.legend();


# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(12,4))
ax1 = sns.countplot(x=regseasoncompactresults['WSeed'],ax=ax1)
ax1.set_title("Seed of Winners - Reg Season")
ax2 = sns.countplot(x=regseasoncompactresults['LSeed'],ax=ax2)
ax2.set_title("Seed of Losers - Reg Season")

plt.legend();


# Based on the visualizations, the top seeds won most of the matches and lost fewer matches relatively. So its safe to assume that this is a key feature for our model both in tourneys and regular seasons.

# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(12,4))
ax1 = sns.countplot(x=regseasoncompactresults['WLoc'],ax=ax1)
ax1.set_title("Reg Season")
ax1.set_xlabel('Winning location')
ax2 = sns.countplot(x=tourneycompactresults['WLoc'],ax=ax2)
ax2.set_title("Tourneys")
ax2.set_xlabel('Winning location')

plt.legend();


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=2, figsize=(12,10))
a = sns.distplot(tourneycompactresults['WScore'],label='Winning Score',ax=ax1)
a = sns.distplot(tourneycompactresults['LScore'],ax=a,label='Losing Score')
a.set_xlabel("Score distribution-Tourney Results")

b = sns.distplot(regseasoncompactresults['WScore'],label='Winning Score',ax=ax2)
b = sns.distplot(regseasoncompactresults['LScore'],ax=b,label='Losing Score')
b.set_xlabel("Score distribution-RegSeason Results")

plt.legend();


# Based on the distribution, the score tends to be higher in Tourney's than in regular seasons.
# 
# Clearly, the score of winners is higher than the score of losers both in tourneys and Regular seasons.

# In[ ]:


# Calculate the Average Team Seed
averageseed = tourneyseeds.groupby(['TeamID']).agg(np.mean).sort_values('SeedNumber')
averageseed = averageseed.merge(teams, left_index=True, right_on='TeamID') #Add Teamnname
averageseed.head(20).plot(x='TeamName',
                          y='SeedNumber',
                          kind='bar',
                          figsize=(15,5),
                         title='Top 20 Average Tournament Seed');

