
# coding: utf-8

# # EDA of the NCAA Women's Basketball Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot
import seaborn as sns
pyplot.style.use('ggplot')


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


# Convert Tourney Seed to a Number
tourneyseeds['SeedNumber'] = tourneyseeds['Seed'].apply(lambda x: int(x[-2:]))

# Credit much of the merge code to Teza (Thanks!)
# https://www.kaggle.com/tejasrinivas/preprocessing-code-to-join-all-the-tables-eda
tourneycompactresults['WSeed'] =     tourneycompactresults[['Season','WTeamID']].merge(tourneyseeds,
                                                      left_on = ['Season','WTeamID'],
                                                      right_on = ['Season','TeamID'],
                                                      how='left')[['SeedNumber']]
tourneycompactresults['LSeed'] =     tourneycompactresults[['Season','LTeamID']].merge(tourneyseeds,
                                                      left_on = ['Season','LTeamID'],
                                                      right_on = ['Season','TeamID'],
                                                      how='left')[['SeedNumber']]

tourneycompactresults =     tourneycompactresults.merge(gamecities,
                                how='left',
                                on=['Season','DayNum','WTeamID','LTeamID'])

regseasoncompactresults['WSeed'] =     regseasoncompactresults[['Season','WTeamID']].merge(tourneyseeds,
                                                        left_on = ['Season','WTeamID'],
                                                        right_on = ['Season','TeamID'],
                                                        how='left')[['SeedNumber']]
regseasoncompactresults['LSeed'] =     regseasoncompactresults[['Season','LTeamID']].merge(tourneyseeds,
                                                        left_on = ['Season','LTeamID'],
                                                        right_on = ['Season','TeamID'],
                                                        how='left')[['SeedNumber']]

regseasoncompactresults =     regseasoncompactresults.merge(gamecities,
                                  how='left',
                                  on=['Season',
                                      'DayNum',
                                      'WTeamID',
                                      'LTeamID'])

# Add Season Results
regseasoncompactresults = regseasoncompactresults.merge(seasons,
                                                        how='left',
                                                        on='Season')
tourneycompactresults = tourneycompactresults.merge(seasons,
                                                    how='left',
                                                    on='Season')

# Add Team Names
regseasoncompactresults['WTeamName'] =     regseasoncompactresults[['WTeamID']].merge(teams,
                                               how='left',
                                               left_on='WTeamID',
                                               right_on='TeamID')[['TeamName']]
regseasoncompactresults['LTeamName'] =     regseasoncompactresults[['LTeamID']].merge(teams,
                                               how='left',
                                               left_on='LTeamID',
                                               right_on='TeamID')[['TeamName']]

tourneycompactresults['WTeamName'] =     tourneycompactresults[['WTeamID']].merge(teams,
                                             how='left',
                                             left_on='WTeamID',
                                             right_on='TeamID')[['TeamName']]
tourneycompactresults['LTeamName'] =     tourneycompactresults[['LTeamID']].merge(teams,
                                             how='left',
                                             left_on='LTeamID',
                                             right_on='TeamID')[['TeamName']]
    
tourneycompactresults['ScoreDiff'] = tourneycompactresults['WScore'] - tourneycompactresults['LScore'] 


# # Start by Looking at Historic Tournament Seeds

# In[ ]:


# Calculate the Average Team Seed
averageseed = tourneyseeds.groupby(['TeamID']).agg(np.mean).sort_values('SeedNumber')
averageseed = averageseed.merge(teams, left_index=True, right_on='TeamID') #Add Teamnname
averageseed.head(20).plot(x='TeamName',
                          y='SeedNumber',
                          kind='bar',
                          figsize=(15,5),
                          title='Top 20 Average Tournament Seed',
                          rot=45)


# In[ ]:


# Pairplot of the Tourney Seed and Scores
sns.pairplot(tourneycompactresults[['WScore',
                                    'LScore',
                                    'ScoreDiff',
                                    'WSeed',
                                    'LSeed',
                                    'Season']], hue='Season')


# ## Regular Season Games of Tourney Teams

# In[ ]:


# Pairplot of Regular Season Games
# Only include teams who are both seeded in the tournament
regseason_in_tourney = regseasoncompactresults.dropna(subset=['WSeed','LSeed'])
sns.pairplot(data = regseason_in_tourney,
             vars=['WScore','LScore','WSeed','LSeed'],
             hue='WSeed')


# > ## Winning Vs. Losing Score Distributions

# In[ ]:


regseason2017 = regseasoncompactresults.loc[regseasoncompactresults['Season'] == 2017]


# In[ ]:


bins = np.linspace(0, 120, 61)
pyplot.figure(figsize=(15,5))
pyplot.title('Distribution of Winning and Losing Scores 2017')
pyplot.hist(regseason2017['WScore'], bins, alpha=0.5, label='Winning Score')
pyplot.hist(regseason2017['LScore'], bins, alpha=0.5, label='Losing Score')
pyplot.legend(loc='upper right')
pyplot.show()


# In[ ]:


bins = np.linspace(0, 120, 61)
pyplot.figure(figsize=(15,5))
pyplot.title('Distribution of Winning and Losing Scores All Years')
pyplot.hist(regseasoncompactresults['WScore'], bins, alpha=0.5, label='Winning Score')
pyplot.hist(regseasoncompactresults['LScore'], bins, alpha=0.5, label='Losing Score')
pyplot.legend(loc='upper right')
pyplot.show()


# > # Teams with the Most Wins and Losses since 1998

# In[ ]:


# Teams with the Most Losses
count_of_losses = regseasoncompactresults.groupby('LTeamID')['LTeamID'].agg('count')
count_of_losses = count_of_losses.sort_values(ascending=False)
team_loss_count = pd.DataFrame(count_of_losses).merge(teams, left_index=True, right_on='TeamID')[['TeamName','LTeamID']]
team_loss_count.rename(columns={'LTeamID':'Loss Count'}).head(10)

# These teams aren't super great at basketball


# In[ ]:


# Teams with the Most Wins
count_of_wins = regseasoncompactresults.groupby('WTeamID')['WTeamID'].agg('count')
count_of_wins = count_of_wins.sort_values(ascending=False)
team_wins_count = pd.DataFrame(count_of_wins).merge(teams, left_index=True, right_on='TeamID')[['TeamName','WTeamID']]
team_wins_count.rename(columns={'WTeamID':'Win Count'}).head(10)
# These teams are super good at basketball


# In[ ]:


winloss_since1998 = pd.merge(team_wins_count, team_loss_count, how='outer')


# In[ ]:


winloss_since1998.sort_values('WTeamID', ascending=False).head(20)


# ## The Winningest Teams of the 2017 Season

# In[ ]:


# Teams with the Most Wins
count_of_wins2017 = regseason2017.groupby('WTeamID')['WTeamID'].agg('count')
count_of_wins2017 = count_of_wins2017.sort_values(ascending=False)
team_wins_count2017 = pd.DataFrame(count_of_wins2017).merge(teams, left_index=True, right_on='TeamID')[['TeamName','WTeamID']]
team_wins_count2017 = team_wins_count2017.rename(columns={'WTeamID':'Win Count'})

count_of_losses2017 = regseason2017.groupby('LTeamID')['LTeamID'].agg('count')
count_of_losses2017 = count_of_losses2017.sort_values(ascending=False)
team_losses_count2017 = pd.DataFrame(count_of_losses2017).merge(teams, left_index=True, right_on='TeamID')[['TeamName','LTeamID']]
team_losses_count2017 = team_losses_count2017.rename(columns={'LTeamID':'Loss Count'})

winloss2017 = pd.merge(team_wins_count2017, team_losses_count2017, how='outer')
winloss2017.sort_values('Win Count', ascending=False).head(26)
winloss2017 = winloss2017.fillna(0)
winloss2017.head(15)


# # Distribution of Game Point Differential 
# Point Differential (Winning Points - Losing Team Points)

# In[ ]:


regseasoncompactresults['ScoreDiff'] = regseasoncompactresults['WScore'] - regseasoncompactresults['LScore'] 


# In[ ]:


regseasoncompactresults['ScoreDiff'].hist(bins=30)


# # Check out the Historic Tourney Results

# In[ ]:


# Point Differential when Home Team wins (favorite)
# The Home Team is the favortie
tourneycompactresults.loc[tourneycompactresults['WLoc'] == 'H']['ScoreDiff'].hist()


# In[ ]:


# Point Differential when Away Team wins (favorite)
# More close games
tourneycompactresults.loc[tourneycompactresults['WLoc'] == 'A']['ScoreDiff'].hist()


# In[ ]:


# Point Differential for neutral site games
tourneycompactresults.loc[tourneycompactresults['WLoc'] == 'N']['ScoreDiff'].hist()

