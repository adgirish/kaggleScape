
# coding: utf-8

# *Last edit by David Lao - 2018/04/04*
# <br>
# <br>
# 
# ![](https://metrouk2.files.wordpress.com/2017/09/fifa-18-ultimate-team-fut-live.jpg?w=748&h=395&crop=1)
# 
# # FIFA Analytics - Find the Best Squad through Data Analysis
# 
# Greeting from Hong Kong! This is an analysis regarding the game FIFA 18, focusing on finding the best squad. The dataset contains all FIFA 18 players **~ 18K players** and **70+** attributes.
# 
# ## Table of Content
# 
# * Data manipulation
# * Players distribution geographically on various measures (*Edit on 2017/10/10*)
# * Squad of best 11 players
# * Age, Potential rating, current rating study of all players
# * Current rating, potential rating and corresponding team value by club
# * Current and future best squad by country (*Edit on 2017/09/30*)
# * Introducing time effect for future rating (*Edit on 2017/10/26*)
# * Player rating vs Value - Regression (*Edit on 2017/10/04*)
# 
# 
# 
# <br>
# Appreciate if you can **Upvote** if this notebook is helpful to you in some ways!
# 

# ## Data manipulation

# First we have some library to load:

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import matplotlib.cm as cm
import re
sns.set_style("darkgrid")
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures



# Then we load the data in:

# In[ ]:


df = pd.read_csv('../input/CompleteDataset.csv')
df.columns


# For simplicity of analysis, I only pull in data I am interested in:

# In[ ]:


df = df[['Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Preferred Positions']]
df.head(10)


# The next step is to manipulate the data for our needs:

# In[ ]:


# get remaining potential
df['Remaining Potential'] = df['Potential'] - df['Overall']


# get only one preferred position (first only)
df['Preferred Position'] = df['Preferred Positions'].str.split().str[0]

# convert K to M
df['Unit'] = df['Value'].str[-1]
df['Value (M)'] = np.where(df['Unit'] == '0', 0, df['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
df['Value (M)'] = df['Value (M)'].astype(float)
df['Value (M)'] = np.where(df['Unit'] == 'M', df['Value (M)'], df['Value (M)']/1000)
df = df.drop('Unit', 1)

df.head(10)


# ## Players distribution geographically on various measures

# Geographical representation of all players distribution:

# In[ ]:


def plot_geo(by_column, measure, sort_column, chart_title, min_rating = 0):
   df_g = df.copy()
   df_g = df_g[df_g['Overall']>min_rating]
   df_geo = df_g.groupby(['Nationality']).agg({by_column: measure})
   df_geo = pd.DataFrame(data = df_geo)
   df_geo = df_geo.rename(columns={by_column: 'Measurement'})
   df_geo['text'] = ''

   df_geo_player = df[['Nationality','Name', sort_column]].groupby(['Nationality']).head(3)
   df_geo_player = df_geo_player.sort_values(['Nationality', sort_column], ascending=[True, False])
   df_geo_player['Name_text'] = df_geo_player['Name'] + ' (' + df_geo_player[sort_column].map(str) + ')'

   for index, row in df_geo.iterrows():
       df_geo['text'].loc[index] = '<br>'.join(df_geo_player[df_geo_player['Nationality'] == index]['Name_text'].values)

   df_geo.rename(index={'England': 'United Kingdom'}, inplace = True)

   data = dict(type='choropleth',
   locations = df_geo.index,
   locationmode = 'country names', z = df_geo['Measurement'],
   text = df_geo['text'], colorbar = {'title':'Scale'},
   colorscale = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],
               [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']], 
   reversescale = False)

   layout = dict(title = chart_title,
   geo = dict(showframe = True, showcoastlines = False, projection={'type':'Mercator'}))

   choromap = go.Figure(data = [data], layout = layout)
   iplot(choromap, validate=False)

plot_geo('Nationality', 'count', 'Overall', 'Total number of players per nationality')
plot_geo('Overall', 'mean', 'Overall', 'Average rating per nationality')
plot_geo('Overall', 'max','Overall',  'Maximum rating per nationality')
plot_geo('Potential', 'max', 'Potential', 'Maximum potential per nationality')
plot_geo('Age', 'mean', 'Age', 'Average age per nationality')


# A few interesting findings:
# * EU and South America are stronger in general
# * We can see that US has relatively greater potential than it's current rating
# * Pretty much all countries have similar average player age

# ## Squad of best 11 players

# Alright, now let's look at something simple, what's the best squad accordingly to FIFA 18 purely based on overall rating?

# In[ ]:


# 'ST', 'RW', 'LW', 'GK', 'CDM', 'CB', 'RM', 'CM', 'LM', 'LB', 'CAM','RB', 'CF', 'RWB', 'LWB'

def get_best_squad(position):
    df_copy = df.copy()
    store = []
    for i in position:
        store.append([i,df_copy.loc[[df_copy[df_copy['Preferred Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Preferred Position'] == i]['Overall'].max()])
        df_copy.drop(df_copy[df_copy['Preferred Position'] == i]['Overall'].idxmax(), inplace = True)
    #return store
    return pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', 'Overall']).to_string(index = False)

# 4-3-3
squad_433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']
print ('4-3-3')
print (get_best_squad(squad_433))


# In[ ]:


# 3-5-2
squad_352 = ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']
print ('3-5-2')
print (get_best_squad(squad_352))


# ## Age, Potential rating, current rating study of all players

# So we have the best squad now, what's next? Well for a club investor, you might not want to just look at the current overall performance of the team, but also have a future-oriented mind - look at the potential of the team. 
# 
# Another concern is the cost of acquiring the squad, it is nearly impossible to get the above squad because the best players come from different club!
# 
# Now let's look at how age, potential rating and current overall rating correlated with each other:

# In[ ]:


df_p = df.groupby(['Age'])['Potential'].mean()
df_o = df.groupby(['Age'])['Overall'].mean()

df_summary = pd.concat([df_p, df_o], axis=1)

ax = df_summary.plot()
ax.set_ylabel('Rating')
ax.set_title('Average Rating by Age')


# ## Current rating, potential rating and corresponding team value by club

# We can see that players have much more potential to grow at their younger age, and grow very slow after around age 28 (average view among all players).
# 
# Now let's look at which team has highest potential and overall rating on best 11 squad. First we have to modify the above function a bit to make the implementation easier:

# In[ ]:


def get_best_squad(position, club = '*', measurement = 'Overall'):
    df_copy = df.copy()
    df_copy = df_copy[df_copy['Club'] == club]
    store = []
    for i in position:
        store.append([df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Preferred Position'].to_string(index = False),df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].max(), float(df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Value (M)'].to_string(index = False))])
        df_copy.drop(df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax(), inplace = True)
    #return store
    return np.mean([x[2] for x in store]).round(1), pd.DataFrame(np.array(store).reshape(11,4), columns = ['Position', 'Player', measurement, 'Value (M)']).to_string(index = False), np.sum([x[3] for x in store]).round(1)

# easier constraint
squad_433_adj = ['GK', 'B$', 'B$', 'B$', 'B$', 'M$', 'M$', 'M$', 'W$|T$', 'W$|T$', 'W$|T$']

# Example Output for Chelsea
rating_433_Chelsea_Overall, best_list_433_Chelsea_Overall, value_433_Chelsea_Overall = get_best_squad(squad_433_adj, 'Chelsea', 'Overall')
rating_433_Chelsea_Potential, best_list_433_Chelsea_Potential, value_433_Chelsea_Potential  = get_best_squad(squad_433_adj, 'Chelsea', 'Potential')

print('-Overall-')
print('Average rating: {:.1f}'.format(rating_433_Chelsea_Overall))
print('Total Value (M): {:.1f}'.format(value_433_Chelsea_Overall))
print(best_list_433_Chelsea_Overall)

print('-Potential-')
print('Average rating: {:.1f}'.format(rating_433_Chelsea_Potential))
print('Total Value (M): {:.1f}'.format(value_433_Chelsea_Potential))
print(best_list_433_Chelsea_Potential)


# Now let's gather the ratings for all clubs available:

# In[ ]:


# very easy constraint since some club do not have strict squad
squad_352_adj = ['GK', 'B$', 'B$', 'B$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'W$|T$|M$', 'W$|T$|M$']

By_club = df.groupby(['Club'])['Overall'].mean()

def get_summary(squad):
    OP = []
    # only get top 100 clubs for shorter run-time
    for i in By_club.sort_values(ascending = False).index[0:100]:
        # for overall rating
        O_temp_rating, _, _  = get_best_squad(squad, club = i, measurement = 'Overall')
        # for potential rating & corresponding value
        P_temp_rating, _, P_temp_value = get_best_squad(squad, club = i, measurement = 'Potential')
        OP.append([i, O_temp_rating, P_temp_rating, P_temp_value])
    return OP

OP_df = pd.DataFrame(np.array(get_summary(squad_352_adj)).reshape(-1,4), columns = ['Club', 'Overall', 'Potential', 'Value of highest Potential squad'])
OP_df.set_index('Club', inplace = True)
OP_df = OP_df.astype(float)


print (OP_df.head(10))
    


# Above shows you the best 11 squad for both current rating, potential rating and corresponding value (M) of top 10 clubs, in table format. How does it show graphically?

# In[ ]:


fig, ax = plt.subplots()
OP_df.plot(kind = 'scatter', x = 'Overall', y = 'Potential', c = 'Value of highest Potential squad', s = 50, figsize = (15,15), xlim = (70, 90), ylim = (70, 90), title = 'Current Rating vs Potential Rating by Club: 3-5-2', ax = ax)


# Some interesting pattern found, is there any strong clubs with relatively low value? Let's drill down to top clubs with potential rating above 85:

# In[ ]:


fig, ax = plt.subplots()
OP_df.plot(kind = 'scatter', x = 'Overall', y = 'Potential', c = 'Value of highest Potential squad', s = 50, figsize = (15,15), xlim = (80, 90), ylim = (85, 90), title = 'Current Rating vs Potential Rating by Club: 3-5-2', ax = ax)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))
       
OP_df['Club_label'] = OP_df.index

OP_df_sub = OP_df[(OP_df['Potential']>=85) & (OP_df['Value of highest Potential squad']<=350)]

label_point(OP_df_sub['Overall'], OP_df_sub['Potential'], OP_df_sub['Club_label'], ax)


# Supposed we have 350M to buy a club with existing squad and are interested in club with at least 85 potential ratings, then we should probably go for Tottenham Hotspur, and Aresenal is relatively a bad choice. Their best 11 squad is as follows:

# In[ ]:


squad_352_adj = ['GK', 'B$', 'B$', 'B$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'M$|W$|T$', 'W$|T$|M$', 'W$|T$|M$']

rating_352_TH_Overall, best_list_352_TH_Overall, value_352_TH_Overall = get_best_squad(squad_352_adj, 'Tottenham Hotspur', 'Overall')
rating_352_TH_Potential, best_list_352_TH_Potential, value_352_TH_Potential  = get_best_squad(squad_352_adj, 'Tottenham Hotspur', 'Potential')

print('-Overall-')
print('Average rating: {:.1f}'.format(rating_352_TH_Overall))
print('Total Value (M): {:.1f}'.format(value_352_TH_Overall))
print(best_list_352_TH_Overall)

print('-Potential-')
print('Average rating: {:.1f}'.format(rating_352_TH_Potential))
print('Total Value (M): {:.1f}'.format(value_352_TH_Potential))
print(best_list_352_TH_Potential)


# ## Current and future best squad by country

# Alright, now let's move onto studying different squad's impact on Nationality teams. First let's modifiy above get_summary and get_best_squad functions for Nationality:

# In[ ]:


def get_best_squad_n(position, nationality, measurement = 'Overall'):
    df_copy = df.copy()
    df_copy = df_copy[df_copy['Nationality'] == nationality]
    store = []
    for i in position:
        store.append([df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Preferred Position'].to_string(index = False),df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].max()])
        df_copy.drop(df_copy[df_copy['Preferred Position'].str.contains(i)][measurement].idxmax(), inplace = True)
    #return store
    return np.mean([x[2] for x in store]).round(2), pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', measurement]).to_string(index = False)

def get_summary_n(squad_list, squad_name, nationality_list):
    OP_n = []

    for i in nationality_list:
        count = 0
        for j in squad_list:
            # for overall rating
            O_temp_rating, _  = get_best_squad_n(position = j, nationality = i, measurement = 'Overall')
            # for potential rating & corresponding value
            P_temp_rating, _ = get_best_squad_n(position = j, nationality = i, measurement = 'Potential')
            OP_n.append([i, squad_name[count], O_temp_rating.round(2), P_temp_rating.round(2)])    
            count += 1
    return OP_n



# Also let's make our squad choices more strict:

# In[ ]:


squad_352_strict = ['GK', 'LB|LWB', 'CB', 'RB|RWB', 'LM|W$', 'RM|W$', 'CM', 'CM|CAM|CDM', 'CM|CAM|CDM', 'W$|T$', 'W$|T$']
squad_442_strict = ['GK', 'LB|LWB', 'CB', 'CB', 'RB|RWB', 'LM|W$', 'RM|W$', 'CM', 'CM|CAM|CDM', 'W$|T$', 'W$|T$']
squad_433_strict = ['GK', 'LB|LWB', 'CB', 'CB', 'RB|RWB', 'CM|LM|W$', 'CM|RM|W$', 'CM|CAM|CDM', 'W$|T$', 'W$|T$', 'W$|T$']
squad_343_strict = ['GK', 'LB|LWB', 'CB', 'RB|RWB', 'LM|W$', 'RM|W$', 'CM|CAM|CDM', 'CM|CAM|CDM', 'W$|T$', 'W$|T$', 'W$|T$']
squad_532_strict = ['GK', 'LB|LWB', 'CB|LWB|RWB', 'CB|LWB|RWB', 'CB|LWB|RWB', 'RB|RWB', 'M$|W$', 'M$|W$', 'M$|W$', 'W$|T$', 'W$|T$']


squad_list = [squad_352_strict, squad_442_strict, squad_433_strict, squad_343_strict, squad_532_strict]
squad_name = ['3-5-2', '4-4-2', '4-3-3', '3-4-3', '5-3-2']


# Below is an example of best 11 squad line-up of England in 3-5-2, both current rating and potential rating:

# In[ ]:


rating_352_EN_Overall, best_list_352_EN_Overall = get_best_squad_n(squad_352_strict, 'England', 'Overall')
rating_352_EN_Potential, best_list_352_EN_Potential = get_best_squad_n(squad_352_strict, 'England', 'Potential')

print('-Overall-')
print('Average rating: {:.1f}'.format(rating_352_EN_Overall))
print(best_list_352_EN_Overall)

print('-Potential-')
print('Average rating: {:.1f}'.format(rating_352_EN_Potential))
print(best_list_352_EN_Potential)


# Now let's explore different squad possibility of England and how it affects the ratings:

# In[ ]:


OP_df_n = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, ['England'])).reshape(-1,4), columns = ['Nationality', 'Squad', 'Overall', 'Potential'])
OP_df_n.set_index('Nationality', inplace = True)
OP_df_n[['Overall', 'Potential']] = OP_df_n[['Overall', 'Potential']].astype(float)

print (OP_df_n)


# Graphcial representation:

# In[ ]:


fig, ax = plt.subplots()


OP_df_n.plot(kind = 'barh', x = 'Squad', y = ['Overall', 'Potential'], edgecolor = 'black', color = ['white', 'lightgrey'], figsize = (15,10), title = 'Current and potential rating (Best 11) by squad (England)', ax = ax)


#print (OP_df_n[OP_df_n['Overall'] == OP_df_n['Overall'].max()]['Squad'])

def get_text_y(look_for):
    count = 0
    for i in squad_name:
        if i == look_for:
            return count
        else:
            count += 1

ax.text(OP_df_n['Overall'].max()/2, get_text_y(OP_df_n[OP_df_n['Overall'] == OP_df_n['Overall'].max()]['Squad'].tolist()[0])-0.2, 'Highest Current Rating: {}'.format(OP_df_n['Overall'].max()))
ax.text(OP_df_n['Potential'].max()/2, get_text_y(OP_df_n[OP_df_n['Potential'] == OP_df_n['Potential'].max()]['Squad'].tolist()[0])+0.1, 'Highest Potential Rating: {}'.format(OP_df_n['Potential'].max()))



# So we can say that England has the best squard as 3-5-2 for both current squad and future squad based on team ratings. How about other countries? Let's see a few more:

# In[ ]:


Country_list = ['Spain','Germany','Brazil','Argentina','Italy']

OP_df_n = pd.DataFrame(np.array(get_summary_n(squad_list, squad_name, Country_list)).reshape(-1,4), columns = ['Nationality', 'Squad', 'Overall', 'Potential'])
OP_df_n.set_index('Nationality', inplace = True)
OP_df_n[['Overall', 'Potential']] = OP_df_n[['Overall', 'Potential']].astype(float)

for i in Country_list:
    OP_df_n_copy = OP_df_n.copy()
    OP_df_n_copy = OP_df_n_copy[OP_df_n_copy.index == i]
    fig, ax = plt.subplots()
    OP_df_n_copy.plot(kind = 'barh', x = 'Squad', y = ['Overall', 'Potential'], edgecolor = 'black', color = ['white', 'lightgrey'], figsize = (15,10), title = 'Current and potential rating (Best 11) by squad ({})'.format(i), ax = ax)

    ax.text(OP_df_n_copy['Overall'].max()/2, get_text_y(OP_df_n_copy[OP_df_n_copy['Overall'] == OP_df_n_copy['Overall'].max()]['Squad'].tolist()[0])-0.2, 'Highest Current Rating: {}'.format(OP_df_n_copy['Overall'].max()))
    ax.text(OP_df_n_copy['Potential'].max()/2, get_text_y(OP_df_n_copy[OP_df_n_copy['Potential'] == OP_df_n_copy['Potential'].max()]['Squad'].tolist()[0])+0.1, 'Highest Potential Rating: {}'.format(OP_df_n_copy['Potential'].max()))


# For Spain, we can see that moving from 4-4-2 to 3-4-3 in long run might benefit the team, more into an attack style. Different countries offer different observations!

# ## Introducing time effect for future rating

# One of the major drawback of above analysis is it ignores the time taken for players to grow to their max potential. Let's try to consider it as well.
# 
# First we need to get the average rating trend of all players:

# In[ ]:


df_summary['Overall age trend factor'] = df_summary['Overall'] / df_summary['Overall'].iloc[0]

# assume players retire at 40
df_summary_trend = df_summary['Overall age trend factor'].loc[16:40]

expand = pd.Series(0, index=range(41,100))

df_summary_trend = df_summary_trend.append(expand)

print(df_summary_trend.head())


# Then we apply the trend factor to estimate the future rating more accurately:

# In[ ]:


def get_best_squad_n_n_yr_later(n, position, nationality):
    df_copy = df.copy()
    df_copy = df_copy[df_copy['Nationality'] == nationality]
    df_copy['Overall_n_yr_later'] = round(df_copy['Age'].apply(lambda x: df_summary_trend.loc[x+n]/df_summary_trend.loc[x])*df_copy['Overall'],1)
    store = []
    for i in position:
        store.append([df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)]['Overall_n_yr_later'].idxmax()]]['Preferred Position'].to_string(index = False),df_copy.loc[[df_copy[df_copy['Preferred Position'].str.contains(i)]['Overall_n_yr_later'].idxmax()]]['Name'].to_string(index = False), df_copy[df_copy['Preferred Position'].str.contains(i)]['Overall_n_yr_later'].max()])
        df_copy.drop(df_copy[df_copy['Preferred Position'].str.contains(i)]['Overall_n_yr_later'].idxmax(), inplace = True)
    #return store
    return np.mean([x[2] for x in store]).round(2), pd.DataFrame(np.array(store).reshape(11,3), columns = ['Position', 'Player', 'Overall_n_yr_later']).to_string(index = False)

# get next 3 years England's best squad for 3-5-2 based on estimate
for n in range(0,4):
    print('{} years later'.format(n))
    rating_352_EN_Overall_later, best_list_352_EN_Overall_later = get_best_squad_n_n_yr_later(n, squad_352_strict, 'England')
    print('Average rating: {:.1f}'.format(rating_352_EN_Overall_later))
    print(best_list_352_EN_Overall_later)
    
    


# We can see the player list changed slightly when time goes on. How about Spain?

# In[ ]:


# get next 3 years Spain's best squad for 3-5-2 based on estimate
for n in range(0,4):
    print('{} years later'.format(n))
    rating_352_SP_Overall_later, best_list_352_SP_Overall_later = get_best_squad_n_n_yr_later(n, squad_352_strict, 'Spain')
    print('Average rating: {:.1f}'.format(rating_352_SP_Overall_later))
    print(best_list_352_SP_Overall_later)


# Let's display them graphically:

# In[ ]:


list_of_countries = ['England', 'Italy', 'Spain', 'Germany', 'Brazil']
n_yr = 15

rating_combine = pd.DataFrame(index = range(0, n_yr), columns = list_of_countries)

for c in list_of_countries:
    for n in range(0,n_yr):
        rating_352_Overall_later, _ = get_best_squad_n_n_yr_later(n, squad_352_strict, c)
        rating_combine[c].iloc[n] = rating_352_Overall_later
        
ax = rating_combine.plot(kind = 'line', figsize = (15,10), title = 'Country 3-5-2 best 11 rating by time')
ax.set_xlabel("n years later")
ax.set_ylabel("team rating")
    


# It drops after some top players are retired and no new player data available, as expected

# ## Player rating vs Value - Regression

# Next, let's look into how player's market value vary with their ratings, by performing regression:

# In[ ]:


X = df['Overall'].values.reshape(-1,1)
y = df['Value (M)'].values.reshape(-1,1)
regr = linear_model.LinearRegression().fit(X, y)

y_pred = regr.predict(X)
print('Coefficients: ', regr.coef_)
print("Mean squared error: %.2f"% mean_squared_error(y, y_pred))
print('Variance score: %.2f'% r2_score(y, y_pred))

def plot_chart(X, y, y_pred, x_l, x_h, y_l, y_h, c):
    plt.figure(figsize = (15,10))
    plt.scatter(X, y, color=c)
    plt.plot(X, y_pred, color='blue', linewidth=3)

    plt.title('Player value (M) vs rating')
    plt.ylim(y_l,y_h)
    plt.xlim(x_l,x_h)
    plt.ylabel('Value (M)')
    plt.xlabel('Player ratings')
    
plot_chart(X, y, y_pred, 40, 100, 0, 130, 'black')


# A few observations:
# 
# * Linear regression is not a good fit to the data (let's try polynomial models)
# * There are a lot of players with zero value that skewed the dataset (let's exclude those from our analysis)

# In[ ]:


df_2 = df[df['Value (M)'] != 0]
X_2 = df_2['Overall'].values.reshape(-1,1)
y_2 = df_2['Value (M)'].values.reshape(-1,1)

poly = PolynomialFeatures(degree=2)
X_2_p = poly.fit_transform(X_2)
clf = linear_model.LinearRegression().fit(X_2_p, y_2)
y_2_pred = clf.predict(X_2_p)

print('Coefficients: ', clf.coef_)
print("Mean squared error: %.2f"% mean_squared_error(y_2, y_2_pred))
print('Variance score: %.2f'% r2_score(y_2, y_2_pred))

color_dict = {'ST': 'red', 'CF': 'red', 'LW': 'red', 'RW': 'red',
              'LM': 'blue', 'RM': 'blue', 'CM': 'blue', 'CAM': 'blue', 'CDM': 'blue',
              'LB': 'green', 'RB': 'green', 'CB': 'green', 'LWB': 'green', 'RWB': 'green',
              'GK': 'purple'}

c = df_2['Preferred Position'].map(color_dict)

plot_chart(X_2, y_2, y_2_pred, 40, 100, 0, 130, 'black')

# by positions
plot_chart(X_2, y_2, y_2_pred, 40, 100, 0, 130, c)
plt.text(50, 120, 'Forward', color = 'red')
plt.text(50, 115, 'Midfielder', color = 'blue')
plt.text(50, 110, 'Defender', color = 'green')
plt.text(50, 105, 'Goalkeeper', color = 'purple')


# From above, we can see that those significantly overpriced players (way above the blue curve) are mostly forward and midfielder players. Let's focus on high rating players and see who are underpriced:

# In[ ]:


x_low = 85
y_max = 60
plot_chart(X_2, y_2, y_2_pred, x_low, 94, 0, y_max, c)
plt.text(92, 20, 'Forward', color = 'red')
plt.text(92, 17, 'Midfielder', color = 'blue')
plt.text(92, 14, 'Defender', color = 'green')
plt.text(92, 11, 'Goalkeeper', color = 'purple')
ax = plt.gca()
for index, row in df_2.iterrows():
    if row['Overall'] > x_low and row['Value (M)'] < y_max:
        ax.text(row['Overall'], row['Value (M)'], '{}, {}'.format(row['Name'],row['Age']))


# We could see that those 'underpriced' players are usually older, and this explains why their market value is relatively lower.

# Please note that the above analysis has a lot of assumption, including the FIFA 18 rating accuracy for each player etc. So just have fun, hopefully it is a fun read!
# 
# Next I will look into applying machine learning models to the data to [predict player's preferred positions](https://www.kaggle.com/laowingkin/fifa-18-predict-positions-logistic-regression), stay tuned!
