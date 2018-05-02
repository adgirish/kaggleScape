
# coding: utf-8

# # Exploring Player Stats
# 
# > * Author: Eric Couto
# > * Date: 17 July 2016
# 
# -------------------
# ### Topics:
# > 1. Loading Data
# > 2. Player Tables
# > 3. t-SNE of Players (FIFA Stats)
# > 4. Evolution of the Best Players (top 20)
# 

# ## 1. Loading Data
# 
# >* Connecting to the database and looking at the available tables

# In[ ]:


import sqlite3
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from bokeh.plotting import figure, ColumnDataSource, show
from bokeh.models import HoverTool
from bokeh.io import output_notebook
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
output_notebook()

database = '../input/database.sqlite'
conn = sqlite3.connect(database)

query = "SELECT name FROM sqlite_master WHERE type='table';"
pd.read_sql(query, conn)


# ## 2. Player Tables
# 
# > * Checking what kind of data are available

# In[ ]:


query = "SELECT * FROM Player;"
a = pd.read_sql(query, conn)
a.head()


# In[ ]:


query = "SELECT * FROM Player_Stats;"
a = pd.read_sql(query, conn)
a.head()


# In[ ]:


query = """SELECT * FROM Player_Stats a
           INNER JOIN (SELECT player_name, player_api_id AS p_id FROM Player) b ON a.player_api_id = b.p_id;"""

drop_cols = ['id','player_fifa_api_id','date_stat','preferred_foot',
             'attacking_work_rate','defensive_work_rate']

players = pd.read_sql(query, conn)
players['date_stat'] = pd.to_datetime(players['date_stat'])
players = players[players.date_stat > pd.datetime(2015,1,1)]
players = players[~players.overall_rating.isnull()].sort_values('date_stat', ascending=False)
players = players.drop_duplicates(subset='player_api_id', keep='first')
players = players.drop(drop_cols, axis=1)

players.info()


# ## 3. t-SNE of Players (FIFA Stats)
# 
# > * Standardization of stats
# > * Dimensionality Reduction using t-SNE: 2 components
# > * Plotting the best players (Overall Rating >=80) in the new space (interactive plots)
# > * Best Players were grouped by attributes -> well separated in clusters of positions: GK, FW, etc

# In[ ]:


players = players.fillna(0)

cols = ['player_api_id','player_name','overall_rating','potential']
stats_cols = [col for col in players.columns if col not in (cols)]

ss = StandardScaler()
tmp = ss.fit_transform(players[stats_cols])
model = TSNE(n_components=2, random_state=0)
tsne_comp = model.fit_transform(tmp)


# In[ ]:


tmp = players[cols]
tmp['comp1'], tmp['comp2'] = tsne_comp[:,0], tsne_comp[:,1]
tmp = tmp[tmp.overall_rating >= 80]

_tools = 'box_zoom,pan,save,resize,reset,tap,wheel_zoom'
fig = figure(tools=_tools, title='t-SNE of Players (FIFA stats)', responsive=True,
             x_axis_label='Component 1', y_axis_label='Component 2')

source = ColumnDataSource(tmp)
hover = HoverTool()
hover.tooltips=[('Jogador','@player_name'),]
fig.scatter(tmp['comp1'], tmp['comp2'], source=source, size=8, alpha=0.6,
            line_color='red', fill_color='red')

fig.add_tools(hover)

show(fig)


# ### Distribution of Overall Rating, Potential and Potential Growth
# > * Checking distributions
# > * Plotting potential top players in the new space (interactive plots)

# In[ ]:


sns.kdeplot(players.overall_rating, shade=True, color="r")


# In[ ]:


sns.kdeplot(players.potential, shade=True, color="r")


# In[ ]:


players['potential_growth'] = players.potential - players.overall_rating
sns.kdeplot(players.potential_growth, shade=True, color="r")


# In[ ]:


tmp = players[cols]
tmp['comp1'], tmp['comp2'] = tsne_comp[:,0], tsne_comp[:,1]
tmp['potential_growth'] = tmp.potential - tmp.overall_rating
tmp = tmp[(tmp.potential_growth >= 5) & (tmp.overall_rating >= 75)]

_tools = 'box_zoom,pan,save,resize,reset,tap,wheel_zoom'
fig = figure(tools=_tools, title='t-SNE of Potential Top Players (FIFA stats)', responsive=True,
             x_axis_label='Component 1', y_axis_label='Component 2')

source = ColumnDataSource(tmp)
hover = HoverTool()
hover.tooltips=[('Jogador','@player_name'),]
fig.scatter(tmp['comp1'], tmp['comp2'], source=source, size=8, alpha=0.6,
            line_color='red', fill_color='red')

fig.add_tools(hover)

show(fig)


# ## 3. Evolution of the Best Players (top20)
# 
# > * How was the evolution of the best players?

# In[ ]:


players = players.sort_values('overall_rating', ascending=False)
best_players = players[['player_api_id','player_name']].head(20)
ids = tuple(best_players.player_api_id.unique())

query = '''SELECT player_api_id, date_stat, overall_rating, potential
           FROM Player_Stats WHERE player_api_id in %s''' % (ids,)

evolution = pd.read_sql(query, conn)
evolution = pd.merge(evolution, best_players)
evolution['year'] = evolution.date_stat.str[:4].apply(int)
evolution = evolution.groupby(['year','player_api_id','player_name']).overall_rating.mean()
evolution = evolution.reset_index()

evolution.head()


# In[ ]:


a = sns.factorplot(data=evolution[evolution.player_api_id.isin(ids[0:5])], x='year',
                   y='overall_rating', hue='player_name', size=6, aspect=2)


# In[ ]:


a = sns.factorplot(data=evolution[evolution.player_api_id.isin(ids[5:10])], x='year',
                   y='overall_rating', hue='player_name', size=6, aspect=2)


# In[ ]:


a = sns.factorplot(data=evolution[evolution.player_api_id.isin(ids[10:15])], x='year',
                   y='overall_rating', hue='player_name', size=6, aspect=2)


# In[ ]:


a = sns.factorplot(data=evolution[evolution.player_api_id.isin(ids[15:20])], x='year',
                   y='overall_rating', hue='player_name', size=6, aspect=2)

