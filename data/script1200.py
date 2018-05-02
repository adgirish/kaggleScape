
# coding: utf-8

# # **Categorizing actors (_hands on plotly_)**
# *Fabien Daniel (August 2017)*
# ___

# In this notebook, I will try to get some insight on the habits of actors and the way they are perceived by spectators. I will not discuss the content of the current dataframe since this was done in many other notebooks (as, picking one up randomly, [this one](https://www.kaggle.com/fabiendaniel/film-recommendation-engine/)).
# ____
# **Acknowledgements:** many thanks to [Tianyi Wang](https://www.kaggle.com/tianyiwang/neighborhood-interaction-with-network-graph) whose kernel gave me the idea to write this notebook, and to [Stéphane Rappeneau](https://www.kaggle.com/stephanerappeneau) for the insights on Plotly's limits. The original dataset from which this kernel was built was originaly updated in September 2017 and [Sohier Dane](https://www.kaggle.com/sohier) made a [guide](https://www.kaggle.com/sohier/getting-imdb-kernels-working-with-tmdb-data) to adapt the old kernels to the new data structure. Many thanks to Sohier for this work !
# ___
# **1. Preparing the data** <br>
# **2. Actors overview** <br>
# **3. A close look at actors  ** <br>
# **4. Actors network ** <br>
# 
# <font color='red'> Warning: currently, the objectives of Section 3 can't be fulfilled due to the developpement progress of the Plotly's package regarding polar charts. For further details, you can have a look at the tickets posted on github (the links are given below). </font>

# ___
# ## 1. Preparing the data
# 
# First, I introduce some functions taken from [Sohier's code](https://www.kaggle.com/sohier/getting-imdb-kernels-working-with-tmdb-data) to interface the kernel with the new data structure:

# In[ ]:


#__________________
import json
import pandas as pd
#__________________
def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries', 'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#____________________________
def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#_______________________________________
def safe_access(container, index_values):
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan
#_______________________________________
LOST_COLUMNS = [
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_facebook_likes',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews']
#_______________________________________
TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',  
    'keywords': 'plot_keywords',
    'vote_count': 'num_voted_users'}
#_______________________________________     
IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}
#_______________________________________
def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])
#_______________________________________
def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])
#_______________________________________
def convert_to_original_format(movies, credits):
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    # I'm assuming that the first production country is equivalent, but have not been able to validate this
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies


# and the load the packages and the dataset:

# In[ ]:


import matplotlib.pyplot as plt
import plotly.offline as pyo
pyo.init_notebook_mode()
from plotly.graph_objs import *
import plotly.graph_objs as go
import numpy as np
import pandas as pd
#_______________________________________________
credits = load_tmdb_credits("../input/tmdb_5000_credits.csv")
movies = load_tmdb_movies("../input/tmdb_5000_movies.csv")
df = convert_to_original_format(movies, credits)


# This dataframe contains around 5000 movies which are described according to 28 variables. In what follows, I will focus on a few of them. I start with the **genres** variable that describes the cinematographic genres (each film can pertain to various categories). As a first step, I extract the list of categorical values:

# In[ ]:


liste_genres = set()
for s in df['genres'].str.split('|'):
    liste_genres = set().union(s, liste_genres)
liste_genres = list(liste_genres)
liste_genres.remove('')


# For the current exercise, the others variables of interest are the **actor_*N*_name** (**_N_** $\in [1:3]$) variables, that list the three main actors appearing in each film. My first goal is to determine the favorite genre of actors. For simplicity, in what follows, I will only consider the actors who appear in the **actor_1_name**. In fact, a more exhaustive view would be
# achieved considering also the two other categories. To proceed with that, I perform some one hot encoding:

# In[ ]:


df_reduced = df[['actor_1_name', 'vote_average',
                 'title_year', 'movie_title']].reset_index(drop = True)
for genre in liste_genres:
    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)
df_reduced[:5]


# and I group according to every actors, taking the mean of all the other variables. Then I check which genre's column takes the highest value and assign the corresponding genre as the actor's favorite genre: 

# In[ ]:


df_actors = df_reduced.groupby('actor_1_name').mean()
df_actors.loc[:, 'favored_genre'] = df_actors[liste_genres].idxmax(axis = 1)
df_actors.drop(liste_genres, axis = 1, inplace = True)
df_actors = df_actors.reset_index()
df_actors[:10]


# At this point, the dataframe contains a list of actors and for each of them, we have a mean IMDB score, its mean year of activity and his favored acting style.
# 
# Then, I create a mask to account only for the actors that played in more than 5 films:

# In[ ]:


df_appearance = df_reduced[['actor_1_name', 'title_year']].groupby('actor_1_name').count()
df_appearance = df_appearance.reset_index(drop = True)
selection = df_appearance['title_year'] > 4
selection = selection.reset_index(drop = True)
most_prolific = df_actors[selection]


# Finally, I look at the percentage of films of each genre to further choose the genres I want to look at:

# In[ ]:


plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(5, 5))
genre_count = []
for genre in liste_genres:
    genre_count.append([genre, df_reduced[genre].values.sum()])
genre_count.sort(key = lambda x:x[1], reverse = True)
labels, sizes = zip(*genre_count)
labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]
ax.pie(sizes, labels=labels_selected,
       autopct = lambda x:'{:2.0f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=0)
ax.axis('equal')
plt.tight_layout()


# ___
# ## 2. Actors overview
# And now, the **magic of plotly** happens:

# In[ ]:


reduced_genre_list = labels[:12]
trace=[]
for genre in reduced_genre_list:
    trace.append({'type':'scatter',
                  'mode':'markers',
                  'y':most_prolific.loc[most_prolific['favored_genre']==genre,'vote_average'],
                  'x':most_prolific.loc[most_prolific['favored_genre']==genre,'title_year'],
                  'name':genre,
                  'text': most_prolific.loc[most_prolific['favored_genre']==genre,'actor_1_name'],
                  'marker':{'size':10,'opacity':0.7,
                            'line':{'width':1.25,'color':'black'}}})
layout={'title':'Actors favored genres',
       'xaxis':{'title':'mean year of activity'},
       'yaxis':{'title':'mean score'}}
fig=Figure(data=trace,layout=layout)
pyo.iplot(fig)


# The above graph lists all the actors that appeared in more than 5 films (only taking into account the **actor_1_name**). The abscissa corresponds to the average of the film release years and the ordinate to the mean IMDB score. Every film is tagged according to its genre and hovering with the mouse, the actors names are displayed.

# ___
# ## 3. A close look at actors

# In the previous section, we had a global overview of all the actors that appeared in more than 5 films and each of them was described with quantites (year, IMDB score, genre) averaged over the films. In this section, the aim is to select some particular actor and to display all its cinematographic biography in a simple view.

# In[ ]:


selection = df_appearance['title_year'] > 10
most_prolific = df_actors[selection]
most_prolific


# In[ ]:


class Trace():
    #____________________
    def __init__(self, color):
        self.mode = 'markers'
        self.name = 'default'
        self.title = 'default title'
        self.marker = dict(color=color, size=110,
                           line=dict(color='white'), opacity=0.7)
        self.r = []
        self.t = []
    #______________________________
    def set_color(self, color):
        self.marker = dict(color = color, size=110,
                           line=dict(color='white'), opacity=0.7)
    #____________________________
    def set_name(self, name):
        self.name = name
    #____________________________
    def set_title(self, title):
        self.na = title
    #__________________________
    def set_values(self, r, t):
        self.r = np.array(r)
        self.t = np.array(t)


# Below, I make a census of the films starring Brad Pitt, identifying the genre of every film. **However, I just take into account the genres with at least 4 films because of a bug in plotly: below this threshold, a few spurious points appear on the graph. A [ticket](https://github.com/plotly/plotly.js/issues/2023#issuecomment-330852374) was sent to report that bug.**

# In[ ]:


df2 = df_reduced[df_reduced['actor_1_name'] == 'Brad Pitt']
total_count  = 0
years = []
imdb_score = []
genre = []
titles = []
for s in liste_genres:
    icount = df2[s].sum()
    #__________________________________________________________________
    # Here, we set the limit to 3 because of a bug in plotly's package
    if icount > 3: 
        total_count += 1
        genre.append(s)
        years.append(list(df2[df2[s] == 1]['title_year']))
        imdb_score.append(list(df2[df2[s] == 1]['vote_average'])) 
        titles.append(list(df2[df2[s] == 1]['movie_title']))
max_y = max([max(s) for s in years])
min_y = min([min(s) for s in years])
year_range = max_y - min_y

years_normed = []
for i in range(total_count):
    years_normed.append( [360/total_count*((an-min_y)/year_range+i) for an in years[i]])


# In[ ]:


color = ['royalblue', 'grey', 'wheat', 'c', 'firebrick', 'seagreen', 'lightskyblue',
          'lightcoral', 'yellowgreen', 'gold', 'tomato', 'violet', 'aquamarine', 'chartreuse']


# In[ ]:


trace = [Trace(color[i]) for i in range(total_count)]
tr    = []
for i in range(total_count):
    trace[i].set_name(genre[i])
    trace[i].set_title(titles[i])
    trace[i].set_values(np.array(imdb_score[i]),
                        np.array(years_normed[i]))
    tr.append(go.Scatter(r      = trace[i].r,
                         t      = trace[i].t,
                         mode   = trace[i].mode,
                         name   = trace[i].name,
                         marker = trace[i].marker,
#                         text   = ['default title' for j in range(len(trace[i].r))], 
                         hoverinfo = 'all'
                        ))        
layout = go.Layout(
    title='Brad Pitt movies',
    font=dict(
        size=15
    ),
    plot_bgcolor='rgb(223, 223, 223)',
    angularaxis=dict(        
        tickcolor='rgb(253,253,253)'
    ),
    hovermode='Closest',
)
fig = go.Figure(data = tr, layout=layout)
pyo.iplot(fig)


# On this graph, we see every film starring Brad Pitt. The radial scale corresponds to the IMDB score and the angular scale indicates both the movie's genre and the films release years. **Unfortunately, polar charts are not any more maintened by plotly's developpement team and it is currently impossible to make the text appearing on hover (see this [ticket](https://github.com/plotly/plotly.js/issues/94#issuecomment-330853403))**. Initially, my objective was to make the films titles and release year appear on hover but it is currently impossible. Hence, this puts strong limits on the usefulness of that kind of representation ...

# ### 4.  Network
# 

# Here, I look at the connections between the most prolific actors:

# In[ ]:


selection = df_appearance['title_year'] > 4
most_prolific = df_actors[selection]
actors_list = most_prolific['actor_1_name'].unique()


# In[ ]:


test = pd.crosstab(df['actor_1_name'], df['actor_2_name'])


# In[ ]:


edge = []
for actor_1, actor_2 in list(test[test > 0].stack().index):
    if actor_1 not in actors_list: continue
    if actor_2 not in actors_list: continue
   
    if actor_1 not in actors_list or actor_2 not in actors_list: continue
    if actor_1 != actor_2:
        edge.append([actor_1, actor_2])


# In[ ]:


num_of_adjacencies = [0 for _ in range(len(df_actors))]
for ind, col in df_actors.iterrows():
    actor = col['actor_1_name']
    nb = sum([1 for i,j in edge if (i == actor) or (j == actor)])
    num_of_adjacencies[ind] = nb


# In[ ]:


def prep(edge, num_of_adjacencies, df, actors_list):
    edge_trace = Scatter(
    x=[],
    y=[],
    line = Line(width=0.5,color='#888'),
    hoverinfo = 'none',
    mode = 'lines')
    
    for actor_1, actor_2 in edge:
        x0, y0 = df[df['actor_1_name'] == actor_1][['title_year', 'vote_average']].unstack()
        x1, y1 = df[df['actor_1_name'] == actor_2][['title_year', 'vote_average']].unstack()
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
             colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    
    for ind, col in df.iterrows():
        if col['actor_1_name'] not in actors_list: continue
        node_trace['x'].append(col['title_year'])
        node_trace['y'].append(col['vote_average'])
        node_trace['text'].append(col['actor_1_name'])
        node_trace['marker']['color'].append(num_of_adjacencies[ind])
        
    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                    title='<br>Connections between actors',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=XAxis(showgrid=True, zeroline=False, showticklabels=True),
                    yaxis=YAxis(showgrid=True, zeroline=False, showticklabels=True)))
    
    return fig


# In[ ]:


fig = prep(edge, num_of_adjacencies, df_actors, actors_list)
pyo.iplot(fig)

