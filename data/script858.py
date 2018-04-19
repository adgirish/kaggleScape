
# coding: utf-8

# Do some simple statics on Death Metal dataset.
# ----------------------------------------------

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


bands = pd.read_csv(
    filepath_or_buffer="../input/bands.csv", 
    sep=",", 
    na_values=["N/A"], 
    dtype={"id": "int32", "formed_in": "float32"}
)


# In[ ]:


albums = pd.read_csv(
    filepath_or_buffer="../input/albums.csv", 
    sep=",", 
    na_values=["N/A"], 
    dtype={"id": "int32", "band": "int32", "year": "int32"}
)


# In[ ]:


np.sort(albums.year.unique())[-1]
print("There are {:,} death metal bands and {:,} albums by {:d}.".format(
    bands.shape[0], albums.shape[0], np.sort(albums.year.unique())[-1]))


# In[ ]:


first_death_metal_band = bands.loc[bands.formed_in.idxmin()]
print("The world's first death metal band is \"{}\" which formed in {:.0f}.".format(first_death_metal_band["name"], first_death_metal_band["formed_in"]))


# What? The first death metal band is not Possessed or Death? Who is Satan's Host? Let's check this band closely.

# In[ ]:


for key, value in zip(first_death_metal_band.index.values, first_death_metal_band):
    print("{:>10}: {:<}".format(key, value))


# Ah. They changed their genre. Their genre was Heavy Metal when they formed.

# In[ ]:


first_death_metal_album = albums.loc[albums.year.idxmin()]
print("The world's first death metal album is \"{}\" released in {:.0f}.".format(first_death_metal_album["title"], first_death_metal_album["year"]))


# Number of Bands and Albums
# --------------------------

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')

bands_count = bands.groupby("formed_in")["id"].count().cumsum().loc[:2015]
albums_count = albums.groupby("year")["id"].count().cumsum().loc[:2015]

df = pd.DataFrame(
    {
        "bands": bands_count,
        "albums": albums_count
    }
)

ax = df.plot(marker="o", markersize=4)
ax.set_xlabel("year", fontsize=8)
ax.set_title("Number of Albums Until the End of That Year", fontsize=10)
_ = ax.legend(fontsize=8, loc="upper left")


# Number of New Bands and Albums
# ------------------------------

# In[ ]:


bands_count = bands.groupby("formed_in")["id"].count().loc[:2015]
albums_count = albums.groupby("year")["id"].count().loc[:2015]

df = pd.DataFrame(
    {
        "bands": bands_count,
        "albums": albums_count
    }
)

ax = df.plot(marker="o", markersize=4)
ax.set_xlabel("year", fontsize=8)
ax.set_title("Number of New Albums Released in That Year", fontsize=10)
ax.annotate("Nirvana's \"nevermind\" was released", xy=(1991, 950), xytext=(1980, 1600), arrowprops={"width": 2.0, "headwidth": 8})
_ = ax.legend(fontsize=8, loc="upper left")


# Recording & distributing is becoming cheaper and easier, so although there are less bands but more album released. 

# Genre Explosion
# -----------------
# Number of distinct genres of bands formed in each year.

# In[ ]:


ax = bands[["formed_in", "genre"]].drop_duplicates().groupby("formed_in")["genre"].count().loc[:2015].plot(color="#ee7621", marker="o", markersize=4, ylim=[0, 400])
_ = ax.set_xlabel("year", fontsize=8)
_ = ax.set_title("Number of Disinct Genres of Albums Released in That Year", fontsize=10)


# In[ ]:


dominant_genres = bands.groupby("genre")["genre"].count().sort_values().tail(10)
dominant_genres["others"] = bands.shape[0] - dominant_genres.sum()
_ = dominant_genres.plot.pie(figsize=(6, 6))


# "Black/Death Metal" and "Death/Black Metal" are two different genres. Those metal nerds.

# Number of New Albums of Each Genre
# ----------------------------------
# 
# The genre field is quite irregular. Many bands have multiple genres (genre changed, like "Satan's Host" 
# and/or combined genre. So a better way is to process genres as short texts. 
# 
# I expelled combined genres by expelled all genres which contain "/". I also expelled genres which contains no more than 50 bands.

# In[ ]:


genre_count = bands.groupby("genre")["id"].count()
main_genres = genre_count[genre_count >= 50].index.values
main_genres = [genre for genre in main_genres if "/" not in genre]

main_genres_bands = bands[bands.genre.isin(main_genres)]
main_genres_albums = pd.merge(
    left=main_genres_bands,
    right=albums,
    left_on="id",
    right_on="band",
    suffixes=["_band", "_album"],
    how="inner"
)[["id_album", "year", "genre"]]

main_genres_albums = main_genres_albums.groupby(["year", "genre"])["id_album"].count().unstack("genre").fillna(0)
main_genres_albums = main_genres_albums.loc[:2015] # data of 2017 is incomplete.
main_genres_albums.drop(["Blackened Death Metal", "Industrial Death Metal", "Experimental Death Metal"], axis=1, inplace=True)

ax = main_genres_albums.plot(figsize=(8, 5), marker="o", markersize=4)
ax.set_xlabel("year", fontsize=8)
_ = ax.set_title("Number of New Albums of Each Main Genres Released in That Year", fontsize=10)
_ = ax.legend(fontsize=8, loc="upper left")


# Which bands are most productive?
# --------------------------------

# We find top 20 bands by the number of albums released.

# In[ ]:


bands_albums = pd.merge(
    left=bands, 
    right=albums, 
    left_on="id", 
    right_on="band", 
    suffixes=["_band", "_album"],
    how="left"
).drop("band", axis=1)


# In[ ]:


bands_albums_count = pd.DataFrame(bands_albums.groupby("id_band")["id_album"].count().sort_values().tail(20))
bands_albums_count.columns = ["albums_count"] 

bands_high_production_top20 = pd.merge(
    left=bands,
    right=bands_albums_count,
    left_on="id",
    right_index=True
)[["name", "albums_count"]].sort_values("albums_count").set_index("name")

_ = bands_high_production_top20.plot.barh(color="#00304e")


# Which albums are most popular?
# ------------------------------

# We find top 20 albums by the number of reviews received.

# In[ ]:


reviews = pd.read_csv(
    filepath_or_buffer="../input/reviews.csv", 
    sep=",", 
    na_values=["N/A"], 
    usecols=["id", "album", "title", "score"],
    dtype={"id": "int32", "album": "int32", "score": "float32"}
)


# In[ ]:


bands_albums_reviews = pd.merge(
    left=bands_albums, 
    right=reviews, 
    left_on="id_album", 
    right_on="album", 
    suffixes=["", "_review"],
    how="left"
).drop("album", axis=1)


# In[ ]:


albums_reviews_count = pd.DataFrame(bands_albums_reviews.groupby("id_album")["id"].count().sort_values().tail(20))
albums_reviews_count.columns = ["reviews_count"] 

reviews_more_reviews_top20 = pd.merge(
    left=bands_albums,
    right=albums_reviews_count,
    left_on="id_album",
    right_index=True
)[["name", "title", "reviews_count"]].sort_values("reviews_count")

reviews_more_reviews_top20["band/album"] = reviews_more_reviews_top20.name + "'s \"" + reviews_more_reviews_top20.title + "\""
_ = reviews_more_reviews_top20[["band/album", "reviews_count"]].set_index("band/album").plot.barh(color="#00304e", legend=False, figsize=(8, 6))


# They are all famous bands and classical or controversial albums

# Which countries have more death metal bands?
# ---------------------------------------

# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[ ]:


country_bands_count = bands.groupby("country")["id"].count()
country_bands_count = pd.DataFrame(country_bands_count)
country_bands_count.columns = ["bands_count"]
country_bands_count.reset_index(inplace=True)


# In[ ]:


# I copy and modify this piece of code from Anisotropico's kernel 
# "Interactive Plotly: Global Youth Unemployment". Thank him.

metricscale = [
    [0, 'rgb(102,194,165)'], 
    [0.05, 'rgb(102,194,165)'], 
    [0.15, 'rgb(171,221,164)'], 
    [0.2, 'rgb(230,245,152)'], 
    [0.25, 'rgb(255,255,191)'], 
    [0.35, 'rgb(254,224,139)'], 
    [0.45, 'rgb(253,174,97)'], 
    [0.55, 'rgb(213,62,79)'], 
    [1.0, 'rgb(158,1,66)']
]

data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale,
        showscale = True,
        locations = country_bands_count['country'].values,
        z = country_bands_count['bands_count'].values,
        locationmode = 'country names',
        text = country_bands_count['country'].values,
        marker = dict(
            line = dict(color = 'rgb(250,250,225)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Number of Death Metal Bands')
            )
       ]

layout = dict(
    title = 'World Map of Number of Death Metal Bands',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,0,52)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmapdeathmetal')


# Bar plot of  top 20 countries by the number of death metal bands

# In[ ]:


ax = bands.groupby("country")["id"].count().sort_values(ascending=False).head(20).sort_values().plot(kind="barh", color="#00304e")
_ = ax.set_xlabel("number of bands")


# Best albums
# -----------

# We find top 30 best reviewed albums in death metal genre. Only consider albums which have received more than or equal to 8 reviews, and rank them by their average score.

# In[ ]:


albums_reviews_count = bands_albums_reviews.groupby("id_album")["id"].count()
popular_albums = albums_reviews_count[albums_reviews_count >= 8].index.values
popular_bands_albums_reviews = bands_albums_reviews[bands_albums_reviews.id_album.isin(popular_albums)]

best_albums = popular_bands_albums_reviews.groupby("id_album")["score"].sum() / popular_bands_albums_reviews.groupby("id_album")["score"].count()

best_albums = pd.DataFrame(best_albums.sort_values().tail(30))
best_albums.columns = ["average_score"] 

bands_albums_best_top = pd.merge(
    left=bands_albums,
    right=best_albums,
    left_on="id_album",
    right_index=True
)[["name", "title", "average_score"]].sort_values("average_score")

bands_albums_best_top["band/album"] = bands_albums_best_top.name + "'s \"" + bands_albums_best_top.title + "\""
ax = bands_albums_best_top[["band/album", "average_score"]].set_index("band/album").plot.barh(color="#00304e", legend=False, figsize=(8, 12), xlim=[0.85, 1.0])
_ = ax.set_xlabel("avg. score")

for y, x in zip(np.arange(0, bands_albums_best_top.shape[0]), bands_albums_best_top.average_score):
    _ = ax.annotate("{:.3f}".format(x), xy=(x-0.008, y-0.1), fontsize=8, color="#eeeeee")


# ## Word Cloud of Lyrics' Themes ##

# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# In[ ]:


catcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=1200,
    height=800
).generate(" ".join(bands.theme.dropna().str.replace("|", ",").values))

plt.imshow(catcloud, alpha=0.8)
plt.axis('off')
plt.show()


# Word cloud for each sub-genres.

# *Brutal Death Metal*

# In[ ]:


catcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=1200,
    height=800
).generate(" ".join(bands[bands.genre=="Brutal Death Metal"].theme.dropna().str.replace("|", ",").values))

plt.imshow(catcloud, alpha=0.8)
plt.axis('off')
plt.show()


# *Melodic Death Metal*

# In[ ]:


catcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=1200,
    height=800
).generate(" ".join(bands[bands.genre=="Melodic Death Metal"].theme.dropna().str.replace("|", ",").values))

plt.imshow(catcloud, alpha=0.8)
plt.axis('off')
plt.show()


# *Technical Death Metal*

# In[ ]:


catcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=1200,
    height=800
).generate(" ".join(bands[bands.genre=="Technical Death Metal"].theme.dropna().str.replace("|", ",").values))

plt.imshow(catcloud, alpha=0.8)
plt.axis('off')
plt.show()


# *Progressive Death Metal*

# In[ ]:


catcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=1200,
    height=800
).generate(" ".join(bands[bands.genre=="Progressive Death Metal"].theme.dropna().str.replace("|", ",").values))

plt.imshow(catcloud, alpha=0.8)
plt.axis('off')
plt.show()

