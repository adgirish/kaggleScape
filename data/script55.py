
# coding: utf-8

# # The Story of Film

# ![](https://media.giphy.com/media/3ohhwDMC187JqL69DG/giphy.gif)

# The birth of the Motion Picture Camera in the late 18th century gave birth to possibly the most potent form of entertainment in existence: Cinema. Movies have managed to enthrall audiences ever since one second clips of racing horses emerged in the 1890s to the introduction of sound in the 1920s to the birth of color in the 1930s to mainstream 3D Movies in the early 2010s.
# 
# Cinema had humble origins in terms of plot, direction and acting (mainly due to its extremely short duration in its early days) but since then, movie industries around the world have been blessed with creative geniuses in the form of directors, screenwriters, actors, sound designers and cinematographers. It has also spread itself into a plethora of genres ranging from romance to comedy to science fiction to horror.
# 
# As with almost every kid born in the last century, I was amazed by movies. I was addicted to it. And I've always wanted to know more about the enigmatic world of cinema. In this notebook, I will try and gain some insights using data. With us, we have a dataset of about 45000 movies with metadata collected from TMDB. Using this data, we will try and answer various questions that I've always had about movies. 
# 
# In this notebook, I will also try and build the following:
# * A **Regressor** that can, to a certain degree, predict the revenue of a particular movie.
# * A **Classifier** that identifies if a movie will be a hit or will make the producers lose money.
# 

# ## Importing Libraries and Loading Our Data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import Image, HTML
import json
import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from wordcloud import WordCloud, STOPWORDS
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
plotly.tools.set_credentials_file(username='rounakbanik', api_key='xTLaHBy9MVv5szF4Pwan')

sns.set_style('whitegrid')
sns.set(font_scale=1.25)
pd.set_option('display.max_colwidth', 50)


# In[ ]:


df = pd.read_csv('../input/movies_metadata.csv')
df.head().transpose()


# ## Understanding the Dataset
# 
# The dataset above was obtained through the TMDB API. The movies available in this dataset are in correspondence with the movies that are listed in the **MovieLens Latest Full Dataset** comprising of 26 million ratings on 45,000 movies from 27,000 users. Let us have a look at the features that are available to us.

# In[ ]:


df.columns


# ### Features
# 
# * **adult:** Indicates if the movie is X-Rated or Adult.
# * **belongs_to_collection:** A stringified dictionary that gives information on the movie series the particular film belongs to.
# * **budget:** The budget of the movie in dollars.
# * **genres:** A stringified list of dictionaries that list out all the genres associated with the movie.
# * **homepage:** The Official Homepage of the move.
# * **id:** The ID of the move.
# * **imdb_id:** The IMDB ID of the movie.
# * **original_language:** The language in which the movie was originally shot in.
# * **original_title:** The original title of the movie.
# * **overview:** A brief blurb of the movie.
# * **popularity:** The Popularity Score assigned by TMDB.
# * **poster_path:** The URL of the poster image.
# * **production_companies:** A stringified list of production companies involved with the making of the movie.
# * **production_countries:** A stringified list of countries where the movie was shot/produced in.
# * **release_date:** Theatrical Release Date of the movie.
# * **revenue:** The total revenue of the movie in dollars.
# * **runtime:** The runtime of the movie in minutes.
# * **spoken_languages:** A stringified list of spoken languages in the film.
# * **status:** The status of the movie (Released, To Be Released, Announced, etc.)
# * **tagline:** The tagline of the movie.
# * **title:** The Official Title of the movie.
# * **video:** Indicates if there is a video present of the movie with TMDB.
# * **vote_average:** The average rating of the movie.
# * **vote_count:** The number of votes by users, as counted by TMDB.

# In[ ]:


df.shape


# In[ ]:


df.info()


# There are a total of **45,466 movies** with **24 features**. Most of the features have very few NaN values (apart from **homepage** and **tagline**). We will attempt at cleaning this dataset to a form suitable for analysis in the next section.

# ## Data Wrangling

# The data that was originally obtained was in the form of a JSON File. This was converted manually into a CSV file to arrive at an input that could be loaded into a Pandas DataFrame effortlessly. In other words, the dataset we have in our hands is already relatively clean. We will however attempt at learning more about our features and performing appropriate wrangling steps to arrive at a form that is more suitable for analysis.
# 
# Let us start by removing the features that are not useful to us.

# In[ ]:


df = df.drop(['imdb_id'], axis=1)


# In[ ]:


df[df['original_title'] != df['title']][['title', 'original_title']].head()


# The original title refers to the title of the movie in the native language in which the movie was shot. As such, I will prefer using the translated, Anglicized name in this analysis and hence, will drop the original titles altogether. We will be able to deduce if the movie is a foreign language film by looking at the **original_language** feature so no tangible information is lost in doing so.

# In[ ]:


df = df.drop('original_title', axis=1)


# In[ ]:


df[df['revenue'] == 0].shape


# We see that the majority of the movies have a recorded revenue of **0**. This indicates that we do not have information about the total revenue for these movies. Although this forms the majority of the movies available to us, we will still use revenue as an extremely important feature going forward from the remaining 7000 moves.

# In[ ]:


df['revenue'] = df['revenue'].replace(0, np.nan)


# The **budget** feature has some unclean values that makes Pandas assign it as a generic object. We proceed to convert this into a numeric variable and replace all the non-numeric values with NaN. Finally, as with budget, we will convert all the values of 0 with NaN to indicate the absence of information regarding budget.

# In[ ]:


df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)
df[df['budget'].isnull()].shape


# As we move forward trying to answer certain questions, we will have to construct several features suitable for that particular query. For now, we will construct two very important features:
# * **year:** The year in which the movie was released.
# * **return:** The ratio of revenue to budget.
# 
# The **return** feature is extremely insightful as it will give us a more accurate picture of the financial success of a movie. Presently, our data will not be able to judge if a \$200 million budget movie that earned \$100 million did better than a \$50,000 budget movie taking in \$200,000. This feature will be able to capture that information.
# 
# A return value > 1 would indicate profit whereas a return value < 1 would indicate a loss.

# In[ ]:


df['return'] = df['revenue'] / df['budget']
df[df['return'].isnull()].shape


# We have close to **5000 movies** for which we have data on revenue and budget ratio. This is close to **10% of the entire dataset**. Although this may seem small, this is enough to perform very useful analysis and discover interesting insights about the world of movies.

# In[ ]:


df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[ ]:


df['adult'].value_counts()


# There are close to **0 adult movies** in this dataset. The **adult** feature therefore is not of much use to us and can be safely dropped.

# In[ ]:


df = df.drop('adult', axis=1)


# In[ ]:


base_poster_url = 'http://image.tmdb.org/t/p/w185/'
df['poster_path'] = "<img src='" + base_poster_url + df['poster_path'] + "' style='height:100px;'>"


# ## Exploratory Data Analysis
# ### Title and Overview Wordclouds
# 
# Are there certain words that figure more often in Movie Titles and Movie Blurbs? I suspect there are some words which are considered more potent and considered more worthy of a title. Let us find out!

# In[ ]:


df['title'] = df['title'].astype('str')
df['overview'] = df['overview'].astype('str')


# In[ ]:


title_corpus = ' '.join(df['title'])
overview_corpus = ' '.join(df['overview'])


# In[ ]:


title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# The word **Love** is the most commonly used word in movie titles. **Girl**, **Day** and **Man** are also among the most commonly occuring words. I think this encapsulates the idea of the ubiquitious presence of romance in movies pretty well.

# In[ ]:


overview_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(overview_corpus)
plt.figure(figsize=(16,8))
plt.imshow(overview_wordcloud)
plt.axis('off')
plt.show()


# **Life** is the most commonly used word in Movie titles. **One** and **Find** are also popular in Movie Blurbs. Together with **Love**, **Man** and **Girl**, these wordclouds give us a pretty good idea of the most popular themes present in movies. 

# ### Production Countries
# 
# The Full MovieLens Dataset consists of movies that are overwhelmingly in the English language (more than 31000). However, these movies may have shot in various locations around the world. It would be interesting to see which countries serve as the most popular destinations for shooting movies by filmmakers, especially those in the United States of America and the United Kingdom.

# In[ ]:


df['production_countries'] = df['production_countries'].fillna('[]').apply(ast.literal_eval)
df['production_countries'] = df['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


s = df.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'countries'


# In[ ]:


con_df = df.drop('production_countries', axis=1).join(s)
con_df = pd.DataFrame(con_df['countries'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_movies', 'country']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(10)


# In[ ]:


con_df = con_df[con_df['country'] != 'United States of America']


# In[ ]:


data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_movies'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(255, 0, 0)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]

layout = dict(
    title = 'Production Countries for the MovieLens Movies (Apart from US)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )


# Unsurprisingly, the **United States** is the most popular destination of production for movies given that our dataset largely consists of English movies. **Europe** is also an extremely popular location with the UK, France, Germany and Italy in the top 5. **Japan** and **India** are the most popular Asian countries when it comes to movie production.

# ### Franchise Movies
# 
# Let us now have a brief look at Franchise movies. I was curious to discover the longest running and the most successful franchises among many other things. Let us wrangle our data to find out!

# In[ ]:


df_fran = df[df['belongs_to_collection'].notnull()]
df_fran['belongs_to_collection'] = df_fran['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)
df_fran = df_fran[df_fran['belongs_to_collection'].notnull()]


# In[ ]:


fran_pivot = df_fran.pivot_table(index='belongs_to_collection', values='revenue', aggfunc={'revenue': ['mean', 'sum', 'count']}).reset_index()


# #### Highest Grossing Movie Franchises 

# In[ ]:


fran_pivot.sort_values('sum', ascending=False).head(10)


# The **Harry Potter** Franchise is the most successful movie franchise raking in more than 7.707 billion dollars from 8 movies. The **Star Wars** Movies come in a close second with a 7.403 billion dollars from 8 movies too. **James Bond**
# is third but the franchise has significantly more movies compared to the others in the list and therefore, a much smaller average gross.

# #### Most Successful Movie Franchises (by Average Gross)
# 
# We will use the average gross per movie to gauge the success of a movie franchise. However, this is not a very potent metric as the revenues in this dataset have not been adjusted for inflation. Therefore, revenue statistics will tend to strongly favor franchises in the recent times.

# In[ ]:


fran_pivot.sort_values('mean', ascending=False).head(10)


# The **Avatar** Collection, although just consisting of one movie at the moment, is the most successful franchise of all time with the sole movie raking in close to 3 billion dollars. The **Harry Potter** franchise is still the most successful franchise with at least 5 movies.

# #### Longest Running Franchises
# 
# Finally, in this subsection, let us take a look at the franchises which have stood the test of time and have managed to deliver the largest number of movies under a single banner. This metric is potent in the way that it isn't affected by inflation. However, this does not imply that successful movie franchises tend to have more movies. Some franchises, such as Harry Potter, have a predefined storyline and it wouldn't make sense to produce more movies despite its enormous success.

# In[ ]:


fran_pivot.sort_values('count', ascending=False).head(10)


# The **James Bond** Movies is the largest franchise ever with over 26 movies released under the banner. **Friday the 13th** and **Pokemon** come in at a distant second and third with 12 and 11 movies respectively.

# ### Production Companies

# In[ ]:


df['production_companies'] = df['production_companies'].fillna('[]').apply(ast.literal_eval)
df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


s = df.apply(lambda x: pd.Series(x['production_companies']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'companies'


# In[ ]:


com_df = df.drop('production_companies', axis=1).join(s)


# In[ ]:


com_sum = pd.DataFrame(com_df.groupby('companies')['revenue'].sum().sort_values(ascending=False))
com_sum.columns = ['Total']
com_mean = pd.DataFrame(com_df.groupby('companies')['revenue'].mean().sort_values(ascending=False))
com_mean.columns = ['Average']
com_count = pd.DataFrame(com_df.groupby('companies')['revenue'].count().sort_values(ascending=False))
com_count.columns = ['Number']

com_pivot = pd.concat((com_sum, com_mean, com_count), axis=1)


# #### Highest Earning Production Companies
# 
# Let us find out which production companies have earned the most money from the movie making business.

# In[ ]:


com_pivot.sort_values('Total', ascending=False).head(10)


# **Warner Bros** is the highest earning production company of all time earning a staggering 63.5 billion dollars from close to 500 movies. **Universal Pictures** and **Paramaount Pictures** are the second and the third highest earning companies with 55 billion dollars and 48 billion dollars in revenue respectively. 

# #### Most Succesful Production Companies
# 
# Which production companies produce the most succesful movies on average? Let us find out. We will only consider those companies that have made at least 15 movies.

# In[ ]:


com_pivot[com_pivot['Number'] >= 15].sort_values('Average', ascending=False).head(10)


# **Pixar Animation Studios** has produced the most successful movies, on average. This is not surprising considering the amazing array of movies that it has produced in the last few decades: Up, Finding Nemo, Inside Out, Wall-E, Ratatouille, the Toy Story Franchise, Cars Franchise, etc. **Marvel Studios** with an average gross of 615 million dollars comes in second with movies such as Iron Man and The Avengers under its banner.

# ### Original Language
# 
# In this section, let us look at the languages of the movies in our dataset. From the production countries, we have already deduced that the majority of the movies in the dataset are English. Let us see what the other major languages represented are.

# In[ ]:


df['original_language'].drop_duplicates().shape[0]


# In[ ]:


lang_df = pd.DataFrame(df['original_language'].value_counts())
lang_df['language'] = lang_df.index
lang_df.columns = ['number', 'language']
lang_df.head()


# There are over 93 languages represented in our dataset. As we had expected, English language films form the overwhelmingly majority. French and Italian movies come at a very distant second and third respectively. Let us represent the most popular languages (apart from English) in the form of a bar plot.

# In[ ]:


plt.figure(figsize=(12,5))
sns.barplot(x='language', y='number', data=lang_df.iloc[1:11])
plt.show()


# As mentioned earlier, **French** and **Italian** are the most commonly occurring languages after English. **Japanese** and **Hindi** form the majority as far as Asian Languages are concerned.

# ### Popularity, Vote Average and Vote Count
# 
# In this section, we will work with metrics provided to us by TMDB users. We will try to gain a deeper understanding of the popularity, vote average and vote count features and try and deduce any relationships between them as well as other numeric features such as budget and revenue.

# In[ ]:


def clean_numeric(x):
    try:
        return float(x)
    except:
        return np.nan


# In[ ]:


df['popularity'] = df['popularity'].apply(clean_numeric).astype('float')
df['vote_count'] = df['vote_count'].apply(clean_numeric).astype('float')
df['vote_average'] = df['vote_average'].apply(clean_numeric).astype('float')


# Let us examine the summary statistics and the distribution of each feature one by one.

# In[ ]:


df['popularity'].describe()


# In[ ]:


sns.distplot(df['popularity'].fillna(df['popularity'].median()))
plt.show()


# In[ ]:


df['popularity'].plot(logy=True, kind='hist')


# The Popularity score seems to be an extremely skewed quentity with a mean of only **2.9** but maximum values reaching as high as 547, which is almost 1800% greater than the mean. However, as can be seen from the distribution plot, almost all movies have a popularity score less than 10 (the 75th percentile is at 3.678902).

# #### Most Popular Movies by Popularity Score

# In[ ]:


df[['title', 'popularity', 'year']].sort_values('popularity', ascending=False).head(10)


# **Minions** is the most popular movie by the TMDB Popularity Score. **Wonder Woman** and **Beauty and the Beast**, two extremely successful woman centric movies come in second and third respectively.

# In[ ]:


df['vote_count'].describe()


# As with popularity scores, the distribution of vote counts is extremely skewed with the median vote count standing at a paltry 10 votes. The most votes a single movie has got stands at 14,075. TMDB Votes, therefore, are not as potent and suggestive as its IMDB Counterpart. Nevertheless, let us check which the most voted on movies on the website are.

# #### Most Voted on Movies

# In[ ]:


df[['title', 'vote_count', 'year']].sort_values('vote_count', ascending=False).head(10)


# **Inception** and **The Dark Knight**, two critically acclaimed and commercially successful Christopher Nolan movies figure at the top of our chart. 

# In[ ]:


df['vote_average'] = df['vote_average'].replace(0, np.nan)
df['vote_average'].describe()


# In[ ]:


sns.distplot(df['vote_average'].fillna(df['vote_average'].median()))


# It appears that TMDB Users are extremely strict in their ratings. The mean rating is only a **5.6** on a scale of 10. Half the movies have a rating of less than or equal to 6. Let us check what the most critically acclaimed movies as per TMDB are. We will only consider those movies that have more than 2000 votes (similar to IMDB's criteria of 5000 votes in selecting its top 250).

# #### Most Critically Acclaimed Movies

# In[ ]:


df[df['vote_count'] > 2000][['title', 'vote_average', 'vote_count' ,'year']].sort_values('vote_average', ascending=False).head(10)


# **The Shawshank Redemption** and **The Godfather** are the two most critically acclaimed movies in the TMDB Database. Interestingly, they are the top 2 movies in IMDB's Top 250 Movies list too. They have a rating of over 9 on IMDB as compared to their 8.5 TMDB Scores.

# Do popularity and vote average share a tangible relationship? In other words, is there a strong positive correlation between these two quanitties? Let us visualise their relationship in the form of a scatterplot.

# In[ ]:


sns.jointplot(x='vote_average', y='popularity', data=df)


# Surprisingly, the Pearson Coefficient of the two aforementioned quantities is a measly **0.097** which suggests that **there is no tangible correlation**. In other words, popularity and vote average and independent quantities. It would be interesting to discover how TMDB assigns numerical popularity scores to its movies.

# In[ ]:


sns.jointplot(x='vote_average', y='vote_count', data=df)


# There is a very small correlation between Vote Count and Vote Average. A large number of votes on a particular movie does not necessarily imply that the movie is good.

# ### Movie Release Dates

# Release Dates can often play a very important role in determining the success and the revenue generated by a particular movie. In this section, we will try and gain insights about release dates in terms of years, months and days of the week. 
# 
# We have already constructed the **year** feature in our preliminary data wrangling step. Let us now extract the month and day too for each movie with a release date.

# In[ ]:


month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


# In[ ]:


def get_month(x):
    try:
        return month_order[int(str(x).split('-')[1]) - 1]
    except:
        return np.nan


# In[ ]:


def get_day(x):
    try:
        year, month, day = (int(i) for i in x.split('-'))    
        answer = datetime.date(year, month, day).weekday()
        return day_order[answer]
    except:
        return np.nan


# In[ ]:


df['day'] = df['release_date'].apply(get_day)
df['month'] = df['release_date'].apply(get_month)


# With these features in hand, let us now check the most popular and most successful months and days.

# In[ ]:


plt.figure(figsize=(12,6))
plt.title("Number of Movies released in a particular month.")
sns.countplot(x='month', data=df, order=month_order)


# It appears that **January** is the most popular month when it comes to movie releases. In Hollywood circles, this is also known as the *the dump month* when sub par movies are released by the dozen. 
# 
# In which months do bockbuster movies tend to release? To answer this question, we will consider all movies that have made in excess of 100 million dollars and calculate the average gross for each month.

# In[ ]:


month_mean = pd.DataFrame(df[df['revenue'] > 1e8].groupby('month')['revenue'].mean())
month_mean['mon'] = month_mean.index
plt.figure(figsize=(12,6))
plt.title("Average Gross by the Month for Blockbuster Movies")
sns.barplot(x='mon', y='revenue', data=month_mean, order=month_order)


# We see that the months of **April**, **May** and **June** have the highest average gross among high grossing movies. This can be attributed to the fact that blockbuster movies are usually released in the summer when the kids are out of school and the parents are on vacation and therefore, the audience is more likely to spend their disposable income on entertainment.
# 
# Do some months tend to be more successful than others? Let us visualise the boxplot between the return and the months.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='month', y='return', data=df[df['return'].notnull()], palette="muted", ax =ax, order=month_order)
ax.set_ylim([0, 12])


# The months of **June** and **July** tend to yield the highest median returns. **September** is the least successful months on the aforementioned metrics. Again, the success of June and July movies can be attributed to them being summer months and times of vacation. September usually denotes the beginning of the school/college semester and hence a slight reduction in the consumption of movies.
# 
# Let us now have a look at the most popular days as we did for months.

# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Number of Movies released on a particular day.")
sns.countplot(x='day', data=df, order=day_order)


# **Friday** is clearly the most popular day for movie releases. This is understandable considering the fact that it usually denotes the beginning of the weekend. **Sunday** and **Monday** are the least popular days and this can be attributed to the same aforementioned reason.

# #### Number of Movies by the year
# 
# The Dataset of 45,000 movies available to us does not represent the entire corpus of movies released since the inception of cinema. However, it is reasomnable to assume that it does include almost every major film released in Hollywood as well as other major film industries across the world (such as Bollywood in India). With this assumption in mind, let us take a look at the number of movies produced by the year.

# In[ ]:


year_count = df.groupby('year')['title'].count()
plt.figure(figsize=(18,5))
year_count.plot()


# We notice that there is a sharp rise in the number of movies **starting the 1990s decade.** However, we will not look too much into this as it is entirely possible that recent movies were oversampled for the purposes of this dataset.
# 
# Next, let us take a look at the earliest movies represented in the dataset.
# 
# #### Earliest Movies Represented

# In[ ]:


df[df['year'] != 'NaT'][['title', 'year']].sort_values('year').head(10)


# The oldest movie, **Passage of Venus**, wss a series of photographs of the transit of the planet Venus across the Sun in 1874. They were taken in Japan by the French astronomer Pierre Janssen using his 'photographic revolver'. This is also the oldest movie on both IMDB and TMDB.

# Finally, in this section, let us construct a heatmap to indicate movie releases by month and year for all movies released in this century. This will give us a good idea of the *hot* and *cold* months for movie buffs.

# In[ ]:


months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


# In[ ]:


df_21 = df.copy()
df_21['year'] = df_21[df_21['year'] != 'NaT']['year'].astype(int)
df_21 = df_21[df_21['year'] >=2000]
hmap_21 = pd.pivot_table(data=df_21, index='month', columns='year', aggfunc='count', values='title')
hmap_21 = hmap_21.fillna(0)


# In[ ]:


sns.set(font_scale=1)
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(hmap_21, annot=True, linewidths=.5, ax=ax, fmt='n', yticklabels=month_order)


# In[ ]:


sns.set(font_scale=1.25)


# ### Movie Status
# 
# Although not entirely relevant to our analysis of movies, gathering information on the various kinds of movies based on their status of release can provide us interesitng insight on the nature of the movies present in our dataset. My preliminary hunch was that almost every movie has the **Released** status. Let's find out.

# In[ ]:


df['status'].value_counts()


# Almost every movie is indeed released. However, it is interesting to see that MovieLens has user ratings for movies that are still in the planning, production and post production stage. We might take this information into account while building our collaborative filtering recommendation engine.

# ### Spoken Languages
# 
# Does the number of spoken languages influence the success of a movie? To do this, we will convert our **spoken_languages** feature to a numeric feature denoting the number of languages spoken in that film.

# In[ ]:


df['spoken_languages'] = df['spoken_languages'].fillna('[]').apply(ast.literal_eval).apply(lambda x: len(x) if isinstance(x, list) else np.nan)


# In[ ]:


df['spoken_languages'].value_counts()


# Most movies have just one language spoken in the entire duration of the film. **19** is the higest number of languages spoken in a film. Let us take a look at all the films with more than 10 spoken languages.

# In[ ]:


df[df['spoken_languages'] >= 10][['title', 'year', 'spoken_languages']].sort_values('spoken_languages', ascending=False)


# The movie with the most number of languages, **Visions of Europe** is actually a collection of 25 short films by 25 different European directors. This explains the sheer diversity of the movie in terms of language.

# In[ ]:


sns.jointplot(x="spoken_languages", y="return", data=df, stat_func=stats.spearmanr, color="m")


# The **Spearman Coefficient** is 0.018 indicating no correlation between the two quantities.

# ### Runtime
# 
# From its humble beginnings of 1 minute slient, black & white clips to epic 3 hour visual delights, movies have a come a long way in terms of runtime. In this section, let us try and gain some additional insights about the nature of movie lengths and their evolution over time.

# In[ ]:


df['runtime'].describe()


# The average length of a movie is about 1 hour and 30 minutes. The longest movie on record in this dataset is a **staggering 1256 minutes (or 20 hours) long.**

# In[ ]:


df['runtime'] = df['runtime'].astype('float')


# We are aware that most movies are less than 5 hours (or 300 minutes) long. Let us plot a distribution of these mainstream movies.

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(df[(df['runtime'] < 300) & (df['runtime'] > 0)]['runtime'])


# Is there any meaningful relationship between runtime and return? Let us find out!

# In[ ]:


df_mat = df[(df['return'].notnull()) & (df['runtime'] > 0) & (df['return'] < 10)]
sns.jointplot('return', 'runtime', data=df_mat)
plt.show()


# There seems to be relationship between the two quantities. **The duration of a movie is independent of its success.** However, I have a feeling this might not be the case with duration and budget. A longer movie should entail a higher budget. Let us find out if this is really the case.

# In[ ]:


df_mat = df[(df['budget'].notnull()) & (df['runtime'] > 0)]
sns.jointplot('budget', 'runtime', data=df_mat)
plt.show()


# The two quantities have a much weaker correlation than I had expected. In retrospect, the genre of the movie tends to have a much greater impact on budget. A 3 hour art film will cost significantly lesser than a 90 minute Sci-Fi movie. 
# 
# Next, I'd like to see the average lengths of movies through time, right from the 1890s to the 2017s. It would be interesting to see the trends in what filmmakers adjudged would be the appropriate length of a movie at that time.

# In[ ]:


plt.figure(figsize=(18,5))
year_runtime = df[df['year'] != 'NaT'].groupby('year')['runtime'].mean()
plt.plot(year_runtime.index, year_runtime)
plt.xticks(np.arange(1874, 2024, 10.0))
plt.show()


# We notice that films started hitting the **60 minute mark as early as 1914**. Starting **1924**, films started having the traiditonal 90 minute duration and has remained more or less constant ever since.
# 
# Finally in this section, let us see the longest and the shortest movies of all time (with respect to the movies in the dataset). 

# #### Shortest Movies

# In[ ]:


df[df['runtime'] > 0][['runtime', 'title', 'year']].sort_values('runtime').head(10)


# We see that every movie in this list except **A Gathering of Cats** were filmed in the late 1890s and the beginning of the 20th century. All these movies were one minute long.

# #### Longest Movies

# In[ ]:


df[df['runtime'] > 0][['runtime', 'title', 'year']].sort_values('runtime', ascending=False).head(10)


# We notice that almost all the entries in the above chart are actually miniseries and hence, do not count as feature length films. We cannot gather too much insight from this list of longest movies as there is no way of distinguishing feature length films from TV Mini Series from our dataset (except, of course, by doing it manually).

# ### Budget
# 
# Let us now turn our attention to budget. We expect budgets to be a skewed quantity and also heavily influenced by inflation. Nevertheless, it would be interesting to gather as much insights as possible from this quantity as budget is often a critical feature in predicting movie revenue and success. As a start, let us gather the summary statistics for our budget.

# In[ ]:


df['budget'].describe()


# The mean budget of a film is 21.6 million dollars whereas the median budget is far smaller at 8 million dollars. This strongly suggests the mean being influenced by outliers. 

# In[ ]:


sns.distplot(df[df['budget'].notnull()]['budget'])


# In[ ]:


df['budget'].plot(logy=True, kind='hist')


# The distribution of movie budgets shows an exponential decay. More than 75% of the movies have a budget smaller than 25 million dollars. Next, let us take a look at the most expensive movies of all time and the revenue & returns that they generated.

# #### Most Expensive Movies of all Time

# In[ ]:


df[df['budget'].notnull()][['title', 'budget', 'revenue', 'return', 'year']].sort_values('budget', ascending=False).head(10)


# Two **Pirates of the Carribean** films occupy the top spots in this list with a staggering budget of over **300 million dollars**. All the top 10 most expensive films made a profit on their investment except for **The Lone Ranger** which managed to recoup less than 35% of its investment, taking in a paltry 90 million dollars on a **255 million dollar** budget.
# 
# How strong a correlation does the budget hold with the revenue? A stronger correlation would directly imply more accurate forecasts.

# In[ ]:


sns.jointplot(x='budget',y='revenue',data=df[df['return'].notnull()])


# The pearson r value of **0.73** between the two quantities indicates a very strong correlation. 

# ### Revenue
# 
# The final numeric feature we will explore is the revenue. The revenue is probably the most important numeric quantity associated with a movie. We will try to predict the revenue for movies given a set of features in a later section. The treatment of revenue will be very similar to that of budget and we will once again begin by studying the summary statistics.

# In[ ]:


df['revenue'].describe()


# The mean gross of a movie is **68.7 million dollars** whereas the median gross is much lower at **16.8 million dollars**, suggesting the skewed nature of revenue. The lowest revenue generated by a movie is **just 1 dollar** whereas the highest grossing movie of all time has raked in an astonishing **2.78 billion dollars.*

# In[ ]:


sns.distplot(df[df['revenue'].notnull()]['revenue'])


# The distribution of revenue undergoes exponential decay just like budget. We also found that the two quantities were strongly correlated. Let us now take a look at the highest and least grossing movies of all time.

# #### Highest Grossing Films of All Time

# In[ ]:


gross_top = df[['poster_path', 'title', 'budget', 'revenue', 'year']].sort_values('revenue', ascending=False).head(10)
pd.set_option('display.max_colwidth', 100)
HTML(gross_top.to_html(escape=False))


# In[ ]:


pd.set_option('display.max_colwidth', 50)


# These figures have not been adjusted for inflation. Therefore, we see a disproportionate number of movies from very recent times in the top 10 list. To get an understanding of the revenue garnered by movies, let us plot the maximum revenue through the years.

# In[ ]:


plt.figure(figsize=(18,5))
year_revenue = df[(df['revenue'].notnull()) & (df['year'] != 'NaT')].groupby('year')['revenue'].max()
plt.plot(year_revenue.index, year_revenue)
plt.xticks(np.arange(1874, 2024, 10.0))
plt.show()


# As can be seen from the figure, the maximum gross has steadily risen over the years. The world of movies broke the 1 billion dollar mark in 1997 with the release of **Titanic**. It took another 12 years to break the 2 billion dollar mark with **Avatar**. Both these movies were directed by James Cameron.

# ### Returns
# 
# We will not look too much into returns for the time being. Let us just check the least and the most successful movies of all time. To do this, we will only consider those movies which have a budget greater than 5 million dollars.

# #### Most Successful Movies

# In[ ]:


df[(df['return'].notnull()) & (df['budget'] > 5e6)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return', ascending=False).head(10)


# #### Worst Box Office Disasters

# In[ ]:


df[(df['return'].notnull()) & (df['budget'] > 5e6) & (df['revenue'] > 10000)][['title', 'budget', 'revenue', 'return', 'year']].sort_values('return').head(10)


# With these analyses in place, we are in a good position to construct our correlation matrix.

# In[ ]:


df['year'] = df['year'].replace('NaT', np.nan)


# In[ ]:


df['year'] = df['year'].apply(clean_numeric)


# In[ ]:


sns.set(font_scale=1)
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    plt.figure(figsize=(9,9))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)


# In[ ]:


sns.set(font_scale=1.25)


# ### Genres

# In[ ]:


df['genres'] = df['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'


# In[ ]:


gen_df = df.drop('genres', axis=1).join(s)


# In[ ]:


gen_df['genre'].value_counts().shape[0]


# TMDB defines 32 different genres for our set of 45,000 movies. Let us now have a look at the most commonly occuring genres in movies.

# In[ ]:


pop_gen = pd.DataFrame(gen_df['genre'].value_counts()).reset_index()
pop_gen.columns = ['genre', 'movies']
pop_gen.head(10)


# In[ ]:


plt.figure(figsize=(18,8))
sns.barplot(x='genre', y='movies', data=pop_gen.head(15))
plt.show()


# **Drama** is the most commonly occurring genre with almost half the movies identifying itself as a drama film. **Comedy** comes in at a distant second with 25% of the movies having adequate doses of humor. Other major genres represented in the top 10 are Action, Horror, Crime, Mystery, Science Fiction, Animation and Fantasy.
# 
# The next question I want to answer is the trends in the share of genres of movies across the world. Has the demand for Science Fiction movies increased? Do certain years have a disproportionate share of Animation Movies? Let's find out!
# 
# We will only be looking at trends starting 2000. We will consider only those themes that appear in the top 15 most popular genres. We will exclude Documentaries, Family and Foreign Movies from our analysis.

# In[ ]:


genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Adventure', 'Science Fiction', 'Mystery', 'Fantasy', 'Mystery', 'Animation']


# In[ ]:


pop_gen_movies = gen_df[(gen_df['genre'].isin(genres)) & (gen_df['year'] >= 2000) & (gen_df['year'] <= 2017)]
ctab = pd.crosstab([pop_gen_movies['year']], pop_gen_movies['genre']).apply(lambda x: x/x.sum(), axis=1)
ctab[genres].plot(kind='bar', stacked=True, colormap='jet', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stacked Bar Chart of Movie Proportions by Genre")
plt.show()


# In[ ]:


ctab[genres].plot(kind='line', stacked=False, colormap='jet', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# The proportion of movies of each genre has remained fairly constant since the beginning of this century except for **Drama**. The proportion of drama films has fallen by over 5%.  **Thriller** movies have enjoyed a slight increase in their share. 
# 
# One question that I have always had is that if some genres are particularly more successful than others. For example, we should expect Science Fiction and Fantasy Movies to bring in more revenue than other genres but when normalized with their budget, do they prove to be as successful? We will visualize two violin plots to answer this question. One will be genres versus the revenue while the other will be versus returns.

# In[ ]:


violin_genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Science Fiction', 'Fantasy', 'Animation']
violin_movies = gen_df[(gen_df['genre'].isin(violin_genres))]


# In[ ]:


plt.figure(figsize=(18,8))
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='genre', y='revenue', data=violin_movies, palette="muted", ax =ax)
ax.set_ylim([0, 3e8])
plt.show()


# **Animation** movies has the largest 25-75 range as well as the median revenue among all the genres plotted. **Fantasy** and **Science Fiction** have the second and third highest median revenue respectively. 

# In[ ]:


plt.figure(figsize=(18,8))
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='genre', y='return', data=violin_movies, palette="muted", ax =ax)
ax.set_ylim([0, 10])
plt.show()


# From the boxplot, it seems like **Animation** Movies tend to yield the highest returns on average. **Horror** Movies also tend to be a good bet. This is partially due to the nature of Horror movies being low budget compared to Fantasy Movies but being capable of generating very high revenues relative to its budget.

# ### Cast and Crew
# 
# Let us now take a look at the cast and crew of our movies. We do not have these details with us in our main dataset. However, we have a separate file consisting of the full cast and crew credits of all the Movielens Movies. Let us take a look at this credits data.

# In[ ]:


credits_df = pd.read_csv('../input/credits.csv')
credits_df.head()


# #### Credits Dataset
# 
# * **cast:** A stringified list of dictionaries consisting of cast names and the corresponding characters they played.
# * **crew:** A stringified list of dictionaries consisting of crew names and the function they performed.
# * **id:** The TMDB ID of movie.

# We need to perform a left join of our original movies metadata dataframe with the credits dataframe on the TMDB Movie ID. Before we are able to perform this join, we need to make sure that the ID column of our main dataframe is clean and of type integer. To do this, let us try to perform an integer conversion of our IDs and if an exception is raised,we will replace the ID with NaN. We will then proceed to drop these rows from our dataframe.

# In[ ]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[ ]:


df['id'] = df['id'].apply(convert_int)


# In[ ]:


df[df['id'].isnull()]


# In[ ]:


df = df.drop([19730, 29503, 35587])


# In[ ]:


df['id'] = df['id'].astype('int')


# In[ ]:


df = df.merge(credits_df, on='id')
df.shape


# In[ ]:


df['cast'] = df['cast'].apply(ast.literal_eval)
df['crew'] = df['crew'].apply(ast.literal_eval)


# In[ ]:


df['cast_size'] = df['cast'].apply(lambda x: len(x))
df['crew_size'] = df['crew'].apply(lambda x: len(x))


# In[ ]:


df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[ ]:


df['director'] = df['crew'].apply(get_director)


# In[ ]:


s = df.apply(lambda x: pd.Series(x['cast']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'actor'
cast_df = df.drop('cast', axis=1).join(s)


# Let us now take a look at the actors and the directors who have raked in the most amount of money with their movies.
# 
# #### Actors with the Highest Total Revenue

# In[ ]:


sns.set_style('whitegrid')
plt.title('Actors with the Highest Total Revenue')
cast_df.groupby('actor')['revenue'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.show()


# #### Directors with the Highest Total Revenue

# In[ ]:


plt.title('Directors with the Highest Total Revenue')
df.groupby('director')['revenue'].sum().sort_values(ascending=False).head(10).plot(kind='bar', colormap='autumn')
plt.show()


# For average revenues, we will consider only actors and directors who have acted and directed in at least 5 movies respectively.

# In[ ]:


actor_list = cast_df.groupby('actor')['revenue'].count().sort_values(ascending=False)
actor_list = list(actor_list[actor_list >= 5].index)
director_list = df.groupby('director')['revenue'].count().sort_values(ascending=False)
director_list = list(director_list[director_list >= 5].index)


# #### Actors with Highest Average Revenue

# In[ ]:


plt.title("Actors with Highest Average Revenue")
cast_df[cast_df['actor'].isin(actor_list)].groupby('actor')['revenue'].mean().sort_values(ascending=False).head(10).plot(kind='bar', colormap='Greens_r')
plt.show()


# #### Directors with Highest Average Revenue

# In[ ]:


plt.title("Directors with Highest Average Revenue")
df[df['director'].isin(director_list)].groupby('director')['revenue'].mean().sort_values(ascending=False).head(10).plot(kind='bar', colormap='autumn_r')
plt.show()


# Which actors and directors are the safest bet? For this, we will consider the average return brought in by a particular director or actor. We will only consider those movies that have raked in at least 10 million dollars. Also, we will only consider actors and directors that have worked in at least 5 films.
# 
# #### Most Successful Actors

# In[ ]:


success_df = cast_df[(cast_df['return'].notnull()) & (cast_df['revenue'] > 1e7) & (cast_df['actor'].isin(actor_list))]
pd.DataFrame(success_df.groupby('actor')['return'].mean().sort_values(ascending=False).head(10))


# #### Most Successful Directors

# In[ ]:


success_df = df[(df['return'].notnull()) & (df['revenue'] > 1e7) & (df['director'].isin(director_list))]
pd.DataFrame(success_df.groupby('director')['return'].mean().sort_values(ascending=False).head(10))


# **John G. Avildsen** has an unnaturally high return. All the other directors in the list are nowhere close to the level of return that he has gained. Let us have a look at his movies.

# In[ ]:


df[(df['director'] == 'John G. Avildsen') & (df['return'].notnull())][['title', 'budget', 'revenue', 'return', 'year']]


# **The Karate Kid, Part II** has a budget of only 113 dollars. This seems like an anomaly since official figures state that the movie cost **13 million dollars**. So, although he has directed amazing movies, he does not belong to this list.
# 
# We will end our Exploratory Data Analysis over here. Let us use some of the insights we gained in this section and build some useful predictive models.

# ## Regression: Predicting Movie Revenues

# In this section, I will attempt at building a regression model to predict movie revenues. Since this isn't one of the main goals of this project, I will not spend too much time feature engineering or hyperparamater tuning my model. 
# 
# Predicting Movie Revenues is an extremely popular problem in Machine Learning which has created a huge amount of literature. Most of the models proposed in these papers use far more potent features than what we possess at the moment. These include Facebook Page Likes, Information on Tweets about the Movie, YouTube Trailer Reaction (Views, Likes, Dislikes, etc.), Movie Rating (MPCAA, CBIFC) among many others.
# 
# For a great project on Movie Box Office Predictions, check out Aarti Annaswamy's excellent project here: [Movies Capstone](https://github.com/aannasw/movies-capstone)
# 
# To compensate for the lack of these features, we are going to cheat a little. We will be using TMDB's **Popularity Score** and **Vote Average** as our features in our model to assign a nuerical value to popularity. However, it must be kept in mind that these metrics will not be available when predicting movie revenues in the real world, when the movie has not been released yet.

# In[ ]:


rgf = df[df['return'].notnull()]
rgf.shape


# We have **5393 records** in our training set. Let us take a look at the features we possess and remove the ones which are unnecessary.

# In[ ]:


rgf.columns


# In[ ]:


rgf = rgf.drop(['id', 'overview', 'poster_path', 'release_date', 'status', 'tagline', 'video', 'return', 'crew'], axis=1)


# We will perform the following feature engineering tasks:
# 
# 1. **belongs_to_collection** will be turned into a Boolean variable. 1 indicates a movie is a part of collection whereas 0 indicates it is not.
# 2. **genres** will be converted into number of genres.
# 3. **homepage** will be converted into a Boolean variable that will indicate if a movie has a homepage or not.
# 4. **original_language** will be replaced by a feature called **is_foreign** to denote if a particular film is in English or a Foreign Language.
# 5. **production_companies** will be replaced with just the number of production companies collaborating to make the movie.
# 6. **production_countries** will be replaced with the number of countries the film was shot in.
# 7. **day** will be converted into a binary feature to indicate if the film was released on a Friday.
# 8. **month** will be converted into a variable that indicates if the month was a holiday season.

# In[ ]:


s = rgf.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_rgf = rgf.drop('genres', axis=1).join(s)
genres_train = gen_rgf['genre'].drop_duplicates()


# In[ ]:


def feature_engineering(df):
    df['belongs_to_collection'] = df['belongs_to_collection'].apply(lambda x: 0 if x == np.nan else 1)
    for genre in genres_train:
        df['is_' + str(genre)] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    df['genres'] = df['genres'].apply(lambda x: len(x))
    df['homepage'] = df['homepage'].apply(lambda x: 0 if x == np.nan else 1)
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x=='en' else 0)
    df = df.drop('original_language', axis=1)
    df['production_companies'] = df['production_companies'].apply(lambda x: len(x))
    df['production_countries'] = df['production_countries'].apply(lambda x: len(x))
    df['is_Friday'] = df['day'].apply(lambda x: 1 if x=='Fri' else 0)
    df = df.drop('day', axis=1)
    df['is_Holiday'] = df['month'].apply(lambda x: 1 if x in ['Apr', 'May', 'Jun', 'Nov'] else 0)
    df = df.drop('month', axis=1)
    df = df.drop(['title', 'cast', 'director'], axis=1)
    df = pd.get_dummies(df, prefix='is')
    df['runtime'] = df['runtime'].fillna(df['runtime'].mean())
    df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())
    return df


# In[ ]:


X, y = rgf.drop('revenue', axis=1), rgf['revenue']


# In[ ]:


X = feature_engineering(X)


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, test_size=0.25)


# In[ ]:


X.shape


# In[ ]:


reg = GradientBoostingRegressor()
reg.fit(train_X, train_y)
reg.score(test_X, test_y)


# We get a Coefficient of Determination of **0.78** which is a pretty score for the basic model that we have built. Let us compare our model's score to a Dummy Regressor.

# In[ ]:


dummy = DummyRegressor()
dummy.fit(train_X, train_y)
dummy.score(test_X, test_y)


# We see that our model performs far more superiorly than the Dummy Regressor. Finally, let us plot the feature importances in the form of a bar plot to deduce which features were the most significant in our making predictions.

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,12))
sns.barplot(x=reg.feature_importances_, y=X.columns)


# We notice that **vote_count**, a feature we *cheated* with, is the most important feature to our Gradient Boosting Model. This goes on to show the improtance of popularity metrics in determining the revenue of a movie. **Budget** was the second most important feature followed by **Popularity** (Literally, a popularity metric) and **Crew Size**.

# ## Classification: Predicting Movie Sucess
# 
# What factors influence whether a movie will be able to recoup its investments? I will try to answer this question by building a binary classifier that predicts if a movie will make a profit or a loss. As with our regression model, we will cheat a little and use features that may not be available to us in the real world for the lack of other useful popularity metrics.
# 
# We have already performed extensive analysis of our data and haven't done a lot with respect to determining factors that make a movie a success. We shall attempt at doing that in this section and follow it up by building our model.

# In[ ]:


cls = df[df['return'].notnull()]
cls.shape


# In[ ]:


cls.columns


# In[ ]:


cls = cls.drop(['id', 'overview', 'poster_path', 'release_date', 'status', 'tagline', 'revenue'], axis=1)


# Let us convert our **return** feature into a binary variable that will serve as our classes: **0** indicating a flop and **1** indicating a hit.

# In[ ]:


cls['return'] = cls['return'].apply(lambda x: 1 if x >=1 else 0)


# In[ ]:


cls['return'].value_counts()


# Our classes seem to be fairly balanced. We do not need to apply any additional methods to deal with the imbalance of classes. Let us now turn our attention to our features.

# In[ ]:


cls['belongs_to_collection'] = cls['belongs_to_collection'].fillna('').apply(lambda x: 0 if x == '' else 1)


# In[ ]:


sns.set(style="whitegrid")
g = sns.PairGrid(data=cls, x_vars=['belongs_to_collection'], y_vars='return', size=5)
g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
g.set(ylim=(0, 1))


# It seems that movies that belong to a franchise have a higher probability of being a success. 

# In[ ]:


cls['homepage'] = cls['homepage'].fillna('').apply(lambda x: 0 if x == '' else 1)
g = sns.PairGrid(data=cls, x_vars=['homepage'], y_vars='return', size=5)
g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
g.set(ylim=(0, 1))


# We see that with homepages, there is not a very huge difference in probability. To avoid the curse of dimensionality, we will eliminate this feature as it is not very useful.

# In[ ]:


s = cls.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_cls = cls.drop('genres', axis=1).join(s)


# In[ ]:


ctab = pd.crosstab([gen_cls['genre']], gen_cls['return'], dropna=False).apply(lambda x: x/x.sum(), axis=1)
ctab.plot(kind='bar', stacked=True, legend=False)


# We find that **TV Movies** have a 0% Failure Rate but that is most probably because they are extremely few in numner. **Foreign Films** have a higher rate of failure than average. Since there isn't anything drastic about a particular genre, we will proceed with one hot encoding all genres.

# In[ ]:


cls.columns


# In[ ]:


def classification_engineering(df):
    for genre in genres_train:
        df['is_' + str(genre)] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    df['genres'] = df['genres'].apply(lambda x: len(x))
    df = df.drop('homepage', axis=1)
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x=='en' else 0)
    df = df.drop('original_language', axis=1)
    df['production_companies'] = df['production_companies'].apply(lambda x: len(x))
    df['production_countries'] = df['production_countries'].apply(lambda x: len(x))
    df['is_Friday'] = df['day'].apply(lambda x: 1 if x=='Fri' else 0)
    df = df.drop('day', axis=1)
    df['is_Holiday'] = df['month'].apply(lambda x: 1 if x in ['Apr', 'May', 'Jun', 'Nov'] else 0)
    df = df.drop('month', axis=1)
    df = df.drop(['title', 'cast', 'director'], axis=1)
    #df = pd.get_dummies(df, prefix='is')
    df['runtime'] = df['runtime'].fillna(df['runtime'].mean())
    df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())
    df = df.drop('crew', axis=1)
    return df
    


# In[ ]:


cls = classification_engineering(cls)


# In[ ]:


cls.columns


# In[ ]:


X, y = cls.drop('return', axis=1), cls['return']


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, test_size=0.25, stratify=y)


# In[ ]:


clf = GradientBoostingClassifier()
clf.fit(train_X, train_y)
clf.score(test_X, test_y)


# Our basic Gradient Boosting Classifier has an accuracy of **80%**. Again, this model can be improved upon through hyperparameter tuning and more advanced feature engineering but since this is not the main objective of this project, I will skip this.

# In[ ]:


dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(train_X, train_y)
dummy.score(test_X, test_y)


# In[ ]:


plt.figure(figsize=(10,12))
sns.barplot(x=clf.feature_importances_, y=X.columns)


# We see that **Vote Count** is once again the most significant feature identified by our Classifier. Other important features include **Budget**, **Popularity** and **Year**. With this, we will conclude our discussion on the classification model and move on to the main part of the project.
# 
# In the next notebook, I will try to build a **Hybrid Recommendation System** based on popularity, content and collaborative filtering. This will use the MovieLens dataset in addition to the TMDB Movies Metadata that we have used here. 
# 
# Next -> [Movie Recommender Systems](https://www.kaggle.com/rounakbanik/movie-recommender-systems)
