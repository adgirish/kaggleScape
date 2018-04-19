
# coding: utf-8

# #A quick review of the IGN reviews

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# We need a nice color palette.

# In[ ]:


sns.set_palette('Set3', 10)
sns.palplot(sns.color_palette())
sns.set_context('talk')


# In[ ]:


raw_data = pd.read_csv('../input/ign.csv')


# Upload the entire dataset. It has a row which contains in release_year column 1970 year. I'll remove this easter egg.

# In[ ]:


raw_data.head()


# In[ ]:


release_date = raw_data.apply(lambda x: pd.datetime.strptime("{0} {1} {2} 00:00:00".format(
            x['release_year'],x['release_month'], x['release_day']), "%Y %m %d %H:%M:%S"),axis=1)
raw_data['release_date'] = release_date


# The easter egg.

# In[ ]:


raw_data[raw_data.release_year == 1970]


# In[ ]:


data = raw_data[raw_data.release_year > 1970]
len(data)


# Let's look at all score phrases in IGN reviews...

# In[ ]:


data.score_phrase.unique()


# ... and average scores of each phrase:

# In[ ]:


data.groupby('score_phrase')['score'].mean().sort_values()


# If you don't know the score phrase put so:
# 
#  - 0 - 1: disaster    
# 
#  - 1 - 2: unbearable 
# 
#  - 2 - 3: painful 
# 
#  - and so on
# 

# In[ ]:


data.platform.unique()


# There are all platforms which are found in the dataset:

# ##Releases and dates

# In[ ]:


plt.figure(figsize=(15,8))
data.groupby(['release_day']).size().plot(c='r')
plt.xticks(range(1,32,3))
plt.tight_layout()


# It is a plot of count of releases per days. Nothing interesting.

# In[ ]:


f, ax = plt.subplots(2,1,figsize=(15,10),sharex=True)
data.release_date.dt.weekday.plot.kde(ax=ax[0],c='g')
data.groupby(data.release_date.dt.weekday).size().plot(ax=ax[1],c='r')
plt.xlim(0.,6.)
plt.xticks(range(7),['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.tight_layout()


# But these plots are much more interesting. We see 2 plots: density of probability and count of releases per weekdays. **Maximum of releases has been in Thuesday, minimum has been in weekends.**

# In[ ]:


plt.figure(figsize=(17,8))
plt.xticks(range(1,13),['January','February','March','April','May','June',
            'July','August','September','October','November','December'])
data.groupby(['release_month']).size().plot(c='r')


# Releases per months. Maximum has been in Fall.

# In[ ]:


plt.figure(figsize=(17,8))
data.groupby(['release_year']).size().plot(kind='bar')


# And per yers.

# In[ ]:


table = data.groupby('release_date').size()
f,ax = plt.subplots(2,1,figsize=(17,10))
#table.rolling(window=30).mean().plot(c='orange',ax=ax[1])
table.plot(ax=ax[0],c='red')
ax[0].set_xlabel('')
table.resample('M').mean().plot(c='orange',ax=ax[1])


# The first plot is Count of releases per days in the entire datset. The second is the average count of releases per days in a month. We can see a **cyclic structure**.

# ##The most populars

# In[ ]:


data.platform.value_counts()[:10].plot.pie(figsize=(10,10))


# The top ten most popular gaming platforms...

# In[ ]:


f, ax = plt.subplots(2,2, figsize=(17,17))
last_games = data[data.release_year == 2014]
last_popular = last_games.platform.value_counts()[last_games.platform.value_counts() > 5]
last_popular.plot.pie(ax=ax[0,0])
ax[0,0].set_title('2014')
ax[0,0].set_ylabel('')
last_games = data[data.release_year == 2015]
last_popular = last_games.platform.value_counts()[last_games.platform.value_counts() > 5]
last_popular.plot.pie(ax=ax[0,1])
ax[0,1].set_title('2015')
ax[0,1].set_ylabel('')
last_games = data[data.release_year == 2016]
last_popular = last_games.platform.value_counts()[last_games.platform.value_counts() > 5]
last_popular.plot.pie(ax=ax[1,0])
ax[1,0].set_title('2016')
ax[1,0].set_ylabel('')
old_games = data[data.release_year <= 2000]
old_popular = old_games.platform.value_counts()[old_games.platform.value_counts() > 5]
old_popular.plot.pie(ax=ax[1,1])
ax[1,1].set_title('2000 and older')
ax[1,1].set_ylabel('')


# ... and top platforms in some years.

# In[ ]:


years = tuple(range(1996,2017))
s = data.groupby([data.release_year,data.platform]).title.count()
top_years_platform = pd.DataFrame([[i,s[i].max(),s[i].argmax()] for i in years], 
                                 columns=['release_year','count_games','platform'])

sc = data.groupby([data.release_year,data.platform]).score
s = sc.median()[sc.count() > 20]
top_scores_platform = pd.DataFrame([[i,s[i].max(),s[i].argmax()] for i in years], 
                                 columns=['release_year','score_game','platform'])


# In[ ]:


f, axes = plt.subplots(1,2,figsize=(18,20))

ax = top_years_platform.count_games.plot(kind='barh',color='orange',ax=axes[0])
ax.set_yticklabels(years) 
ax.set_xlabel('Count of releases')
rects = ax.patches
for i, v in enumerate(top_years_platform.platform): 
    ax.text(10, i-.1, v, fontweight='bold')

ax2 = top_scores_platform.score_game.plot(kind='barh',color='blue',ax=axes[1])
ax2.set_yticklabels(years) 
ax2.set_xlabel('Average score')
rects = ax2.patches
for i, v in enumerate(top_scores_platform.platform): 
    ax2.text(0.3, i-.1, v, fontweight='bold', color='white')


# The left plot is the most popular platform in year. 
# The right is a platform with the most high average score. For this purpose games are chosen that had more 20 releases per year.
# **It is interesting that platforms on the left and right are not the same usually.**

# In[ ]:


data_pc = data[data.platform == 'PC']
data_ps = data[data.platform == 'PlayStation']
data_ps2 = data[data.platform == 'PlayStation 2']
data_ps3 = data[data.platform == 'PlayStation 3']
data_ps4 = data[data.platform == 'PlayStation 4']
data_xbox = data[data.platform == 'Xbox']
data_xbox360 = data[data.platform == 'Xbox 360']
data_xbox_one = data[data.platform == 'Xbox One']
df = pd.DataFrame({'PC' : data_pc.groupby('release_year').size(),
                   'PS' : data_ps.groupby('release_year').size(),
                   'PS2' : data_ps2.groupby('release_year').size(),
                   'PS3' : data_ps3.groupby('release_year').size(),
                   'PS4' : data_ps4.groupby('release_year').size(),
                   'Xbox' : data_xbox.groupby('release_year').size(),
                   'Xbox 360' : data_xbox360.groupby('release_year').size(),
                   'Xbox One' : data_xbox_one.groupby('release_year').size()
                  })


# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,20))
df.plot(kind='barh',stacked=True,ax=ax)


# The plot of the platforms by the year of the review

# In[ ]:


data_pc = data[data.platform == 'PC']
plt.figure(figsize=(15,8))
data_pc.groupby('release_year').platform.size().plot(kind='bar',color='green')


# And for PC only.

# ##Scores

# In[ ]:


plt.figure(figsize=(15,8))
plt.xlim(1995,2017)
plt.ylim(1.8,10)
sns.kdeplot(data.release_year, data.score, n_levels=20, cmap="Reds", shade=True, shade_lowest=False)


# In[ ]:


plt.figure(figsize=(15,8))
plt.ylim(1.5,10.5)
plt.xticks(range(1,13),['January','February','March','April','May','June',
            'July','August','September','October','November','December'])
sns.kdeplot(data.release_month, data.score, n_levels=20, cmap="Blues", shade=True, shade_lowest=False)


# In[ ]:


plt.figure(figsize=(15,8))
plt.ylim(1.5,10.5)
sns.kdeplot(data.release_day, data.score, n_levels=20, cmap="Greens", shade=True, shade_lowest=False)


# There are joint distribution of density of scores and dates. Darker areas correspond to more typical values. Well we once again see that **November is the most popular month** among game developers. But now we see one more thing: **a typical score is approximately 8**. Let's check it out.

# In[ ]:


plt.figure(figsize=(17,8))
#sns.kdeplot(data.score, shade=True, c='g', label='Density')
plt.xticks(np.linspace(0,10,21))
plt.xlim(0,10)
data.score.plot.kde(c='g', label='Density')
plt.legend()


# Yes, we have been right. The most typical score is 8. Moreover **reviewers like to put scores near with whole numbers.** I suppose it is a feature of human behavior.

# We can create probability distribution graphics to different platforms. Let's make it for some actual platforms.

# In[ ]:


plt.figure(figsize=(17,10))
plt.xticks(np.linspace(0,10,21))
plt.xlim(0,10)
data.score.plot.kde(label='All platform')
data[data.platform == 'PC'].score.plot.kde(label='PC')
#data[data.platform == 'PlayStation'].score.plot.kde(label='PlayStation')
#data[data.platform == 'PlayStation 2'].score.plot.kde(label='PlayStation 2')
data[data.platform == 'PlayStation 3'].score.plot.kde(label='PlayStation 3')
data[data.platform == 'PlayStation 4'].score.plot.kde(label='PlayStation 4')
plt.legend(loc='upper left')


# In[ ]:


plt.figure(figsize=(17,10))
plt.xticks(np.linspace(0,10,21))
plt.xlim(0,10)
data.score.plot.kde(label='All platform')
data[data.platform == 'PC'].score.plot.kde(label='PC')
#data[data.platform == 'Xbox'].score.plot.kde(label='Xbox')
data[data.platform == 'Xbox 360'].score.plot.kde(label='Xbox 360')
data[data.platform == 'Xbox One'].score.plot.kde(label='Xbox One')
plt.legend(loc='upper left')


# In[ ]:


plt.figure(figsize=(17,10))
plt.xticks(np.linspace(0,10,21))
plt.xlim(0,10)
data.score.plot.kde(label='All platform')
data[data.platform == 'Android'].score.plot.kde(label='Android')
data[data.platform == 'iPhone'].score.plot.kde(label='iPhone')
data[data.platform == 'iPad'].score.plot.kde(label='iPad')
plt.legend(loc='upper left')


# In[ ]:


plt.figure(figsize=(17,10))
plt.xticks(np.linspace(0,10,21))
plt.xlim(0,10)
data.score.plot.kde(label='All platform',c='black')
data[data.platform == 'PC'].score.plot.kde(label='PC')
data[data.platform == 'PlayStation 4'].score.plot.kde(label='PlayStation 4')
data[data.platform == 'Xbox One'].score.plot.kde(label='Xbox One')
data[data.platform == 'iPad'].score.plot.kde(label='iPad')
plt.legend(loc='upper left')


# PC has a peak little bit right than all platforms. **The platforms with the highest average scores are Playstation 4, Xbox One and iPad.** Playstation 4 has won.

# ## Genres

# In[ ]:


genres = data.groupby('genre')['genre']
genres_count=genres.count()
large_genres=genres_count[genres_count>=150]
large_genres.sort_values(ascending=False,inplace=True)
large_genres


# There are all genres in the dataset which have 150 games at least

# In[ ]:


data_genre = data[data.genre.isin(large_genres.keys())]
table_score = pd.pivot_table(data_genre,values=['score'],index=['release_year'],columns=['genre'],aggfunc='mean',margins=False)
table_count = pd.pivot_table(data_genre,values=['score'],index=['release_year'],columns=['genre'],aggfunc='count',margins=False)
table = table_score[table_count > 10]
plt.figure(figsize=(19,16))
sns.heatmap(table.score,linewidths=.5,annot=True,vmin=0,vmax=10,cmap='YlGnBu')
plt.title('Average scores of games (cell exists if a genre has at least 10 releases in year)')


# In[ ]:


plt.figure(figsize=(19,16))
sns.heatmap(table_count.score,linewidths=.5,annot=True,fmt='2.0f',vmin=0)
plt.title('Count of games')


# **People love actions, adventures, shooters, sports, strategies and RPG** more than another genres. We see it in the second plot. Also <b>action-adventures and RPG have higher average scores</b> than another ones. It is interesting that <b>some genres have "intervals" of popularity</b>, e.g. such like Music.

# ## Some about titles

# In[ ]:


import nltk


# In[ ]:


t = data.title.apply(nltk.word_tokenize).sum()


# In[ ]:


from collections import Counter
from string import punctuation

def content_text(text):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    without_stp  = Counter()
    for word in text:
        word = word.lower()
        if len(word) < 3:
            continue
        if word not in stopwords:
            without_stp.update([word])
    return [(y,c) for y,c in without_stp.most_common(20)]

without_stop = content_text(t)
without_stop


# There are <b>the most common words in games titles.</b> Do you want to create a nice tag cloud? I want :)

# In[ ]:


from PIL import Image
import random
from wordcloud import WordCloud, STOPWORDS

text = ' '.join(t)
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='white', max_font_size=110, stopwords=stopwords, 
                      random_state=3, relative_scaling=.5).generate(text)
plt.figure(figsize=(15,18))
plt.imshow(wordcloud)
plt.axis('off')


# If you call your game something like <b>"World War II. Adventure Game: Star Edition"</b>, then you have problems with imagination :) 

# ## Let's talk about masterpieces

# Let's just look at the table of masterpieces.

# In[ ]:


master = data[data.score == 10][['title','platform','genre','release_year']]
master


# Note that a game can occur more than once, as the output for multiple platforms. Let's draw charts of genres and platforms.

# In[ ]:


f, ax = plt.subplots(2,1, figsize=(10,20))
master.groupby('genre').size().plot.pie(ax=ax[0],cmap='Set3')
master.groupby('platform').size().plot.pie(ax=ax[1],cmap='terrain')
ax[0].set_ylabel('')
ax[1].set_ylabel('')


# Do you know all these platforms?

# ## Conclusion

# Using this dataset we find out about cyclic structure of game releases, the most popular platforms, some details about game scoring, naming and much more. I hope you were interested. Thank you for watching!
# 
# P.S. Special thanks to other participants for some greats ideas.
