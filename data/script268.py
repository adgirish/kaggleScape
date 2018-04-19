
# coding: utf-8

# # The World of TED

# ![](https://09c449efca3bbeb52dcea716-ddjaey2ypcfdo.netdna-ssl.com/wp-content/uploads/2017/03/TED.gif)

# Founded in 1984 by Richard Saulman as a non profit organisation that aimed at bringing experts from the fields of Technology, Entertainment and Design together, TED Conferences have gone on to become the Mecca of ideas from virtually all walks of life. As of 2015, TED and its sister TEDx chapters have published more than 2000 talks for free consumption by the masses and its speaker list boasts of the likes of Al Gore, Jimmy Wales, Shahrukh Khan and Bill Gates.
# 
# Ted, which operates under the slogan 'Ideas worth spreading' has managed to achieve an incredible feat of bringing world renowned experts from various walks of life and study and giving them a platform to distill years of their work and research into talks of 18 minutes in length. What's even more incredible is that their invaluable insights is available on the Internet for free.
# 
# Since the time I begin watching TED Talks in high school, they have never ceased to amaze me. I have learned an incredible amount, about fields I was completely alien to, in the form of poignant stories, breathtaking visuals and subtle humor. So in this notebook, I wanted to attempt at finding insights about the world of TED, its speakers and its viewers and try to answer a few questions that I had always had in the back of my mind.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pandas.io.json import json_normalize
from wordcloud import WordCloud, STOPWORDS


# In[ ]:


month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


# The data has been obtained by running a custom web scraper on the official TED.com website. The data is shared under the Creative Commons License (just like the TED Talks) and hosted on Kaggle. You can download it here: https://www.kaggle.com/rounakbanik/ted-talks

# ## The Main TED Dataset

# The main dataset contains metadata about every TED Talk hosted on the TED.com website until September 21, 2017. Let me give you a brief walkthrough of the kind of data available so as to give you an idea of what are the possibilities with this dataset.

# In[ ]:


df = pd.read_csv('../input/ted_main.csv')
df.columns


# ## Features Available
# 
# * **name:** The official name of the TED Talk. Includes the title and the speaker.
# * **title:** The title of the talk
# * **description:** A blurb of what the talk is about.
# * **main_speaker:** The first named speaker of the talk.
# * **speaker_occupation:** The occupation of the main speaker.
# * **num_speaker:** The number of speakers in the talk.
# * **duration:** The duration of the talk in seconds.
# * **event:** The TED/TEDx event where the talk took place.
# * **film_date:** The Unix timestamp of the filming.
# * **published_date:** The Unix timestamp for the publication of the talk on TED.com
# * **comments:** The number of first level comments made on the talk.
# * **tags:** The themes associated with the talk.
# * **languages:** The number of languages in which the talk is available.
# * **ratings:** A stringified dictionary of the various ratings given to the talk (inspiring, fascinating, jaw dropping, etc.)
# * **related_talks:** A list of dictionaries of recommended talks to watch next.
# * **url:** The URL of the talk.
# * **views:** The number of views on the talk.
# 
# I'm just going to reorder the columns in the order I've listed the features for my convenience (and OCD).

# In[ ]:


df = df[['name', 'title', 'description', 'main_speaker', 'speaker_occupation', 'num_speaker', 'duration', 'event', 'film_date', 'published_date', 'comments', 'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]


# Before we go any further, let us convert the Unix timestamps into a human readable format. 

# In[ ]:


import datetime
df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))


# In[ ]:


df.head()


# We also have another dataset which contains the transcript of every talk but we will get to that later. For now, let us begin with the analysis of TED Talks!

# In[ ]:


len(df)


# We have over 2550 talks at our disposal. These represent all the talks that have ever been posted on the TED Platform until September 21, 2017 and has talks filmed in the period between 1994 and 2017. It has been over two glorious decades of TED.

# ## Most Viewed Talks of All Time
# 
# For starters, let us perform some easy analysis. I want to know what the 15 most viewed TED talks of all time are. The number of views gives us a good idea of the popularity of the TED Talk.

# ![](https://pi.tedcdn.com/r/pf.tedcdn.com/images/playlists/20_most_popular_2683333149_1200x627.jpg?c=1050%2C550&w=1050)

# In[ ]:


pop_talks = df[['title', 'main_speaker', 'views', 'film_date']].sort_values('views', ascending=False)[:15]
pop_talks


# ### Observations
# 
# * Ken Robinson's talk on **Do Schools Kill Creativity?** is the most popular TED Talk of all time with 47.2 million views.
# * Also coincidentally, it is also one of the first talks to ever be uploaded on the TED Site (the main dataset is sorted by published date). 
# * Robinson's talk is closely followed by Amy Cuddy's talk on **Your Body Language May Shape Who You Are**. 
# * There are only 2 talks that have surpassed the 40 million mark and 4 talks that have crossed the 30 million mark. 
# 
# Let us make a bar chart to visualise these 15 talks in terms of the number of views they garnered.

# In[ ]:


pop_talks['abbr'] = pop_talks['main_speaker'].apply(lambda x: x[:3])
sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
sns.barplot(x='abbr', y='views', data=pop_talks)


# Finally, in this section, let us investigate the summary statistics and the distibution of the views garnered on various TED Talks.

# In[ ]:


sns.distplot(df['views'])


# In[ ]:


sns.distplot(df[df['views'] < 0.4e7]['views'])


# In[ ]:


df['views'].describe()


# The average number of views on TED Talks in **1.6 million.** and the median number of views is **1.12 million**. This suggests a very high average level of popularity of TED Talks. We also notice that the majority of talks have views less than **4 million**. We will consider this as the cutoff point when costructing box plots in the later sections.

# ## Comments
# 
# Although the TED website gives us access to all the comments posted publicly, this dataset only gives us the number of comments. We will therefore have to restrict our analysis to this feature only. You could try performing textual analysis by scraping the website for comments.

# In[ ]:


df['comments'].describe()


# ### Observations
# 
# * On average, there are **191.5 comments** on every TED Talk. Assuming the comments are constructive criticism, we can conclude that the TED Online Community is highly involved in discussions revolving TED Talks.
# * There is a **huge standard deviation** associated with the comments. In fact, it is even larger than the mean suggesting that the measures may be sensitive to outliers. We shall plot this to check the nature of the distribution.
# * The minimum number of comments on a talk is **2** and the maximum is **6404**. The **range is 6402.**. The minimum number, though, may be as a result of the talk being posted extremely recently.

# In[ ]:


sns.distplot(df['comments'])


# In[ ]:


sns.distplot(df[df['comments'] < 500]['comments'])


# From the plot above, we can see that the bulk of the talks have **fewer than 500 comments**. This clearly suggests that the mean obtained above has been heavily influenced by outliers. This is possible because the number of samples is **only 2550 talks**.
# 
# Another question that I am interested in is if the number of views is correlated with the number of comments. We should think that this is the case as more popular videos tend to have more comments. Let us find out.

# In[ ]:


sns.jointplot(x='views', y='comments', data=df)


# In[ ]:


df[['views', 'comments']].corr()


# As the scatterplot and the correlation matrix show, the pearson coefficient is **slightly more than 0.5**. This suggests a **medium to strong correlation** between the two quantities. This result was pretty expected as mentioned above. Let us now check the number of views and comments on the 10 most commented TED Talks of all time.

# In[ ]:


df[['title', 'main_speaker','views', 'comments']].sort_values('comments', ascending=False).head(10)


# As can be seen above, Richard Dawkins' talk on **Militant Atheism'** generated the greatest amount of discussion and opinions despite having significantly lesser views than Ken Robinson's talk, which is second in the list. This raises some interesting questions.
# 
# Which talks tend to attract the largest amount of discussion?
# 
# To answer this question, we will define a new feature **discussion quotient** which is simply the ratio of the number of comments to the number of views. We will then check which talks have the largest discussion quotient.

# In[ ]:


df['dis_quo'] = df['comments']/df['views']


# In[ ]:


df[['title', 'main_speaker','views', 'comments', 'dis_quo', 'film_date']].sort_values('dis_quo', ascending=False).head(10)


# This analysis has actually raised extremely interesting insights. Half of the talks in the top 10 are on the lines of **Faith and Religion**. I suspect science and religion is still a very hotly debated topic even in the 21st century. We shall come back to this hypothesis in a later section.
# 
# The most discusses talk, though, is **The Case for Same Sex Marriage** (which has religious undertones). This is not that surprising considering the amount of debate the topic caused back in 2009 (the time the talk was filmed).

# ## Analysing TED Talks by the month and the year
# 
# TED (especially TEDx) Talks tend to occur all throughout the year. Is there a hot month as far as TED is concerned? In other words, how are the talks distributed throughout the months since its inception? Let us find out.

# ![](https://media.giphy.com/media/d1GpZTVp2eV7gQk8/giphy.gif)

# In[ ]:


df['month'] = df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])

month_df = pd.DataFrame(df['month'].value_counts()).reset_index()
month_df.columns = ['month', 'talks']


# In[ ]:


sns.barplot(x='month', y='talks', data=month_df, order=month_order)


# **February** is clearly the most popular month for TED Conferences whereas **August** and **January** are the least popular. February's popularity is largely due to the fact that the official TED Conferences are held in February. Let us check the distribution for TEDx talks only.

# In[ ]:


df_x = df[df['event'].str.contains('TEDx')]
x_month_df = pd.DataFrame(df_x['month'].value_counts().reset_index())
x_month_df.columns = ['month', 'talks']


# In[ ]:


sns.barplot(x='month', y='talks', data=x_month_df, order=month_order)


# As far as TEDx talks are concerned, **November** is the most popular month. However, we cannot take this result at face value as very few of the TEDx talks are actually uploaded to the TED website and therefore, it is entirely possible that the sample in our dataset is not at all representative of all TEDx talks. A slightly more accurate statement would be that **the most popular TEDx talks take place the most in October and November.**
# 
# The next question I'm interested in is the most popular days for conducting TED and TEDx conferences. The tools applied are very sensible to the procedure applied for months.

# In[ ]:


def getday(x):
    day, month, year = (int(i) for i in x.split('-'))    
    answer = datetime.date(year, month, day).weekday()
    return day_order[answer]


# In[ ]:


df['day'] = df['film_date'].apply(getday)


# In[ ]:


day_df = pd.DataFrame(df['day'].value_counts()).reset_index()
day_df.columns = ['day', 'talks']


# In[ ]:


sns.barplot(x='day', y='talks', data=day_df, order=day_order)


# The distribution of days is almost a bell curve with **Wednesday and Thursday** being the most popular days and **Sunday** being the least popular. This is pretty interesting because I was of the opinion that most TED Conferences would happen sometime in the weekend.

# Let us now visualize the number of TED talks through the years and check if our hunch that they have grown significantly is indeed true.

# In[ ]:


df['year'] = df['film_date'].apply(lambda x: x.split('-')[2])
year_df = pd.DataFrame(df['year'].value_counts().reset_index())
year_df.columns = ['year', 'talks']


# In[ ]:


plt.figure(figsize=(18,5))
sns.pointplot(x='year', y='talks', data=year_df)


# ### Obervations
# 
# * As expected, the number of TED Talks have gradually increased over the years since its inception in 1984. 
# * There was a **sharp increase** in the number if talks in **2009**. It might be interesting to know the reasons behind 2009 being the tipping point where the number of talks increased more than twofold.
# * The number of talks have been pretty much constant since 2009. 
# 
# Finally, to put it all together, let us construct a heat map that shows us the number of talks by month and year. This will give us a good summary of the distribution of talks.

# In[ ]:


months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


# In[ ]:


hmap_df = df.copy()
hmap_df['film_date'] = hmap_df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1] + " " + str(x.split('-')[2]))
hmap_df = pd.pivot_table(hmap_df[['film_date', 'title']], index='film_date', aggfunc='count').reset_index()
hmap_df['month_num'] = hmap_df['film_date'].apply(lambda x: months[x.split()[0]])
hmap_df['year'] = hmap_df['film_date'].apply(lambda x: x.split()[1])
hmap_df = hmap_df.sort_values(['year', 'month_num'])
hmap_df = hmap_df[['month_num', 'year', 'title']]
hmap_df = hmap_df.pivot('month_num', 'year', 'title')
hmap_df = hmap_df.fillna(0)


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(hmap_df, annot=True, linewidths=.5, ax=ax, fmt='n', yticklabels=month_order)


# ## TED Speakers
# 
# In this section, we will try and gain insight about all the amazing speakers who have managed to inspire millions of people through their talks on the TED Platform. The first question we shall ask in this section is who are the most popular TED Speakers. That is, which speakers have given the most number of TED Talks.

# ![](https://tedconfblog.files.wordpress.com/2011/05/tedsalonmay2011-speakers.jpg)

# In[ ]:


speaker_df = df.groupby('main_speaker').count().reset_index()[['main_speaker', 'comments']]
speaker_df.columns = ['main_speaker', 'appearances']
speaker_df = speaker_df.sort_values('appearances', ascending=False)
speaker_df.head(10)


# **Hans Rosling**, the Swiss Health Professor is clearly the most popular TED Speaker, with more than **9 appearances** on the TED Forum. **Juan Enriquez** comes a close second with **7 appearances**. Rives and Marco Tempest have graced the TED platform **6 times**.
# 
# Which occupation should you choose if you want to become a TED Speaker? Let us have a look what kind of people TED is most interested in inviting to its events.

# In[ ]:


occupation_df = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]
occupation_df.columns = ['occupation', 'appearances']
occupation_df = occupation_df.sort_values('appearances', ascending=False)


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x='occupation', y='appearances', data=occupation_df.head(10))
plt.show()


# ### Observations
# 
# * **Writers** are the most popular with more than 45 speakers identifying themselves as the aforementioned.
# * **Artists** and **Designers** come a distant second with around 35 speakers in each category.
# * This result must be taken with a pinch of salt as a considerable number of speakers identify themselves with multiple professions (for example, writer/entrepreneur). Performing an analysis taking this into consideraion is left as an exercise to the reader.
# 
# Do some professions tend to attract a larger number of viewers? Do answer this question let us visualise the relationship between the top 10 most popular professions and the views thet garnered in the form of a box plot.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='speaker_occupation', y='views', data=df[df['speaker_occupation'].isin(occupation_df.head(10)['occupation'])], palette="muted", ax =ax)
ax.set_ylim([0, 0.4e7])
plt.show()


# On average, out of the top 10 most popular professions, **Psychologists** tend to garner the most views. **Writers** have the greatest range of views between the first and the third quartile.. 
# 
# Finally, let us check the number of talks which have had more than one speaker.

# In[ ]:


df['num_speaker'].value_counts()


# Almost every talk has just one speaker. There are close to 50 talks where two people shared the stage. The maximum number of speakers to share a single stage was 5. I suspect this was a dance performance. Let's have a look.

# In[ ]:


df[df['num_speaker'] == 5][['title', 'description', 'main_speaker', 'event']]


# My hunch was correct. It is a talk titled **A dance to honor Mother Earth** by Jon Boogz and Lil Buck at the TED 2017 Conference.

# ## TED Events
# 
# Which TED Events tend to hold the most number of TED.com upload worthy events? We will try to answer that question in this section.

# In[ ]:


events_df = df[['title', 'event']].groupby('event').count().reset_index()
events_df.columns = ['event', 'talks']
events_df = events_df.sort_values('talks', ascending=False)
events_df.head(10)


# As expected, the official TED events held the major share of TED Talks published on the TED.com platform. **TED2014 had the most number of talks** followed by TED2009. There isn't too much insight to be gained from this. 

# ## TED Languages
# 
# One remarkable aspect of TED Talks is the sheer number of languages in which it is accessible. Let us perform some very basic data visualisation and descriptive statistics about languages at TED.

# In[ ]:


df['languages'].describe()


# On average, a TED Talk is available in 27 different languages. The maximum number of languages a TED Talk is available in is a staggering 72. Let us check which talk this is.

# In[ ]:


df[df['languages'] == 72]


# The most translated TED Talk of all time is Matt Cutts' **Try Something New in 30 Days**. The talk does have a very universal theme of exploration. The sheer number of languages it's available in demands a little more inspection though as it has just over 8 million views, far fewer than the most popular TED Talks. 
# 
# Finally, let us check if there is a correlation between the number of views and the number of languages a talk is availbale in. We would think that this should be the case since the talk is more accessible to a larger number of people but as Matt Cutts' talk shows, it may not really be the case.

# In[ ]:


sns.jointplot(x='languages', y='views', data=df)
plt.show()


# The Pearson coefficient is 0.38 suggesting a **medium correlation** between the aforementioned quantities. 

# ## TED Themes
# 
# In this section, we will try to find out the most popular themes in the TED conferences. Although TED started out as a conference about technology, entertainment and design, it has since diversified into virtually every field of study and walk of life. It will be interesting to see if this conference with Silicon Valley origins has a bias towards certain topics.
# 
# To answer this question, we need to wrangle our data in a way that it is suitable for analysis. More specifically, we need to split the related_tags list into separate rows.

# ![](https://media.giphy.com/media/rPMTIZOBVnlD2/giphy.gif)

# In[ ]:


import ast
df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))


# In[ ]:


s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'theme'


# In[ ]:


theme_df = df.drop('tags', axis=1).join(s)
theme_df.head()


# In[ ]:


len(theme_df['theme'].value_counts())


# TED defines a staggering **416 different categories** for its talks. Let us now check the most popular themes.

# In[ ]:


pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()
pop_themes.columns = ['theme', 'talks']
pop_themes.head(10)


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x='theme', y='talks', data=pop_themes.head(10))
plt.show()


# As may have been expected, **Technology** is the most popular topic for talks. The other two original factions, Design and Entertainment, also make it to the list of top 10 themes. **Science** and **Global Issues** are the second and the third most popular themes respectively.
# 
# The next question I want to answer is the trends in the share of topics of TED Talks across the world. Has the demand for Technology talks increased? Do certain years have a disproportionate share of talks related to global issues? Let's find out! 
# 
# We will only be considering the top 7 themes, excluding TEDx and talks after 2009, the year when the number of TED Talks really peaked.

# In[ ]:


pop_theme_talks = theme_df[(theme_df['theme'].isin(pop_themes.head(8)['theme'])) & (theme_df['theme'] != 'TEDx')]
pop_theme_talks['year'] = pop_theme_talks['year'].astype('int')
pop_theme_talks = pop_theme_talks[pop_theme_talks['year'] > 2008]


# In[ ]:


themes = list(pop_themes.head(8)['theme'])
themes.remove('TEDx')
ctab = pd.crosstab([pop_theme_talks['year']], pop_theme_talks['theme']).apply(lambda x: x/x.sum(), axis=1)
ctab[themes].plot(kind='bar', stacked=True, colormap='rainbow', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[ ]:


ctab[themes].plot(kind='line', stacked=False, colormap='rainbow', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# The proportion of technology talks has steadily increased over the years with a slight dip in 2010. This is understandable considering the boom of technologies such as blockchain, deep learning and augmented reality capturing people's imagination.
# 
# Talks on culture have witnessed a dip, decreasing steadily starting 2013. The share of culture talks has been the least in 2017.  Entertainment talks also seem to have witnessed a slight decline in popularity since 2009.
# 
# Like with the speaker occupations, let us investigate if certain topics tend to garner more views than certain other topics. We will be doing this analysis for the top ten categories that we discovered in an earlier cell. As with the speaker occupations, the box plot will be used to deduce this relation.

# In[ ]:


pop_theme_talks = theme_df[theme_df['theme'].isin(pop_themes.head(10)['theme'])]
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='theme', y='views', data=pop_theme_talks, palette="muted", ax =ax)
ax.set_ylim([0, 0.4e7])


# Although culture has lost its share in the number of TED Talks over the years, they garner the highest median number of views.

# ## Talk Duration and Word Counts
# 
# In this section, we will perform analysis on the length of TED Talks. TED is famous for imposing a very strict time limit of 18 minutes. Although this is the suggested limit, there have been talks as short as 2 minutes and some have stretched to as long as 24 minutes. Let us get an idea of the distribution of TED Talk durations.

# In[ ]:


#Convert to minutes
df['duration'] = df['duration']/60
df['duration'].describe()


# TED Talks, on average are **13.7 minutes long**. I find this statistic surprising because TED Talks are often synonymous with 18 minutes and the average is a good 3 minutes shorter than that.
# 
# The shortest TED Talk on record is **2.25 minutes** long whereas the longest talk is **87.6 minutes** long. I'm pretty sure the longest talk was not actually a TED Talk. Let us look at both the shortest and the longest talk.

# In[ ]:


df[df['duration'] == 2.25]


# In[ ]:


df[df['duration'] == 87.6]


# The shortest talk was at TED2007 titled **The ancestor of language** by Murray Gell-Mann. The longest talk on TED.com, as we had guessed, is not a TED Talk at all. Rather, it was a talk titled **Parrots, the universe and everything** delivered by Douglas Adams at the University of California in 2001.
# 
# Let us now check for any correlation between the popularity and the duration of a TED Talk. To make sure we only include TED Talks, we will consider only those talks which have a duration less than 25 minutes.

# In[ ]:


sns.jointplot(x='duration', y='views', data=df[df['duration'] < 25])
plt.xlabel('Duration')
plt.ylabel('Views')
plt.show()


# There seems to be almost no correlation between these two quantities. This strongly suggests that **there is no tangible correlation between the length and the popularity of a TED Talk**. Content is king at TED.
# 
# Next, we look at transcripts to get an idea of word count. For this, we introduce our second dataset, the one which contains all transcripts.

# In[ ]:


df2 = pd.read_csv('../input/transcripts.csv')
df2.head()


# In[ ]:


len(df2)


# It seems that we have data available for 2467 talks. Let us perform a join of the two dataframes on the url feature to include word counts for every talk.

# In[ ]:


df3 = pd.merge(left=df,right=df2, how='left', left_on='url', right_on='url')
df3.head()


# In[ ]:


df3['transcript'] = df3['transcript'].fillna('')
df3['wc'] = df3['transcript'].apply(lambda x: len(x.split()))


# In[ ]:


df3['wc'].describe()


# We can see that the average TED Talk has around **1971 words** and there is a significantly large standard deviation of a **1009 words**. The longest talk is more than **9044 words** in length.
# 
# Like duration, there shouldn't be any correlation between number of words and views. We will proceed to look at a more interesting statstic: **the number of words per minute.**

# In[ ]:


df3['wpm'] = df3['wc']/df3['duration']
df3['wpm'].describe()


# The average TED Speaker enunciates **142 words per minute**. The fastest talker spoke a staggering **247 words a minute** which is much higher than the average of 125-150 words per minute in English. Let us see who this is!
# 

# In[ ]:


df3[df3['wpm'] > 245]


# The person is **Mae Jemison** with a talk on **Teach arts and sciences together** at the TED2002 conference. We should take this result with a pinch of salt because I went ahead and had a look at the talk and she didn't really seem to speak that fast.
# 
# Finally, in this section, I'd like to see if there is any correlation between words per minute and popularity. 

# In[ ]:


sns.jointplot(x='wpm', y='views', data=df3[df3['duration'] < 25])
plt.show()


# Again, there is practically no correlation. If you are going to give a TED Talk, you probably shouldn't worry if you're speaking a little faster or a little slower than usual.

# ## TED Ratings
# 
# TED allows its users to rate a particular talk on a variety of metrics. We therefore have data on how many people found a particular talk funny, inspiring, creative and a myriad of other verbs. Let us inspect how this ratings dictionary actually looks like.

# In[ ]:


df.iloc[1]['ratings']


# In[ ]:


df['ratings'] = df['ratings'].apply(lambda x: ast.literal_eval(x))


# In this section, I want to find out which talks were rated the funniest, the most beautiful, the most confusing and most jaw dropping of all time. The rest is left to the reader to explore. We now need to define three extra features to accomplish this task.

# In[ ]:


df['funny'] = df['ratings'].apply(lambda x: x[0]['count'])
df['jawdrop'] = df['ratings'].apply(lambda x: x[-3]['count'])
df['beautiful'] = df['ratings'].apply(lambda x: x[3]['count'])
df['confusing'] = df['ratings'].apply(lambda x: x[2]['count'])
df.head()


# ### Funniest Talks of all time

# In[ ]:


df[['title', 'main_speaker', 'views', 'published_date', 'funny']].sort_values('funny', ascending=False)[:10]


# ### Most Beautiful Talks of all time

# In[ ]:


df[['title', 'main_speaker', 'views', 'published_date', 'beautiful']].sort_values('beautiful', ascending=False)[:10]


# ### Most Jaw Dropping Talks of all time

# In[ ]:


df[['title', 'main_speaker', 'views', 'published_date', 'jawdrop']].sort_values('jawdrop', ascending=False)[:10]


# ### Most Confusing Talks of all time

# In[ ]:


df[['title', 'main_speaker', 'views', 'published_date', 'confusing']].sort_values('confusing', ascending=False)[:10]


# ## Related Videos
# 
# In this last section, we will have a look at how every TED Talk is related to every other TED Talk by constructing a graph with all the talks as nodes and edges defined between two talks if one talk is on the list of recommended watches of the other. Considering the fact that TED Talks are extremely diverse, it would be interesting to see how dense or sparse our graph will be.

# ![](https://dragonlaw.io/wp-content/uploads/2016/11/networking-dribbble.gif)

# In[ ]:


df['related_talks'] = df['related_talks'].apply(lambda x: ast.literal_eval(x))


# In[ ]:


s = df.apply(lambda x: pd.Series(x['related_talks']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'related'


# In[ ]:


related_df = df.drop('related_talks', axis=1).join(s)
related_df['related'] = related_df['related'].apply(lambda x: x['title'])


# In[ ]:


d = dict(related_df['title'].drop_duplicates())
d = {v: k for k, v in d.items()}


# In[ ]:


related_df['title'] = related_df['title'].apply(lambda x: d[x])
related_df['related'] = related_df['related'].apply(lambda x: d[x])


# In[ ]:


related_df = related_df[['title', 'related']]
related_df.head()


# In[ ]:


edges = list(zip(related_df['title'], related_df['related']))


# In[ ]:


import networkx as nx
G = nx.Graph()
G.add_edges_from(edges)


# In[ ]:


plt.figure(figsize=(25, 25))
nx.draw(G, with_labels=False)


# The graph is reasonably dense with every node connected to at least 3 other nodes. This shows us the real beauty of the conference. No matter how different the subjects of the talks are, the common theme of spreading ideas and inspiring people seems to be a adhesive force between them.

# ## The TED Word Cloud
# 
# I was curious about which words are most often used by TED Speakers. Could we create a Word Cloud out of all TED Speeches? Luckily, Python has a very useful word cloud generating library that allows us to do just that. 

# In[ ]:


corpus = ' '.join(df2['transcript'])
corpus = corpus.replace('.', '. ')


# In[ ]:


wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# The word **One** is the most popular word across the corpus of all TED Transcripts which I think encapsulates the idea of TED pretty well.  **Now**, **Think**, **See**, **People**, **Laughter** and **Know** are among the most popular words used in TED Speeches. TED speeched seem to place a lot of emphasis on knowledge, insight, the present and of course, the people.
# 
# I shall end my analysis here. Of course, a lot more can be done using the data we have right now. I will leave this up to the community to come up with more interesting analysis and insights than what I was capable of producing. 
