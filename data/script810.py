
# coding: utf-8

# ![presentation.jpg](https://lh3.googleusercontent.com/-ha00LTVcBpE/Wp376iB_0eI/AAAAAAAADMo/qtS1NkE0msc25eLi1psYkpNOAPxl7MIIwCJoC/w530-h298-n-rw/Untitled%2Bpresentation.jpg)

# # **TOP  VIEWS  VIDEOS  WALL**

# In[8]:


from IPython.display import HTML, display
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

files = [i for i in glob.glob('../input/*.{}'.format('csv'))]
dfs = list()
for csv in files:
    df = pd.read_csv(csv)
    df['country'] = csv[9:11]
    dfs.append(df)

my_df = pd.concat(dfs)
my_df = my_df.dropna()
to_int = ['views']
for column in to_int:
    my_df[column] = my_df[column].astype(int)
my_df = my_df.drop_duplicates('video_id',keep='last').sort_values('views',ascending=False)
most_views_thumb = my_df['thumbnail_link'].head(100).reset_index()['thumbnail_link']

ROW = 10
COLUMN = 10
tempHTML = ''
innerHTML = '<div style="block">'
    
for r in range(ROW):
    rowHTML = ''
    for c in range(COLUMN):
        #if c != COLUMN-1:
        tempHTML = '<img src="' + most_views_thumb[c*10+r] + '"style="float:left;width:80px;height:80px;margin:0">'
        rowHTML += tempHTML
        #else:
            #tempHTML = '<img src="' + most_views_thumb[c*10+r] + '"style="float:left;width:80px;height:80px;margin:0">'
            #rowHTML += tempHTML
    innerHTML += rowHTML #'<div>' +rowHTML + '</div>'
innerHTML += '</div>'
display(HTML(innerHTML))

#my_df.head()


# # **Deep Analysis on Trending Youtube Video Statistic**
# This Notebook will walk you through some data exploration and deeper analysis on Youtube Trending Video Statistic in between countries like US, Canada, United Kingdom, France and Germany.
# We will go through 2 session in this notebook:
# ## 1. Overview of statistics
# ## 2. Analysis
# > ### How long usually a video can trend in different countries? 
# > ### How many likes, dislikes, views and comments get by different countries? 
# > ### Correlation of trending video in between countries
# > ### Videos from which category has longer trend?
# > ### Correlation between Days of Publish to Trend v/s Trending Duration
# > ### Users like videos from which CATEGORY the most?
# > ### What is the ratio of Likes-Dislikes and Views-Comments in different categories?
# > ### What's the tags in the most negative and most positive category? What's the most discuss words for Science & Technology?
# > ### To Be Continued......

# # Importing Libraries

# In[9]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import glob
import seaborn as sns
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# Any results you write to the current directory are saved as output.


# ## Import all the CSV files (US, CA, DE, GB, FR)

# In[6]:


files = [i for i in glob.glob('../input/*.{}'.format('csv'))]
sorted(files)


# ## Combine all the statistics from different regions
# Let's see what we got in this combined sheet.

# In[10]:


dfs = list()
for csv in files:
    df = pd.read_csv(csv, index_col='video_id')
    df['country'] = csv[9:11]
    dfs.append(df)

my_df = pd.concat(dfs)
my_df.head(3)


# ## Reformat the date time and removed the incompleted rows

# In[11]:


my_df['trending_date'] = pd.to_datetime(my_df['trending_date'],errors='coerce', format='%y.%d.%m')
my_df['publish_time'] = pd.to_datetime(my_df['publish_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%fZ')

my_df = my_df[my_df['trending_date'].notnull()]
my_df = my_df[my_df['publish_time'].notnull()]

my_df = my_df.dropna(how='any',inplace=False, axis = 0)

my_df.insert(4, 'publish_date', my_df['publish_time'].dt.date)
my_df['publish_time'] = my_df['publish_time'].dt.time

my_df_full = my_df.reset_index().sort_values('trending_date').set_index('video_id')
my_df = my_df.reset_index().sort_values('trending_date').drop_duplicates('video_id',keep='last').set_index('video_id')
my_df[['publish_date','publish_time']].head()


# We keep two set of data here:
# 1. ***my_df*** (which only keep the last entry if duplicated because it carries latest stat)
# 2. ***my_df_full*** ( full set of combined data, keep for later use)

# ## Insert category column

# In[12]:


my_df['category_id'] = my_df['category_id'].astype(str)
my_df_full['category_id'] = my_df['category_id'].astype(str)

category_id = {}

with open('../input/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        category_id[category['id']] = category['snippet']['title']

my_df.insert(4, 'category', my_df['category_id'].map(category_id))
my_df_full.insert(4, 'category', my_df_full['category_id'].map(category_id))
category_list = my_df['category'].unique()
category_list


# 
# # Analysis : How long usually a video can trend in different countries?
# The statistic provided has a key features to reflect how long a video trending which is the **number of appearances**. The greater the number of apperances indicate the long-last the video trend is. From here we can see that United Kingdom's list has the most long trended Youtube videos (Top 5 in the list is from GB). 

# In[13]:


fre_df = pd.DataFrame(my_df_full.groupby([my_df_full.index,'country']).count()['title'].sort_values(ascending=False)).reset_index()
fre_df.head(), fre_df.tail()


# ### Let's visualize the stats above using line graphs

# In[14]:


video_list,max_list = list(),list()
country_list = my_df.groupby(['country']).count().index

for c in country_list:
    video_list.append(fre_df[fre_df['country']==c]['title'].value_counts().sort_index())
    max_list.append(max(fre_df[fre_df['country']==c]['title'].value_counts().sort_index().index))

fig, [ax0, ax1, ax2, ax3, ax4] = plt.subplots(nrows=5,figsize=(15, 30))
st = fig.suptitle("How long a video trend in different countries?", fontsize=32)
st.set_y(0.9)
for i, pt in enumerate([ax0, ax1, ax2, ax3, ax4]):
    pt.plot(video_list[i].index, video_list[i])
    pt.spines['right'].set_visible(False)
    pt.spines['top'].set_visible(False)
    pt.set_xlabel("appearances",fontsize=14)
    pt.set_ylabel(country_list[i],fontsize=24)
    pt.axes.set_xlim(1, max(max_list))

# Tweak spacing between subplots to prevent labels from overlapping
plt.subplots_adjust(hspace=0.2)
plt.subplots_adjust(wspace=0)


# ### Observation :
# From the line graphs above we can see that United Kingdom has numbers of enduring video in trend follow by US and Canada. In contrast, both France and Germany have very few video can last long in trending with max of 5 appearances in statistics.

# ## Ratio of Youtube Trending Videos in 5 countries
# Since we have combined statistics from multiple countries, it's good to have a look on the number and ratio of videos we have in different countries. In this plot we keep only last entry for duplicated videos. That's why we can clearly observe that **the country with the most long-trending video end up having lesser videos**.

# In[17]:


labels = my_df.groupby(['country']).count().index
sizes = my_df.groupby(['country']).count()['title']
explode = (0, 0, 0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig, ax = plt.subplots(figsize=(10,10))
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=False, explode=explode, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
sizes


# # Analysis : How many likes, dislikes, views and comments get by different countries? 

# In[16]:


to_int = ['views', 'likes', 'dislikes', 'comment_count']
for column in to_int:
    my_df[column] = my_df[column].astype(int)
    
measures = list()
n_groups = len(country_list)
for i, typ in enumerate(to_int):
    measure = list()
    for c in country_list:
        measure.append(my_df[my_df['country']==c][typ].agg('sum')/len(my_df[my_df['country']==c].index.unique()))
    measures.append(measure)

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(ncols=2,nrows=2, figsize=(15,10))

index = np.arange(n_groups)
bar_width = 0.1
opacity = 0.4
error_config = {'ecolor': '0.3'}

for i, axs in enumerate([[ax1, ax2], [ax3, ax4]]):
    for j, ax in enumerate(axs):
        ax.bar(index + (bar_width), measures[(i+j)+i], bar_width*4,
                alpha=opacity, color=['b','y','g','r','k'],
                error_kw=error_config)
        ax.set_title(to_int[(i+j)+i])
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(country_list)


# **Observation:**
# Obviously, four of the graphs share the similar trend in numbers. One possible reason to this is due to the video's trending duration. Enduring trending videos have the advantages in getting more views, likes, dislikes and comments.

# # Analysis : Correlation of trending video in between countries
# 
# 
# > **Question : Does the video trending in one country will trend in other countries too? If yes,how much  do they correlate to each other? **
# 
#  Let's find out the answer!

# In[ ]:


corr_list = pd.DataFrame(fre_df['video_id'].unique(), columns=['video_id'])
for country_code in fre_df['country'].unique():
    corr_list[country_code] = 0
corr_list['total']=0
corr_list=corr_list.set_index('video_id')
#print new_list
for index , item in corr_list.iterrows():
    #print index
    total = 0
    for i ,row in fre_df[fre_df['video_id'] == index][['country','title']].iterrows():
        total += row['title']
        corr_list.loc[[index],[row['country']]] = row['title']
    corr_list.loc[[index],['total']] = total
corr_list.head()


# Suprisingly the answer is **YES**. They do coexist in multiple list.
# Next, let's see how much do they correlate to each other.

# ## Plot the Heatmaps of Correlation

# In[ ]:


countries = ['GB', 'US', 'CA', 'DE', 'FR'] #looking at correlations between these countries
corr_matrix = corr_list[countries].corr()

fig, ax = plt.subplots(figsize=(15,15))
fig.suptitle('Correlation between countries', fontsize=20).set_y(0.85)
heatmap = ax.imshow(corr_matrix, interpolation='nearest', cmap=cm.OrRd)

# making the colorbar on the side
cbar_min = corr_matrix.min().min()
cbar_max = corr_matrix.max().max()
cbar = fig.colorbar(heatmap, ticks=[cbar_min, cbar_max])

# making the labels
labels = ['']
for column in countries:
    labels.append(column)
#print labels
ax.set_yticklabels(labels, minor=False, fontsize=15)
ax.set_xticklabels(labels, minor=False, fontsize=15)
corr_matrix


# Here we got the heatmap of correlation of Youtube Trending Videos between countries. Not suprisingly, the video from **United Kingdom, US and Canada** is highly correlate to each other compare to **Germany and France**. This might due to the sharing of common language in these contries. As compare, Germany and France seems like more isolate where they did not follow the trend from english speaking countries (Just my personal opinion by observing the heatmap). This can also explain why United Kingdom has the highest number in long-trend videos, as it is contributes by multiple contries at the same time.

# # Analysis : Users like videos from which **CATEGORY** the most?

# ## From United Kingdom Users : 

# In[ ]:


cat_df_gb = my_df[my_df['country']=='GB']['category'].value_counts().reset_index()
plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=cat_df_gb['index'],x=cat_df_gb['category'], data=cat_df_gb,orient='h')
plt.xlabel("Number of Videos")## From United Kingdom users : 
plt.ylabel("Categories")
plt.title("Catogories of trend videos in United Kingdom")


# ## From US Users : 

# In[ ]:


cat_df_us = my_df[my_df['country']=='US']['category'].value_counts().reset_index()
plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=cat_df_us['index'],x=cat_df_us['category'], data=cat_df_us,orient='h')
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Catogories of trend videos in US")


# # From Canada Users:

# In[ ]:


cat_df_ca = my_df[my_df['country']=='CA']['category'].value_counts().reset_index()
plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=cat_df_ca['index'],x=cat_df_ca['category'], data=cat_df_ca,orient='h')
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Catogories of trend videos in CANADA")


# # From Germany Users:

# In[ ]:


cat_df_de = my_df[my_df['country']=='DE']['category'].value_counts().reset_index()
plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=cat_df_de['index'],x=cat_df_de['category'], data=cat_df_de,orient='h')
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Catogories of trend videos in GERMANY")


# # From France Users:

# In[ ]:


cat_df_fr = my_df[my_df['country']=='FR']['category'].value_counts().reset_index()
plt.figure(figsize=(15,10))
sns.set_style("whitegrid")
ax = sns.barplot(y=cat_df_fr['index'],x=cat_df_fr['category'], data=cat_df_fr,orient='h')
plt.xlabel("Number of Videos")
plt.ylabel("Categories")
plt.title("Catogories of trend videos in FRANCE")


# ## Observation:
# 1. Top category of all 5 countries is Entertainment.
# 2. Music's videos ranked insignificantly in Canada, Germany and France compare to US and UK.
# 3. Sport's videos are more popular in Canada, Germany and France.
# 4. All top 8 categories in United Kingdom entertainment-related, this might not be a good sign.
# 5. Show's and Activism's video get the bottom rank in all most countries.
# 
# Remarks: Please feel free to comment in below if you observe any interesting facts.

# # Analysis : Videos from which category has longer trend?

# ## Getting the days on trend for videos

# In[ ]:


publish_to_trend = {}
my_df_first = my_df_full.reset_index().drop_duplicates('video_id',keep ='first').set_index('video_id')
diff_first = (my_df_first['trending_date']).astype('datetime64[ns]')-my_df_first['publish_date'].astype('datetime64[ns]')

diff_first = diff_first.reset_index()
diff_first.columns = ['video_id','publish_to_trend']

for i, row in diff_first.iterrows():
    publish_to_trend[row['video_id']] = row['publish_to_trend'].days

my_df_last = my_df
diff_last = my_df['trending_date'].astype('datetime64[ns]')-my_df['publish_date'].astype('datetime64[ns]')
diff_last = diff_last.reset_index()
diff_last.columns = ['video_id','publish_to_trend_last']
my_df = my_df.reset_index()
my_df.insert(4,'publish_to_trend_last', diff_last['publish_to_trend_last'].astype('timedelta64[D]').astype(int))
my_df.insert(4, 'publish_to_trend', my_df['video_id'].map(publish_to_trend))
my_df.insert(4, 'trend_duration', 0)
my_df['trend_duration'] = (my_df['publish_to_trend_last']-my_df['publish_to_trend'])+1
my_df.set_index('video_id')[['publish_to_trend','trend_duration']].sort_values('trend_duration',ascending=False).head()


# ## Heatmap : Categories v/s Trending Duration

# In[ ]:


cat_trend_duration= my_df.groupby(['category','trend_duration']).count()['video_id'].unstack().clip(upper=300)
plt.figure(figsize=(15,15))#You can Arrange The Size As Per Requirement
sns.heatmap(cat_trend_duration, cmap='plasma_r')
plt.title("Category v/s Trending Duration")


# We clip number larger than 300 to have better intrepretation with smaller scale from (0 to 6000) -> (0 to 300). From the heatmap, we get the ranking of long trend categories:
# 1. Music
# 2. Entertainment
# 3. People & Blogs
# 4. Comedy
# 5. Sports, New & Politics and Howto & Style

# # Analysis : Correlation between Days of Publish to Trend v/s Trending Duration?

# ## Heatmap : Days of Publish to Trend v/s Trending Duration

# In[ ]:


my_df['publish_to_trend'] = my_df['publish_to_trend'].clip(upper=365)
cat_trend_duration= my_df.groupby(['trend_duration','publish_to_trend']).count()['video_id'].unstack().clip(upper=300)
plt.figure(figsize=(15,15))
ax = sns.heatmap(cat_trend_duration, cmap='viridis_r')
ax.invert_yaxis()
plt.title("Correlation between Days from Publish v/s Trend and Trending Duration")


# We clip the days from publish to trend for any number larger than 365  (0 to 4825) -> (0 to 365). Suprisingly, there are some videos take few years to be on trend. These are some obeservation:
# 1. The less days needed for a video from publish to trend, the longer the trend duration.
# 2. Videos that can get into trending within 5 days will have higher probability to be trending for longer time.
# 

# # Analysis : What is the ratio of Likes-Dislikes and Views-Comments in different categories?

# ## Likes-Dislikes Ratio

# In[18]:


like_dislike_ratio = my_df.groupby('category')['likes'].agg('sum') / my_df.groupby('category')['dislikes'].agg('sum')
like_dislike_ratio = like_dislike_ratio.sort_values(ascending=False).reset_index()
like_dislike_ratio.columns = ['category','ratio']
plt.subplots(figsize=(10, 15))
sns.barplot(x="ratio", y="category", data=like_dislike_ratio,
            label="Likes-Dislikes Ratio", color="b")


# ## Views-Comments Ratio

# In[19]:


views_comment_ratio = my_df.groupby('category')['views'].agg('sum') / my_df.groupby('category')['comment_count'].agg('sum')
views_comment_ratio = views_comment_ratio.sort_values(ascending=False).reset_index()
views_comment_ratio.columns = ['category','ratio']
plt.subplots(figsize=(10, 15))
sns.barplot(x="ratio", y="category", data=views_comment_ratio,
            label="Views-Comments Ratio", color="r")


# ## Dislikes-Views Ratio

# In[25]:


view_dislike_ratio = my_df.groupby('category')['dislikes'].agg('sum') / my_df.groupby('category')['views'].agg('sum')
view_dislike_ratio = view_dislike_ratio.sort_values(ascending=False).reset_index()
view_dislike_ratio.columns = ['category','ratio']
plt.subplots(figsize=(10, 15))
sns.barplot(x="ratio", y="category", data=view_dislike_ratio,
            label="Views-Dislikes Ratio", color="y")


# ## Likes-Views Ratio

# In[24]:


view_like_ratio = my_df.groupby('category')['likes'].agg('sum') / my_df.groupby('category')['views'].agg('sum')
view_like_ratio = view_like_ratio.sort_values(ascending=False).reset_index()
view_like_ratio.columns = ['category','ratio']
plt.subplots(figsize=(10, 15))
sns.barplot(x="ratio", y="category", data=view_like_ratio,
            label="Views-Likes Ratio", color="g")


# ## Observation:
# 1. Pets & Animals videos have highest likes-dislikes ratio. Not suprisingly, people find difficult to hate pets and animals.
# 2. Nonprofit & Activism's videos have lowest likes-dislike ratio and views-comments ratio. People relatively hate these video and comment a lot.
# 3. Obviosly, people still prefer implicit feedback than explicit. The ratio of views to comments is so large that only a comment written for hundreds of views.
# 
# Remarks: Also, I would like to hear any interesting facts that you found. Please comment below.
# 

# # Analysis: Sentiment analysis on Video's tags 

# ## Import libraries

# In[ ]:


from wordcloud import WordCloud
import nltk
#nltk.download()
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS

from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re


# In[ ]:


MAX_N = 1000

#remove all the stopwords from the text
en_stopwords = list(stopwords.words('english'))
de_stopwords = list(stopwords.words('german'))   
fr_stopwords = list(stopwords.words('french'))   
en_stopwords.extend(de_stopwords)
en_stopwords.extend(fr_stopwords)

polarities = list()

for cate in category_list:
    tags_word = my_df[my_df['category']==cate]['tags'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
    tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)
    word_tokens = word_tokenize(tags_word)
    filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]

# Remove numbers
    cleaned_data_title = [word for word in without_single_chr if not word.isdigit()]      
    
# Calculate frequency distribution
    word_dist = nltk.FreqDist(cleaned_data_title)
    hnhk = pd.DataFrame(word_dist.most_common(MAX_N),
                    columns=['Word', 'Frequency'])

    compound = .0
    for word in hnhk['Word'].head(MAX_N):
        compound += SentimentIntensityAnalyzer().polarity_scores(word)['compound']

    polarities.append(compound)

category_list = pd.DataFrame(category_list)
polarities = pd.DataFrame(polarities)
tags_sentiment = pd.concat([category_list,polarities],axis=1)
tags_sentiment.columns = ['category','polarity']
tags_sentiment=tags_sentiment.sort_values('polarity').reset_index()

plt.figure(figsize=(16,10))
sns.set(style="white",context="talk")
ax = sns.barplot(x=tags_sentiment['polarity'],y=tags_sentiment['category'], data=tags_sentiment,orient='h',palette="RdBu")
plt.xlabel("Categories")
plt.ylabel("polarity")
plt.title("Polarity of Categories in Youtube videos")


# By using sentiment analyzer from NLTK, we can examine the polarities of tags from all Youtube Trending Videos. All the 1000 most frequent tag from each categories were examined and result in form of number. *NEGATIVE* values indicate that most of the tags have negative sentiment and *POSITIVE* in contrast. However, I did not expected New & Politics videos to have most negative sentiment with its tags.

# # Analysis : What's the tags in the most negative and most positive category? What's the most discuss words for Science & Technology?

# In[ ]:


def wcloud(data,bgcolor):
    plt.figure(figsize = (20,15))
    cloud = WordCloud(background_color = bgcolor, max_words = 100,  max_font_size = 50)
    cloud.generate(' '.join(data))
    plt.imshow(cloud)
    plt.axis('off')


# ## News & Politics

# In[ ]:


tags_word = my_df[my_df['category']=='News & Politics']['tags'].str.lower().str.cat(sep=' ')

tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)
word_tokens = word_tokenize(tags_word)
filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
without_single_chr = [word for word in filtered_sentence if len(word) > 2]
cleaned_data_title = [word for word in without_single_chr if not word.isdigit()]

wcloud(cleaned_data_title,'white')


# ## People & Blogs

# In[ ]:


tags_word = my_df[my_df['category']=='People & Blogs']['tags'].str.lower().str.cat(sep=' ')

tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)
word_tokens = word_tokenize(tags_word)
filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
without_single_chr = [word for word in filtered_sentence if len(word) > 2]
cleaned_data_title = [word for word in without_single_chr if not word.isdigit()]

wcloud(cleaned_data_title,'white')


# ## Science & Technology

# In[ ]:


tags_word = my_df[my_df['category']=='Science & Technology']['tags'].str.lower().str.cat(sep=' ')

tags_word = re.sub('[^A-Za-z]+', ' ', tags_word)
word_tokens = word_tokenize(tags_word)
filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
without_single_chr = [word for word in filtered_sentence if len(word) > 2]
cleaned_data_title = [word for word in without_single_chr if not word.isdigit()]

wcloud(cleaned_data_title,'white')


# # **To Be Continued.....**
