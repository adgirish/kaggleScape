
# coding: utf-8

# # Sentimental analysis 

# ##### Importing the textblob for checking the sentiments of the comments.v

# ##### Importig the modulesv

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from textblob import TextBlob


# In[ ]:


videos = pd.read_csv('../input/USvideos.csv',encoding='utf8',error_bad_lines = False);#opening the file USvideos
comm = pd.read_csv('../input/UScomments.csv',encoding='utf8',error_bad_lines=False);#opening the file UScomments


# ##### Making the BOB classifier and using it to test the sentiments of the sentence

# In[ ]:


pol=[] # list which will contain the polarity of the comments
for i in comm.comment_text.values:
    try:
        analysis =TextBlob(i)
        pol.append(analysis.sentiment.polarity)
        
    except:
        pol.append(0)


# ##### Converting the continuous variable to categorical one!

# In[ ]:


comm['pol']=pol

comm['pol'][comm.pol==0]= 0

comm['pol'][comm.pol > 0]= 1
comm['pol'][comm.pol < 0]= -1


# ### Lets perform EDA for the Positve sentences
# 

# In[ ]:


df_positive = comm[comm.pol==1]
df_positive.head()


# In[ ]:


k= (' '.join(df_positive['comment_text']))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# ### Its time to go for negative sentences

# In[ ]:


df_negative = comm[comm.pol==-1]
k= (' '.join(df_negative['comment_text']))
wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# # OOPS !

# # lets count the number of data with each type

# In[ ]:


comm['pol'].replace({1:'positive',0:'Neutral',-1:'negative'}).value_counts().plot(kind='bar',figsize=(7,4));
plt.title('Number of types of commets');
plt.xlabel('Comment_type');
plt.ylabel('number');


# ## Lets generate the dataframe which has unique_id and the info about its comment

# In[ ]:


id=[]
pos_comm=[]
neg_comm=[]
neutral_comm =[]
for i in set(comm.video_id):
    id.append(i)
    try:
        pos_comm.append(comm[comm.video_id==i].pol.value_counts()[1])
    except:
        pos_comm.append(0)
    try:    
        neg_comm.append(comm[comm.video_id==i].pol.value_counts()[-1])
    except:
        neg_comm.append(0)
    try:    
        neutral_comm.append(comm[comm.video_id==i].pol.value_counts()[0])
    except:
        neutral_comm.append(0)


# ## WOW! this is the most handful thing for analysis. Lets see its EDA in EDA section

# In[ ]:


df_unique = pd.DataFrame(id)
df_unique.columns=['id']
df_unique['pos_comm'] =pos_comm
df_unique['neg_comm'] = neg_comm
df_unique['neutral_comm'] = neutral_comm
df_unique['total_comments']=df_unique['pos_comm']+df_unique['neg_comm']+df_unique['neutral_comm']
df_unique.head(6)


# #### Storing this file for the EDA in another file

# In[ ]:


df_unique.to_csv('unique.csv',index=False,)


# # Exploratory Data Analysis :
#                                                                           

# In[ ]:


videos.head()


# In[ ]:


comm.head()


# # The data set is from 13.09 to 26.09. Total 14 days(2 week) Dataset

# In[ ]:


videos.date.value_counts()


# ## Surprisingly the videos_ id is not unique. lets see why

# In[ ]:


print(videos.video_id.value_counts()[:12]) # these videos have become 7 times the most trending videos of these 2 weeks.
most_trending = videos.video_id.value_counts()[:12].index


# In[ ]:


videos[videos.video_id=='mlxdnyfkWKQ']


# ## For first 2 minute i was like how its possible???..  Yes we are analysing the videos that were trending, So yes it is possible to have multiple ids

# Lets analyse which video was most trending of this time.

# In[ ]:


for i in most_trending:
    info =videos[videos.video_id== i][['title','channel_title','views','likes','dislikes','comment_total']].tail(1)# get the last row of the dataframe(total like,views,dislikes)
    print(info)
    print('****************************************************************************************')


# ## Explanation- Ofcourse it should be decreasing because the trending videos
# Looks like there is a clear and steady decline. If you are trending today, then you have good  chance of trending tomorrow, but such probability will fall off steadily as you look further out in the future.

# # tags

# In[ ]:


# slpitting the tags
tags = videos['tags'].map(lambda k: k.lower().split('|')).values 

# joining and making a complete list
k= (' '.join(videos['tags']))  
wordcloud = WordCloud(width = 1000, height = 500).generate((' '.join(k.lower().split('|'))))# word cloud


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# # Tags like Makeup,iphone,tutorial,beauty are the most common among all

# # Now its time to see which of the channel among all is best

# In[ ]:


videos.columns


# In[ ]:


df1 =pd.DataFrame(videos.channel_title.value_counts())
df1.columns=['times channel got trenidng']# how many times the channel got trending'
df1.head(6)


# In[ ]:


df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['views'].mean()).sort_values(by='views',ascending=False)
df_channel.head(10).plot(kind='bar');
plt.title('Most viewed channels');


# ## Inference -The Zayn is mostly viewed channel among all !

# In[ ]:


df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['likes'].mean()).sort_values(by='likes',ascending=False)
df_channel.head(10).plot(kind='bar');
plt.title('Most liked channels');


# #### Inference -Zayn vevo also leads for the mostly liked vidoes

# In[ ]:


videos['likes_per_view']=videos['likes']/videos['views']
df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['likes_per_view'].mean()).sort_values(by='likes_per_view',ascending=False)
df_channel.head(10).plot(kind='bar');
plt.title('Most liked channels');


# ## Videos which are disliked among all
# CBS Subday morning seems to be pretty boring channel

# In[ ]:


videos['dislikes_per_view']=videos['dislikes']/videos['views']
df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['dislikes_per_view'].mean()).sort_values(by='dislikes_per_view',ascending=False)
df_channel.head(10).plot(kind='bar');
plt.title('Most disliked channels');


# # Now i am going import a Unique file created in sentimental analysis section

# In[ ]:


unique = pd.read_csv('unique.csv',)


# In[ ]:


unique.sort_values(by='pos_comm',ascending=False).head(5)


# In[ ]:


videos[videos.video_id == 'eERPlIdPJtI'].title[225]


# ### Inference- Mostly 'Weight Update: 6 weeks Post Surgery! 93 pounds!' have very large number of positive reviews

# In[ ]:


sns.barplot(data=unique.sort_values(by='pos_comm',ascending=False).head(10),x='id',y='pos_comm')
plt.xticks(rotation=45);
plt.figure(figsize=(5,4));


# In[ ]:


sns.barplot(data=unique.sort_values(by='neg_comm',ascending=False).head(10),x='id',y='neg_comm')
plt.xticks(rotation=45);
plt.figure(figsize=(5,4));


# In[ ]:


sns.barplot(data=unique.sort_values(by='total_comments',ascending=False).head(10),x='id',y='total_comments')
plt.xticks(rotation=45);
plt.figure(figsize=(5,4));


# #### Lets find out the relation among continuous variables

# As quite obvious the number of likes have very strong relation with views

# In[ ]:


sns.regplot(data=videos,x='views',y='likes');
plt.title("Regression plot for likes & views");


# Number of dislikes are related but the relation is not as much strong.

# In[ ]:


sns.regplot(data=videos,x='views',y='dislikes');
plt.title("Regression plot for dislikes & views");


# ### Correlation matrix is the evidence of above analysis!

# In[ ]:


df_corr = videos[['views','likes','dislikes']]

sns.heatmap(df_corr.corr(),annot=True)


# <--------------------------------------END------------------------------------------->
# ###### If you like my work,
# ###### Your upvotes will be appreciated!
