
# coding: utf-8

# # Top Spotify Tracks of 2017: What Makes Songs Popular?
# In this kernel, I'll be examining the audio features of the tracks in Spotify's Top Songs of 2017 playlist. I extracted these audio features using Spotify's Web API and the spotipy library, and created the featuresdf.csv file. Please consider upvoting if this is interesting!
# 
# **Contents:**
# 1. Import Necessary Libraries
# 2. Read In and Explore the Data
# 3. Data Analysis

# ## 1) Import Necessary Libraries
# We'll start off by importing several Python libraries such as `numpy`, `pandas`, `matplotlib.pylot` and `seaborn`.

# In[1]:


#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## 2) Read In and Explore the Data
# It's time to read in our data using `pd.read_csv`, and take a first look using the `head()` and `describe()` functions.

# In[2]:


#read in featuresdf.csv and store in variable named music
music = pd.read_csv("../input/featuresdf.csv")
#look at first five rows of dataset
music.head()


# In[3]:


#look at summary of dataset
music.describe(include="all")


# ## 3) Data Analysis
# It's time to analyze each of the features in search of the answer to our question: What makes top Spotify songs popular?

# In[4]:


#we'll start off by creating two datasets, numeric and small, with some values removed
numeric = music.drop(['id','name','artists'], axis=1)
#i'm removing the below values since all the other values range between 0.0 and 1.0
#the other values are much larger, making it hard to see these values
small = numeric.drop(['tempo','duration_ms','key','loudness','time_signature'], axis=1)


# In[5]:


#set color palette to pastel
sns.set_palette('pastel')


# ## First Look
# To take our first look at the dataset, we'll create a bar chart of the mean values for danceability, energy, mode, speechiness, acousticness, instrumentalness, liveness and valence.

# In[6]:


#create a bar chart of the mean values of the audio features in the small dataset
small.mean().plot.bar()
plt.title('Mean Values of Audio Features')
plt.show()


# ## Danceability

# In[7]:


#mean value and distplot for danceability feature
print("Mean value for danceability:", music['danceability'].mean())
sns.distplot(music['danceability'])
plt.show()


# With a mean value of 0.697, it's clear that the majority of the top tracks have a high danceability rating.  
# **Conclusion:** People like to stream songs they can dance to. I wonder if this says anything about when/where people stream songs? (Parties etc.?) 

# ## Energy

# In[8]:


#mean value and distplot for energy feature
print("Mean value for energy:", music['energy'].mean())
sns.distplot(music['energy'])
plt.show()


# Again, people seem like energetic songs more than calm ones (mean of 0.661), although this feature seems to be a bit more evenly distributed than danceability.  
# **Conclusion:** People like energetic songs. I wonder what the ages of Spotify users are?

# ## Key

# In[9]:


#map the numeric values of key to notes
key_mapping = {0.0: 'C', 1.0: 'C♯,D♭', 2.0: 'D', 3.0: 'D♯,E♭', 4.0: 'E', 5.0: 'F', 6.0: 'F♯,G♭', 7.0: 'G', 8.0: 'G♯,A♭', 9.0: 'A', 10.0: 'A♯,B♭', 11.0: 'B'}
music['key'] = music['key'].map(key_mapping)

sns.countplot(x = 'key', data=music, order=music['key'].value_counts().index)
plt.title("Count of Song Keys")
plt.show()


# **Conclusion:** The most common key among top tracks is C♯/D♭. 

# ## Loudness

# In[10]:


#mean value and distplot for loudness feature
print("Mean value for loudness:", music['loudness'].mean())
sns.distplot(music['loudness'])
plt.show()


# So the mean value for loudness is -5.653. I'm still kind of confused on how to interpret this (Why are the values negative? Does a more negative value mean less loud or more loud?). If anybody has some insight, it would be greatly appreciated!  
# **Conclusion:** ??

# ## Mode

# In[11]:


#print mean value for mode
print("Mean value for mode feature:", music['mode'].mean())

#map the binary value of mode to major/minor
mode_mapping = {1.0: "major", 0.0: "minor"}
music['mode'] = music['mode'].map(mode_mapping)

#draw a countplot of the values
sns.countplot(x = 'mode', data=music)
plt.title("Count of Major/Minor Songs")
plt.show()


# People lean more towards songs with a major mode than those with a minor mode. Does this mean people like happier songs? (Maybe we'll find out with the valence feature.)  
# **Conclusion:** Major is preferred over minor. 

# ## Speechiness

# In[12]:


#mean value and distplot for speechiness feature
print("Mean value for speechiness:", music['speechiness'].mean())
sns.distplot(music['speechiness'])
plt.show()


# The mean value for speechiness is pretty low (only 0.104). This indicates that people prefer actual music. (I wonder if rapping counts as spoken lyrics?)  
# **Conclusion:** Actual music is more popular than, say, audiobooks. (Can't say I didn't see this coming..) 

# ## Acousticness

# In[13]:


#mean value and distplot for acousticness feature
print("Mean value for acousticness:", music['acousticness'].mean())
sns.distplot(music['acousticness'])
plt.show()


# Once again, the mean value for acousticness is low at 0.166.  
# **Conclusion:** People don't seem to stream acoustic songs as much non-acoustic ones. Sorry, acoustic covers!

# ## Instrumentalness

# In[14]:


#mean value and distplot for instrumentalness feature
print("Mean value for instrumentalness:", music['instrumentalness'].mean())
sns.distplot(music['instrumentalness'])
plt.show()


# The mean value for instrumentalness is *really, really* low at 0.00479.  
# **Conclusion:** People like songs that have lyrics. 

# ## Liveness

# In[15]:


#mean value and distplot for liveness feature
print("Mean value for liveness:valen", music['liveness'].mean())
sns.distplot(music['liveness'])
plt.show()


# As expected, the mean value for liveness is pretty low at 0.151. I wouldn't expect people to listen to live music on Spotify with the audience cheering in the background.  
# **Conclusion:** People like to listen to  live music at concerts, not on Spotify.

# ## Valence

# In[16]:


#mean value and distplot for valence feature
print('Mean value for valence feature:', music['valence'].mean())
sns.distplot(music['valence'])
plt.show()


# Happy and sad songs are actually pretty evenly distributed at 0.517.  
# **Conclusion:** Some days are happy, some days are sad. Music reflects that. 

# ## Tempo

# In[17]:


#mean value and distplot for tempo feature
print('Mean value for tempo feature:', music['tempo'].mean())
sns.distplot(music['tempo'])
plt.show()


# The mean value for tempo is 119.202 bpm, which is actually pretty fast.  
# **Conclusion:** People like fast songs more than slow ones.

# ## Duration

# In[18]:


#mean value and distplot for duration_ms feature
print('Mean value for duration_ms feature:', music['duration_ms'].mean())
sns.distplot(music['duration_ms'])
plt.show()


# The mean value for duration is 218387 milliseconds, which is around 3 minutes and 38 seconds.  
# **Conclusion:** People don't like it when songs are too short or too long. (Duh - although I'd say that 3 mins and 38 secs is pretty long already) 

# ## Time Signature

# In[19]:


#mean value and distplot for time_signature feature
print('Mean value for time_signature feature:', music['time_signature'].mean())
sns.distplot(music['time_signature'])
plt.show()


# Basically all the songs in the playlist are 4/4.  
# **Conclusion:** People really like songs that are 4/4? (I wonder if we subconsciously notice this or something.) 

# ## Correlation Heatmap

# In[26]:


plt.figure(figsize = (16,5))
sns.heatmap(numeric.corr(), cmap="coolwarm", annot=True)
plt.show()


# Energy and loudness seem to be pretty correlated, which is not surprising. What I did find surprising is that there seems to be little correlation between energy and danceability. 
# 
# **To be continued**

# # This is a work-in-progress! Any and all feedback is appreciated! 
