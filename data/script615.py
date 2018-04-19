
# coding: utf-8

# Text Preprocessing
# --------
# Text preprocessing made on the competition datasets.
# The preprocessing consists of 4 steps:
# 
#  1. **Removing tags and URIs from contents**
#  2. **Removing punctuation from titles and contents**
#  3. **Removing stopwords from titles and contents**
#  4. **Converting the tags from string to a list of tags**
# 
# This type of operations can be used as a first step for any other process regarding the competition.

# In[ ]:


import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string


# Datasets loading
# ---------

# In[ ]:


dataframes = {
    "cooking": pd.read_csv("../input/cooking.csv"),
    "crypto": pd.read_csv("../input/crypto.csv"),
    "robotics": pd.read_csv("../input/robotics.csv"),
    "biology": pd.read_csv("../input/biology.csv"),
    "travel": pd.read_csv("../input/travel.csv"),
    "diy": pd.read_csv("../input/diy.csv"),
}


# For simplicity, i'll show an example of the steps of the preprocessing on an item of the robotics dataset

# In[ ]:


print(dataframes["robotics"].iloc[1])


# Removing html tags and uris from contents
# -----------

# In[ ]:


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""


# In[ ]:


# This could take a while
for df in dataframes.values():
    df["content"] = df["content"].map(stripTagsAndUris)


# In[ ]:


print(dataframes["robotics"].iloc[1])


# Removing punctuation from titles and contents
# -----------

# In[ ]:


def removePunctuation(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)


# In[ ]:


for df in dataframes.values():
    df["title"] = df["title"].map(removePunctuation)
    df["content"] = df["content"].map(removePunctuation)


# In[ ]:


print(dataframes["robotics"].iloc[1])


# Removing stopwords from titles and contents
# -----------

# In[ ]:


stops = set(stopwords.words("english"))
def removeStopwords(x):
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)


# In[ ]:


for df in dataframes.values():
    df["title"] = df["title"].map(removeStopwords)
    df["content"] = df["content"].map(removeStopwords)


# In[ ]:


print(dataframes["robotics"].iloc[1])


# Splitting tags string in a list of tags
# -----------

# In[ ]:


for df in dataframes.values():
    # From a string sequence of tags to a list of tags
    df["tags"] = df["tags"].map(lambda x: x.split())


# In[ ]:


print(dataframes["robotics"].iloc[1])


# Saving preprocessed dataframes to csv
# -----------

# In[ ]:


for name, df in dataframes.items():
    # Saving to file
    df.to_csv(name + "_light.csv", index=False)

