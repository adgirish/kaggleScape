
# coding: utf-8

# For this kind of NLP problem, word means a lot for it, but there are some abbreviation in some sentence, such as 'What's' and 'What is',  literally, they are the same thing, but they will cause some problem when you do some statistics method, so let's replace them.

# In[ ]:


import numpy as np
import pandas as pd
from IPython.display import display
pd.set_option('display.max_colwidth',-1) # set max col width in order we can some more content


# First, let define some replace words, and maybe I missed some other abbreviations, if you have some more, please add it in the comment.

# In[ ]:


punctuation='["\'?,\.]' # I will replace all these punctuation with ''
abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    punctuation:'',
    '\s+':' ', # replace multi space with one single space
}


# I define a function in order can be reused

# In[ ]:


def process_data(file_name):
    data=pd.read_csv(file_name)
    data.question1=data.question1.str.lower() # conver to lower case
    data.question2=data.question2.str.lower()
    data.question1=data.question1.astype(str)
    data.question2=data.question2.astype(str)
    data.replace(abbr_dict,regex=True,inplace=True)
    display(data.head(2))
    return data


# In[ ]:


train_df=process_data('../input/train.csv')


# Let's check the result.

# In[ ]:


train_df.head()

