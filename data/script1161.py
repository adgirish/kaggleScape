
# coding: utf-8

# # A Laconic Approach - EDA (Version 1)
# In this notebook, I will be doing exploratory data analysis for given data for text normalization in laconic fashion. The task given to us in this competition is to convert written text into spoken forms, for example - 6ft will be converted into 6 feets, or $1.13 will convert into one dollar and thirteen cents.  I will work first by taking a glimpse of dataset shared and then go towards features extraction or rule based approach or selecting a RNN for this task. ** Lastly I will explain my analysis in laconic fashion, less words and more information**
# 
# - PS - My Second public kernel, do upvote if you find this analysis useful.
# 
# 

# ## Loading packages 

# In[ ]:


import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import time          #to get the system time
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
import math          #for basic maths operations
import seaborn as sns#for making plots
import matplotlib.pyplot as plt # for plotting
import os            # for operating system commands
import plotly.plotly as py # for Ploting 
import plotly.graph_objs as go # for ploting 
import plotly # for ploting 
plotly.offline.init_notebook_mode() # for using plotly in offline mode


# ## Importing input data
# **Train** - The dataset provided has following fields 
# 1. Setence_id - it signifies the id of sentence
# 2. Token id - it signifies the word's id inside that particular sentence 
# 3. class - TBU
# 4. before/ after - they shows how the token is getting changed after
# 
# **Test** - It won't have the field after 

# In[ ]:


s = time.time()
train_df = pd.read_csv("../input/en_train.csv")
test_df = pd.read_csv("../input/en_test.csv")
end = time.time()
print("time taken by above cell is {}.".format(end -s))
train_df.head()


# In[ ]:


train_seq = train_df.copy() # storing an original copy for later use


# ## Sanity check -
# Let's check three things - 
# 1. Number of rows in training and test dataets provided in this competition
# 2. Number of sentence in training and test dataets provided in this competition
# 3. Number of Nulls in training and test data and column wise nulls distribution

# In[ ]:


start = time.time()
print("Total number of rows in given training data is {}.".format(train_df.shape[0]))
print("Total number of sentence in given training data is {}".format(len(set(train_df.sentence_id))))
print("Total number of Nulls in given training data is \n{}.".format(train_df.isnull().sum()))
print("Total number of rows in given test data is {}.".format(test_df.shape[0]))
print("Total number of sentence in given test data is {}".format(len(set(test_df.sentence_id))))
print("Total number of Nulls in given test data is \n{}.".format(test_df.isnull().sum()))
end = time.time()
print("Time taken by above cell is {}.".format(end - start))


# ## Lets explore given variables in training data 
# **1. Sentence_id and Token_id ** - Let's plot a hoistogram and check the number of words in a given sentence and their frequency 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
temp_tr = pd.DataFrame(train_df.groupby('sentence_id')['token_id'].count())
sns.distplot(temp_tr['token_id'], axlabel = 'Number of words in a sentence', label = 'Number of words in a sentence', color="r")
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Min and Max of word per sentence is {} and {}.".format(temp_tr.token_id.min(),temp_tr.token_id.max()))
del temp_tr
print("Time taken by above cell is {}.".format((end-start)))
plt.show()


# **Findings** 
# - From the above plot, it is clear that the most of the sentence has less than or equal to 30 tokenper sentence and very few sentence s have more than  30 token per sentence. 
# - Minimum words per sentence is 2 
# - Maximum words per sentence is 256

# **2. Class** - Lets make box plots of classes and check the distributions of class variable

# In[ ]:


start = time.time()
temp_tr = pd.DataFrame(train_df.groupby('class')['token_id'].count())
temp_tr = temp_tr.reset_index()
X = list(temp_tr['class'])
Y = list(temp_tr['token_id'])
data = [go.Bar(
            x=X,
            y=Y
    )]
del temp_tr
plotly.offline.iplot(data, filename='basic-bar')
end = time.time()
print("Total number of different classes in training data is {}.".format(len(X)))
print("Time taken by above cell is {}.".format((end-start)))


# ** Findings - **
# - We can see that most frequent classes are plain, punct, letters, verbatim, date and cardinal ( total 6) 
# - Rest 10 classes are occuring very less frequently 
# - ** Class vaiable isn't present in test data => We have to assign class variable to test (you got it, right ? - cool  ;) )**

# **3. Lets see change before/ after with class** - 
# - Lets create a flag variable for token and check if before after is same or not 
# - Summarize over class varaible and see the effect of class type on normalization

# In[ ]:


# Lets first assign a variable change as 0 and if there is any change we will modify this change varaible to 1
start = time.time()
def isChange(row):
    """function to check if before after is getting changed or not"""
    chan = 0 
    if row['before'] == row['after']:
        chan = 0
    else:
        chan = 1
    return chan
train_df['change'] = 0
train_df['change'] = train_df.apply(lambda row: isChange(row), axis = 1)
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
train_df.head()


# In[ ]:


start = time.time()
temp_chn = train_df.loc[train_df['change']==1]
temp_nchn = train_df.loc[train_df['change']==0]

temp_tr1 = pd.DataFrame(temp_chn.groupby('class')['token_id'].count())
temp_tr1 = temp_tr1.reset_index()
X1 = list(temp_tr1['class'])
Y1 = list(temp_tr1['token_id'])

temp_tr2 = pd.DataFrame(temp_nchn.groupby('class')['token_id'].count())
temp_tr2 = temp_tr2.reset_index()
X2 = list(temp_tr2['class'])
Y2 = list(temp_tr2['token_id'])
trace1 = go.Bar(
    x=X1,
    y=Y1,
    name='Change'
)
trace2 = go.Bar(
    x=X2,
    y=Y2,
    name='NO Change'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='grouped-bar')
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))


# ** Findings ** - 
# - Most of the things that are getting changed are in all the difference classes but in plain and punct, and few in verbatim
# - Implies most of the data that is given to us in this competition is not changing and is redundent 
# - **Cardinal is changing => cardinal is getting spoken in english, 24 becomes twenty-four**
# - ** Date is changing => date 2Jan or 2/01/2001 is spoken as second January two thousand one**
# - ** Same is the case with letters, time, telephone** 

# **4. Class vs changes of token in sentence** - Lets plot changes in sentence grouped by over class and see the distribution using swarmplots in seaborn packages 
# - first plot is when the token_id change is considered as it is
# - second plot, limit on y is set, for better visulization of data 

# In[ ]:


start = time.time()
temp_tr = pd.DataFrame(train_df.groupby(['class', 'sentence_id', 'change'])['token_id'].count())
temp_tr.reset_index(inplace = True)
sns.set(style="ticks")
sns.set_context("poster")
sns.boxplot(x="class", y="token_id", hue="change", data=temp_tr, palette="PRGn")
plt.ylim(0, 150)
sns.despine(offset=10, trim=True)
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))


# In[ ]:


start = time.time()
temp_tr = pd.DataFrame(train_df.groupby(['class', 'sentence_id', 'change'])['token_id'].count())
temp_tr.reset_index(inplace = True)
sns.set(style="ticks")
sns.set_context("poster")
sns.boxplot(x="class", y="token_id", hue="change", data=temp_tr, palette="PRGn")
plt.ylim(0, 15)
sns.despine(offset=10, trim=True)
end = time.time()
print(temp_tr['class'].unique())
print("Time taken by above cell is {}.".format((end-start)))


# In[ ]:


start = time.time()
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
temp_tr1 = pd.DataFrame(temp_chn.groupby('sentence_id')['token_id'].count())
temp_tr2 = pd.DataFrame(temp_nchn.groupby('sentence_id')['token_id'].count())
sns.distplot(temp_tr1['token_id'], ax=ax[0], color='blue', label='With Change')
sns.distplot(temp_tr2['token_id'], ax=ax[1], color='green', label='Without Change')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))


# **Findings ** - 
# - From the above plot, it is clear that the distribution of sentences having change is somewhat similar to complete data 
# - Distribution of data for which there is no change is completely different than dist of complete data 

# In[ ]:


print("Fraction of token in complete data that are being changed are {}.".format(temp_tr1.shape[0]*100/train_df.shape[0]))


# **Findings ** - 
# - Fraction of data that is being changed is around ~4%, and anyway plain class data is redundent, **be careful of the class**

# In[ ]:


# lets check overlap between train and test 
train_list = train_df['before'].tolist()
test_list = test_df['before'].tolist()
s1 = set(train_list)
s2 = set(test_list)
common = s1.intersection(s2)
print("Common tokens between train and test is {}".format(len(common)/len(s2)))


# In[ ]:


def Assign(test, train):
    """ function to assign results"""
    token_dict = {}
    token_dict = dict(zip(train.before, train.after))
    #test['after'] = ''
    print("test shape {}".format(test.shape[0]))
    train.sort_values('before', ascending = True, inplace = True)
    train.drop_duplicates(subset='before', keep='first', inplace=True)
    train_new = train[['before', 'after']]
    print(train_new.head())
    print(test.head())
    test_new = pd.merge(test, train_new, how = 'left', on = 'before')
    print(test_new.head())

    #test_new['after'] = list(map(str, test_new['after']))
    def isNaN(num):
        return num != num
    test_new.after = np.where(isNaN(test_new.after), test_new.before, test_new.after)
    return(test_new)

start = time.time()
sub = Assign(test_df, train_df)
end = time.time()
sub.head(5)
#sub1.shape[0]


# In[ ]:


def submission(row):
    a = str(row['sentence_id'])+ "_"+ str(row['token_id'])
    return(a)

sub['id'] = sub.apply(lambda row: submission(row), axis =1)
sub[['id', 'after']].to_csv("mahesh_common_token.csv", index = False)


# ## Data preprocessing for Seq2Seq Modeling using RNN 
# My plan is now is to make a RNN for seq2seq modelling, As there can be contextual information and to capture that you must have the idea of context which can only be there is you are seeing sequences and not the words. Now for sequence to sequence modelling the first task is to convert the output sequence to correct output format. 

# In[ ]:


# I am defining the functions and will work on it later when I get time
print(train_seq.head(2))
def words_to_sequence(train_sub):
    """function takes the input dataframe and outputs a df which has sequence/sentences"""
    seq_ids = list(train_sub.sentence_id.unique())
    seq_df = pd.DataFrame(columns = ['sentence_id', 'before', 'after'])
    for i in seq_ids:
        temp = train_sub.loc[train_sub['sentence_id']==i]
        before_ = list(temp.before)
        #print(before_)
        before_list = ' '.join(word for word in before_)
        #print(before_list)
        after_ = list(temp.after)
        after_list = ' '.join(word for word in after_)
        seq_dict = {}
        seq_dict['sentence_id'] =i
        seq_dict['before'] = before_list
        seq_dict['after'] = after_list
        seq_temp = pd.DataFrame([seq_dict], columns=seq_dict.keys())
        seq_df = seq_df.append(seq_temp, ignore_index=True)
    return(seq_df)   


train_sub_seq = words_to_sequence(train_seq.loc[train_seq.sentence_id < 25].copy())
train_sub_seq.head(10)


# In[ ]:


def seq_to_words(seq_df):
    """function to convert seq dataframe to input kind of df"""
    return(words_df)

# Will finish this function later..


# # To be continued ....
