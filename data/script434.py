
# coding: utf-8

# In[2]:


# Import libraries and set desired options

from __future__ import division, print_function
# Disable Anaconda warnings
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# ##### Approaching the Problem
# We will be solving the intruder detection problem analyzing his behavior on the Internet. It is a complicated and interesting problem combining the data analysis and behavioral psychology.
# 
# For example: Yandex solves the mailbox intruder detection problem based on the user's behavior patterns. In a nutshell, intruder's behaviour pattern might differ from the owner's one: 
# - the breaker might not delete emails right after they are read, as the mailbox owner might do
# - the intruder might mark emails and even move the cursor differently
# - etc.
# 
# So the intruder could be detected and thrown out from the mailbox proposing the owner to be authentificated via SMS-code.
# This pilot project is described in the Habrahabr article.
# 
# Similar things are being developed in Google Analytics and described in scientific researches. You can find more on this topic by searching "Traversal Pattern Mining" and "Sequential Pattern Mining".
# 
# In this competition we are going to solve a similar problem: our algorithm is supposed to analyze the sequence of websites consequently visited by a particular person and to predict whether this person is Alice or an intruder (someone else). As a metric we will use [ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic). We will reveal who Alice is at the end of the course.

# ### 1. Data Downloading and Transformation
# Register on [Kaggle](www.kaggle.com), if you have not done it before.
# Go to the competition [page](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) and download the data.
# 
# First, read the training and test sets. Then explore the data and perform a couple of simple exercises:

# In[4]:


# Read the training and test data sets
train_df = pd.read_csv('../input/train_sessions.csv',
                       index_col='session_id')
test_df = pd.read_csv('../input/test_sessions.csv',
                      index_col='session_id')

# Switch time1, ..., time10 columns to datetime type
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# Sort the data by time
train_df = train_df.sort_values(by='time1')

# Look at the first rows of the training set
train_df.head()


# The training data set contains the following features:
# 
# - **site1** – id of the first visited website in the session
# - **time1** – visiting time for the first website in the session
# - ...
# - **site10** – id of the tenth visited website in the session
# - **time10** – visiting time for the tenth website in the session
# - **target** – target variable, possesses value of 1 for Alice's sessions, and 0 for the other users' sessions
#     
# User sessions are chosen in the way they are not longer than half an hour or/and contain more than ten websites. I.e. a session is considered as ended either if a user has visited ten websites or if a session has lasted over thirty minutes.
# 
# There are some empty values in the table, it means that some sessions contain less than ten websites. Replace empty values with 0 and change columns types to integer. Also load the websites dictionary and check how it looks like:

# In[5]:


# Change site1, ..., site10 columns type to integer and fill NA-values with zeros
sites = ['site%s' % i for i in range(1, 11)]
train_df[sites] = train_df[sites].fillna(0).astype('int')
test_df[sites] = test_df[sites].fillna(0).astype('int')

# Load websites dictionary
with open(r"../input/site_dic.pkl", "rb") as input_file:
    site_dict = pickle.load(input_file)

# Create dataframe for the dictionary
sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
print(u'Websites total:', sites_dict.shape[0])
sites_dict.head()


# In[6]:


# Answer
print(test_df.shape, train_df.shape)


# In[7]:


# Our target variable
y_train = train_df['target']

# United dataframe of the initial data 
full_df = pd.concat([train_df.drop('target', axis=1), test_df])

# Index to split the training and test data sets
idx_split = train_df.shape[0]


# For the very basic model, we will use only the visited websites in the session (but we will not take into account timestamp features). The point behind this data selection is: *Alice has her favorite sites, and the more often you see these sites in the session, the higher probability that this is an Alice's session, and vice versa.*
# 
# Let us prepare the data, we will take only features `site1, site2, ... , site10` from the whole dataframe. Keep in mind that the missing values are replaced with zero. Here is how the first rows of the dataframe look like:

# In[8]:


# Dataframe with indices of visited websites in session
full_sites = full_df[sites]
full_sites.head()


# Sessions are the sequences of website indices, and data in this representation is inconvenient for linear methods. According to our hypothesis (Alice has favorite websites) we need to transform this dataframe so each website has corresponding feature (column) and its value is equal to number of this website visits in the session. It can be done in two lines:

# In[9]:


# sequence of indices
sites_flatten = full_sites.values.flatten()

# and the matrix we are looking for
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0],
                                sites_flatten,
                                range(0, sites_flatten.shape[0]  + 10, 10)))[:, 1:]


# ### 3. Training the first model
# 
# So, we have an algorithm and data for it. Let us build our first model, using [logistic regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) implementation from ` Sklearn` with default parameters. We will use the first 90% of the data for training (the training data set is sorted by time), and the remaining 10% for validation. Let's write a simple function that returns the quality of the model and then train our first classifier:

# In[10]:


def get_auc_lr_valid(X, y, C=1.0, seed=17, ratio = 0.9):
    # Split the data into the training and validation sets
    idx = int(round(X.shape[0] * ratio))
    # Classifier training
    lr = LogisticRegression(C=C, random_state=seed, n_jobs=-1).fit(X[:idx, :], y[:idx])
    # Prediction for validation set
    y_pred = lr.predict_proba(X[idx:, :])[:, 1]
    # Calculate the quality
    score = roc_auc_score(y[idx:], y_pred)
    
    return score


# In[11]:


get_ipython().run_cell_magic('time', '', '# Select the training set from the united dataframe (where we have the answers)\nX_train = full_sites_sparse[:idx_split, :]\n\n# Calculate metric on the validation set\nprint(get_auc_lr_valid(X_train, y_train))')


# The first model demonstrated the quality  of 0.91952 on the validation set. Let's take it as the first baseline and starting point. To make a prediction on the test data set ** we need to train the model again on the entire training data set ** (until this moment, our model used only part of the data for training), which will increase its generalizing ability:

# In[12]:


# Function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[13]:


# Train the model on the whole training data set
# Use random_state=17 for repeatability
# Parameter C=1 by default, but here we set it explicitly
lr = LogisticRegression(C=1.0, random_state=17).fit(X_train, y_train)

# Make a prediction for test data set
X_test = full_sites_sparse[idx_split:,:]
y_test = lr.predict_proba(X_test)[:, 1]

# Write it to the file which could be submitted
write_to_submission_file(y_test, 'baseline_1.csv')

