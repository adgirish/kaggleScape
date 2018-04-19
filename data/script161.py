
# coding: utf-8

# # Identifying authors - Who wrote that?
# Started on 30 Oct 2017
# 
# This notebook is inspired by:
# * Machine Learning: Classification - Coursera course by University of Washington,
# https://www.coursera.org/learn/ml-classification
# * Machine Learning with Text in scikit-learn - Kevin Markham's tutorial at Pycon 2016, 
# https://m.youtube.com/watch?t=185s&v=ZiKMIuYidY0
# * Kernel by bshivanni - "Predict the author of the story", 
# https://www.kaggle.com/bsivavenu/predict-the-author-of-the-story
# * Kernel by SRK - "Simple Engg Feature Notebook - Spooky Author",
# https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author

# Comments:
# 
# * In this kernel, I did a weighted averaging of the 'proba' of the 2 models to see the performance.
# * I added character counts as features to the sparse matrix to see if prediction performance will improve.
# 

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read "train.csv" and "test.csv into pandas

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Examine the train data

# In[ ]:


# check the class distribution for the author label in train_df?
train_df['author'].value_counts()


# #### The class distribution looks balanced.

# In[ ]:


# compute the character length for the rows and record these
train_df['text_length'] = train_df['text'].str.len()


# In[ ]:


# look at the histogram plot for text length
train_df.hist()
plt.show()


# #### Most of the text length are 500 characters and less. Let's look at the summary statistics of the text lengths by author.

# In[ ]:


EAP = train_df[train_df['author'] =='EAP']['text_length']
EAP.describe()


# In[ ]:


EAP.hist()
plt.show()


# In[ ]:


MWS = train_df[train_df['author'] == 'MWS']['text_length']
MWS.describe()


# In[ ]:


MWS.hist()
plt.show()


# In[ ]:


HPL = train_df[train_df['author'] == 'HPL']['text_length']
HPL.describe()


# In[ ]:


HPL.hist()
plt.show()


# ## Similarly examine the text length & distribution in test data

# In[ ]:


# examine the text characters length in test_df and record these
test_df['text_length'] = test_df['text'].str.len()


# In[ ]:


test_df.hist()
plt.show()


# #### The proportion of text which are long in the test data is very similar to that in the train data.

# ## Some preprocessing of the target variable to facilitate modelling

# In[ ]:


# convert author labels into numerical variables
train_df['author_num'] = train_df.author.map({'EAP':0, 'HPL':1, 'MWS':2})
# Check conversion for first 5 rows
train_df.head()


# #### Let's limit all text length to 700 characters for both train and test (for less outliers in data).

# In[ ]:


train_df = train_df.rename(columns={'text':'original_text'})
train_df['text'] = train_df['original_text'].str[:700]
train_df['text_length'] = train_df['text'].str.len()


# In[ ]:


test_df = test_df.rename(columns={'text':'original_text'})
test_df['text'] = test_df['original_text'].str[:700]
test_df['text_length'] = test_df['text'].str.len()


# ## Define X and y from train data for use in tokenization by Vectorizers

# In[ ]:


X = train_df['text']
y = train_df['author_num']


# ## Split train data into a training and a test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[ ]:


# examine the class distribution in y_train and y_test
print(y_train.value_counts(),'\n', y_test.value_counts())


# ## Vectorize the data using Vectorizer

# In[ ]:


# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# vect = CountVectorizer()
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b')
vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\?|\;|\:|\!|\'')
vect


# In[ ]:


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix created from X_train
X_train_dtm


# In[ ]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
X_test_dtm = vect.transform(X_test)
# examine the document-term matrix from X_test
X_test_dtm


# ### Add character counts as a features to the sparse matrix using function `add_feature`

# In[ ]:


def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[ ]:


from string import punctuation
X_train_chars = X_train.str.len()
X_train_punc = X_train.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_test_chars = X_test.str.len()
X_test_punc = X_test.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_train_dtm = add_feature(X_train_dtm, [X_train_chars, X_train_punc])
X_test_dtm = add_feature(X_test_dtm, [X_test_chars, X_test_punc])


# In[ ]:


X_train_dtm


# In[ ]:


X_test_dtm


# ## Build and evaluate an author classification model using Multinomial Naive Bayes

# In[ ]:


# import and instantiate the Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb


# In[ ]:


# tune hyperparameter alpha = [0.01, 0.1, 1, 10, 100]
from sklearn.model_selection import GridSearchCV
grid_values = {'alpha':[0.01, 0.1, 1.0, 10.0, 100.0]}
grid_nb = GridSearchCV(nb, param_grid=grid_values, scoring='neg_log_loss')
grid_nb.fit(X_train_dtm, y_train)
grid_nb.best_params_


# In[ ]:


# set with recommended hyperparameters
nb = MultinomialNB(alpha=1.0)
# train the model using X_train_dtm & y_train
nb.fit(X_train_dtm, y_train)


# In[ ]:


# make author (class) predictions for X_test_dtm
y_pred_test = nb.predict(X_test_dtm)


# In[ ]:


# compute the accuracy of the predictions with y_test
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_test)


# In[ ]:


# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)


# In[ ]:


# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)


# In[ ]:


# calculate predicted probabilities for X_test_dtm
y_pred_prob = nb.predict_proba(X_test_dtm)
y_pred_prob[:10]


# In[ ]:


# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)


# ## Build and evaluate an author classification model using Logistic Regression

# In[ ]:


# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=8)
logreg


# In[ ]:


# tune hyperparameter
grid_values = {'C':[0.01, 0.1, 1.0, 3.0, 5.0]}
grid_logreg = GridSearchCV(logreg, param_grid=grid_values, scoring='neg_log_loss')
grid_logreg.fit(X_train_dtm, y_train)
grid_logreg.best_params_


# In[ ]:


# set with recommended parameter
logreg = LogisticRegression(C=1.0, random_state=8)
# train the model using X_train_dtm & y_train
logreg.fit(X_train_dtm, y_train)


# In[ ]:


# make class predictions for X_test_dtm
y_pred_test = logreg.predict(X_test_dtm)


# In[ ]:


# compute the accuracy of the predictions
metrics.accuracy_score(y_test, y_pred_test)


# In[ ]:


# compute the accuracy of predictions with the training data
y_pred_train = logreg.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)


# In[ ]:


# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)


# In[ ]:


# compute the predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)
y_pred_prob[:10]


# In[ ]:


# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)


# ## Train the Logistic Regression model with the entire dataset from "train.csv"

# In[ ]:


# Learn the vocabulary in the entire training data, and create the document-term matrix
X_dtm = vect.fit_transform(X)
# Examine the document-term matrix created from X_train
X_dtm


# In[ ]:


# Add character counts features
X_chars = X.str.len()
X_punc = X.apply(lambda x: len([c for c in str(x) if c in punctuation]))
X_dtm = add_feature(X_dtm, [X_chars, X_punc])
X_dtm


# In[ ]:


# Train the Logistic Regression model using X_dtm & y
logreg.fit(X_dtm, y)


# In[ ]:


# Compute the accuracy of training data predictions
y_pred_train = logreg.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)


# ## Make predictions on the test data and compute the probabilities for submission

# In[ ]:


test = test_df['text']
# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_dtm = vect.transform(test)
# examine the document-term matrix from X_test
test_dtm


# In[ ]:


# Add character counts features
test_chars = test.str.len()
test_punc = test.str.count(r'\W')
test_dtm = add_feature(test_dtm, [test_chars, test_punc])
test_dtm


# In[ ]:


# make author (class) predictions for test_dtm
LR_y_pred = logreg.predict(test_dtm)
print(LR_y_pred)


# In[ ]:


# calculate predicted probabilities for test_dtm
LR_y_pred_prob = logreg.predict_proba(test_dtm)
LR_y_pred_prob[:10]


# ## Train the Naive Bayes model with the entire dataset "train.csv"

# In[ ]:


nb.fit(X_dtm, y)


# In[ ]:


# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)


# ## Make predictions on test data

# In[ ]:


# make author (class) predictions for test_dtm
NB_y_pred = nb.predict(test_dtm)
print(NB_y_pred)


# In[ ]:


# calculate predicted probablilities for test_dtm
NB_y_pred_prob = nb.predict_proba(test_dtm)
NB_y_pred_prob[:10]


# ## Create submission file
# #### Here I am combining the probabilities from the two models, using parameter alpha. 

# In[ ]:


alpha = 0.6
y_pred_prob = ((1-alpha)*LR_y_pred_prob + alpha*NB_y_pred_prob)
y_pred_prob[:10]


# In[ ]:


result = pd.DataFrame(y_pred_prob, columns=['EAP','HPL','MWS'])
result.insert(0, 'id', test_df['id'])
result.head()


# In[ ]:


# Generate submission file in csv format
result.to_csv('rhodium_submission_16.csv', index=False, float_format='%.20f')


# ### Thank you for reading this.
# ### Comments and tips are most welcomed.
# ### Please upvote if you find it useful. Cheers!
