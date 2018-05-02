
# coding: utf-8

# # Classifying multi-label comments with Logistic Regression
# #### Rhodium Beng
# Started on 20 December 2017
# 
# This kernel is inspired by:
# - kernel by Jeremy Howard : _NB-SVM strong linear baseline + EDA (0.052 lb)_
# - kernel by Issac : _logistic regression (0.055 lb)_
# - _Solving Multi-Label Classification problems_, https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/

# In[8]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re


# ## Load training and test data

# In[9]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# ## Examine the data (EDA)

# In[10]:


train_df.sample(5)


# In the training data, the comments are labelled as one or more of the six categories; toxic, severe toxic, obscene, threat, insult and identity hate. This is essentially a multi-label classification problem.

# In[11]:


cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']


# In[12]:


# check missing values in numeric columns
train_df.describe()


# There are no missing numeric values. 
# As the mean values are very small (some way below 0.05), there would be many not labelled as positive in the six categories. From this I guess that there would be many comments which are not labelled in any of the six categories. Let's take a look.

# In[13]:


unlabelled_in_all = train_df[(train_df['toxic']!=1) & (train_df['severe_toxic']!=1) & (train_df['obscene']!=1) & 
                            (train_df['threat']!=1) & (train_df['insult']!=1) & (train_df['identity_hate']!=1)]
print('Percentage of unlabelled comments is ', len(unlabelled_in_all)/len(train_df)*100)


# In[14]:


# check for any 'null' comment
no_comment = train_df[train_df['comment_text'].isnull()]
len(no_comment)


# In[15]:


test_df.head()


# In[16]:


no_comment = test_df[test_df['comment_text'].isnull()]
no_comment


# All rows in the training and test data contain comments, so there's no need to clean up null fields.

# In[17]:


# let's see the total rows in train, test data and the numbers for the various categories
print('Total rows in test is {}'.format(len(test_df)))
print('Total rows in train is {}'.format(len(train_df)))
print(train_df[cols_target].sum())


# As mentioned earlier, majority of the comments in the training data are not labelled in one or more of these categories.

# In[18]:


# Let's look at the character length for the rows in the training data and record these
train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))


# In[19]:


# look at the histogram plot for text length
sns.set()
train_df['char_length'].hist()
plt.show()


# Most of the text length are within 500 characters, with some up to 5,000 characters long.

# Next, let's examine the correlations among the target variables.

# In[20]:


data = train_df[cols_target]


# In[21]:


colormap = plt.cm.plasma
plt.figure(figsize=(7,7))
plt.title('Correlation of features & targets',y=1.05,size=14)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,
           linecolor='white',annot=True)


# Indeed, it looks like some of the labels are higher correlated, e.g. insult-obscene has the highest at 0.74, followed by toxic-obscene and toxic-insult.

# What about the character length & distribution of the comment text in the test data?

# In[22]:


test_df['char_length'] = test_df['comment_text'].apply(lambda x: len(str(x)))


# In[23]:


plt.figure()
plt.hist(test_df['char_length'])
plt.show()


# Now, the shape of character length distribution looks similar between the training data and the train data. For the training data, I guess the train data were clipped to 5,000 characters to facilitate the folks who did the labelling of the comment categories.

# ## Clean up the comment text

# In[24]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


# In[25]:


# clean the comment_text in train_df [Thanks to Pulkit Jha for the useful pointer.]
train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))


# In[26]:


# clean the comment_text in test_df [Thanks, Pulkit Jha.]
test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))


# 
# ## Define X from entire train & test data for use in tokenization by Vectorizer

# In[27]:


train_df = train_df.drop('char_length',axis=1)


# In[28]:


X = train_df.comment_text
test_X = test_df.comment_text


# In[29]:


print(X.shape, test_X.shape)


# ## Vectorize the data

# In[30]:


# import and instantiate TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(max_features=5000,stop_words='english')
vect


# In[31]:


# learn the vocabulary in the training data, then use it to create a document-term matrix
X_dtm = vect.fit_transform(X)
# examine the document-term matrix created from X_train
X_dtm


# In[32]:


# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_X_dtm = vect.transform(test_X)
# examine the document-term matrix from X_test
test_X_dtm


# ## Solving a multi-label classification problem
# One way to approach a multi-label classification problem is to transform the problem into separate single-class classifier problems. This is known as 'problem transformation'. There are three methods:
# * _**Binary Relevance.**_ This is probably the simplest which treats each label as a separate single classification problems. The key assumption here though, is that there are no correlation among the various labels.
# * _**Classifier Chains.**_ In this method, the first classifier is trained on the input X. Then the subsequent classifiers are trained on the input X and all previous classifiers' predictions in the chain. This method attempts to draw the signals from the correlation among preceding target variables.
# * _**Label Powerset.**_ This method transforms the problem into a multi-class problem  where the multi-class labels are essentially all the unique label combinations. In our case here, where there are six labels, Label Powerset would in effect turn this into a 2^6 or 64-class problem. {Thanks Joshua for pointing out.}

# ## Binary Relevance - build a multi-label classifier using Logistic Regression

# In[33]:


# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(C=12.0)

# create submission file
submission_binary = pd.read_csv('../input/sample_submission.csv')

for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm, y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # compute the predicted probabilities for X_test_dtm
    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
    submission_binary[label] = test_y_prob


# ### Create submission file

# In[34]:


submission_binary.head()


# In[35]:


# generate submission file
submission_binary.to_csv('submission_binary.csv',index=False)


# #### Binary Relevance with Logistic Regression classifier scored 0.074 on the public leaderboard.

# ## Classifier Chains - build a multi-label classifier using Logistic Regression

# In[36]:


# create submission file
submission_chains = pd.read_csv('../input/sample_submission.csv')

# create a function to add features
def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[37]:


for label in cols_target:
    print('... Processing {}'.format(label))
    y = train_df[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm,y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))
    # make predictions from test_X
    test_y = logreg.predict(test_X_dtm)
    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]
    submission_chains[label] = test_y_prob
    # chain current label to X_dtm
    X_dtm = add_feature(X_dtm, y)
    print('Shape of X_dtm is now {}'.format(X_dtm.shape))
    # chain current label predictions to test_X_dtm
    test_X_dtm = add_feature(test_X_dtm, test_y)
    print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))


# ### Create submission file

# In[38]:


submission_chains.head()


# In[39]:


# generate submission file
submission_chains.to_csv('submission_chains.csv', index=False)


# ## Create a combined submission

# In[40]:


# create submission file
submission_combined = pd.read_csv('../input/sample_submission.csv')


# Combine using simple average from Binary Relevance and Classifier Chains.

# In[41]:


# corr_targets = ['obscene','insult','toxic']
for label in cols_target:
    submission_combined[label] = 0.5*(submission_chains[label]+submission_binary[label])


# In[42]:


submission_combined.head()


# In[43]:


# generate submission file
submission_combined.to_csv('submission_combined.csv', index=False)


# ### Thanks for reading my kernel.
# ### Tips and comments are most welcomed & appreciated.
# ### Please upvote if you find it useful.
