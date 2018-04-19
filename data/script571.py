
# coding: utf-8

# Adversarial validation is a mean to check if train and test datasets have significant differences. The idea is to use the dataset features to try and separate train and test samples.
# 
# So you would create a binary target that would be 1 for train samples and 0 for test samples and fit a classifier on the features to predict if a given sample is in train or test datasets!
# 
# Here we will use a LogisticRegression and a TF-IDF vectorizer to check if text features distributions are different and see if we can separate the samples. 
# 
# The best kernel on this is certainly [here](https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms) by [Konrad Banachewicz](https://www.kaggle.com/konradb)
# 
# Other resources can be found [on fastML](http://fastml.com/adversarial-validation-part-one/)
# 

# In[1]:


import numpy as np
import pandas as pd
trn = pd.read_csv("../input/train.csv", encoding="utf-8")
sub = pd.read_csv("../input/test.csv", encoding="utf-8")


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import regex
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 1), 
    max_features=20000
)
trn_idf = vectorizer.fit_transform(trn.comment_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.comment_text)
sub_vocab = vectorizer.vocabulary_
all_idf = vectorizer.fit_transform(pd.concat([trn.comment_text, sub.comment_text], axis=0))
all_vocab = vectorizer.vocabulary_


# Convert vocab dictionnaries to list of words

# In[3]:


trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
all_words = [word for word in all_vocab.keys()]


# Check a few figures on words not in train or test

# In[4]:


common_words = set(trn_words).intersection(set(sub_words)) 
print("number of words in both train and test : %d "
      % len(common_words))
print("number of words in all_words not in train : %d "
      % (len(trn_words) - len(set(trn_words).intersection(set(all_words)))))
print("number of words in all_words not in test : %d "
      % (len(sub_words) - len(set(sub_words).intersection(set(all_words)))))


# This means there are substantial differences between train and test vocabularies or term frequencies
# 
# Let's check if a LinearRegression can make a difference between train and test using this.
# 
# We would take the output of the TF-IDF vectorizer fitted on train + test.

# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# Create target where all train samples are ones and all test samples are zeros
target = np.hstack((np.ones(trn.shape[0]), np.zeros(sub.shape[0])))
# Shuffle samples to mix zeros and ones
idx = np.arange(all_idf.shape[0])
np.random.seed(1)
np.random.shuffle(idx)
all_idf = all_idf[idx]
target = target[idx]
# Train a Logistic Regression
folds = StratifiedKFold(5, True, 1)
for trn_idx, val_idx in folds.split(all_idf, target):
    lr = LogisticRegression()
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))


# I must say I find these results alarming. There are significant differences between train and test sets even with  a vectorizer fitted on train and test samples. 
# 
# I did not see any wrong doing in the code so far but if you do please shout as loud as possible !
# 
# This to me means we may have surprises on the private LB.
# 
# Another experiment is to see if using common words (i.e. words in both train and test) would help reduce difference in words distribution. I will use a min_df of 3 

# In[6]:


vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 1), 
    max_features=20000,
    min_df=3
)
trn_idf = vectorizer.fit_transform(trn.comment_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.comment_text)
sub_vocab = vectorizer.vocabulary_


# Now use common vocab in TFIDF

# In[7]:


trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
print("Number of words in common : ", len(set(trn_words).intersection(set(sub_words))))
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 1), 
    vocabulary=list(set(trn_words).intersection(set(sub_words)))
)
all_idf = vectorizer.fit_transform(pd.concat([trn.comment_text, sub.comment_text], axis=0))
# Create target where all train samples are ones and all test samples are zeros
target = np.hstack((np.ones(trn.shape[0]), np.zeros(sub.shape[0])))
# Shuffle samples to mix zeros and ones
idx = np.arange(all_idf.shape[0])
np.random.seed(1)
np.random.shuffle(idx)
all_idf = all_idf[idx]
target = target[idx]
# Train a Logistic Regression
folds = StratifiedKFold(5, True, 1)
for trn_idx, val_idx in folds.split(all_idf, target):
    lr = LogisticRegression()
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))


# It's not really better.
# 
# Would using a smaller vocab finally help ?

# In[8]:


vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 1), 
    max_features=500,
    min_df=3
)
trn_idf = vectorizer.fit_transform(trn.comment_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.comment_text)
sub_vocab = vectorizer.vocabulary_

trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
print("Number of words in common : ", len(set(trn_words).intersection(set(sub_words))))
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 1), 
    vocabulary=list(set(trn_words).intersection(set(sub_words)))
)
all_idf = vectorizer.fit_transform(pd.concat([trn.comment_text, sub.comment_text], axis=0))
# Create target where all train samples are ones and all test samples are zeros
target = np.hstack((np.ones(trn.shape[0]), np.zeros(sub.shape[0])))
# Shuffle samples to mix zeros and ones
idx = np.arange(all_idf.shape[0])
np.random.seed(1)
np.random.shuffle(idx)
all_idf = all_idf[idx]
target = target[idx]
# Train a Logistic Regression
folds = StratifiedKFold(5, True, 1)
for trn_idx, val_idx in folds.split(all_idf, target):
    lr = LogisticRegression()
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))


# Even with an extreme reduction in vocabulary we still have significant differences between train and test. 
# 
# My belief is that it is due to the wide range of topics covered by the dataset with only 300000 samples in total.
# 
# Now you need to be aware that tokenization is important. To demonstrate that I'm going to use CountVectorizer and 2 different tokenization methods: a very simple oe and the TweetTokenizer from the nltk package.
# 
# The very simple tokenization will display a 0.60 adversarial AUC when TweetTokenizer will reach 0.80+ !
# 
# Let's start with the simple Tokenization

# In[15]:


from nltk.tokenize import TweetTokenizer
vectorizer = CountVectorizer(
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 2), 
    max_features=500,
    min_df=3
)
trn_idf = vectorizer.fit_transform(trn.comment_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.comment_text)
sub_vocab = vectorizer.vocabulary_

trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
print("Number of words in common : ", len(set(trn_words).intersection(set(sub_words))))
vectorizer = CountVectorizer(
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 2), 
    vocabulary=list(set(trn_words).intersection(set(sub_words)))
)
all_idf = vectorizer.fit_transform(pd.concat([trn.comment_text, sub.comment_text], axis=0))
# Create target where all train samples are ones and all test samples are zeros
target = np.hstack((np.ones(trn.shape[0]), np.zeros(sub.shape[0])))
# Shuffle samples to mix zeros and ones
idx = np.arange(all_idf.shape[0])
np.random.seed(1)
np.random.shuffle(idx)
all_idf = all_idf[idx]
target = target[idx]
# Train a Logistic Regression
folds = StratifiedKFold(5, True, 1)
for trn_idx, val_idx in folds.split(all_idf, target):
    lr = LogisticRegression()
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))


# Here is the corresponding list of words

# In[10]:


list(set(trn_words).intersection(set(sub_words)))


# Now let's use TweetTokenizer

# In[ ]:


from nltk.tokenize import TweetTokenizer
vectorizer = CountVectorizer(
    strip_accents='unicode',
    tokenizer=TweetTokenizer().tokenize,
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 2), 
    max_features=500,
    min_df=3
)
trn_idf = vectorizer.fit_transform(trn.comment_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.comment_text)
sub_vocab = vectorizer.vocabulary_

trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
print("Number of words in common : ", len(set(trn_words).intersection(set(sub_words))))
vectorizer = CountVectorizer(
    strip_accents='unicode',
    tokenizer=TweetTokenizer().tokenize,
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 2), 
    vocabulary=list(set(trn_words).intersection(set(sub_words)))
)
all_idf = vectorizer.fit_transform(pd.concat([trn.comment_text, sub.comment_text], axis=0))
# Create target where all train samples are ones and all test samples are zeros
target = np.hstack((np.ones(trn.shape[0]), np.zeros(sub.shape[0])))
# Shuffle samples to mix zeros and ones
idx = np.arange(all_idf.shape[0])
np.random.seed(1)
np.random.shuffle(idx)
all_idf = all_idf[idx]
target = target[idx]
# Train a Logistic Regression
folds = StratifiedKFold(5, True, 1)
for trn_idx, val_idx in folds.split(all_idf, target):
    lr = LogisticRegression()
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))


# Here is the corresponding list of words

# In[17]:


print(list(set(trn_words).intersection(set(sub_words))))


# You may want to push all this further and see the impact of a cleaner dataset.
