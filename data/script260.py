
# coding: utf-8

# The goal of this script is to improve classification by extending the dataset. My inspiration came from Pavel Ostyakov's [A simple technique for extending dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038).
# 
# My method involves creating synthetic training data using a Markov chain generator. People far smarter than me figured out how to adapt Markov chains for this purpose and create a package called [Markovify](https://github.com/jsvine/markovify) for easy use.
# 
# The technique here does not improve the AUC score over a baseline script, at least not the way I'm doing it. Even so, I found the computer-generated comments to be fascinating!
# 
#    - UPDATE Jan 29: Changed the eval to use a single, stratified validation set for 'threat'. Added logloss metric.
#    - UPDATE Feb 03: Changed base algorithm. Changed metric to AUC. Changed some content.
# 
# ### 1. Create Some Data
# 
# Import packages and data as usual...

# In[ ]:


# %autosave 600
import numpy as np
import pandas as pd
import markovify as mk

train = pd.read_csv('../input/train.csv')
train.head()


# Jagan shows us the breakdown of classes in his [Stop the S@# EDA](https://www.kagg**le.com/jagangupta/stop-the-s-toxic-comments-eda). 'threat' is very imbalanced, which may make it a good candidate for upsampling.
# 
# <img src="https://www.kaggle.io/svf/2240904/531400eed7c5659dbfa7af1ebaf64e45/__results___files/__results___10_0.png" />
# 
# 
# We're looking at one category at a time so I don't think we care about the class of the other categories at this time. I could be wrong. 
# 

# In[ ]:


tox = train.loc[train['threat'] == 1, ['comment_text']].reset_index(drop=True)
tox.head(10)


# Nasty stuff! These people need to relax.
# 
# From here I take a simple approach and load the raw texts into one document. No text processing or anything. 
# Also I'll get the median length of the comments to provide a more consistent output.

# In[ ]:


doc = tox['comment_text'].tolist()

nchar = int(tox.comment_text.str.len().median())
nchar


# Now comes the magic which is so easy with Markovify. Create a text_model object and produce some comments...

# In[ ]:


text_model = mk.Text(doc)
for i in range(10):
    print(text_model.make_short_sentence(nchar))


# Only two lines of code and you too can sound like an angry 5th grader!

# ### 2. Check for Improvement
# 
# This is fun and all, but we want to see if it helps classify the comments. I'll plagiarize [Logistic regression with words and char n-grams](https://www.kaggle.com/thousandvoices/logistic-regression-with-words-and-char-n-grams) by thousandvoices.
# 

# In[ ]:


import numpy as np
import pandas as pd
import markovify as mk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit


# In[ ]:


runrows = 40000  # None


train = pd.read_csv('../input/train.csv', nrows=runrows).fillna(' ')
test = pd.read_csv('../input/test.csv', nrows=runrows).fillna(' ')

class_names = list(train)[-6:]
train_base_text = train['comment_text']
test_text = test['comment_text']

maxfeats = 200  # 20000


# This part takes a little over an hour to run, which is why I used the limits above. I had to run the script on my machine to finish.

# In[ ]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=maxfeats)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=maxfeats)

predictions = {'id': test['id']}
losses = []

for cl in class_names:
    
    #preprocess
    print('starting {}'.format(cl))
    class_df = train.loc[train[cl] == 1, ['comment_text']].reset_index(drop=True)
    class_list = class_df['comment_text'].tolist() 
    
    nchar = int(class_df.comment_text.str.len().median())
    
    # only augment the smallest classes
    ll = class_df.shape[0]
    if ll < 7500:
        count = int(ll/2)
    else:
        count = 0
        
    # generate markovified text
    mkv_text = []
    text_model = mk.Text(class_list)
    for i in range(count):
        new = text_model.make_short_sentence(nchar)
        mkv_text.append(new)
       
    train_ls = train_base_text.tolist() + mkv_text
    train_text = pd.Series(train_ls)    
    all_text = train_text.append(test_text)
    
    # create tfidf features
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)
    
    train_features = hstack([train_char_features, train_word_features])
    test_features = hstack([test_char_features, test_word_features])
    
    train_base_tgt = train[cl]
    train_class = np.ones(count)
    train_target = np.append(train_base_tgt, train_class)
    
    # train and predict
    classifier = LogisticRegression(solver='sag', n_jobs=-1)
    classifier.fit(train_features, train_target)
    predictions[cl] = expit(logit(classifier.predict_proba(test_features)[:, 1]) - 0.5)


# In[ ]:


submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('submission_mk.csv', index=False)


# As mentioned at the beginning, the final result is not as good as the baseline. This script scores 0.977 vs 0.978 on the test set. ryches says below in the comments:
# > Very interesting to generate data like this but I think that the bag of words is holding it back because the markov is just picking things likely to go together in different orders so on a certain level it is just shuffling words around and then your bag of words model is ignoring the order anyway.
# 
# There may be other features that survive the model, but not these ones. Thanks for reading!
