
# coding: utf-8

# <h3>Note:  in the Discussion section they said that data from figshare has some overlap with the current test set (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46177). So it's possible that using features/scores based on this data may overfit to the current test set.  Once they change the test set, the LB scores may change.
# So at this point, I think it's hard to tell whether using features based on these datasets will ultimately help your LB score. It may still help, but we won't know for sure until the new test set is released.<h3>

# **The idea for this kernel is to use the public datasets at https://conversationai.github.io/ to train models and use those models to score the train and test sets for this challenge. You can then use the scores as features when training the real models. So the output of this kernel isn't meant to be submitted as is. The output is the original train/test datasets, with additional columns/features.**
# 
# Using these enhanced train/test sets improved my logistic-regression based models from 0.047 to 0.044 log-loss. I haven't done much if any tuning for these models below, so you should be able to tweak things and get even better results.
# 
# I understand that there are PerspectiveAPI models that may be similar. But rather than wait for an API key, and so I could play around with the models more myself, I trained the models in this kernel.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


toxic_cmt = pd.read_table('../input/conversationaidataset/toxicity_annotated_comments.tsv')
toxic_annot = pd.read_table('../input/conversationaidataset/toxicity_annotations.tsv')
aggr_cmt = pd.read_table('../input/conversationaidataset/aggression_annotated_comments.tsv')
aggr_annot = pd.read_table('../input/conversationaidataset/aggression_annotations.tsv')
attack_cmt = pd.read_table('../input/conversationaidataset/attack_annotated_comments.tsv')
attack_annot = pd.read_table('../input/conversationaidataset/attack_annotations.tsv')


# **Find the mean score for toxicity, aggression, attack, and join with the corresponding comment**
# For each comment/rev_id, multiple workers have labeld/annotated. So then you have to decide what your overall label is for a given comment/rev_id. I simply took the mean value, and will train a regression model. You could try other aggregations/methods. You could, e.g., instead go with majority vote, and train binary classifiers, etc.

# In[ ]:


def JoinAndSanitize(cmt, annot):
    df = cmt.set_index('rev_id').join(annot.groupby(['rev_id']).mean())
    df = Sanitize(df)
    return df


# **Basic cleaning/standardizing -- can potentially do more (or less) here**

# In[ ]:


def Sanitize(df):
    comment = 'comment' if 'comment' in df else 'comment_text'
    df[comment] = df[comment].str.lower().str.replace('newline_token', ' ')
    df[comment] = df[comment].fillna('erikov')
    return df


# In[ ]:


toxic = JoinAndSanitize(toxic_cmt, toxic_annot)
attack = JoinAndSanitize(attack_cmt, attack_annot)
aggression = JoinAndSanitize(aggr_cmt, aggr_annot)


# **The attack and aggression labeled datasets are actually the same with only very slightly different annotations/labels**
# So probably only the scores from one model will be needed, but I left both here for completeness.

# In[ ]:


len(attack), len(aggression)


# In[ ]:


attack['comment'].equals(aggression['comment'])


# Check how correlated the mean value for the annotations between the attack and aggression datasets are

# In[ ]:


attack['attack'].corr(aggression['aggression'])


# **Check dataset**

# In[ ]:


toxic.head()
#attack.head()
#aggression.head()


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

def Tfidfize(df):
    # can tweak these as desired
    max_vocab = 200000
    split = 0.1

    comment = 'comment' if 'comment' in df else 'comment_text'
    
    tfidfer = TfidfVectorizer(ngram_range=(1,2), max_features=max_vocab,
                   use_idf=1, stop_words='english',
                   smooth_idf=1, sublinear_tf=1 )
    tfidf = tfidfer.fit_transform(df[comment])

    return tfidf, tfidfer


# Get the tfidf values for the training sets, as well as the fit tfidf vectorizer to be used later to transform the train/test sets for the real challenge datasets.

# In[ ]:


X_toxic, tfidfer_toxic = Tfidfize(toxic)
y_toxic = toxic['toxicity'].values
X_attack, tfidfer_attack = Tfidfize(attack)
y_attack = attack['attack'].values
X_aggression, tfidfer_aggression = Tfidfize(aggression)
y_aggression = aggression['aggression'].values


# **Model Training Strategy**
# 
# Rather than converting the 'toxicity', 'attack', 'aggression' into a binary label (e.g., >= 0.5), let's train a regression model to use as much information as possible. The output score from these models could be used as features in training the further refined models in the current challenge ('severe_toxic', 'obscene', etc.).
# 
# The toxicity/attack/aggression may not have a 1-1 mapping with the desired targets for the challenge, but they may be features that can help.

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

ridge = Ridge()
mse_toxic = -cross_val_score(ridge, X_toxic, y_toxic, scoring='neg_mean_squared_error')
mse_attack = -cross_val_score(ridge, X_attack, y_attack, scoring='neg_mean_squared_error')
mse_aggression = -cross_val_score(ridge, X_aggression, y_aggression, scoring='neg_mean_squared_error')


# In[ ]:


mse_toxic.mean(), mse_attack.mean(), mse_aggression.mean()


# **If the cross-validation scores look okay, train on the full dataset**

# In[ ]:


model_toxic = ridge.fit(X_toxic, y_toxic)
model_attack = ridge.fit(X_attack, y_attack)
model_aggression = ridge.fit(X_aggression, y_aggression)


# **Now score the original train and test sets, and save out as an additional feature for those datasets. (These can then be used when training/scoring with our real model**

# In[ ]:


train_orig = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test_orig = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')


# In[ ]:


train_orig = Sanitize(train_orig)
test_orig = Sanitize(test_orig)


# In[ ]:


def TfidfAndPredict(tfidfer, model):
    tfidf_train = tfidfer.transform(train_orig['comment_text'])
    tfidf_test = tfidfer.transform(test_orig['comment_text'])
    train_scores = model.predict(tfidf_train)
    test_scores = model.predict(tfidf_test)
    
    return train_scores, test_scores


# In[ ]:


toxic_tr_scores, toxic_t_scores = TfidfAndPredict(tfidfer_toxic, model_toxic)


# In[ ]:


toxic_tr_scores.shape, toxic_t_scores.shape


# In[ ]:


attack_tr_scores, attack_t_scores = TfidfAndPredict(tfidfer_attack, model_attack)


# In[ ]:


attack_tr_scores.shape, attack_t_scores.shape


# In[ ]:


aggression_tr_scores, aggression_t_scores = TfidfAndPredict(tfidfer_aggression, model_aggression)


# In[ ]:


aggression_tr_scores.shape, aggression_t_scores.shape


# **Ok, now write out these scores alongside the original train and test datasets**

# In[ ]:


# toxic_level, to not be confused with original label 'toxic'
train_orig['toxic_level'] = toxic_tr_scores
train_orig['attack'] = attack_tr_scores
train_orig['aggression'] = aggression_tr_scores
test_orig['toxic_level'] = toxic_t_scores
test_orig['attack'] = attack_t_scores
test_orig['aggression'] = aggression_t_scores


# In[ ]:


train_orig.to_csv('train_with_convai.csv', index=False)
test_orig.to_csv('test_with_convai.csv', index=False)

