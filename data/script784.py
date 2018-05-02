
# coding: utf-8

# ## Word Share Model
# The word_match_share feature is surprisingly predictive. In this notebook, we make a quick Pandas-only model that gets 0.356 LB without machine learning. It just groups on binned word_match_share values and then averages the label.

# In[ ]:


import numpy as np
import pandas as pd
tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')
from nltk.corpus import stopwords
SCALE = 0.3627


# As noted in the kernel [How many 1's are in the public LB][1], the mean label on the test set is about 0.175, which is quite different from the training mean of 0.369. The constant SCALE above is an odds ratio. We use it to shift the mean label of the predictions from that of the training set to that of the test set. This is needed because the evaluation metric for this competition is log loss, which is sensitive to the mean prediction.       
# $$SCALE = \frac{\frac{y_{te}}{1 - y_{te}}}{\frac{y_{tr}}{1 - y_{tr}}}$$    
# Where y_te is the mean test label and y_tr is the mean training set label.
# 
# 
#   [1]: https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb

# This model uses only one feature, which is the ever-popular word_match_share feature. The version below is Pandas-centric, but is equivalent to other versions available in the kernels.

# In[ ]:


def word_match_share(x):
    '''
    The much-loved word_match_share feature.

    Args:
        x: source data with question1/2
        
    Returns:
        word_match_share as a pandas Series
    '''
    stops = set(stopwords.words('english'))
    q1 = x.question1.fillna(' ').str.lower().str.split()
    q2 = x.question2.fillna(' ').str.lower().str.split()
    q1 = q1.map(lambda l : set(l) - stops)
    q2 = q2.map(lambda l : set(l) - stops)
    q = pd.DataFrame({'q1':q1, 'q2':q2})
    q['len_inter'] = q.apply(lambda row : len(row['q1'] & row['q2']), axis=1)
    q['len_tot'] = q.q1.map(len) + q.q2.map(len)
    return (2 * q.len_inter / q.len_tot).fillna(0)


# To make our model, all we do is group on binned values of word_match_share, and then count the number of positives and total values, scaling the positives by SCALE to adjust the mean prediction. Then we compute binned word_match_share for the test set and apply those values, with a little bit of Laplace smoothing to get rid of extreme values based on low counts.

# In[ ]:


def bin_model(tr, te, bins=100, vpos=1, vss=3):
    '''
    Runs a Pandas table model using the word_match_share feature.
    
    Args:
        tr: pandas DataFrame with question1/2 in it
        te: test data frame
        bins: word shares are rounded to whole numbers after multiplying by bins.
        v_pos: number of virtual positives for smoothing (can be non-integer)
        vss: virtual sample size for smoothing (can be non-integer)
        
    Returns:
        submission in a Pandas Data Frame.
    '''
    tr['word_share'] = word_match_share(tr)
    tr['binned_share'] = (bins * tr.word_share).round()
    pos = tr.groupby('binned_share').is_duplicate.sum()
    cts = tr.binned_share.value_counts()
    te['word_share'] = word_match_share(te)
    te['binned_share'] = (bins * te.word_share).round()
    te_pos = te.binned_share.map(pos, na_action='ignore').fillna(0)
    te_cts = te.binned_share.map(cts, na_action='ignore').fillna(0)
    prob = (te_pos + vpos) / (te_cts + vss)
    odds = prob / (1 - prob)
    scaled_odds = SCALE * odds
    scaled_prob = scaled_odds / (1 + scaled_odds)
    sub = te[['test_id']].copy()
    sub['is_duplicate'] = scaled_prob
    return sub


# In[ ]:


sub = bin_model(tr, te)
sub.to_csv('no_ml_model.csv', index=False, float_format='%.6f')
sub.head(10)


# And that should do it... :)
