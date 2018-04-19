
# coding: utf-8

# A lot of people have looked at blending kernels and asked where the weights come from.  And the answer is usually something like, "Intuition, after taking into account their public leaderboard scores and the correlations among their forecasts."  Understandably, people don't find that answer very satisfying.  So I made this notebook to show the elegant way of choosing blending weights.  I call it "stacking" instead of "blending" because it fits an explicit metamodel on top of the base models, but the distinction in meaning isn't clear cut.  Some will say it's still blending becuase the "stacking model" is linear and the validation data used to fit the weights are separate from the original training data.  (With k-fold validation, you could use out-of-fold predictions from the original training data, but k-fold validation may be the wrong approach given the time component in the data.)

# In[56]:


# File containing validation data
# (These are selected from the last day of the original training set
#  to correspond to the times of day used in the test set.)
VAL_FILE = '../input/training-and-validation-data-pickle/validation.pkl.gz'


# In[57]:


import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

print(os.listdir("../input"))


# In[58]:


almost_zero = 1e-10
almost_one = 1 - almost_zero


# In[59]:


# Just names to identify the models
base_models = {
    'lgb1 ': "Python LGBM based on Pranav Pandya's R version",
    'wbftl': "anttip's Wordbatch FM-FTRL",
    'nngpu': "Downampled Neural Network run on GPU"
    }


# In[60]:


# Files with validation set predictions from each of the base models
# (These were fit on a subset of the training data that ends a day before
#  the end of the full training set.)
cvfiles = {
    'lgb1 ': '../input/validate-pranav-lgb-model/pranav_lgb_val_nostop.csv',
    'wbftl': '../input/validate-anttip-s-wordbatch-fm-ftrl-9711-version/wordbatch_fm_ftrl_val.csv',
    'nngpu': '../input/gpu-validation/gpu_val1.csv'
    }


# In[61]:


# Files with test set predictions
# (These were fit on the full training set
#  or on a subset at the end, to accommodate memory limits.)
subfiles = {
    'lgb1 ': '../input/try-pranav-s-r-lgbm-in-python/sub_lgbm_r_to_python_nocv.csv',
    'wbftl': '../input/anttip-s-wordbatch-fm-ftrl-9711-version/wordbatch_fm_ftrl.csv',
    'nngpu': '../input/talkingdata-gpu-example-with-multiple-runs/gpu_test2.csv'
    }


# In[62]:


# Public leaderbaord scores, for comparison
lbscores = {
    'lgb1 ': .9694,
    'wbftl': .9711,
    'nngpu': .9678
}


# You can click on the "Data" tab and follow the links to the kernels that generated each of the outputs above. Usually there are "forked from" links that you can follow to see where the models originated. (In the case of Pranav's LGBM, though, the link is misleading, because the original was [a separate kernel in R](https://www.kaggle.com/pranav84/single-lightgbm-in-r-with-75-mln-rows-lb-0-9690), and the fork was from a different Python kernel.)

# To train my stacker, I use logit transformations of each base model's validation predictions.  IMO this is the most straightforward way of doing it: my stacking model is a logistic regression, which takes a linear combination of its inputs and does a logistic (inverse logit) transformation on the result. It is essentially interpreting its inputs to be in units of logit (log odds) and then converting the result back into units of probability.
# 
# But you could also experiment with using other kinds of inputs.  Although the base models generate results that are nominally expressed as probabilities, they are mostly optimized for rank (i.e., using AUC) rather than probability value.  So in a sense they aren't *really* probabilities and maybe don't deserve to be treated as probabilities by having log odds taken.  Also, logit takes extreme predictions at face value, but maybe you don't trust the base models enough to believe their extreme predictions: maybe you want some regularization of the base model predicitons, and one way to get that regularization is to take the raw probabilities, or the ranks, instead of the logits.

# In[63]:


model_order = [m for m in base_models]  # To make sure order is consistent when converting to array


# In[64]:


cvdata = pd.DataFrame( { 
    m:pd.read_csv(cvfiles[m])['is_attributed'].clip(almost_zero,almost_one).apply(logit) 
    for m in base_models
    } )
X_train = np.array(cvdata[model_order])
y_train = pd.read_pickle(VAL_FILE)['is_attributed']  # relies on validation cases being in same order


# In[65]:


cvdata.head()


# In[66]:


cvdata.corr()


# In[67]:


stack_model = LogisticRegression()
stack_model.fit(X_train, y_train)
stack_model.coef_


# Note that the evaluation criterion for this competition (AUC) depends only on rank. Therefore:
# 1. Any linear transformation applied to the coefficients won't affect the score of the result.
# 2. We don't care much about the value of the intercept term (not shown above).
# 3. It doesn't matter whether the coefficients sum to 1.
# 4. If we normalize the coefficients to sum to 1, the result has the same interpretation as blending weights.

# In[68]:


weights = stack_model.coef_/stack_model.coef_.sum()
columns = cvdata[model_order].columns
scores = [ roc_auc_score( y_train, expit(cvdata[c]) )  for c in columns ]
names = [ base_models[c] for c in columns ]
lb = [ lbscores[c] for c in columns ]
pd.DataFrame( data={'LB score': lb, 'CV score':scores, 'weight':weights.reshape(-1)}, index=names )


# In[69]:


print(  'Stacker score: ', roc_auc_score( y_train, stack_model.predict_proba(X_train)[:,1] )  )


# Take the stacker score with a grain of salt. It's based on a directly optimized fit, so it's kind of like training set performance. (The validation set is a training set for the stacker.) You can't expect the leaderboard score to improve as much (relative to single model performance) as the stacker score does.  And when you make changes, you shouldn't expect a close relationship between changes in the stacker score and changes in leaderboard score.  (One would expect them typically to move in the same direction, but there are a lot of exceptions.)  The stacker score will almost always be higher than the individual model CV scores, since the stacker could have chosen to use just one model.  (Not quite always, because the stacker is optimizing log likelihood rather than AUC.)  But the leaderbaord score of the stacked model is not necessarily going to be higher than best individual model leaderbaord score (though it usually is, which is why we do stacking).  

# In[70]:


final_sub = pd.DataFrame()
subs = {m:pd.read_csv(subfiles[m]).rename({'is_attributed':m},axis=1) for m in base_models}
first_model = list(base_models.keys())[0]
final_sub['click_id'] = subs[first_model]['click_id']


# In[71]:


df = subs[first_model]
for m in base_models:
    if m != first_model:
        df = df.merge(subs[m], on='click_id')  # being careful in case clicks are in different order
df.head()


# In[72]:


X_test = np.array( df.drop(['click_id'],axis=1)[model_order].clip(almost_zero,almost_one).apply(logit) )
final_sub['is_attributed'] = stack_model.predict_proba(X_test)[:,1]
final_sub.head(10)


# In[ ]:


final_sub.to_csv("sub_stacked.csv", index=False, float_format='%.9f')

