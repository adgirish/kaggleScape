
# coding: utf-8

# Hi everyone ! My brand new Python package for Auto Machine Learning is now available on github/PyPI/Kaggle kernels ! :)
# 
# **https://github.com/AxeldeRomblay/MLBox**
# 
# - It is very easy to use (see **documentation** on github)
# - It provides state-of-the-art algorithms and technics such as deep learning/entity embedding, stacking, leak detection, parallel processing, hyper-parameters optimization...
# - It has already been tested on Kaggle and performs well (see Kaggle "Two Sigma Connect: Rental Listing Inquiries" | Rank : **85/2488**)
# 
# **Please put a star on github and fork the script if you like it !** 
# 
# Enjoy :) 

# # Inputs & imports : that's all you need to give !

# In[1]:


from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


# In[ ]:


paths = ["../input/train.csv", "../input/test.csv"]
target_name = "target"


# # Now let MLBox do the job ! 

# ## ... to read and clean all the files 

# In[ ]:


rd = Reader(sep = ",")
df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)


# In[ ]:


dft = Drift_thresholder()
df = dft.fit_transform(df)   #removing non-stable features (like ID,...)


# ## ... to tune all the hyper-parameters

# In[ ]:


def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
 
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return np.abs(gini(a, p) / gini(a, a))


opt = Optimiser(scoring = make_scorer(gini_normalized, greater_is_better=True, needs_proba=True), n_folds=2)


# In[ ]:


space = {
    
        'est__strategy':{"search":"choice",
                                  "space":["LightGBM"]},    
        'est__n_estimators':{"search":"choice",
                                  "space":[700]},    
        'est__colsample_bytree':{"search":"uniform",
                                  "space":[0.77,0.82]},
        'est__subsample':{"search":"uniform",
                                  "space":[0.73,0.8]},
        'est__max_depth':{"search":"choice",
                                  "space":[5,6,7]},
        'est__learning_rate':{"search":"uniform",
                                  "space":[0.008, 0.02]} 
    
        }

params = opt.optimise(space, df, 7)


# But you can also tune the whole Pipeline ! Indeed, you can choose:
# 
# * different strategies to impute missing values
# * different strategies to encode categorical features (entity embeddings, ...)
# * different strategies and thresholds to select relevant features (random forest feature importance, l1 regularization, ...)
# * to add stacking meta-features !
# * different models and hyper-parameters (XGBoost, Random Forest, Linear, ...)

# ## ... to predict

# In[ ]:


prd = Predictor()
prd.fit_predict(params, df)


# ### Formatting for submission

# In[ ]:


submit = pd.read_csv("../input/sample_submission.csv",sep=',')
preds = pd.read_csv("save/"+target_name+"_predictions.csv")

submit[target_name] =  preds["1.0"].values

submit.to_csv("mlbox.csv", index=False)


# # That's all !!
# 
# If you like my new auto-ml package, please **put a star on github and fork/vote the Kaggle script :)**
