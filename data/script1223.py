
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

import numpy as np

import sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn import metrics

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from IPython.display import display


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **GINI calculations**

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
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]


# In[ ]:


PATH = "../input/train.csv"
data_raw= pd.read_csv(f'{PATH}', low_memory=False)


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)
display_all(data_raw.head(5))


# In[ ]:


# Describe the data set
display_all(data_raw.describe(include='all'))


# **Important observations:**
# - Target variable (*target*) has its mean 3.65% meaning that only 3.65% of data is classified as the target. Hence, the dataset is unbalanced

# In[ ]:


# Distribution of target variable
import matplotlib.pyplot as plt
plt.hist(data_raw['target'])
plt.show()

print('Percentage of claims filed :' , str(np.sum(data_raw['target'])/data_raw.shape[0]*100), '%')


# But before jumping into balancing the data let's **look at the NA's, which in this dataset are denoted as -1**.

# In[ ]:


nas = np.sum(data_raw == -1)/len(data_raw) *100
print("The percentage of missing values is")
print (nas[nas>0].sort_values(ascending = False))


# Now, for categorical variables we will create dummy variables (aka **one-hot encoding**)

# In[ ]:


# make a copy of the initial dataset
data_clean = data_raw.copy()
#data_clean.columns
cat_cols = [c for c in data_clean.columns if c.endswith('cat')]
for column in cat_cols:
    temp=pd.get_dummies(data_clean[column], prefix=column, prefix_sep='_')
    data_clean=pd.concat([data_clean,temp],axis=1)
    data_clean=data_clean.drop([column],axis=1)

print('data_clean shape is:',data_clean.shape)


# In[ ]:


# Impute missing values with medians

num_cols = ['ps_reg_03','ps_car_14', 'ps_car_11', 'ps_car_12' ]

for n in num_cols:
    dummy_name = str(n) + 'NA'
    data_clean[dummy_name] = (data_clean[n]==-1).astype(int)
    med = data_clean[data_clean[n]!=-1][n].median()
    data_clean.loc[data_clean[n]==-1,n] = med
    

    


# In[ ]:


#Make transformation to ps_car_13, as suggested here: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41489
data_clean['ps_car_13_trans'] = round(data_clean['ps_car_13']* data_clean['ps_car_13']* 90000,2)


# **Undersampling**:
# Let's take 25% of abundant data (target = 0) and stack together with the rare data (target = 1)

# In[ ]:


sub_df_0= data_clean[(data_clean['target']==0)]
sub_df_1= data_clean[(data_clean['target']==1)]
sub_df_1.shape


# In[ ]:


sub_df = sub_df_0.sample(frac = 0.25, random_state = 42)
data_sub = pd.concat([sub_df_1,sub_df])


# ## XGBoost model

# In[ ]:


# First split the data into training and validation (test) sets
training_features, test_features, training_target, test_target, = train_test_split(data_sub.drop(['id','target'], axis=1),
                                               data_sub['target'],
                                               test_size = .2,
                                               random_state=12)

# Now further split the training test into training and validation to 
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .2,
                                                  random_state=12)


# We first run an XGB trainng model with some baseline parameters and then proceed to tuning the key parameters via a series of loops. 
# 
# To grasp an idea of what parameters to tune and in what order, look here:
# - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# - https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1?next_slideshow=1

# Since it takes time (10-30) minutes to run loops, I'll comment out the code and hardcode the results below each code cell with a loop. To actually run the code, please uncomment it.

# In[ ]:


xgb_params = {'eta': 0.02, 
              'max_depth': 6, 
              'subsample': 1.0, 
              'colsample_bytree': 0.3,
              'min_child_weight': 1,
              'objective': 'binary:logistic', 
              'eval_metric': 'auc', 
              'seed': 99, 
              'silent': True}
d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_val,y_val)
d_test = xgb.DMatrix(test_features)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#model = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=100, early_stopping_rounds=200)
#print(model.best_score, model.best_iteration, model.best_ntree_limit)


# [0]	train-gini:0.196601	valid-gini:0.19429
# 
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# 
# [100]	train-gini:0.321987	valid-gini:0.278464
# 
# [200]	train-gini:0.359511	valid-gini:0.285658
# 
# [300]	train-gini:0.392516	valid-gini:0.289204
# 
# [400]	train-gini:0.419991	valid-gini:0.289849
# 
# [500]	train-gini:0.443408	valid-gini:0.291014
# 
# [600]	train-gini:0.458003	valid-gini:0.290918
# 
# [700]	train-gini:0.473734	valid-gini:0.290754
# 
# [800]	train-gini:0.487845	valid-gini:0.290586
# 
# Stopping. Best iteration:
# [
# 648]	train-gini:0.465592	valid-gini:0.291336
# 
# 0.291336 648 649

# In[ ]:


#results = {'best_score':[],'best_iter':[],'best_ntree_limit':[]}


# train-gini:0.465592	valid-gini:0.291336
# Best_ntree_limit = 649
# 
# Now, let's tune the **learning rate (eta)**

# In[ ]:


results = {'eta':[],'best_score':[],'best_ntree_limit':[]}
for e in [0.01, 0.02, 0.03,0.05,0.1,0.2]:
    xgb_params = {'eta': e, 
                  'max_depth': 6, 
                  'subsample': 1.0, 
                  'colsample_bytree': 0.3,
                  'min_child_weight': 1,
                  'objective': 'binary:logistic', 
                  'seed': 99, 
                  'silent': True}

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
   # m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=100, early_stopping_rounds=200)
    #results['best_score'].append(m.best_score)
    #results['best_ntree_limit'].append(m.best_ntree_limit)
    #results['eta'].append(e)
    
#print('eta:',results['eta'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])


# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [100]	train-gini:0.306974	valid-gini:0.276946
# [200]	train-gini:0.324915	valid-gini:0.279907
# [300]	train-gini:0.341885	valid-gini:0.284432
# [400]	train-gini:0.360752	valid-gini:0.286026
# [500]	train-gini:0.379185	valid-gini:0.2878
# [600]	train-gini:0.394244	valid-gini:0.288596
# [700]	train-gini:0.408409	valid-gini:0.289171
# [800]	train-gini:0.420627	valid-gini:0.289688
# [900]	train-gini:0.433574	valid-gini:0.289878
# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [100]	train-gini:0.321987	valid-gini:0.278464
# [200]	train-gini:0.359511	valid-gini:0.285658
# [300]	train-gini:0.392516	valid-gini:0.289204
# [400]	train-gini:0.419991	valid-gini:0.289849
# [500]	train-gini:0.443408	valid-gini:0.291014
# [600]	train-gini:0.458003	valid-gini:0.290918
# [700]	train-gini:0.473734	valid-gini:0.290754
# [800]	train-gini:0.487845	valid-gini:0.290586
# Stopping. Best iteration:
# [648]	train-gini:0.465592	valid-gini:0.291336
# 
# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [100]	train-gini:0.337376	valid-gini:0.283497
# [200]	train-gini:0.390756	valid-gini:0.289544
# [300]	train-gini:0.429966	valid-gini:0.292974
# [400]	train-gini:0.456514	valid-gini:0.292849
# [500]	train-gini:0.481765	valid-gini:0.291735
# Stopping. Best iteration:
# [340]	train-gini:0.44194	valid-gini:0.293288
# 
# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [100]	train-gini:0.37337	valid-gini:0.288152
# [200]	train-gini:0.440843	valid-gini:0.289715
# [300]	train-gini:0.48632	valid-gini:0.287219
# [400]	train-gini:0.520738	valid-gini:0.286422
# Stopping. Best iteration:
# [201]	train-gini:0.441803	valid-gini:0.289792
# 
# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [100]	train-gini:0.431092	valid-gini:0.28708
# [200]	train-gini:0.516016	valid-gini:0.28225
# [300]	train-gini:0.579317	valid-gini:0.275065
# Stopping. Best iteration:
# [102]	train-gini:0.433424	valid-gini:0.287461
# 
# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [100]	train-gini:0.512916	valid-gini:0.273241
# [200]	train-gini:0.638731	valid-gini:0.261126
# Stopping. Best iteration:
# [32]	train-gini:0.389099	valid-gini:0.282492
# 
# eta: [0.01, 0.02, 0.03, 0.05, 0.1, 0.2] best_score: [0.290367, 0.291336, 0.293288, 0.289792, 0.287461, 0.282492] best_ntree_limit: [988, 649, 341, 202, 103, 33]

# We see that **$\eta = 0.03$** gives better score of 0.293288 and at such learning rate **n_trees** = 341
# 
# We can now tune ```max_depth``` parameter

# In[ ]:


results = {'max_depth':[],'best_score':[],'best_ntree_limit':[]}
for md in range(3,9,1):
    xgb_params = {'eta': 0.03, 
                  'max_depth': md, 
                  'subsample': 1.0, 
                  'colsample_bytree': 0.3,
                  'min_child_weight': 1,
                  'objective': 'binary:logistic', 
                  'seed': 99, 
                  'silent': True}

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    #m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=200)
    #results['best_score'].append(m.best_score)
    #results['best_ntree_limit'].append(m.best_ntree_limit)
    #results['max_depth'].append(md)
    
#print('max_depth:',results['max_depth'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])


# [0]	train-gini:0.152895	valid-gini:0.148037
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [50]	train-gini:0.254714	valid-gini:0.261111
# [100]	train-gini:0.264271	valid-gini:0.268258
# [150]	train-gini:0.275104	valid-gini:0.274721
# [200]	train-gini:0.283029	valid-gini:0.278809
# [250]	train-gini:0.290808	valid-gini:0.282106
# [300]	train-gini:0.296478	valid-gini:0.284243
# [350]	train-gini:0.301258	valid-gini:0.285575
# [400]	train-gini:0.305827	valid-gini:0.286782
# [450]	train-gini:0.310241	valid-gini:0.287885
# [500]	train-gini:0.313672	valid-gini:0.288415
# [550]	train-gini:0.31689	valid-gini:0.288889
# [600]	train-gini:0.320219	valid-gini:0.289408
# [650]	train-gini:0.322747	valid-gini:0.289832
# [700]	train-gini:0.325932	valid-gini:0.290298
# [750]	train-gini:0.328984	valid-gini:0.290569
# [800]	train-gini:0.331656	valid-gini:0.29101
# [850]	train-gini:0.334356	valid-gini:0.291343
# [900]	train-gini:0.336958	valid-gini:0.291606
# [950]	train-gini:0.339361	valid-gini:0.291371
# [0]	train-gini:0.171693	valid-gini:0.166814
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [50]	train-gini:0.269709	valid-gini:0.270024
# [100]	train-gini:0.280589	valid-gini:0.276234
# [150]	train-gini:0.294328	valid-gini:0.280918
# [200]	train-gini:0.306189	valid-gini:0.283884
# [250]	train-gini:0.316871	valid-gini:0.286329
# [300]	train-gini:0.325312	valid-gini:0.287213
# [350]	train-gini:0.333742	valid-gini:0.288468
# [400]	train-gini:0.340944	valid-gini:0.289095
# [450]	train-gini:0.347357	valid-gini:0.288774
# [500]	train-gini:0.353035	valid-gini:0.289154
# [550]	train-gini:0.358306	valid-gini:0.289405
# [600]	train-gini:0.363562	valid-gini:0.289747
# [650]	train-gini:0.368137	valid-gini:0.289983
# [700]	train-gini:0.373225	valid-gini:0.290168
# [750]	train-gini:0.377744	valid-gini:0.290104
# [800]	train-gini:0.382511	valid-gini:0.289967
# [850]	train-gini:0.38713	valid-gini:0.28973
# Stopping. Best iteration:
# [691]	train-gini:0.372335	valid-gini:0.290245
# 
# [0]	train-gini:0.186317	valid-gini:0.186526
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [50]	train-gini:0.287977	valid-gini:0.274809
# [100]	train-gini:0.304387	valid-gini:0.280157
# [150]	train-gini:0.323393	valid-gini:0.284439
# [200]	train-gini:0.340842	valid-gini:0.287747
# [250]	train-gini:0.356246	valid-gini:0.290444
# [300]	train-gini:0.369436	valid-gini:0.2913
# [350]	train-gini:0.381322	valid-gini:0.291347
# [400]	train-gini:0.390526	valid-gini:0.291576
# [450]	train-gini:0.398911	valid-gini:0.291635
# [500]	train-gini:0.406795	valid-gini:0.291134
# [550]	train-gini:0.414509	valid-gini:0.290781
# [600]	train-gini:0.422203	valid-gini:0.290599
# Stopping. Best iteration:
# [435]	train-gini:0.396286	valid-gini:0.291927
# 
# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [50]	train-gini:0.311761	valid-gini:0.278182
# [100]	train-gini:0.337376	valid-gini:0.283497
# [150]	train-gini:0.365534	valid-gini:0.287479
# [200]	train-gini:0.390756	valid-gini:0.289544
# [250]	train-gini:0.412683	valid-gini:0.292106
# [300]	train-gini:0.429966	valid-gini:0.292974
# [350]	train-gini:0.444596	valid-gini:0.293222
# [400]	train-gini:0.456514	valid-gini:0.292849
# [450]	train-gini:0.469462	valid-gini:0.292862
# [500]	train-gini:0.481765	valid-gini:0.291735
# Stopping. Best iteration:
# [340]	train-gini:0.44194	valid-gini:0.293288
# 
# [0]	train-gini:0.208237	valid-gini:0.200162
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [50]	train-gini:0.346494	valid-gini:0.279076
# [100]	train-gini:0.384323	valid-gini:0.284499
# [150]	train-gini:0.424827	valid-gini:0.28727
# [200]	train-gini:0.460108	valid-gini:0.288491
# [250]	train-gini:0.490432	valid-gini:0.288809
# [300]	train-gini:0.511943	valid-gini:0.288354
# [350]	train-gini:0.527088	valid-gini:0.288391
# [400]	train-gini:0.543902	valid-gini:0.288251
# [450]	train-gini:0.561581	valid-gini:0.288526
# Stopping. Best iteration:
# [266]	train-gini:0.496139	valid-gini:0.289732
# 
# [0]	train-gini:0.218514	valid-gini:0.196553
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [50]	train-gini:0.395258	valid-gini:0.279268
# [100]	train-gini:0.448654	valid-gini:0.285241
# [150]	train-gini:0.50154	valid-gini:0.286381
# [200]	train-gini:0.547344	valid-gini:0.285872
# [250]	train-gini:0.580952	valid-gini:0.286312
# [300]	train-gini:0.605303	valid-gini:0.286118
# Stopping. Best iteration:
# [134]	train-gini:0.487134	valid-gini:0.287129
# 
# max_depth: [3, 4, 5, 6, 7, 8] best_score: [0.291721, 0.290245, 0.291927, 0.293288, 0.289732, 0.287129] best_ntree_limit: [911, 692, 436, 341, 267, 135]

# We can see that **```max_depth``` = 6** is the best choice, valid-gini:0.293288, n_trees_341
# Now, let's tweak ```min_child_weight``` parameter
# 

# In[ ]:


results = {'min_child_w':[],'best_score':[],'best_ntree_limit':[]}
for mcw in range(1,10,1):
    xgb_params = {'eta': 0.03, 
                  'max_depth': 6, 
                  'subsample': 1.0, 
                  'colsample_bytree': 0.3,
                  'min_child_weight': mcw,
                  'objective': 'binary:logistic', 
                  'seed': 99, 
                  'silent': True}

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    #m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=200, early_stopping_rounds=200)
    #results['best_score'].append(m.best_score)
    #results['best_ntree_limit'].append(m.best_ntree_limit)
    #results['min_child_w'].append(mcw)
    
#print('min_child_w:',results['min_child_w'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])


# [0]	train-gini:0.196601	valid-gini:0.19429
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.390756	valid-gini:0.289544
# [400]	train-gini:0.456514	valid-gini:0.292849
# Stopping. Best iteration:
# [340]	train-gini:0.44194	valid-gini:0.293288
# 
# [0]	train-gini:0.197209	valid-gini:0.19521
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.389033	valid-gini:0.290442
# [400]	train-gini:0.455562	valid-gini:0.293512
# Stopping. Best iteration:
# [391]	train-gini:0.452666	valid-gini:0.293653
# 
# [0]	train-gini:0.197141	valid-gini:0.194548
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.38573	valid-gini:0.288662
# [400]	train-gini:0.448931	valid-gini:0.28902
# Stopping. Best iteration:
# [354]	train-gini:0.438576	valid-gini:0.289972
# 
# [0]	train-gini:0.197227	valid-gini:0.194243
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.384145	valid-gini:0.289345
# [400]	train-gini:0.445069	valid-gini:0.29142
# Stopping. Best iteration:
# [321]	train-gini:0.426088	valid-gini:0.29215
# 
# [0]	train-gini:0.197234	valid-gini:0.194579
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.381508	valid-gini:0.290414
# [400]	train-gini:0.442692	valid-gini:0.292785
# Stopping. Best iteration:
# [354]	train-gini:0.430505	valid-gini:0.29324
# 
# [0]	train-gini:0.197748	valid-gini:0.194716
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.378905	valid-gini:0.289523
# [400]	train-gini:0.436356	valid-gini:0.292219
# Stopping. Best iteration:
# [388]	train-gini:0.433689	valid-gini:0.29251
# 
# [0]	train-gini:0.198659	valid-gini:0.199186
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.377616	valid-gini:0.289681
# [400]	train-gini:0.432548	valid-gini:0.292334
# [600]	train-gini:0.476844	valid-gini:0.289811
# Stopping. Best iteration:
# [429]	train-gini:0.438909	valid-gini:0.292701
# 
# [0]	train-gini:0.19869	valid-gini:0.199082
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.375943	valid-gini:0.289621
# [400]	train-gini:0.43302	valid-gini:0.291149
# [600]	train-gini:0.474083	valid-gini:0.289815
# Stopping. Best iteration:
# [406]	train-gini:0.434396	valid-gini:0.291263
# 
# [0]	train-gini:0.198665	valid-gini:0.199127
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.375206	valid-gini:0.28903
# [400]	train-gini:0.431248	valid-gini:0.290529
# Stopping. Best iteration:
# [261]	train-gini:0.395649	valid-gini:0.291537
# 
# min_child_w: [1, 2, 3, 4, 5, 6, 7, 8, 9] best_score: [0.293288, 0.293653, 0.289972, 0.29215, 0.29324, 0.29251, 0.292701, 0.291263, 0.291537] best_ntree_limit: [341, 392, 355, 322, 355, 389, 430, 407, 262]

# So, ```min_child_weight``` = 2 is the best, valid-gini:0.293653, n_tree = 392
# Finally, we can tweak ```colsample_bytree``` parameter
# 

# In[ ]:


results = {'colsample_bytree':[],'best_score':[],'best_ntree_limit':[]}
for cst in [0.3,0.4,0.5]:
    xgb_params = {'eta': 0.03, 
                  'max_depth': 6, 
                  'subsample': 1.0, 
                  'colsample_bytree': cst,
                  'min_child_weight': 2,
                  'objective': 'binary:logistic', 
                  'seed': 99, 
                  'silent': True}

    #watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    #m = xgb.train(xgb_params, d_train, 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=200, early_stopping_rounds=200)
    #results['best_score'].append(m.best_score)
    #results['best_ntree_limit'].append(m.best_ntree_limit)
    #results['colsample_bytree'].append(cst)
    
#print('colsample_bytree:',results['colsample_bytree'],'best_score:',results['best_score'],'best_ntree_limit:', results['best_ntree_limit'])


# [0]	train-gini:0.197209	valid-gini:0.19521
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.389033	valid-gini:0.290442
# [400]	train-gini:0.455562	valid-gini:0.293512
# Stopping. Best iteration:
# [391]	train-gini:0.452666	valid-gini:0.293653
# 
# [0]	train-gini:0.204759	valid-gini:0.189515
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.393239	valid-gini:0.288466
# [400]	train-gini:0.459999	valid-gini:0.289528
# Stopping. Best iteration:
# [248]	train-gini:0.413525	valid-gini:0.290517
# 
# [0]	train-gini:0.21643	valid-gini:0.209827
# Multiple eval metrics have been passed: 'valid-gini' will be used for early stopping.
# 
# Will train until valid-gini hasn't improved in 200 rounds.
# [200]	train-gini:0.399235	valid-gini:0.289937
# [400]	train-gini:0.463524	valid-gini:0.290695
# Stopping. Best iteration:
# [259]	train-gini:0.4212	valid-gini:0.291762
# 
# colsample_bytree: [0.3, 0.4, 0.5] best_score: [0.293653, 0.290517, 0.291762] best_ntree_limit: [392, 249, 260]

# So, ```colsample_bytree``` = 0.3 is the best option with valid-gini = 0.293653 and n_tree = 392

# Now let's train the model on the full data set.

# In[ ]:


training_features, test_features, training_target, test_target, = train_test_split(data_clean.drop(['id','target'], axis = 1),
                                               data_clean['target'],
                                               test_size = .2,
                                               random_state=12)

# Now further split the training test into training and validation to 
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .2,
                                                  random_state=12)


# In[ ]:


#Final model
xgb_params = {'eta': 0.03, 
                  'max_depth': 6, 
                  'subsample': 1.0, 
                  'colsample_bytree': 0.3,
                  'min_child_weight': 2,
                  'objective': 'binary:logistic', 
                  'seed': 99, 
                  'silent': True}
d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_val,y_val)

#watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#model = xgb.train(xgb_params, d_train, 392,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=200, early_stopping_rounds=200)


# Now let's see what features are important

# In[ ]:


#Feature importance
#feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
#feat_imp.plot(kind='bar', title='Feature Importances')
#feat_imp[:60]


# Looks like features with the score under 100 look equally unimportant. 
# Thus, let's only keep those features that have a feature importance score >= 100.

# In[ ]:


#to_keep = feat_imp[feat_imp>=100].index
#df = data_clean[to_keep]
#x_train = df
#y_train = data_clean['target']


# In[ ]:


xgb_params = {'eta': 0.03, 
                  'max_depth': 6, 
                  'subsample': 1.0, 
                  'colsample_bytree': 0.3,
                  'min_child_weight': 2,
                  'objective': 'binary:logistic', 
                  'seed': 99, 
                  'silent': True}
#xgb.DMatrix(x_train[predictors].values, label=y_train.values)
#d_train = xgb.DMatrix(x_train, y_train)
#d_valid = xgb.DMatrix(x_val,y_val)

#watchlist = [(d_train, 'train'), (d_valid, 'valid')]
#model = xgb.train(xgb_params, d_train, 392, feval=gini_xgb, maximize=True, verbose_eval=False)


# ## Prepare Test Set

# In[ ]:


#Download and transform test set
test = pd.read_csv('../input/test.csv', low_memory=False)


# In[ ]:


test.head(5)


# In[ ]:


nas = np.sum(test == -1)/len(test) *100
print("The percentage of missing values is")
print (nas[nas>0].sort_values(ascending = False))


# In[ ]:


#Transformations
test_clean = test.copy()

cat_cols = [c for c in test_clean.columns if c.endswith('cat')]

# Creating dummies for missing values in categorical features
for column in cat_cols:
    temp=pd.get_dummies(test_clean[column], prefix=column, prefix_sep='_')
    test_clean=pd.concat([test_clean,temp],axis=1)
    test_clean=test_clean.drop([column],axis=1)

print('test_clean shape is:',test_clean.shape)

    
# Impute missing values with medians

num_cols = ['ps_reg_03','ps_car_14', 'ps_car_11']

for n in num_cols:
    dummy_name = str(n) + 'NA'
    test_clean[dummy_name] = (test_clean[n]==-1).astype(int)
    med = test_clean[test_clean[n]!=-1][n].median()
    test_clean.loc[test_clean[n]==-1,n] = med
    print(n,np.sum(data_clean[n] == -1)/len(data_clean) *100)


# ### Make predictions

# In[ ]:


#x_test = test_clean[to_keep]
#dtest = xgb.DMatrix(x_test)
#xgb_pred = model.predict(dtest)

#id_test = test_clean['id'].values
#output = pd.DataFrame({'id': id_test, 'target': xgb_pred})

