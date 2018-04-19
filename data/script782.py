
# coding: utf-8

# *Mei-Cheng Shih, 2016*

# This kernel is inspired by the post of **JMT5802**.  The aim of this kernel is to use XGBoost to replace RF which was used as the core of the Boruta package.  Since XGBoost generates better quality predictions than RF in this casw, the output of this kernel is expected to be mor representative.   Moreover, the code also includes the data cleaning process I used to build my model

# **First, import packages for data cleaning and read the data**

# In[ ]:


from scipy.stats.mstats import mode
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder

"""
Read Data
"""
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
target=train['SalePrice']
train=train.drop(['SalePrice'],axis=1)
trainlen=train.shape[0]


# **Combined the train and test set for cleaning**

# In[ ]:


alldata=pd.concat([train, test], axis=0, join='outer',ignore_index=True)
alldata=alldata.drop(['Id','Utilities'],axis=1)
alldata.ix[:,(alldata.dtypes=='int64') & (alldata.columns!='MSSubClass')]=alldata.ix[:,(alldata.dtypes=='int64') & (alldata.columns!='MSSubClass')].astype('float64')


# **Dealing with the NA values in the variables, some of them equal to 0 and some equal to median, based on the txt descriptions**

# In[ ]:


fMedlist=['LotFrontage']
fArealist=['MasVnrArea','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath', 'BsmtHalfBath','MasVnrArea','Fireplaces','GarageArea','GarageYrBlt','GarageCars']

for i in fArealist:
    alldata.ix[pd.isnull(alldata.ix[:,i]),i]=0
        
for i in fMedlist:
   alldata.ix[pd.isnull(alldata.ix[:,i]),i]=np.nanmedian(alldata.ix[:,i])    
       
### Transforming Data
le=LabelEncoder()
nacount_category=np.array(alldata.columns[((alldata.dtypes=='int64') | (alldata.dtypes=='object')) & (pd.isnull(alldata).sum()>0)])
category=np.array(alldata.columns[((alldata.dtypes=='int64') | (alldata.dtypes=='object'))])
Bsmtset=set(['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'])
MasVnrset=set(['MasVnrType'])
Garageset=set(['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond'])
Fireplaceset=set(['FireplaceQu'])
Poolset=set(['PoolQC'])
NAset=set(['Fence','MiscFeature','Alley'])

for i in nacount_category:
    if i in Bsmtset:
        alldata.ix[pd.isnull(alldata.ix[:,i]) & (alldata['TotalBsmtSF']==0),i]='Empty'
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]=alldata.ix[:,i].value_counts().index[0]
    elif i in MasVnrset:
        alldata.ix[pd.isnull(alldata.ix[:,i]) & (alldata['MasVnrArea']==0),i]='Empty'
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]=alldata.ix[:,i].value_counts().index[0]
    elif i in Garageset:
        alldata.ix[pd.isnull(alldata.ix[:,i]) & (alldata['GarageArea']==0),i]='Empty'
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]=alldata.ix[:,i].value_counts().index[0]
    elif i in Fireplaceset:
        alldata.ix[pd.isnull(alldata.ix[:,i]) & (alldata['Fireplaces']==0),i]='Empty'
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]=alldata.ix[:,i].value_counts().index[0]
    elif i in Poolset:
        alldata.ix[pd.isnull(alldata.ix[:,i]) & (alldata['PoolArea']==0),i]='Empty'
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]=alldata.ix[:,i].value_counts().index[0]
    elif i in NAset:
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]='Empty'
    else:
        alldata.ix[pd.isnull(alldata.ix[:,i]),i]=alldata.ix[:,i].value_counts().index[0]

for i in category:
    alldata.ix[:,i]=le.fit_transform(alldata.ix[:,i])

train=alldata.ix[0:trainlen-1,:]
test=alldata.ix[trainlen:alldata.shape[0],:]


# **Import required packages for Feature Selection Process**

# In[ ]:


import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


# **Start the code, drop some outliers.  The outliers were detected by package statsmodel in python, skip details here**

# In[ ]:


o=[30,462,523,632,968,970, 1298, 1324]

train=train.drop(o,axis=0)
target=target.drop(o,axis=0)

train.index=range(0,train.shape[0])
target.index=range(0,train.shape[0])


# **Set XGB model, the parameters were obtained from CV based on a Bayesian Optimization Process**   

# In[ ]:


est=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0.045,                 
                 learning_rate=0.07,
                 max_depth=20,
                 min_child_weight=1.5,
                 n_estimators=300,                                                                    
                 reg_alpha=0.65,
                 reg_lambda=0.45,
                 subsample=0.95)


# **Start the test process, the basic idea is to permutate the order of elements in each of the columns randomly and see the impact of the permutation**

# **For the evaluation metric of feature importance, I used ((MSE of pertutaed data)-(MSE of original data))/(MSE of original data)**

# In[ ]:


n=200

scores=pd.DataFrame(np.zeros([n,train.shape[1]]))
scores.columns=train.columns
ct=0

for train_idx, test_idx in ShuffleSplit(train.shape[0], n, .25):
    ct+=1
    X_train, X_test = train.ix[train_idx,:], train.ix[test_idx,:]
    Y_train, Y_test = target.ix[train_idx], target.ix[test_idx]
    r = est.fit(X_train, Y_train)
    acc = mean_squared_error(Y_test, est.predict(X_test))
    for i in range(train.shape[1]):
        X_t = X_test.copy()
        X_t.ix[:,i]=shuffle(np.array(X_t.ix[:, i]))
        shuff_acc =  mean_squared_error(Y_test, est.predict(X_t))
        scores.ix[ct-1,i]=((acc-shuff_acc)/acc)


# Generate output, the mean, median, max and min of the scores fluctuation 

# In[ ]:


fin_score=pd.DataFrame(np.zeros([train.shape[1],4]))
fin_score.columns=['Mean','Median','Max','Min']
fin_score.index=train.columns
fin_score.ix[:,0]=scores.mean()
fin_score.ix[:,1]=scores.median()
fin_score.ix[:,2]=scores.min()
fin_score.ix[:,3]=scores.max()


# See the importances of features.  The higher the value, the less important the factor.

# In[ ]:


pd.set_option('display.max_rows', None)
fin_score.sort_values('Mean',axis=0)


# **The result is a little bit difference from what JMT5802 got, but in general they are similar.  For example, OverallQual, GrLivArea are important in both cases, and PoolArea and PoolQC are not important in both cases.  Also, based on the test conducted in link below, it is reasonable to say the differences are not obvious in both cases**

# Also, the main code was modified from the example in the link below, special thanks to the author of the blog

# http://blog.datadive.net/selecting-good-features-part-iii-random-forests/

# **Updates:**
# 
# After several tests, I removed the variables in the list below, and this action did improve my score a little bit.
# ['Exterior2nd', 'EnclosedPorch', 'RoofMatl', 'PoolQC', 'BsmtHalfBath',
#   'RoofStyle', 'PoolArea', 'MoSold', 'Alley', 'Fence', 'LandContour',
#   'MasVnrType', '3SsnPorch', 'LandSlope']
# 
