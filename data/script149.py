
# coding: utf-8

# Updates:
# 
# The stacking model:
# https://www.kaggle.com/schoolpal/nn-stacking-magic-no-magic-30409-private-31063
# 
# https://www.kaggle.com/schoolpal/modifications-to-reynaldo-s-script/notebook
# I just added two XGB models, they were used together with in this one and one more DNN model in the stacking
# 
# --------------------------------------------------------
# 
# This is a simple LightGBM script. It got two magic number, one took from Andy's script (proposed by Louis?). The second one actually down scale only the "old" investment properties, as the new ones are supported by the mortgage subsidy program? The LB score is at 0.3094. 
# 
# 
# One nice thing is that the classical BoxCox transformation can further improve the performance to 0.3093. It can also be verified by local skewness.  I wonder why no one bring this up in the kernel/forum.
# 
# This script (log version) serves as one of the basis model for the later stacking.

# In[ ]:


from sklearn.model_selection import train_test_split,KFold,TimeSeriesSplit
from sklearn import model_selection, preprocessing
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn import model_selection, preprocessing
import pdb

def process(train,test):
    RS=1
    np.random.seed(RS)
    ROUNDS = 1500 # 1300,1400 all works fine
    params = {
        'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'learning_rate': 0.01 , #small learn rate, large number of iterations
            'verbose': 0,
            'num_leaves': 2 ** 5,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': RS,
            'feature_fraction': 0.7,
            'feature_fraction_seed': RS,
            'max_bin': 100,
            'max_depth': 7,
            'num_rounds': ROUNDS,
        }
    #Remove the bad prices as suggested by Radar
    train=train[(train.price_doc>1e6) & (train.price_doc!=2e6) & (train.price_doc!=3e6)]
    train.loc[(train.product_type=='Investment') & (train.build_year<2000),'price_doc']*=0.9 
    train.loc[train.product_type!='Investment','price_doc']*=0.969 #Louis/Andy's magic number
    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

  
    id_test = test.id
    times=pd.concat([train.timestamp,test.timestamp])
    num_train=train.shape[0]
    y_train = train["price_doc"]
    train.drop(['price_doc'],inplace=True,axis=1)
    da=pd.concat([train,test])
    da['na_count']=da.isnull().sum(axis=1)
    df_cat=None
    to_remove=[]
    for c in da.columns:
        if da[c].dtype=='object':
            oh=pd.get_dummies(da[c],prefix=c)
            if df_cat is None:
                df_cat=oh
            else:
                df_cat=pd.concat([df_cat,oh],axis=1)
            to_remove.append(c)
    da.drop(to_remove,inplace=True,axis=1)

    #Remove rare features,prevent overfitting
    to_remove=[]
    if df_cat is not None:
        sums=df_cat.sum(axis=0)
        to_remove=sums[sums<200].index.values
        df_cat=df_cat.loc[:,df_cat.columns.difference(to_remove)]
        da = pd.concat([da, df_cat], axis=1)
    x_train=da[:num_train].drop(['timestamp','id'],axis=1)
    x_test=da[num_train:].drop(['timestamp','id'],axis=1)
    #Log transformation, boxcox works better.
    y_train=np.log(y_train)
    train_lgb=lgb.Dataset(x_train,y_train)
    model=lgb.train(params,train_lgb,num_boost_round=ROUNDS)
    predict=model.predict(x_test)
    predict=np.exp(predict)
    return predict,id_test
if __name__=='__main__':
    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])
    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])
    predict,id_test=process(train,test)
    output=pd.DataFrame({'id':id_test,'price_doc':predict})
    output.to_csv('lgb.csv',index=False)

