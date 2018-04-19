# Starter code for multiple regressors implemented by Leandro dos Santos Coelho
# Source code based on Forecasting Favorites, 1owl
# https://www.kaggle.com/the1owl/forecasting-favorites , version 10

# Part II

import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, metrics
import gc; gc.enable()
import random

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import TheilSenRegressor, BayesianRidge

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

import time

np.random.seed(165)

# store the total processing time
start_time = time.time()
tcurrent   = start_time

print('Three regressors - Neural network (MLP), Bayesian Ridge, Bagging(4x) and XGBoost (2x)\n')
print('Datasets reading')


# read datasets
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}
data = {
    'tra': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),
    'tes': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),
    'ite': pd.read_csv('../input/items.csv'),
    'sto': pd.read_csv('../input/stores.csv'),
    'trn': pd.read_csv('../input/transactions.csv', parse_dates=['date']),
    'hol': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),
    'oil': pd.read_csv('../input/oil.csv', parse_dates=['date']),
    }


# dataset processing
print('Datasets processing')

train = data['tra'][(data['tra']['date'].dt.month == 8) & (data['tra']['date'].dt.day > 15)]
del data['tra']; gc.collect();
target = train['unit_sales'].values
target[target < 0.] = 0.
train['unit_sales'] = np.log1p(target)

def df_lbl_enc(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            df[c] = lbl.fit_transform(df[c])
            print(c)
    return df

def df_transform(df):
    df['date'] = pd.to_datetime(df['date'])
    df['yea'] = df['date'].dt.year
    df['mon'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['date'] = df['date'].dt.dayofweek
    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})
    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})
    df = df.fillna(-1)
    return df

data['ite'] = df_lbl_enc(data['ite'])
train = pd.merge(train, data['ite'], how='left', on=['item_nbr'])
test = pd.merge(data['tes'], data['ite'], how='left', on=['item_nbr'])
del data['tes']; gc.collect();
del data['ite']; gc.collect();

train = pd.merge(train, data['trn'], how='left', on=['date','store_nbr'])
test = pd.merge(test, data['trn'], how='left', on=['date','store_nbr'])
del data['trn']; gc.collect();
target = train['transactions'].values
target[target < 0.] = 0.0001
train['transactions'] = np.log1p(target)

data['sto'] = df_lbl_enc(data['sto'])
train = pd.merge(train, data['sto'], how='left', on=['store_nbr'])
test = pd.merge(test, data['sto'], how='left', on=['store_nbr'])
del data['sto']; gc.collect();

data['hol'] = data['hol'][data['hol']['locale'] == 'National'][['date','transferred']]
data['hol']['transferred'] = data['hol']['transferred'].map({'False': 0, 'True': 1})
train = pd.merge(train, data['hol'], how='left', on=['date'])
test = pd.merge(test, data['hol'], how='left', on=['date'])
del data['hol']; gc.collect();

train = pd.merge(train, data['oil'], how='left', on=['date'])
test = pd.merge(test, data['oil'], how='left', on=['date'])
del data['oil']; gc.collect();

train = df_transform(train)
test = df_transform(test)
col = [c for c in train if c not in ['id', 'unit_sales','perishable','transactions']]

x1 = train[(train['yea'] != 2016)]
x2 = train[(train['yea'] == 2016)]
del train; gc.collect();

y1 = x1['transactions'].values
y2 = x2['transactions'].values

def NWRMSLE(y, pred, w):
    return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5


#------------------- forecasting based on multiple regressors (r) models
    
print('\nRunning the basic regressors ...')    

number_regressors_to_test = 5

for method in range(1, number_regressors_to_test+1):
    
    # set the seed to generate random numbers
    ra1 = round(method + 15*method + 564*method) 
    np.random.seed(ra1)
    
    
    print('\nmethod = ', method)
    
    if (method==1):
        print('Multilayer perceptron (MLP) neural network 01')
        str_method = 'MLP model01'    
        r = MLPRegressor(hidden_layer_sizes=(4,), max_iter=40)

    if (method==2):
        print('Multilayer perceptron (MLP) neural network 02')
        str_method = 'MLP model02'    
        r = MLPRegressor(hidden_layer_sizes=(5,), max_iter=30)

    if (method==3):
        print('Bayesian Ridge')
        str_method = 'BayesianRidge'
        r = BayesianRidge(compute_score=True)
        
        
    # class sklearn.ensemble.BaggingRegressor(base_estimator=None, n_estimators=10, 
    #    max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, 
    # oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)        
    
    if (method==4):
        print('Bagging Regressor 01')
        str_method = 'BaggingRegressor01'
        r = BaggingRegressor(DecisionTreeRegressor(max_depth=6,max_features=0.75))        

    if (method==5):
        print('GradientBoosting 01')
        str_method = 'GradientBoosting01'
        r = GradientBoostingRegressor(n_estimators=95, max_depth=6, learning_rate = 0.04, 
                                       random_state=ra2, verbose=0, warm_start=True,
                                       subsample= 0.7, max_features = 0.8)        
                                       
    r.fit(x1[col], y1)


    a1 = NWRMSLE(y2, r.predict(x2[col]), x2['perishable'])
    # part of the output file name
    N1 = str(a1)
    
    test['transactions'] = r.predict(test[col])
    test['transactions'] = test['transactions'].clip(lower=0.+1e-12)

    col = [c for c in x1 if c not in ['id', 'unit_sales','perishable']]
    y1 = x1['unit_sales'].values
    y2 = x2['unit_sales'].values


    # set a new seed to generate random numbers
    ra2 = round(method + 31*method + 51*method) 
    np.random.seed(ra2)

    if (method==1):
        r = MLPRegressor(hidden_layer_sizes=(3,), max_iter=50)
 
    if (method==2):
        r = MLPRegressor(hidden_layer_sizes=(4,), max_iter=30)
 
    if (method==3):    
        r = BayesianRidge(compute_score=True)

    if (method==4):
        r = BaggingRegressor(DecisionTreeRegressor(max_depth=5,max_features=0.85))        
        
    if (method==5):
        r = GradientBoostingRegressor(n_estimators=75, max_depth=6, learning_rate = 0.04, 
                                       random_state=ra2, verbose=0, warm_start=True,
                                       subsample= 0.82, max_features = 0.7)        

    r.fit(x1[col], y1)
    
    a2 = NWRMSLE(y2, r.predict(x2[col]), x2['perishable'])
    # part of the output file name
    N2 = str(a2)

    print('Performance: NWRMSLE(1) = ',a1,'NWRMSLE(2) = ',a2)

    test['unit_sales'] = r.predict(test[col])
    cut = 0.+1e-12 # 0.+1e-15
    
    
    test['unit_sales'] = (np.exp(test['unit_sales']) - 1).clip(lower=cut) # adopted in https://www.kaggle.com/the1owl/forecasting-favorites , version 10
    
    #test['unit_sales'] = (np.expm1(test['unit_sales']) - 1).clip(lower=cut) # if use log1p

    output_file = 'sub 4regr v19 ' + str(str_method) + ' method ' + str(method) + N1 + ' - ' + N2 + '.csv'
 
    test[['id','unit_sales']].to_csv(output_file, index=False, float_format='%.2f')


print( "\nFinished ...")
nm=(time.time() - start_time)/60
print ("Total time %s min" % nm)