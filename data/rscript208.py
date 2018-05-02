"""
Genetic programming using gplearn

Genetic Programming in Python, with a scikit-learn inspired API
Details in: https://pypi.python.org/pypi/gplearn/

Based on https://www.kaggle.com/the1owl/surprise-me
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from sklearn.ensemble import ExtraTreesRegressor

#See example in http://nbviewer.jupyter.org/github/trevorstephens/gplearn/blob/master/doc/gp_examples.ipynb
from gplearn.genetic import SymbolicRegressor

import time
start_time = time.time()
tcurrent   = start_time

np.random.seed(429)   

data = {
    'tra':   pd.read_csv('../input/air_visit_data.csv'),
    'as':    pd.read_csv('../input/air_store_info.csv'),
    'hs':    pd.read_csv('../input/hpg_store_info.csv'),
    'ar':    pd.read_csv('../input/air_reserve.csv'),
    'hr':    pd.read_csv('../input/hpg_reserve.csv'),
    'id':    pd.read_csv('../input/store_id_relation.csv'),
    'tes':   pd.read_csv('../input/sample_submission.csv'),
    'hol':   pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date' })
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime']   = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime']   = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
        
    #--- begin  new features    
    data[df]['reserve_datetime_diff_2'] = data[df].apply(
        lambda r: ( (r['visit_datetime'] - r['reserve_datetime']).days)**2.1, axis=1)
    data[df]['reserve_datetime_diff_3'] = data[df].apply(
        lambda r: ( (r['visit_datetime'] - r['reserve_datetime']).days)**3.2, axis=1)
    #--- end new features        
        
    data[df] = data[df].groupby(
        ['air_store_id', 'visit_datetime'], as_index=False)[[
            'reserve_datetime_diff', 'reserve_visitors'
        ]].sum().rename(columns={
            'visit_datetime': 'visit_date'
        })
        
    show_data = 0    
    if (show_data==1):
        print(data[df].head())

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow']        = data['tra']['visit_date'].dt.dayofweek
data['tra']['year']       = data['tra']['visit_date'].dt.year
data['tra']['month']      = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(
    lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(
    lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat(
    [
        pd.DataFrame({
            'air_store_id': unique_stores,
            'dow': [i] * len(unique_stores)
        }) for i in range(7)
    ],
    axis=0,
    ignore_index=True).reset_index(drop=True)

#sure it can be compressed...
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].min().rename(columns={
        'visitors': 'min_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].mean().rename(columns={
        'visitors': 'mean_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].median().rename(columns={
        'visitors': 'median_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].max().rename(columns={
        'visitors': 'max_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].count().rename(columns={
        'visitors': 'count_observations'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])


stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])

for df in ['ar', 'hr']:
    train = pd.merge(
        train, data[df], how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(
        test, data[df], how='left', on=['air_store_id', 'visit_date'])

col = [
    c for c in train
    if c not in ['id', 'air_store_id', 'visit_date', 'visitors']
]
train = train.fillna(-1)
test = test.fillna(-1)

# XGB starter template borrowed from @anokas: https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)

for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32)

train_x = train.drop(['air_store_id', 'visit_date', 'visitors'], axis=1)
train_y = np.log1p(train['visitors'].values)

if (show_data==1):
    print(train_x.shape, train_y.shape)
    
test_x  = test.drop(['id', 'air_store_id', 'visit_date', 'visitors'], axis=1)

print('Columns in the train_x dataset')

print(train_x.columns.values)
#------------------------------------------------ starting regressors

regressor = 10

print('\n\nAdopted regressor = ', regressor,'\n')

if (regressor == 1):
    print('Starting XGBoost')
    boost_params = {'eval_metric': 'rmse'}
    xgb0 = xgb.XGBRegressor(
        max_depth        = 8,
        learning_rate    = 0.01,
        n_estimators     = 10000,
        objective        = 'reg:linear',
        gamma            = 0,
        min_child_weight = 1,
        subsample        = 1,
        colsample_bytree = 1,
        scale_pos_weight = 1,
        seed             = 27,
        **boost_params)
        
    xgb0.fit(train_x, train_y)
    predict_y = xgb0.predict(test_x)
    print('Finished XGBoost')
    
    
if (regressor == 2):    
    print('Starting Extra trees')
    et = ExtraTreesRegressor (n_estimators         = 10000, 
                                 max_depth         = 8, 
                                 n_jobs            = -1, 
                                 random_state      = 11, 
                                 verbose           = 0, 
                                 warm_start        = True,
                                 min_samples_leaf  = 120, 
                                 max_features      = 0.8)    
    et.fit(train_x, train_y)
    predict_y = et.predict(test_x)
    print('Finished Extra trees')
    

if (regressor == 3):    
    print('Starting Genetic Programming')
    
    '''
    http://gplearn.readthedocs.io/en/stable/reference.html
    
    The sum of p_crossover, p_subtree_mutation, p_hoist_mutation and p_point_mutation 
    should total to 1.0 or less.
    '''
    
    gp = SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log','sqrt'),
                           population_size       = 100, 
                           const_range           = (-10, 100),
                           generations           = 100, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.5, 
                           p_subtree_mutation    = 0.25,
                           p_hoist_mutation      = 0.05, 
                           p_point_mutation      = 0.20, 
                           init_depth            = (6, 12),
                           max_samples           = 0.7, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.0001, 
                           random_state          = 1121)  
                           # seed is relevant to stochastic approaches such as genetic programming
                           
if (regressor == 4):    
    print('Starting Genetic Programming')
    
    '''
    http://gplearn.readthedocs.io/en/stable/reference.html
    
    The sum of p_crossover, p_subtree_mutation, p_hoist_mutation and p_point_mutation 
    should total to 1.0 or less.
    '''
    
    gp = SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log','sqrt'),
                           population_size       = 100, 
                           const_range           = (-10, 100),
                           generations           = 400, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.55, 
                           p_subtree_mutation    = 0.20,
                           p_hoist_mutation      = 0.05, 
                           p_point_mutation      = 0.20, 
                           init_depth            = (6, 12),
                           max_samples           = 0.9, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.0001, 
                           random_state          = 343)  
                           # seed is relevant to stochastic approaches such as genetic programming


if (regressor == 5):    
    print('Starting Genetic Programming')
    
    '''
    http://gplearn.readthedocs.io/en/stable/reference.html
    
    The sum of p_crossover, p_subtree_mutation, p_hoist_mutation and p_point_mutation 
    should total to 1.0 or less.
    '''
    
    gp = SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log','sqrt'),
                           population_size       = 120, 
                           const_range           = (-10, 110),
                           generations           = 500, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.55, 
                           p_subtree_mutation    = 0.20,
                           p_hoist_mutation      = 0.05, 
                           p_point_mutation      = 0.20, 
                           init_depth            = (6, 12),
                           max_samples           = 0.85, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.0001, 
                           random_state          = 888)  
                           # seed is relevant to stochastic approaches such as genetic programming

if (regressor == 6):    
    print('Starting Genetic Programming')

    gp = SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log','sqrt'),
                           population_size       = 200, 
                           const_range           = (-15, 130),
                           generations           = 500, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.50, 
                           p_subtree_mutation    = 0.20,
                           p_hoist_mutation      = 0.10, 
                           p_point_mutation      = 0.20, 
                           init_depth            = (6, 12),
                           max_samples           = 0.75, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.0001, 
                           random_state          = 567)  

if (regressor == 7):    
    print('Starting Genetic Programming')

    gp = SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log'),
                           population_size       = 200, 
                           const_range           = (-20, 140),
                           generations           = 700, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.50, 
                           p_subtree_mutation    = 0.15,
                           p_hoist_mutation      = 0.13, 
                           p_point_mutation      = 0.22, 
                           init_depth            = (6, 11),
                           max_samples           = 0.8, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.00015, 
                           random_state          = 433)  

if (regressor == 8):    
    print('Starting Genetic Programming')

    gp = SymbolicRegressor(function_set=('sqrt','add', 'sub', 'mul', 'div','max','min','log'),
                           population_size       = 150, 
                           const_range           = (-30, 280),
                           generations           = 600, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.45, 
                           p_subtree_mutation    = 0.15,
                           p_hoist_mutation      = 0.15, 
                           p_point_mutation      = 0.25, 
                           init_depth            = (6, 13),
                           max_samples           = 0.5, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.0002, 
                           random_state          = 12)  
                           
if (regressor == 9):    
    print('Starting Genetic Programming')

    gp = SymbolicRegressor(function_set=('sqrt','add', 'sub', 'mul', 'div','max','min','log'),
                           population_size       = 180, 
                           const_range           = (-40, 400),
                           generations           = 700, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.50, 
                           p_subtree_mutation    = 0.15,
                           p_hoist_mutation      = 0.10, 
                           p_point_mutation      = 0.25, 
                           init_depth            = (6, 13),
                           max_samples           = 0.55, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.005, 
                           random_state          = 1542)  
  
if (regressor == 10):    
    print('Starting Genetic Programming')

    gp = SymbolicRegressor(function_set=('sqrt','add', 'sub', 'mul', 'div','max','min','log'),
                           population_size       = 150, 
                           const_range           = (-10, 210),
                           generations           = 700, 
                           stopping_criteria     = 0.001,
                           p_crossover           = 0.45, 
                           p_subtree_mutation    = 0.15,
                           p_hoist_mutation      = 0.15, 
                           p_point_mutation      = 0.25, 
                           init_depth            = (6, 13),
                           max_samples           = 0.4, 
                           verbose               = 1, 
                           n_jobs                = -1, 
                           metric                = 'rmse',
                           parsimony_coefficient = 0.0005, 
                           random_state          = 1234)  
                           
if (regressor >= 3):                           
    gp.fit(train_x, train_y)
    predict_y = gp.predict(test_x)  
    predict_y[predict_y < 0] = 0        # only positive values
    
    print ('\nDetails about the results using Genetic Programming\n')  
    print (gp._program)
    print ('R2(max)    = ',gp.score(train_x, train_y))  

    # summary of the results
    print('Raw fitness = ',gp._program.raw_fitness_)    
    #print('Fitness     = ',gp._program.fitness_)    
    print('OOB fitness = ',gp._program.oob_fitness_)    
    print('Depth       = ',gp._program.depth_)    
    print('Length      = ',gp._program.length_,'\n')    

    '''
    Comments:
    raw_fitness_ : The raw fitness of the individual program.
    fitness_     : The penalized fitness of the individual program.
    oob_fitness_ : The out-of-bag raw fitness of the individual program for the held-out samples. 
                     Only present when sub-sampling was used in the estimator by 
                     specifying max_samples < 1.0.
    depth_       : The maximum depth of the program tree.
    length_      : The number of functions and terminals in the program.
    '''
    print('Finished Genetic Programming')


#------------------------------------------------ end regressors

test['visitors'] = np.expm1(predict_y)

fname = 'submissionr v01 regressor ' + str(regressor) + '.csv'

test[['id', 'visitors']].to_csv(fname, index=False, float_format='%.3f')  

nm=(time.time() - start_time)/60
print ("Total processing time %s min" % nm)