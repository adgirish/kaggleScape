
# coding: utf-8

# 
# 
# Version 24: Added Nikunj's features and retuned<br>
# Version 25: Added more Nikunj features and retuned again. <br>
# Version 26: Deleted some of Nikunj features and retuned again.<br>
# Version 27: Remove Niknuj features and go to tuning that was optimal without them, as baseline<br>
# Version 28: Same as version 27 but after having tested some Nikunj features individually<br>
# Version 29: Add 2 best Nikunj features (zip_count, city_count)<br>
# Version 30: Add 3rd feature (GarPoolAC), and some cleanup<br>
# Version 32: Retune: colsample .7 -> .8<br>
# Version 33: Retune: lambda=10, subsample=.55<br>
# Version 34: Revert subsample=.5<br>
# Version 35: Fine tune: lambda=9<br>
# Version 36: Revert: colsample .7<br>
# Version 37: Cleanup<br>
# Version 38: Make boosting rounds and stopping rounds inversely proportional to learning rate<br>
# Version 40: Add city_mean and zip_mean features<br>
# Version 41: Fix comments (Previously mis-stated logerror as "sale price" in feature descriptions)<br>
# Version 42: Fix bug in city_mean definition<br>
# Version 43: Get rid of city_mean<br>
# Version 44: Retune: alpha=0.5<br>
# Version 45: fine tune: lambda=9.5<br>
# Version 46: Roll back to version 39 model, because zip_mean had a data leak, and the corrected version doesn't help<br>
# Version 47: Add additional aggregation features, including by neighborhood<br>
# Verison 48: Put test set features in the correct order<br>
# Version 49: Retune: lambda=5, colsample=.55<br>
# Version 50: Retune: alpha=.65, colsample=.50<br>
# Version 51: Retune: max_depth=7<br>
# Version 52: Make it optional to generate submission file when running full notebook<br>
# Version 53. Option to do validation only<br>
# Version 54. Starting to clean up the code<br>
# Version 55. Option to fit final model to full training set<br>
# Version 56. Optimize fudge factor<br>
# Version 57. Allow change to validation set cutoff date<br>
# Version 59. Try September 15 as validation cutoff<br>
# Version 62. Allow final fit on 2017 (no correction for data leak)<br>
# Version 68. Add seasonal features<br>
#  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Turns out the seasonal features make the fudge factor largely irrelevant,<br>
#  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;but that's partly because I chose the basedate to fit the fudge factors.)<br>
#  Version 71. Make separate predictions for 2017 using 2017 properties data<br>
#  Version 72. Run with FIT_2017_TRAIN_SET = False<br>
#  Version 73. Remove outliers from 2017 data and set FIT_2017_TRAIN_SET = True<br>
#  Version 74. Set FIT_2017_TRAIN_SET = False again<br>
#   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Removing outliers helps, but 2017 data still generate bad 2016 predictions.)<br>
#   Version 76. Allow fitting combined training set<br>

# In[ ]:


MAKE_SUBMISSION = True          # Generate output file.
CV_ONLY = False                 # Do validation only; do not generate predicitons.
FIT_FULL_TRAIN_SET = True       # Fit model to full training set after doing validation.
FIT_2017_TRAIN_SET = False      # Use 2017 training data for full fit (no leak correction)
FIT_COMBINED_TRAIN_SET = True   # Fit combined 2016-2017 training set
USE_SEASONAL_FEATURES = True
VAL_SPLIT_DATE = '2016-09-15'   # Cutoff date for validation split
LEARNING_RATE = 0.007           # shrinkage rate for boosting roudns
ROUNDS_PER_ETA = 20             # maximum number of boosting rounds times learning rate
OPTIMIZE_FUDGE_FACTOR = False   # Optimize factor by which to multiply predictions.
FUDGE_FACTOR_SCALEDOWN = 0.3    # exponent to reduce optimized fudge factor for prediction


# In[ ]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import datetime as dt
from datetime import datetime
import gc
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg


# In[ ]:


properties16 = pd.read_csv('../input/properties_2016.csv', low_memory = False)
properties17 = pd.read_csv('../input/properties_2017.csv', low_memory = False)

# Number of properties in the zip
zip_count = properties16['regionidzip'].value_counts().to_dict()
# Number of properties in the city
city_count = properties16['regionidcity'].value_counts().to_dict()
# Median year of construction by neighborhood
medyear = properties16.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()
# Mean square feet by neighborhood
meanarea = properties16.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()
# Neighborhood latitude and longitude
medlat = properties16.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()
medlong = properties16.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()

train = pd.read_csv("../input/train_2016_v2.csv")
for c in properties16.columns:
    properties16[c]=properties16[c].fillna(-1)
    if properties16[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties16[c].values))
        properties16[c] = lbl.transform(list(properties16[c].values))


# In[ ]:


train_df = train.merge(properties16, how='left', on='parcelid')
select_qtr4 = pd.to_datetime(train_df["transactiondate"]) >= VAL_SPLIT_DATE
if USE_SEASONAL_FEATURES:
    basedate = pd.to_datetime('2015-11-15').toordinal()


# In[ ]:


del train
gc.collect()


# In[ ]:


# Inputs to features that depend on target variable
# (Ideally these should be recalculated, and the dependent features recalculated,
#  when fitting to the full training set.  But I haven't implemented that yet.)

# Standard deviation of target value for properties in the city/zip/neighborhood
citystd = train_df[~select_qtr4].groupby('regionidcity')['logerror'].aggregate("std").to_dict()
zipstd = train_df[~select_qtr4].groupby('regionidzip')['logerror'].aggregate("std").to_dict()
hoodstd = train_df[~select_qtr4].groupby('regionidneighborhood')['logerror'].aggregate("std").to_dict()


# In[ ]:


def calculate_features(df):
    # Nikunj's features
    # Number of properties in the zip
    df['N-zip_count'] = df['regionidzip'].map(zip_count)
    # Number of properties in the city
    df['N-city_count'] = df['regionidcity'].map(city_count)
    # Does property have a garage, pool or hot tub and AC?
    df['N-GarPoolAC'] = ((df['garagecarcnt']>0) &                          (df['pooltypeid10']>0) &                          (df['airconditioningtypeid']!=5))*1 

    # More features
    # Mean square feet of neighborhood properties
    df['mean_area'] = df['regionidneighborhood'].map(meanarea)
    # Median year of construction of neighborhood properties
    df['med_year'] = df['regionidneighborhood'].map(medyear)
    # Neighborhood latitude and longitude
    df['med_lat'] = df['regionidneighborhood'].map(medlat)
    df['med_long'] = df['regionidneighborhood'].map(medlong)

    df['zip_std'] = df['regionidzip'].map(zipstd)
    df['city_std'] = df['regionidcity'].map(citystd)
    df['hood_std'] = df['regionidneighborhood'].map(hoodstd)
    
    if USE_SEASONAL_FEATURES:
        df['cos_season'] = ( (pd.to_datetime(df['transactiondate']).apply(lambda x: x.toordinal()-basedate)) *                              (2*np.pi/365.25) ).apply(np.cos)
        df['sin_season'] = ( (pd.to_datetime(df['transactiondate']).apply(lambda x: x.toordinal()-basedate)) *                              (2*np.pi/365.25) ).apply(np.sin)


# In[ ]:


dropvars = ['airconditioningtypeid', 'buildingclasstypeid',
            'buildingqualitytypeid', 'regionidcity']
droptrain = ['parcelid', 'logerror', 'transactiondate']
droptest = ['ParcelId']


# In[ ]:


calculate_features(train_df)

x_valid = train_df.drop(dropvars+droptrain, axis=1)[select_qtr4]
y_valid = train_df["logerror"].values.astype(np.float32)[select_qtr4]

print('Shape full training set: {}'.format(train_df.shape))
print('Dropped vars: {}'.format(len(dropvars+droptrain)))
print('Shape valid X: {}'.format(x_valid.shape))
print('Shape valid y: {}'.format(y_valid.shape))

train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.419 ]
print('\nFull training set after removing outliers, before dropping vars:')     
print('Shape training set: {}\n'.format(train_df.shape))

if FIT_FULL_TRAIN_SET:
    full_train = train_df.copy()

train_df=train_df[~select_qtr4]
x_train=train_df.drop(dropvars+droptrain, axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)
n_train = x_train.shape[0]
print('Training subset after removing outliers:')     
print('Shape train X: {}'.format(x_train.shape))
print('Shape train y: {}'.format(y_train.shape))

if FIT_FULL_TRAIN_SET:
    x_full = full_train.drop(dropvars+droptrain, axis=1)
    y_full = full_train["logerror"].values.astype(np.float32)
    n_full = x_full.shape[0]
    print('\nFull trainng set:')     
    print('Shape train X: {}'.format(x_train.shape))
    print('Shape train y: {}'.format(y_train.shape))


# In[ ]:


if not CV_ONLY:
    # Generate test set data
    
    sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)
    
    # Process properties for 2016
    test_df = pd.merge( sample_submission[['ParcelId']], 
                        properties16.rename(columns = {'parcelid': 'ParcelId'}), 
                        how = 'left', on = 'ParcelId' )
    if USE_SEASONAL_FEATURES:
        test_df['transactiondate'] = '2016-10-31'
        droptest += ['transactiondate']
    calculate_features(test_df)
    x_test = test_df.drop(dropvars+droptest, axis=1)
    print('Shape test: {}'.format(x_test.shape))

    # Process properties for 2017
    for c in properties17.columns:
        properties17[c]=properties17[c].fillna(-1)
        if properties17[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(properties17[c].values))
            properties17[c] = lbl.transform(list(properties17[c].values))
    zip_count = properties17['regionidzip'].value_counts().to_dict()
    city_count = properties17['regionidcity'].value_counts().to_dict()
    medyear = properties17.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()
    meanarea = properties17.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()
    medlat = properties17.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()
    medlong = properties17.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()

    test_df = pd.merge( sample_submission[['ParcelId']], 
                        properties17.rename(columns = {'parcelid': 'ParcelId'}), 
                        how = 'left', on = 'ParcelId' )
    if USE_SEASONAL_FEATURES:
        test_df['transactiondate'] = '2017-10-31'
    calculate_features(test_df)
    x_test17 = test_df.drop(dropvars+droptest, axis=1)

    del test_df


# In[ ]:


del train_df
del select_qtr4
gc.collect()


# In[ ]:


xgb_params = {  # best as of 2017-09-28 13:20 UTC
    'eta': LEARNING_RATE,
    'max_depth': 7, 
    'subsample': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 5.0,
    'alpha': 0.65,
    'colsample_bytree': 0.5,
    'base_score': y_mean,'taxdelinquencyyear'
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dvalid_x = xgb.DMatrix(x_valid)
dvalid_xy = xgb.DMatrix(x_valid, y_valid)
if not CV_ONLY:
    dtest = xgb.DMatrix(x_test)
    dtest17 = xgb.DMatrix(x_test17)
    del x_test


# In[ ]:


del x_train
gc.collect()


# In[ ]:


num_boost_rounds = round( ROUNDS_PER_ETA / xgb_params['eta'] )
early_stopping_rounds = round( num_boost_rounds / 20 )
print('Boosting rounds: {}'.format(num_boost_rounds))
print('Early stoping rounds: {}'.format(early_stopping_rounds))


# In[ ]:


evals = [(dtrain,'train'),(dvalid_xy,'eval')]
model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds,
                  evals=evals, early_stopping_rounds=early_stopping_rounds, 
                  verbose_eval=10)


# In[ ]:


valid_pred = model.predict(dvalid_x, ntree_limit=model.best_ntree_limit)
print( "XGBoost validation set predictions:" )
print( pd.DataFrame(valid_pred).head() )
print("\nMean absolute validation error:")
mean_absolute_error(y_valid, valid_pred)


# In[ ]:


if OPTIMIZE_FUDGE_FACTOR:
    mod = QuantReg(y_valid, valid_pred)
    res = mod.fit(q=.5)
    print("\nLAD Fit for Fudge Factor:")
    print(res.summary())

    fudge = res.params[0]
    print("Optimized fudge factor:", fudge)
    print("\nMean absolute validation error with optimized fudge factor: ")
    print(mean_absolute_error(y_valid, fudge*valid_pred))

    fudge **= FUDGE_FACTOR_SCALEDOWN
    print("Scaled down fudge factor:", fudge)
    print("\nMean absolute validation error with scaled down fudge factor: ")
    print(mean_absolute_error(y_valid, fudge*valid_pred))
else:
    fudge=1.0


# In[ ]:


if FIT_FULL_TRAIN_SET and not CV_ONLY:
    if FIT_COMBINED_TRAIN_SET:
        # Merge 2016 and 2017 data sets
        train16 = pd.read_csv('../input/train_2016_v2.csv')
        train17 = pd.read_csv('../input/train_2017.csv')
        train16 = pd.merge(train16, properties16, how = 'left', on = 'parcelid')
        train17 = pd.merge(train17, properties17, how = 'left', on = 'parcelid')
        train17[['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']] = np.nan
        train_df = pd.concat([train16, train17], axis = 0)
        # Generate features
        citystd = train_df.groupby('regionidcity')['logerror'].aggregate("std").to_dict()
        zipstd = train_df.groupby('regionidzip')['logerror'].aggregate("std").to_dict()
        hoodstd = train_df.groupby('regionidneighborhood')['logerror'].aggregate("std").to_dict()
        calculate_features(train_df)
        # Remove outliers
        train_df=train_df[ train_df.logerror > -0.4 ]
        train_df=train_df[ train_df.logerror < 0.419 ]
        # Create final training data sets
        x_full = train_df.drop(dropvars+droptrain, axis=1)
        y_full = train_df["logerror"].values.astype(np.float32)
        n_full = x_full.shape[0]     
    elif FIT_2017_TRAIN_SET:
        train = pd.read_csv('../input/train_2017.csv')
        train_df = train.merge(properties17, how='left', on='parcelid')
        # Generate features
        citystd = train_df.groupby('regionidcity')['logerror'].aggregate("std").to_dict()
        zipstd = train_df.groupby('regionidzip')['logerror'].aggregate("std").to_dict()
        hoodstd = train_df.groupby('regionidneighborhood')['logerror'].aggregate("std").to_dict()
        calculate_features(train_df)
        # Remove outliers
        train_df=train_df[ train_df.logerror > -0.4 ]
        train_df=train_df[ train_df.logerror < 0.419 ]
        # Create final training data sets
        x_full = train_df.drop(dropvars+droptrain, axis=1)
        y_full = train_df["logerror"].values.astype(np.float32)
        n_full = x_full.shape[0]     
    dtrain = xgb.DMatrix(x_full, y_full)
    num_boost_rounds = int(model.best_ntree_limit*n_full/n_train)
    full_model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds, 
                           evals=[(dtrain,'train')], verbose_eval=10)


# In[ ]:


del properties16
del properties17
gc.collect()


# In[ ]:


if not CV_ONLY:
    if FIT_FULL_TRAIN_SET:
        pred = fudge*full_model.predict(dtest)
        pred17 = fudge*full_model.predict(dtest17)
    else:
        pred = fudge*model.predict(dtest, ntree_limit=model.best_ntree_limit)
        pred17 = fudge*model.predict(dtest17, ntree_limit=model.best_ntree_limit)
        
    print( "XGBoost test set predictions for 2016:" )
    print( pd.DataFrame(pred).head() )
    print( "XGBoost test set predictions for 2017:" )
    print( pd.DataFrame(pred17).head() )    


# In[ ]:


if MAKE_SUBMISSION and not CV_ONLY:
   y_pred=[]
   y_pred17=[]

   for i,predict in enumerate(pred):
       y_pred.append(str(round(predict,4)))
   for i,predict in enumerate(pred17):
       y_pred17.append(str(round(predict,4)))
   y_pred=np.array(y_pred)
   y_pred17=np.array(y_pred17)

   output = pd.DataFrame({'ParcelId': sample_submission['ParcelId'].astype(np.int32),
           '201610': y_pred, '201611': y_pred, '201612': y_pred,
           '201710': y_pred17, '201711': y_pred17, '201712': y_pred17})
   # set col 'ParceID' to first col
   cols = output.columns.tolist()
   cols = cols[-1:] + cols[:-1]
   output = output[cols]

   output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)


# In[ ]:


print("Mean absolute validation error without fudge factor: ", )
print( mean_absolute_error(y_valid, valid_pred) )
if OPTIMIZE_FUDGE_FACTOR:
    print("Mean absolute validation error with fudge factor:")
    print( mean_absolute_error(y_valid, fudge*valid_pred) )

