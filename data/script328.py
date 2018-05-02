
# coding: utf-8

# ## Sklearn Basic Neural Network (0.583 LB)

# In the following network I build a NN based on the Sklearn interface. This is a brief summary of the steps I follow to build it:
# 
# - Load data
# - Generate location features
# - Feature engineering ('basic_preprocess' function)
# - Normalize features
# - GridSearch on NN parametes to find the optimum
# - Generate predictions on the test dataset
# 
# The generated predictions got a 0.584 log-loss (LB).
# 
# Some of the feature engineering here is based on two previous notebooks:
# 
# - [Unsupervised and supervised neighborhood encoding](https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/unsupervised-and-supervised-neighborhood-encoding)
# - [Price/Bedrooms/Bathrooms](https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths)
# 
# 
# *Any feedback or comment will be appreciated! Upvote if you found it interesting/useful :)
# Thanks!*

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import time as time
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import make_pipeline

def get_skf_indexes(df, target, kfold=4):
    X = df.values
    y = df[target].values
    skf = StratifiedKFold(n_splits=4);
    skf.get_n_splits(X, y);
    indexes = [[],[]]
    for train_index, test_index in skf.split(X, y):
        indexes[0].append(train_index)
        indexes[1].append(test_index)
    return indexes


def output_results(clf, x_test, listing, fname):
    preds = clf.predict_proba(x_test)
    preds = pd.DataFrame(preds)
    cols = ['low', 'medium', 'high']
    preds.columns = cols
    preds['listing_id'] = listing
    preds.to_csv(fname, index=None)
    print(preds[cols].mean().values)


def basic_preprocess(df_train, df_test, n_min=50, precision=3):
    
    # Interest: Numerical encoding of interest level
    df_train['y'] = 0.0
    df_train.loc[df_train.interest_level=='medium', 'y'] = 1.0
    df_train.loc[df_train.interest_level=='high', 'y'] = 2.0
    
    # Location features: Latitude, longitude
    df_train['num_latitude'] = df_train.latitude.values
    df_test['num_latitude'] = df_test.latitude.values
    df_train['num_longitude'] = df_train.longitude.values
    df_test['num_longitude'] = df_test.longitude.values
    x = np.sqrt(((df_train.latitude - df_train.latitude.median())**2) + (df_train.longitude - df_train.longitude.median())**2)
    df_train['num_dist_from_center'] = x.values
    x = np.sqrt(((df_test.latitude - df_train.latitude.median())**2) + (df_test.longitude - df_train.longitude.median())**2)
    df_test['num_dist_from_center'] = x.values
    df_train['pos'] = df_train.longitude.round(precision).astype(str) + '_' + df_train.latitude.round(precision).astype(str)
    df_test['pos'] = df_test.longitude.round(precision).astype(str) + '_' + df_test.latitude.round(precision).astype(str)
    
    # Degree of "outlierness"
    OutlierAggregated = (df_train.bedrooms > 4).astype(float)
    OutlierAggregated2 = (df_test.bedrooms > 4).astype(float)
    OutlierAggregated += (df_train.bathrooms > 3).astype(float)
    OutlierAggregated2 += (df_test.bathrooms > 3).astype(float)
    OutlierAggregated += (df_train.bathrooms < 1).astype(float)
    OutlierAggregated2 += (df_test.bathrooms < 1).astype(float)
    x = np.abs((df_train.price - df_train.price.median())/df_train.price.std()) > 0.30
    OutlierAggregated += x.astype(float)
    x2 = np.abs((df_test.price - df_train.price.median())/df_train.price.std()) > 0.30
    OutlierAggregated2 += x2.astype(float)
    x = np.log1p(df_train.price/(df_train.bedrooms.clip(1,3) + df_train.bathrooms.clip(1,2))) > 8.2
    OutlierAggregated += x.astype(float)
    x2 = np.log1p(df_test.price/(df_test.bedrooms.clip(1,3) + df_test.bathrooms.clip(1,2))) > 8.2
    OutlierAggregated2 += x2.astype(float)
    x = np.sqrt(((df_train.latitude - df_train.latitude.median())**2) + (df_train.longitude - df_train.longitude.median())**2) > 0.30
    OutlierAggregated += x.astype(float)
    x2 = np.sqrt(((df_test.latitude - df_train.latitude.median())**2) + (df_test.longitude - df_train.longitude.median())**2) > 0.30
    OutlierAggregated2 += x2.astype(float)
    df_train['num_OutlierAggregated'] = OutlierAggregated.values
    df_test['num_OutlierAggregated'] = OutlierAggregated2.values
    
    # Average interest in unique locations at given precision
    x = df_train.groupby('pos')['y'].aggregate(['count', 'mean'])
    d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
    impute = df_train.y.mean()
    df_train['num_pos'] = df_train.pos.apply(lambda x: d.get(x, impute))
    df_test['num_pos'] = df_test.pos.apply(lambda x: d.get(x, impute))
    
    # Density in unique locations at given precision
    vals = df_train['pos'].value_counts()
    dvals = vals.to_dict()
    df_train['num_pos_density'] = df_train['pos'].apply(lambda x: dvals.get(x, vals.min()))
    df_test['num_pos_density'] = df_test['pos'].apply(lambda x: dvals.get(x, vals.min()))

    # Building null
    df_train['num_building_null'] = (df_train.building_id=='0').astype(float)
    df_test['num_building_null'] = (df_test.building_id=='0').astype(float)
    
    # Building supervised
    x = df_train.groupby('building_id')['y'].aggregate(['count', 'mean'])
    d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
    impute = df_train.y.mean()
    df_train['num_building_id'] = df_train.building_id.apply(lambda x: d.get(x, impute))
    df_test['num_building_id'] = df_test.building_id.apply(lambda x: d.get(x, impute))
    
    # Building frequency
    d = np.log1p(df_train.building_id.value_counts()).to_dict()
    impute = np.min(np.array(list(d.values())))
    df_train['num_fbuilding'] = df_train.building_id.apply(lambda x: d.get(x, impute))
    df_test['num_fbuilding'] = df_test.building_id.apply(lambda x: d.get(x, impute))
    
    # Manager supervised
    x = df_train.groupby('manager_id')['y'].aggregate(['count', 'mean'])
    d = x.loc[x['count'] >= n_min, 'mean'].to_dict()
    impute = df_train.y.mean()
    df_train['num_manager'] = df_train.manager_id.apply(lambda x: d.get(x, impute))
    df_test['num_manager'] = df_test.manager_id.apply(lambda x: d.get(x, impute))

    # Manager frequency
    d = np.log1p(df_train.manager_id.value_counts()).to_dict()
    impute = np.min(np.array(list(d.values())))
    df_train['num_fmanager'] = df_train.manager_id.apply(lambda x: d.get(x, impute))
    df_test['num_fmanager'] = df_test.manager_id.apply(lambda x: d.get(x, impute))
    
    # Creation time features
    df_train['created'] = pd.to_datetime(df_train.created)
    df_train['num_created_weekday'] = df_train.created.dt.dayofweek.astype(float)
    df_train['num_created_weekofyear'] = df_train.created.dt.weekofyear
    df_test['created'] = pd.to_datetime(df_test.created)
    df_test['num_created_weekday'] = df_test.created.dt.dayofweek
    df_test['num_created_weekofyear'] = df_test.created.dt.weekofyear
    
    # Bedrooms/Bathrooms/Price
    df_train['num_bathrooms'] = df_train.bathrooms.clip_upper(4)
    df_test['num_bathrooms'] = df_test.bathrooms.clip_upper(4)
    df_train['num_bedrooms'] = df_train.bedrooms.clip_upper(5)
    df_test['num_bedrooms'] = df_test.bedrooms.clip_upper(5)
    df_train['num_price'] = df_train.price.clip_upper(10000)
    df_test['num_price'] = df_test.price.clip_upper(10000)
    bins = df_train.price.quantile(np.arange(0.05, 1, 0.05))
    df_train['num_price_q'] = np.digitize(df_train.price, bins)
    df_test['num_price_q'] = np.digitize(df_test.price, bins)
    
    # Composite features based on: 
    # https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths
    df_train['num_priceXroom'] = (df_train.price / (1 + df_train.bedrooms.clip(1, 4) + 0.5*df_train.bathrooms.clip(0, 2))).values
    df_test['num_priceXroom'] = (df_test.price / (1 + df_test.bedrooms.clip(1, 4) + 0.5*df_test.bathrooms.clip(0, 2))).values
    df_train['num_even_bathrooms'] = ((np.round(df_train.bathrooms) - df_train.bathrooms)==0).astype(float)
    df_test['num_even_bathrooms'] = ((np.round(df_test.bathrooms) - df_test.bathrooms)==0).astype(float)
    
    # Other features
    df_train['num_features'] = df_train.features.apply(lambda x: len(x))
    df_test['num_features'] = df_test.features.apply(lambda x: len(x))
    df_train['num_photos'] = df_train.photos.apply(lambda x: len(x))
    df_test['num_photos'] = df_test.photos.apply(lambda x: len(x))
    df_train['num_desc_length'] = df_train.description.str.split(' ').str.len()
    df_test['num_desc_length'] = df_test.description.str.split(' ').str.len()
    df_train['num_desc_length_null'] = (df_train.description.str.len()==0).astype(float)
    df_test['num_desc_length_null'] = (df_test.description.str.len()==0).astype(float)
    
    # Features/Description Features
    bows = {'nofee': ['no fee', 'no-fee', 'no  fee', 'nofee', 'no_fee'],
            'lowfee': ['reduced_fee', 'low_fee','reduced fee', 'low fee'],
            'furnished': ['furnished'],
            'parquet': ['parquet', 'hardwood'],
            'concierge': ['concierge', 'doorman', 'housekeep','in_super'],
            'prewar': ['prewar', 'pre_war', 'pre war', 'pre-war'],
            'laundry': ['laundry', 'lndry'],
            'health': ['health', 'gym', 'fitness', 'training'],
            'transport': ['train', 'subway', 'transport'],
            'parking': ['parking'],
            'utilities': ['utilities', 'heat water', 'water included']
          }
    for fname, bow in bows.items():
        x1 = df_train.description.str.lower().apply(lambda x: np.sum([1 for i in bow if i in x]))
        x2 = df_train.features.apply(lambda x: np.sum([1 for i in bow if i in ' '.join(x).lower()]))
        df_train['num_'+fname] = ((x1 + x2) > 0).astype(float).values
        x1 = df_test.description.str.lower().apply(lambda x: np.sum([1 for i in bow if i in x]))
        x2 = df_test.features.apply(lambda x: np.sum([1 for i in bow if i in ' '.join(x).lower()]))
        df_test['num_'+fname] = ((x1 + x2) > 0).astype(float).values

    return df_train, df_test


# ### A. Load and preprocess datasets

# Load data:

# In[ ]:


df = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')
df['created'] = pd.to_datetime(df.created)
df_test['created'] = pd.to_datetime(df_test.created)


# Location encoding based on:
# 
# - [Unsupervised and supervised neighborhood encoding](https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/unsupervised-and-supervised-neighborhood-encoding)
# - [Price/Bedrooms/Bathrooms](https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths)

# In[ ]:


dftemp = df.copy()
for i in ['latitude', 'longitude']:
    while(1):
        x = dftemp[i].median()
        ix = abs(dftemp[i] - x) > 3*dftemp[i].std()
        if ix.sum()==0:
            break
        dftemp.loc[ix, i] = np.nan
dftemp = dftemp.loc[dftemp[['latitude', 'longitude']].isnull().sum(1) == 0, :]

dfm = DataFrameMapper([(['latitude'], [StandardScaler()]), (['longitude'], [StandardScaler()])])

for i in [5, 10, 20, 40]:
    pipe_location = make_pipeline(dfm, KMeans(n_clusters=i, random_state=1))
    pipe_location.fit(dftemp);
    df['location_'+str(i)] = pipe_location.predict(df).astype(str)
    df_test['location_'+str(i)] = pipe_location.predict(df_test).astype(str)
for i in df.location_10.unique():
    df['num_location_10_'+str(i)] = (df.location_10==i).astype(float)
    df_test['num_location_10_'+str(i)] = (df_test.location_10==i).astype(float)


# ### B. Keep only relevant numerical features and normalize

# In[ ]:


# Get relevant features
df, df_test = basic_preprocess(df, df_test, n_min=15, precision=3)
feats = [i for i in df.columns.values if i.startswith('num_')]
x_train = df[feats].values
x_test = df_test[feats].values
print(x_train.shape, x_test.shape)


# In[ ]:


# Normalize
for i in range(x_train.shape[1]):
    x_test[:, i] = (x_test[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])
    x_train[:, i] = (x_train[:, i] - np.mean(x_train[:, i]))/np.std(x_train[:, i])


# ### C. Build and evaluate Neural Network

# In[ ]:


# Classifier
clf_nn = MLPClassifier(solver='lbfgs', random_state=1)
params = {
    'alpha': [1e-6], # 1e-5, 1e-4...
    'activation': ['tanh'], # 'relu', 'sigmoid'....
    'hidden_layer_sizes': [(10, 30, 5)]#, (30, 30, 5), (20, 20, 20), (30, 30, 5)]
}
gs_nn = GridSearchCV(clf_nn, param_grid=params, scoring='neg_log_loss', n_jobs=2, cv=2, verbose=2, refit=True) # cv=5
start = time.time()
gs_nn.fit(x_train, df.y.values)
print('- Time: %.2f minutes' % ((time.time() - start)/60))
print('- Best score: %.4f' % gs_nn.best_score_)
print('- Best params: %s' % gs_nn.best_params_)


# In[ ]:


output_results(gs_nn, x_test, df_test.listing_id.values, 'basic_nn.csv') # 0.58372 LB

