
# coding: utf-8

# # Intro
# 
# I'm trying to learn the very basics with this exercise. My goal is to train a linear regression model with a subset of columns from this interesting dataset in order to predict the value of a used car.
# 
# Any help or advice is welcome!!!
# 
# ### Changelist
# 
# * left only random forest with gridsearchcv
# * rewritten all the notebook
# * added name length feature
# * better study on the data
# * used seaborn to plot
# * added random forest and xgboost algorithms

# In[ ]:


import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, preprocessing, svm
from sklearn.preprocessing import StandardScaler, Normalizer
import math
import matplotlib
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Useful functions

# In[ ]:


def category_values(dataframe, categories):
    for c in categories:
        print('\n', dataframe.groupby(by=c)[c].count().sort_values(ascending=False))
        print('Nulls: ', dataframe[c].isnull().sum())

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )



# # Preparing data
# 
# ## Reading from file
# 
# Just reading the file and printing some lines.

# In[ ]:


df = pd.read_csv('../input/autos.csv', sep=',', header=0, encoding='cp1252')
#df = pd.read_csv('autos.csv.gz', sep=',', header=0, compression='gzip',encoding='cp1252')
df.sample(10)


# Let's see some info from numeric fields

# In[ ]:


df.describe()


# ## Dropping some useless columns
# 
# Some column can already be dropped.

# In[ ]:


print(df.seller.unique())
print(df.offerType.unique())
print(df.abtest.unique())
print(df.nrOfPictures.unique())


# Seller has only one value, while offerType and abtest has no relevance for the analysis. So far, I still don't know how to use the `dateCrawled` column.
# 
# Therefore I modify the dataframe dropping all those features.
# 
# I remove `lastSeen`, `dateCreated` and `postalCode` as well as I don't think they will be useful for a price prediction.

# In[ ]:


df.drop(['seller', 'offerType', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 'postalCode', 'dateCreated'], axis='columns', inplace=True)


# ## Cleaning data
# 
# Cleaning data from duplicates, NaNs and selecting reasonable ranges for columns
# 

# In[ ]:


print("Too new: %d" % df.loc[df.yearOfRegistration >= 2017].count()['name'])
print("Too old: %d" % df.loc[df.yearOfRegistration < 1950].count()['name'])
print("Too cheap: %d" % df.loc[df.price < 100].count()['name'])
print("Too expensive: " , df.loc[df.price > 150000].count()['name'])
print("Too few km: " , df.loc[df.kilometer < 5000].count()['name'])
print("Too many km: " , df.loc[df.kilometer > 200000].count()['name'])
print("Too few PS: " , df.loc[df.powerPS < 10].count()['name'])
print("Too many PS: " , df.loc[df.powerPS > 500].count()['name'])
print("Fuel types: " , df['fuelType'].unique())
#print("Offer types: " , df['offerType'].unique())
#print("Sellers: " , df['seller'].unique())
print("Damages: " , df['notRepairedDamage'].unique())
#print("Pics: " , df['nrOfPictures'].unique()) # nrOfPictures : number of pictures in the ad (unfortunately this field contains everywhere a 0 and is thus useless (bug in crawler!) )
#print("Postale codes: " , df['postalCode'].unique())
print("Vehicle types: " , df['vehicleType'].unique())
print("Brands: " , df['brand'].unique())

# Cleaning data
#valid_models = df.dropna()

#### Removing the duplicates
dedups = df.drop_duplicates(['name','price','vehicleType','yearOfRegistration'
                         ,'gearbox','powerPS','model','kilometer','monthOfRegistration','fuelType'
                         ,'notRepairedDamage'])

#### Removing the outliers
dedups = dedups[
        (dedups.yearOfRegistration <= 2016) 
      & (dedups.yearOfRegistration >= 1950) 
      & (dedups.price >= 100) 
      & (dedups.price <= 150000) 
      & (dedups.powerPS >= 10) 
      & (dedups.powerPS <= 500)]

print("-----------------\nData kept for analisys: %d percent of the entire set\n-----------------" % (100 * dedups['name'].count() / df['name'].count()))


# ## Working on the `null` values
# 
# Checking if theree are NaNs to fix or drop

# In[ ]:


dedups.isnull().sum()


# Some decisions to take for the nulls in the following fields: vehicleType (37422 nulls), gearbox (19803 nulls), model (20288 nulls), fuelType (33081 nulls), notRepairedDamage (70770 nulls).
# 
# ### `model`-`brand`-`vehicleType`
# If we have the `model` we could determine the `brand` and the `vehicleType` calculating the mode for the corresponding fields in the rest of the dataset. The opposite combinations are not true. So I think the actions should be:
# 
#     | vehicleType | brand | model | Action
#     | ---           | ---     | ---     |
#     | null        |  null | [value] | Set the other fields
#     | null        | [value] | null  | Delete
#     | [value]       |  null | null  | Delete
# 
# __So far, I'll drop all the NaNs in these 3 fields.__
# 
# ### `notRepairedDamage`
# Those with null `notRepairedDamage` field could be set to "`not-declared`" value for example.
# 
# ### `fuelType`
# Null `fuelType`s could be set to "`not-declared`" value again.
# 
# ### `gearbox`
# Null `fuelType`s could be set to "`not-declared`" value again.
# 

# In[ ]:


dedups['notRepairedDamage'].fillna(value='not-declared', inplace=True)
dedups['fuelType'].fillna(value='not-declared', inplace=True)
dedups['gearbox'].fillna(value='not-declared', inplace=True)
dedups['vehicleType'].fillna(value='not-declared', inplace=True)
dedups['model'].fillna(value='not-declared', inplace=True)


# Checking if all the nulls have been filled or dropped.

# In[ ]:


dedups.isnull().sum()


# OK, we're clear. Let's do some visualization now.

# ## Visualizations
# ### Categories distribution
# Let's see some charts to understand how data is distributed across the categories

# In[ ]:


categories = ['gearbox', 'model', 'brand', 'vehicleType', 'fuelType', 'notRepairedDamage']

for i, c in enumerate(categories):
    v = dedups[c].unique()
    
    g = dedups.groupby(by=c)[c].count().sort_values(ascending=False)
    r = range(min(len(v), 5))

    print( g.head())
    plt.figure(figsize=(5,3))
    plt.bar(r, g.head()) 
    #plt.xticks(r, v)
    plt.xticks(r, g.index)
    plt.show()


# ### Feature engineering

# Adding the name length to see how much does a long description influence the price

# In[ ]:


dedups['namelen'] = [min(70, len(n)) for n in dedups['name']]

ax = sns.jointplot(x='namelen', 
                   y='price',
                   data=dedups[['namelen','price']], 
#                   data=dedups[['namelen','price']][dedups['model']=='golf'], 
                    alpha=0.1, 
                    size=8)


# It seems that a name length between 15 and 30 characters is better for the sale price. An explanation could be that a longer name includes more optionals and accessories and therefore the price is obviously higher.
# Very short and very long names do not work well.

# In[ ]:


labels = ['name', 'gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
les = {}

for l in labels:
    les[l] = preprocessing.LabelEncoder()
    les[l].fit(dedups[l])
    tr = les[l].transform(dedups[l]) 
    dedups.loc[:, l + '_feat'] = pd.Series(tr, index=dedups.index)

labeled = dedups[ ['price'
                        ,'yearOfRegistration'
                        ,'powerPS'
                        ,'kilometer'
                        ,'monthOfRegistration'
                        , 'namelen'] 
                    + [x+"_feat" for x in labels]]


# In[ ]:


len(labeled['name_feat'].unique()) / len(labeled['name_feat'])


# Labels for the name column account for 62% of the total. I think it's too much, so I remove the feature.

# In[ ]:


labeled.drop(['name_feat'], axis='columns', inplace=True)


# ### Correlations
# Let's see how features are correlated each other and, more important, with the price.

# In[ ]:


plot_correlation_map(labeled)
labeled.corr()


# This is the list of the most influencing features for the price

# In[ ]:


labeled.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


# I don't know why the model does not influence the car price more...

# # Playing with different models

# ## Prepare data for training
# Here I split the dataset in train and validation data and tune the right-skewed sale price column.

# In[ ]:



Y = labeled['price']
X = labeled.drop(['price'], axis='columns', inplace=False)


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"1. Before":Y, "2. After":np.log1p(Y)})
prices.hist()

Y = np.log1p(Y)


# ### Basic imports and functions
# 
# Trying with some model from scikit learn: LinearRegression, LR with L2 regularization and others.

# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, train_test_split

def cv_rmse(model, x, y):
    r = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = 5))
    return r

# Percent of the X array to use as training set. This implies that the rest will be test set
test_size = .33

#Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state = 3)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

r = range(2003, 2017)
km_year = 10000



# ## Random forests
# 
# I use the GridSearch to set the optimal parameteres for the regressor, then train the final model.
# 
# I've removed the other parameters to quickly make this point pass online while I keep working on many parameters offline.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf = RandomForestRegressor()

param_grid = { "criterion" : ["mse"]
              , "min_samples_leaf" : [3]
              , "min_samples_split" : [3]
              , "max_depth": [10]
              , "n_estimators": [500]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)
gs = gs.fit(X_train, y_train)


# #### Predicting samples

# In[ ]:


print(gs.best_score_)
print(gs.best_params_)
 


# In[ ]:


bp = gs.best_params_
forest = RandomForestRegressor(criterion=bp['criterion'],
                              min_samples_leaf=bp['min_samples_leaf'],
                              min_samples_split=bp['min_samples_split'],
                              max_depth=bp['max_depth'],
                              n_estimators=bp['n_estimators'])
forest.fit(X_train, y_train)
# Explained variance score: 1 is perfect prediction
print('Score: %.2f' % forest.score(X_val, y_val))


# #### Predicting samples

# ### Features importance

# In[ ]:


importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

print(X_train.columns.values)
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center",tick_label = X_train.columns.values)
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()



# # Conclusions
# 
# I've tried to play with as much stuff as I could with this dataset in order to understand the very basic topics about:
# 
# * data interpretation and selection
# * feature selection and labeling
# * data visualization
# * very rough ML algorithms application
# 
# There's very much to improve both in how I managed all these steps and in the different outcomes of the predictions on the sale price. I'll experiment a bit more in the next few days, then I'll move on another dataset to learn more.
# 
