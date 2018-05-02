
# coding: utf-8

# ### <font color='salmon'>  Corporación Favorita Grocery Sales Forecasting
# 

# ### <font color='salmon'>Importing all the libraries that we will need</font>

# In[ ]:


# import necessary modules
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import gc

import seaborn as sns
sns.set(style = 'whitegrid', color_codes = True)
get_ipython().run_line_magic('matplotlib', 'inline')

#For statistical tests
import scipy.stats as st

#For formula notation (similar to R)
import statsmodels.formula.api as smf

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator


# ### <font color='salmon'>Read the Data</font>

# In[ ]:


df_train = pd.read_csv("../input/corporitasampled-train-data/train_rd.csv")


# In[ ]:


df_train.head()


# In[ ]:


print("we have taken ",len(df_train), "rows")


# In[ ]:


Input_Path = '../input/favorita-grocery-sales-forecasting/'


# In[ ]:


test  = pd.read_csv("../input/favorita-grocery-sales-forecasting/test.csv")
testg  = pd.read_csv("../input/favorita-grocery-sales-forecasting/test.csv")
store = pd.read_csv("../input/favorita-grocery-sales-forecasting/stores.csv")
holiday = pd.read_csv("../input/favorita-grocery-sales-forecasting/holidays_events.csv")
item = pd.read_csv("../input/favorita-grocery-sales-forecasting/items.csv")
oil = pd.read_csv("../input/favorita-grocery-sales-forecasting/oil.csv")
trans = pd.read_csv("../input/favorita-grocery-sales-forecasting/transactions.csv")


# ### <font color='salmon'>Getting familiar with the data</font>

# ### <font color='salmon'>train data</font>
# 
# * Training data, which includes the target unit_sales by date, store_nbr, and item_nbr and a unique id to label rows.
# * The target unit_sales can be integer (e.g., a bag of chips) or float (e.g., 1.5 kg of cheese).
# * Negative values of unit_sales represent returns of that particular item.
# * The onpromotion column tells whether that item_nbr was on promotion for a specified date and store_nbr.
# * Approximately 16% of the onpromotion values in this file are NaN.

# In[ ]:


df_train.head()


# ### <font color='salmon'>Items data</font>
# 
# * Item metadata, including family, class, and perishable.
# * NOTE: Items marked as perishable have a score weight of 1.25; otherwise, the weight is 1.0

# In[ ]:


item.head()


# In[ ]:


print("There are",len(item['family'].unique()),"families of products or items")


# ### <font color='salmon'>Oil data</font>
# 
# Daily oil price. Includes values during both the train and test data timeframe. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)

# In[ ]:


oil.head()


# ### <font color='salmon'>Transaction data</font>
# * The count of sales transactions for each date, store_nbr combination. 
# * Only included for the training data timeframe.

# In[ ]:


trans.head()


# ### <font color='salmon'>Holiday data</font>
# 
# * Holidays and Events, with metadata
# * A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. 
# A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.

# In[ ]:


holiday.head()


# ### <font color='salmon'>Store data</font>
# * Store metadata, including city, state, type, and cluster.
# * cluster is a grouping of similar stores.

# In[ ]:


store.head()


# In[ ]:


print("There are",len(store['type'].unique()),"type of stores")


# In[ ]:


print("Stores are in ",len(store['city'].unique()),"cities in ", len(store['state'].unique()),"states")


# ### <font color='salmon'>Test data</font>

# In[ ]:


test.head()


# ### <font color='salmon'>Join Data</font>
# 
# * Join stores and items data to Trian data with respect to Store number and Item number
# * Join holiday and oil data to Train with respect to Date

# In[ ]:


train = pd.merge(df_train, store, on= "store_nbr")
train = pd.merge(train, item, on= "item_nbr")
train = pd.merge(train, holiday, on="date")
train = pd.merge(train, oil, on ="date")


# In[ ]:


train.head()


# Join Train and item data

# In[ ]:


train_items = pd.merge(df_train, item, how='inner')
train_items1 = pd.merge(df_train, item, how='inner')
train_items2 = pd.merge(df_train, item, how='inner')


# In[ ]:


train_items.head()


# ### <font color='salmon'> Data Pre Processing</font>

# ### <font color='salmon'> Checking for missing Data</font>

# In[ ]:


oil_nan = (oil.isnull().sum() / oil.shape[0]) * 100
oil_nan


# Only 3.5% missing data on oil price

# In[ ]:


store_nan = (store.isnull().sum() / store.shape[0]) * 100
store_nan


# No missing store data

# In[ ]:


item_nan = (item.isnull().sum() / item.shape[0]) * 100
item_nan


# No missing item data

# In[ ]:


df_train_nan = (df_train.isnull().sum() / df_train.shape[0]) * 100
df_train_nan


# 17.3% missing data on onpromotion

# On promotion NAN values are UNKNOWN - if item is on promotion 
# 
# So replacing Nan of "on promotion" with 2 to indicate the items have unknown status on promotion

# In[ ]:


train['onpromotion'] = train['onpromotion'].fillna(2)
train['onpromotion'] = train['onpromotion'].replace(True,1)
train['onpromotion'] = train['onpromotion'].replace(False,0)


# In[ ]:


(train['onpromotion'].unique())


# Unknown oil price - putting 0 for now

# In[ ]:


train['dcoilwtico'] = train['dcoilwtico'].fillna(0)


# In[ ]:


train['Year']  = train['date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['date'].apply(lambda x: int(str(x)[5:7]))
train['date']  = train['date'].apply(lambda x: (str(x)[8:]))


test['Year']  = test['date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['date'].apply(lambda x: int(str(x)[5:7]))
test['date']  = test['date'].apply(lambda x: (str(x)[8:]))

train.head()


# In[ ]:


train_items1['date'] = pd.to_datetime(train_items1['date'], format='%Y-%m-%d')
train_items1['day_item_purchased'] = train_items1['date'].dt.day
train_items1['month_item_purchased'] =train_items1['date'].dt.month
train_items1['quarter_item_purchased'] = train_items1['date'].dt.quarter
train_items1['year_item_purchased'] = train_items1['date'].dt.year
train_items1.drop('date', axis=1, inplace=True)

train_items2['date'] = pd.to_datetime(train_items2['date'], format='%Y-%m-%d')
train_items2['day_item_purchased'] = train_items2['date'].dt.day
train_items2['month_item_purchased'] =train_items2['date'].dt.month
train_items2['quarter_item_purchased'] = train_items2['date'].dt.quarter
train_items2['year_item_purchased'] = train_items2['date'].dt.year
train_items2.drop('date', axis=1, inplace=True)


# In[ ]:


#train_items['Year']  = train_items['date'].apply(lambda x: int(str(x)[:4]))
#train_items['Month'] = train_items['date'].apply(lambda x: int(str(x)[5:7]))
#train_items['date']  = train_items['date'].apply(lambda x: (str(x)[8:]))


# In[ ]:


train_items1.loc[(train_items1.unit_sales<0),'unit_sales'] = 1 
train_items1['unit_sales'] =  train_items1['unit_sales'].apply(pd.np.log1p) 

train_items1['family'] = train_items1['family'].astype('category')
train_items1['onpromotion'] = train_items1['onpromotion'].astype('category')
train_items1['perishable'] = train_items1['perishable'].astype('category')
cat_columns = train_items1.select_dtypes(['category']).columns
train_items1[cat_columns] = train_items1[cat_columns].apply(lambda x: x.cat.codes)

train_items2.loc[(train_items2.unit_sales<0),'unit_sales'] = 1 
train_items2['unit_sales'] =  train_items2['unit_sales'].apply(pd.np.log1p) 

train_items2['family'] = train_items2['family'].astype('category')
train_items2['onpromotion'] = train_items2['onpromotion'].astype('category')
train_items2['perishable'] = train_items2['perishable'].astype('category')
cat_columns = train_items2.select_dtypes(['category']).columns
train_items2[cat_columns] = train_items2[cat_columns].apply(lambda x: x.cat.codes)


# In[ ]:


train_items1.head()


# ### <font color='salmon'> EDA </font>

# We have taken a sample of train data to plot the graphs - the entire data is taking too much time

# In[ ]:


strain = train.sample(frac=0.01,replace=True)


# ### <font color='salmon'> Plotting Sales with promotion </font>

# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.barplot(x='onpromotion', y='unit_sales', data=strain, ax=axis1)


# ### <font color='salmon'> Plotting Sales per Item Family </font>

# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.barplot(x='family', y='unit_sales', data=strain, ax=axis1)


# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x=strain['family'], data=strain, ax=axis1)


# ### <font color='salmon'> Plotting Sales per Store Type </font>

# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='type_x', y='unit_sales', data=strain, ax=axis1)


# Store type "A" is highest selling

# ### <font color='salmon'> Plotting Stores in Cities and states </font>

# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x=store['city'], data=store, ax=axis1)


# Quito and Guayaquil have the most number of stores

# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x=store['state'], data=store, ax=axis1)


# Pichincha and Guayas have highest number of stores

# In[ ]:


fig, (axis1) = plt.subplots(1,1,figsize=(30,4))
sns.countplot(x='cluster', data=store, ax=axis1)


# In[ ]:


g = sns.FacetGrid(train, col='cluster', hue='cluster', size=4)
g.map(sns.barplot, 'type_x', 'unit_sales');


# #### Plotting Oil Price

# In[ ]:


fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(15,8))

ax1 = oil.plot(legend=True,ax=axis1,marker='o',title="Oil Price")


# ### Plotting Sales with date

# In[ ]:


average_sales = train.groupby('date')["unit_sales"].mean()
average_promo = train.groupby('date')["onpromotion"].mean()

fig, (axis1, axis2) = plt.subplots(2,1,figsize=(15,4))

ax1 = average_sales.plot(legend=True,ax=axis1,marker='o',title="Average Sales")
ax2 = average_promo.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Average Promo")


# In[ ]:


train.plot(kind='scatter',x='store_nbr',y='unit_sales',figsize=(15,4))


# Store number 44 / 45 - maximum sales

# In[ ]:


train.plot(kind='scatter',x='item_nbr',y='unit_sales',figsize=(15,4))


# In[ ]:


store_number = train.groupby('store_nbr')["unit_sales"].mean()
item_number = train.groupby('item_nbr')["unit_sales"].mean()

fig, (axis1, axis2) = plt.subplots(2,1,figsize=(30,4))

ax1 = store_number.plot(legend=True,ax=axis1,marker='o',title="Sales with store")
ax2 = item_number.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Sales with item")


# Store 45 - interesting

# ### <font color='salmon'> chi-squared Test </font>

# #### <font color='navy'> Question 1 - Is there any statistically significant relation between Store Type and Cluster of the stores ?
#  
# * Null Hypothesis H0        = Store Type (a, b, c, d, e) and Cluster (1 to 17) are independent from each other.
# * Alternative Hypothesis HA = Store Tpe and cluster are not independent of each other. There is a relationship between them.
# 
#  
# * Store Type - categorical variable
# * Cluster - categorical variable
# 
# Now, to determine if there is a statistically significant correlation between the variables, 
# we use a chi-square test of independence of variables in a contingency table
# 
# Here, we create a contingency table, with the frequencies of all possible values
# 

# In[ ]:


# Contingency table
ct = pd.crosstab(store['type'], store['cluster'])
ct


# In[ ]:


ct.plot.bar(figsize = (15, 6), stacked=True)
plt.legend(title='cluster vs Type')
plt.show()


# Finally, we compute the chi-square test statistic and the associated p-value. The null hypothesis is the independence between the variables. SciPy implements this test in scipy.stats.chi2_contingency, which returns several objects. We're interested in the second result, which is the p-value.

# In[ ]:


st.chi2_contingency(ct)


# ####  Interpretation of Result:
# 
# * The p-value is much lower than 0.05.
# * There is strong evidence that the null hypothesis is False.
# * We reject the null hypothesis and conclude that there is a statistically significant correlation between the Store Type and cluster of the stores.

# #### <font color='navy'> Question 1A - Is there any statistically significant relation between City and Cluster of the stores ?
#  
# * Null Hypothesis H0        = City and Cluster are independent from each other.
# * Alternative Hypothesis HA = City Tpe and cluster are not independent of each other. There is a relationship between them.
# 
#  
# * City - categorical variable
# * Cluster - categorical variable
# 
# Now, to determine if there is a statistically significant correlation between the variables, 
# we use a chi-square test of independence of variables in a contingency table
# 
# Here, we create a contingency table, with the frequencies of all possible values

# In[ ]:


# Contingency table
ct2 = pd.crosstab(store['city'], store['cluster'])
ct2


# In[ ]:


ct2.plot.bar(figsize = (15, 6), stacked=True)
plt.legend(title='cluster')
plt.show()


# In[ ]:


st.chi2_contingency(ct2)


# ####  Interpretation of Result:
# 
# * The p-value is higher than 0.05.
# * There is NO evidence to reject Null Hypothesis.
# * We can continue with the null hypothesis and conclude that there is no dependence of cluster and cities

# ### <font color='salmon'> t-Test

# #### <font color='navy'> Question 2 - Is there any statistically significant relation between Store Type and Sales of the stores ?

# * Null Hypothesis H0        = Promotion and Sales are independent from each other.
# * Alternative Hypothesis HA = Promotion and Sales are not independent of each other. There is a relationship between them.
# 
# 
# * Promotion - categorical variable - Independent variable
# * Sales - continuous variable - Dependent variable
# 
# Now, to determine if there is a statistically significant correlation between the variables, 
# we use a student t test
# 
# 2-sample t-test: testing for difference across populations

# In[ ]:


promo_sales = train[train['onpromotion'] == 1.0]['unit_sales']
nopromo_sales = train[train['onpromotion'] == 0.0]['unit_sales']
st.ttest_ind(promo_sales, nopromo_sales, equal_var = False)


# ### <font color='salmon'> Correlation / Regression

# #### <font color='navy'> Question 3 - Is there any statistically significant relation between  Oil price and Sales of the stores ?
# 
# * Null Hypothesis H0        = Oil price and Sales are independent from each other.
# * Alternative Hypothesis HA = Oil price and Sales are not independent of each other. There is a relationship between them.
# 
# 
# * Oil Price - Independent continuous variable
# * Sales - Dependent continuous variable
# 
# ##### We will do Simple Linear Regression now

# In[ ]:


lm0 = smf.ols(formula = 'unit_sales ~ dcoilwtico', data = train).fit()


# In[ ]:


#print the Result 
print(lm0.summary())


# No relation between oil price on sales

# > ### <font color='salmon'>Random Forest - Trying to predict Sales

# In[ ]:


X_train = train.drop(['unit_sales', 'description', 'locale_name','locale','city','state','family','type_x','type_y','cluster','class','perishable','transferred', 'dcoilwtico'], axis = 1)
y_train = train.unit_sales


# In[ ]:


rf = RandomForestRegressor(n_jobs = -1, n_estimators = 15)
y = rf.fit(X_train, y_train)
print('model fit')


# In[ ]:


X_test = test
y_test = rf.predict(X_test)


# In[ ]:


result = pd.DataFrame({'id':test.id, 'unit_sales': y_test}).set_index('id')
result = result.sort_index()
result[result.unit_sales < 0] = 0
result.to_csv('submissionR.csv', index=False)
print('submission created')


# ### <font color='salmon'> XGBOOST - Try to predict Sales

# In[ ]:


train_items1 = train_items1.drop(['unit_sales','family','class','perishable'], axis = 1)


# In[ ]:


train_items1.head()


# In[ ]:


train_items2 = train_items2.drop(['id','store_nbr','item_nbr','onpromotion', 'day_item_purchased','month_item_purchased','quarter_item_purchased','year_item_purchased','family','class','perishable'], axis = 1)


# In[ ]:


train_items2.head()


# In[ ]:


Xg_train, Xg_valid = train_test_split(train_items1, test_size=0.012, random_state=10)
Yg_train, Yg_valid = train_test_split(train_items2, test_size=0.012, random_state=10)
features = list(train_items1.columns.values)
features2 = list(train_items2.columns.values)


# In[ ]:


features 


# In[ ]:


features2


# In[ ]:


#dtrain = xgb.DMatrix(Xg_train[features], Xg_train.unit_sales)
#dvalid = xgb.DMatrix(Xg_valid[features], Xg_valid.unit_sales)
#Xg_train.dtypes


# In[ ]:


dtrain = xgb.DMatrix(Xg_train[features], Yg_train[features2])
dvalid = xgb.DMatrix(Xg_valid[features], Yg_valid[features2])


# In[ ]:


def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))


# In[ ]:


def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)


# In[ ]:


params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 30
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]


# In[ ]:


gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,   early_stopping_rounds=20, feval=rmspe_xg, verbose_eval=True)


# In[ ]:


print("Validating")
yhat = gbm.predict(xgb.DMatrix(Xg_valid[features]))
error = rmspe(Yg_valid.unit_sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))


# In[ ]:


testg.head()


# In[ ]:


testg['date'] = pd.to_datetime(testg['date'], format='%Y-%m-%d')
testg['day_item_purchased'] = testg['date'].dt.day
testg['month_item_purchased'] =testg['date'].dt.month
testg['quarter_item_purchased'] = testg['date'].dt.quarter
testg['year_item_purchased'] = testg['date'].dt.year
testg.drop('date', axis=1, inplace=True)


# In[ ]:


testg.head()


# In[ ]:


features


# In[ ]:


testg.loc[(train_items.unit_sales<0),'unit_sales'] = 1 
#testg['unit_sales'] =  train_items['unit_sales'].apply(pd.np.log1p) 
testg['onpromotion'] = testg['onpromotion'].astype('category')
cat_columns = testg.select_dtypes(['category']).columns
testg[cat_columns] = testg[cat_columns].apply(lambda x: x.cat.codes)


# In[ ]:


dtest = xgb.DMatrix(testg[features])


# In[ ]:


test_probs = gbm.predict(dtest)
print("Make predictions on the test set")


# In[ ]:


result = pd.DataFrame({"id": test["id"], 'unit_sales': np.expm1(test_probs)})
result.to_csv("submissionX2.csv", index=False)
print("Submission created")

