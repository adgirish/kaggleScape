
# coding: utf-8

# # Journey to the top 10%

# With most kernels/notebooks, you get the following steps, presented one after another:
# 
# 1. Data Exploration
# 2. Data Engineering
# 3. Data Modelling
# 4. Results
# 
# This doesn't reflect the actual process of how you take part Kaggle competitions; in reality parts 1 through to 3 are cycled through repeatedly, in varying orders, until we finally hit stage 4.
# 
# Therefore, this is a notebook which more closely documents the actual process I took. The aim is twofold:
# 
# 1. Posterity, as I will want to revisit the exact thoughts/rationales I had behind making certain choices.
# 2. Pedagogical, as it may benefit others to understand how I hacked and stumbled my way through this competition.
# 
# Of course, as much as it pains me, the process will be abridged as otherwise it would be far too long (it took me about 2 weeks of effort with 5 different notebooks).
# 
# At the time of writing, the below code will get you into the top 10% of the "House Prices: Advanced Regression Techniques" tutorial competition on Kaggle. I tried to do as much of this on my own as possible, but give credit where it's due when I've adopted ideas from other Kagglers.
# 
# I also understand that some people will want a TL;DR, so see below for a summary of the contents so you can skip to the juicy bits (probably sections 5.C and 6.A-C):

# # Contents:

# 1. Initial Data Exploration
# 2. Create a Baseline Model
# 3. Feature Engineering: First Pass
# 4. Different Models
# 5. More Data Modelling and Exploration
#     1. Outliers (manual and statistical detection)
#     2. Overfitting
#     3. The FINAL engineering steps in one section
# 6. Final Modelling
#     1. Lasso
#     2. Ensembled XGBoost
#     3. Ensembled Everything
# 7. Afterword

# # 1. Initial Data Exploration

# The first step is always to read the data description if avaialble. We need to build a foundation of knowledge on which we can stand, so that we can make informed decisions further down the line. The approach I will take is relatively straightforward; look for interesting things in the data which we can test out later.

# The first thing to notice is that this dataset is that there are a fair few variables. Furthermore, there seems to be categorical features listed as numbers/ordinals (MSSubClass, MoSold), and the exact opposite issue (Quality measures). We will make a note of this and see if it improves our models later.
# 
# Interestingly, it forewarns us to the presence of missing variables, which for the most case are due to a parameter not being relevant (i.e.: a one story flat will never have a properties relating to a second story).
# 
# The next thing to do is to load in the training data and understand what we will be fitting our models to.

# In[ ]:


import numpy as np
import pandas as pd
import math
from scipy import stats


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.info()


# As expected, there are a number of fields with missing values, and some fields with very few actual values. This is not necessarily a bad thing, as we can impute values where necessary. Furthermore, perhaps setting them straight to zero may be better?

# ## Correlations

# As a starting point, let's do a quick heatmap plot. Primarily because they look nice, but also because they can give us a bit of information about the data :)

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.heatmap(data=train.corr())
plt.show()
plt.gcf().clear()


# Well what does this show us? For starters, 'Id' seems to be pretty irrelevant. Sometimes in Kaggle, the Id can represent some sort of temporal/ordinal factor, but it does not seem to be the case here. In any case, if some sort of temporal factor was present, we'd be able to get it from MoSold/YrSold, neither of which seem to possess much correlation.
# 
# Luckily, SalePrice seems to have correlations with a fair few other features. Ones which particularly jump out are OverallQual and GrLivArea, which is not hugely surprising. We should expect to see these appearing high in feature importance measures later on.
# 
# Finally, LotArea seems to be highly correlated, but missing a lot of values (over 200). This is important to note during imputation, and means we ought to spend a bit longer on this variable.

# ## Skews

# Recall the form of a standard linear regression
# 
# $$
# y = ax_1 + bx_2 + cx_3 + ...
# $$
# 
# Clearly our dependant ($y$) needs to be proportional to the other variables. This cannot be the case if the $y$ is say skewed, but the other parameters are normally distributed. Whilst this shouldn't affect tree-based regressors, it can have a drastic impact on the performance of regression based algorithms (OLS, Lasso, Ridge, KRR, etc.).
# 
# With this in mind, let's view the skews:

# In[ ]:


train.skew()


# Clearly some variables display intense skews, so we'll need to address this later on. Note at this point our target variable is skewed as well (SalePrice), so the potential issue highlighted above does indeed apply.

# # 2. Create a Baseline Model

# Right, we've identified a few different things to try out, namely:
# 
# 1. Careful treatment of missing value imputation
# 2. Converting categoricals into numericals
# 3. Converting numericals into categoricals
# 4. Unskewing variables
# 
# Before we bulldoze in with these, it is best to get a quick baseline model with little to no data engineering so we can at least have some sort of benchmark against which we can assess the effectiveness of our engineering steps.

# ## Train

# First steps, create a copy of the data, and turn the categorical data into dummy variables.

# In[ ]:


train_d = train.copy()
train_d = pd.get_dummies(train_d)


# Now remove the original categorical variables (these will cause our SKLearn to spew out errors if left in)

# In[ ]:


keep_cols = train_d.select_dtypes(include=['number']).columns
train_d = train_d[keep_cols]


# Now fill the NAs with means (NAs will similarly cause errors)

# In[ ]:


train_d = train_d.fillna(train_d.mean())


# All done, let's do the test data.

# ## Test

# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


test_d = test.copy()
test_d = pd.get_dummies(test_d)


# In[ ]:


test_d = test_d.fillna(test_d.mean())


# Some dummy variables exist in train but not test; create them in the test set and set to zero.

# In[ ]:


for col in keep_cols:
    if col not in test_d:
        test_d[col] = 0


# In[ ]:


test_d = test_d[keep_cols]


# ## Modelling

# Let's use a random forest as this should partially remove the dependancy on skews that we have with linear regression based modelling. Furthermore, RFs tend to perform quite well in general. Let's take measures such as CrossVal score and perform a GridSearch over the hyperparameter space.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

rf_test = RandomForestRegressor(n_jobs = -1)
params = {'max_depth': [20,30,40], 'n_estimators': [500], 'max_features': [100,140,160]}
gsCV = GridSearchCV(estimator = rf_test, param_grid = params, cv = 5, n_jobs = -1, verbose = 3)
gsCV.fit(train_d.drop('SalePrice',axis = 1),train_d['SalePrice'])print(gsCV.best_estimator_)
# (NB: I will stop Grid Searching henceforth and just print the best hyperparameters as it takes ages and only needs to be demonstrated once)

# In[ ]:


rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, random_state=1234)
cv_score = cross_val_score(rf_test, train_d.drop('SalePrice', axis = 1), train_d['SalePrice'], cv = 5, n_jobs = -1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# The R^2 score seems strong. Submitting the outputs from the model nets us a score of 0.14636, which is actually pretty decent, but puts us in the top 65%. This also shows the strength of this competition; whilst the difference in this and a top 10% score (0.116x) is minimal, in reality 55% of the competition stands between us and our goal! A bit harder than the Titanic one I think...
# 
# It's a good thing then that we've got a few ideas down our sleaves, let's get on with it.

# # 3. Feature Engineering: First Pass

# We'll try a few things:
# 
# 1. LotFrontage Imputation
# 2. Medians vs Means
# 3. Numeric to Categoric
# 4. Categoric to Numerical
# 5. Dropping columns
# 6. 'None' vs NA
# 7. Unskewing data
# 
# and see where we end up.

# ## 1. LotFrontage Imputation

# We've identified LotFrontage as a potential sticking point, so let's think of how to address this. Either we can use just 0s (as these houses don't have fronts, so this is technically true), or impute values based on medians, be it a global median or some sort of conditional median. Let's try 0s and conditional medians, and see what works better.

# ### 0s

# In[ ]:


train_0 = train.copy()


# In[ ]:


null_index = train_0.LotFrontage.isnull()
train_0.loc[null_index,'LotFrontage'] = 0


# In[ ]:


train_0 = pd.get_dummies(train_0)


# In[ ]:


keep_cols = train_0.select_dtypes(include=['number']).columns
train_0 = train_0[keep_cols]


# In[ ]:


train_0 = train_0.fillna(train_0.mean())


# In[ ]:


rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state=1234)
cv_score = cross_val_score(rf_test, train_0.drop('SalePrice', axis = 1), train_0['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# This scores basically no difference in-sample, so let's refrain from using this.

# ### Conditional Medians

# Perhaps we should impute medians, i.e.: pretend that these properties had lots. In reality it's usually good enough to just use a generic median, but let's try to be clever. In my experience, it would make sense for lot size to be related to the areas in which houses are located. Let's see if there's any significant difference conditioning on neighbourhood. 

# In[ ]:


sns.barplot(data=train,x='Neighborhood',y='LotFrontage', estimator=np.median)
plt.xticks(rotation=90)
plt.show()
plt.gcf().clear()


# Looks like a good thing to condition on. Let's use this instead.

# In[ ]:


gb_neigh_LF = train['LotFrontage'].groupby(train['Neighborhood'])


# In[ ]:


train_LFm = train.copy()


# In[ ]:


# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 
# associated with it...
for key,group in gb_neigh_LF:
    # find where we are both simultaneously missing values and where the key exists
    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)
    # fill in those blanks with the median of the key's group object
    train_LFm.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()


# Does this help? Let's see.

# In[ ]:


train_LFm = pd.get_dummies(train_LFm)


# In[ ]:


keep_cols = train_LFm.select_dtypes(include=['number']).columns
train_LFm = train_LFm[keep_cols]


# In[ ]:


train_LFm = train_LFm.fillna(train_LFm.mean())


# In[ ]:


rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state= 1234)
cv_score = cross_val_score(rf_test, train_LFm.drop('SalePrice', axis = 1), train_LFm['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# Again very similar. We will need to impute this at some point, as the non-tree based algorithms will require this. It seems to perform mildly better than turning it into 0s, but it's incredibly close. Let's go with this for now.

# ## 2. Medians vs Means

# In my haste I ran with means instead of the medians. Of course as any good statistician will tell you, a median is a more robust statistic than a mean as it's less/not affected by outliers. Let's see if converting to a different statistic will affect our model's ability to generalise.

# In[ ]:


train_med = train.copy()


# In[ ]:


# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 
# associated with it...
for key,group in gb_neigh_LF:
    # find where we are both simultaneously missing values and where the key exists
    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)
    # fill in those blanks with the median of the key's group object
    train_med.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()


# In[ ]:


train_med = pd.get_dummies(train_med)


# In[ ]:


keep_cols = train_med.select_dtypes(include=['number']).columns
train_med = train_med[keep_cols]


# In[ ]:


train_med = train_med.fillna(train_med.median())


# In[ ]:


rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, n_jobs=-1, random_state=1234)
cv_score = cross_val_score(rf_test, train_med.drop('SalePrice', axis = 1), train_med['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# Again a slight increase. Let's see how we're doing on the leaderboard now as a quick check.

# In[ ]:


rf_test.fit(train_med.drop('SalePrice',axis = 1),train_med['SalePrice'])


# The score is now 0.14559, which is a very marginal improvement. Having said this, such small gains will make all the difference between the top scores near the top 10%!

# ## 3. Numeric to Categoric

# Now lets examine the numerics which ought to be categorics. In the interests of being concise, I experimented with the following:
# 
# 1. MSSubClass
# 2. MoSold
# 3. YrSold
# 
# And found improved performance with only MSSubClass converted. This can be illustrated below:

# In[ ]:


sns.barplot(data=train, x='MSSubClass', y='SalePrice')
plt.show()
plt.gcf().clear()


#  Therefore we will be using this going forward.

# ## INTERJECTION: Change models?

# At this point I'd thought it would be wise to try a different modelling technique. Even with the best will in the world, I was getting issues with the ever-reliable random forest. Namely, there was huge noise in my cross validation score, which was making choosing the right data engineering steps a bit of a nightmare. Also RF regression is tediously slow. Enter XGBoost.
# 
# I won't explain how XGBoost works, as there is literature online which can explain it better than I ever will, but suffice to say it is similar to RF in that it combines a lot of trees together, but unlike RF it doesn't build them in a random manner.

# In[ ]:


from xgboost.sklearn import XGBRegressor


# In[ ]:


xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
cv_score = cross_val_score(xgb_test, train_med.drop(['SalePrice','Id'], axis = 1), train_med['SalePrice'], cv = 5, n_jobs = -1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# This is quite a bit stronger than RF, submitting yields 0.13031, which puts us strongly in the top 50%. A while to go yet, but we are moving in the right direction. Furthermore, let's move to using XGBoost as our regression method now.

# ## 4. Categoric to Numeric

# This is interesting. Some of the fields regarding the quality of the property are 'secretly' ordinal. Case in point, the field entitled BsmtCond, which has different quality ratings. Perhaps turning these into their numeric correspondent will improve performance, as we will be able to mine out better trends.

# In[ ]:


has_rank = [col for col in train if 'TA' in list(train[col])]


# In[ ]:


dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}


# In[ ]:


train_c2n = train.copy()


# In[ ]:


train_c2n['MSSubClass'] = train_c2n['MSSubClass'].astype('category')


# In[ ]:


# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 
# associated with it...
for key,group in gb_neigh_LF:
    # find where we are both simultaneously missing values and where the key exists
    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)
    # fill in those blanks with the median of the key's group object
    train_c2n.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()


# In[ ]:


for col in has_rank:
    train_c2n[col+'_2num'] = train_c2n[col].map(dic_num)


# In[ ]:


train_c2n = pd.get_dummies(train_c2n)


# In[ ]:


train_cols = train_c2n.select_dtypes(include=['number']).columns
train_c2n = train_c2n[train_cols]


# In[ ]:


train_c2n = train_c2n.fillna(train_c2n.median())


# In[ ]:


xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
cv_score = cross_val_score(xgb_test, train_c2n.drop(['SalePrice','Id'], axis = 1), train_c2n['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# This is a strong increase. Let's move onto dropping columns.

# ## 5. Dropping Columns

# One thing to note about this dataset is the lack of data (~1500 rows) and with it the curse of dimensionality. One way around this is to remove the number of columns, specifically ones which are carry little information.
# 
# I experimented with this, and found that removing columns where 97% of the data was a single class to be a good method.

# In[ ]:


from statistics import mode


# In[ ]:


low_var_cat = [col for col in train.select_dtypes(exclude=['number']) if 1 - sum(train[col] == mode(train[col]))/len(train) < 0.03]
low_var_cat


# Let's drop these columns.

# In[ ]:


has_rank = [col for col in train if 'TA' in list(train[col])]


# In[ ]:


dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}


# In[ ]:


train_col = train.copy()


# In[ ]:


train_col = train_col.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)


# In[ ]:


train_col['MSSubClass'] = train_col['MSSubClass'].astype('category')


# In[ ]:


# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 
# associated with it...
for key,group in gb_neigh_LF:
    # find where we are both simultaneously missing values and where the key exists
    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)
    # fill in those blanks with the median of the key's group object
    train_col.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()


# In[ ]:


for col in has_rank:
    train_col[col+'_2num'] = train_col[col].map(dic_num)


# In[ ]:


train_col = pd.get_dummies(train_col)


# In[ ]:


train_cols = train_col.select_dtypes(include=['number']).columns
train_col = train_col[train_cols]


# In[ ]:


train_col = train_col.fillna(train_col.median())


# In[ ]:


xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
cv_score = cross_val_score(xgb_test, train_col.drop(['SalePrice','Id'], axis = 1), train_col['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# A moderate increase, so let's continue.

# ## 6. 'None' vs NA

# So we got around the issue with NAs in the numerical data by replacing with the medians. However, what about with categoric? Up to now we've encoded these as NA, which means when the dummy variables are created they effectively get dropped. Instead, let's create a new class called 'None' for when the feature doesn't exist for a property (i.e.: a property without a basement), or in the case of a genuine missing value, replace with the mode.

# In[ ]:


cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]
cat_hasnull


# We see that all these fields apart from 'Electrical' have NAs denoting that there is none of that item in the house (i.e.: no Garage). Therefore let's replace these with the string 'None' and the Electrical field with the mode value.

# In[ ]:


cat_hasnull.remove('Electrical')


# In[ ]:


mode_elec = mode(train['Electrical'])
mode_elec


# So the mode is SBrkr. Let's use this as the replacement.

# In[ ]:


has_rank = [col for col in train if 'TA' in list(train[col])]


# In[ ]:


dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}


# In[ ]:


cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]


# In[ ]:


cat_hasnull.remove('Electrical')


# In[ ]:


train_none = train.copy()


# In[ ]:


train_none = train_none.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)


# In[ ]:


train_none['MSSubClass'] = train_none['MSSubClass'].astype('category')


# In[ ]:


for col in cat_hasnull:
    null_idx = train_none[col].isnull()
    train_none.loc[null_idx, col] = 'None'


# In[ ]:


null_idx_el = train_none['Electrical'].isnull()
train_none.loc[null_idx_el, 'Electrical'] = 'SBrkr'


# In[ ]:


# for the key (the key is neighborhood in this case), and the group object (group is LotFrontage grouped by Neighborhood) 
# associated with it...
for key,group in gb_neigh_LF:
    # find where we are both simultaneously missing values and where the key exists
    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)
    # fill in those blanks with the median of the key's group object
    train_none.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()


# In[ ]:


for col in has_rank:
    train_none[col+'_2num'] = train_none[col].map(dic_num)


# In[ ]:


train_none = pd.get_dummies(train_none)


# In[ ]:


train_cols = train_none.select_dtypes(include=['number']).columns
train_none = train_none[train_cols]


# In[ ]:


train_none = train_none.fillna(train_none.median())


# In[ ]:


xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
cv_score = cross_val_score(xgb_test, train_none.drop(['SalePrice','Id'], axis = 1), train_none['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# A nice increase! Let's keep going.

# ## 7. Unskewing Data

# As aforementioned, if we want to look at more traditional regression techniques, we need to address the skewness that exists in our data. Interestingly, if you were to try a Lasso model with the current data, you'd get something in between XGBoost and RandomForest. This bodes well :)

# In[ ]:


cols_skew = [col for col in train_none if '_2num' in col or '_' not in col]
train_none[cols_skew].skew()


# In addition to the original values, we have some new skewed values of the _2num variety (i.e.: our categorics converted into numbers). In the interests of being concise, I found that unskewing values with a skewness magnitude above 1 to give the best results.

# In[ ]:


cols_unskew = train_none[cols_skew].columns[abs(train_none[cols_skew].skew()) > 1]


# In[ ]:


train_unskew = train_none.copy()


# In[ ]:


for col in cols_unskew:
    train_unskew[col] = np.log1p(train_none[col])


# In[ ]:


xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)
cv_score = cross_val_score(xgb_test, train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# Interestingly enough the CV score for the XGBoost model improves. Perhaps this is down to unskewing the sale price, which reduces the heteroskedacisity of the target, making it easier to predict.
# 
# For reference, let us submit. We now get a score of 0.12741, which is a big jump in the right direction, putting us in the top 39%.

# # 4. Different Models: Lasso

# We've addressed a lot of the issues holding us back when using a linear model. In terms of which linear models to use, there are really 3 which are popular:
# 
# 1. Lasso
# 2. Ridge
# 3. Elastic Net (combination of Lasso and Ridge)
# 
# I've tried all 3 with this data set, and Lasso is by far and away the most performant. If you were to try the other 2, you will find a huge amount of overfit on the training set, even with cross validation.
# 
# With this knowledge, let's get lazy and use the LassoCV algorithm. This is effectively the same as the standard Lasso algorithm, but it will use cross validation to figure out the best parameters each time you use the model. This saves having to perform gridsearch, and typically gives identical performance. An important thing to note with these types of algorithms is the need to scale the data so that the feature space has the same sort of magnitudes in each feature's direction.

# In[ ]:


from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
LCV = LassoCV()
scale_LCV = Pipeline([('scaler',scaler),('LCV',LCV)])

cv_score = cross_val_score(scale_LCV, train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'], cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# A strong score. Submitting gives us 0.12303. Incredibly this destroys the XGBoost score, which goes to show that you can't always trust your CV score. In any case, we are in the top 30%.
# 
# The beauty of Lasso is that it will drop features thanks to the use of the L1-norm. Of course this can be misleading, in the same way that a random forest's variable importances can be misleading, as having two competing but highly correlated features will mean one will get suppressed despite being highly predictive.
# 
# For our interest, let's see what didn't get dropped, and had high weights:

# In[ ]:


scale_LCV.fit(train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'])


# In[ ]:


lasso_w = scale_LCV.named_steps['LCV'].coef_
cols= train_unskew.drop(['SalePrice','Id'], axis=1).columns


# In[ ]:


cols_w = pd.DataFrame()
cols_w['Features'] = cols
cols_w['LassoWeights'] = lasso_w
cols_w['LassoWeightsMag'] = abs(lasso_w)


# In[ ]:


cols_w[cols_w.LassoWeights==0]


# Funnily enough, LotFrontage gets dropped! That will teach us next time to spend so much time data engineering :) Joking aside, it looks like a lot of dummy variables get removed. What about the important features?

# In[ ]:


cols_w.sort_values(by='LassoWeightsMag',ascending=False)[:15]


# Not too many surprises here. Properties in commercial areas are worth less, bigger houses and garages are worth more, as are good quality houses. Furthermore, a bit of in depth cyberstalking of the neighborhoods seems to back up the numbers.

# # 5. More Data Exploration and Engineering

# We've done well but we've hit a bit of an impasse. We've scrabbled up into the top 30%, which is the easy part. Now however, we will need to take careful consideration about the data we have and creative ways of treating it which will provide us with the final push that we need to make the top 10%.

# ## Outliers?

# For those of you familiar with classification problems, outliers aren't usually too big a deal. Since the target is only 1 or 0, the impact of these points isn't that large as long as we have a decently sized population (which we just about do with 1500). However with regression it's a different story.
# 
# I can highly recommend reading the following [lecture notes](https://quantoid.net/files/702/lecture9.pdf); in short, we can now turn to econometrics to help us deal with the effect of outliers. I'm going to outline two approaches, both of which achieve similar results.

# ### Manually Detect Outliers

# Heuristic but effective. The way we ought to approach this is to look at our Lasso model and identify which features have the largest weightings, and therefore effect on the final prediction. Luckily we've just done this, so let's plot univariate distributions against the SalePrice for the top 5 important variables:

# In[ ]:


top5_feats = list(cols_w.sort_values(by='LassoWeightsMag',ascending=False)[:5].Features)


# In[ ]:


def print_scatters(df_in, cols, against):   
    plt.figure(1)
    # sets the number of figure row (ie: for 10 variables, we need 5, for 9 we 
    # need 5 as well)
    rows = math.ceil(len(cols)/2)
    f, axarr = plt.subplots(rows, 2, figsize=(10, rows*3))
    # for each variable you inputted, plot it against the dependant
    for col in cols:
        ind = cols.index(col)
        i = math.floor(ind/2)
        j = 0 if ind % 2 == 0 else 1
        if col != against:
            sns.regplot(data = df_in, x=col, y=against, fit_reg=False, ax=axarr[i,j])
        else:
            sns.distplot(a = df_in[col], ax=axarr[i,j])
        axarr[i, j].set_title(col)
    f.text(-0.01, 0.5, against, va='center', rotation='vertical', fontsize = 12)
    plt.tight_layout()
    plt.show()
    plt.gcf().clear()


# In[ ]:


print_scatters(df_in=train,cols=['SalePrice']+top5_feats,against='SalePrice')


# Immediately two points jump out, both of which are GrLivArea, the most important feature. In case you haven't spotted them, look at the two houses above 4500 sqft in living area. These are the two biggest houses yet are also very cheap. This is a problem, as if you look at the lecture notes linked above, they are outliers with high leverage, meaning they can cause huge overfit to them. Let's do a bit more analysis:

# In[ ]:


sns.lmplot(data=train, x='GrLivArea',y='SalePrice')
plt.show()
plt.gcf().clear()


# Now removing these two:

# In[ ]:


sns.lmplot(data=train[train.GrLivArea < 4500], x='GrLivArea',y='SalePrice')
plt.show()
plt.gcf().clear()


# This is a remarkable change, as simply removing two points has resulted in a visible shift in the curve, despite the effect of the remaining data. Furthermore, the confidence interval of our plots has improved massively.
# 
# Submitting confirms the necessecity to remove these points, with a nice improved score of 0.12174. We are now in the top 27%.
# 
# Clearly these outliers were preventing our model from generalising onto new data!

# ### A more statistical approach?

# Of course we might have juts gotten lucky by stumbling upon these two outliers. Why don't we try something more analytical, and therefore repeatable?
# 
# If you refer to the notes linked above, you can find an interesting statistical test described on slides 29 onwards. Effectively we can do Leave-One-Out with a OLS, and then perform hypothesis tests on the standard deviation of the residuals (they are t-distributed), allowing us to determine which points have a significant effect on the fit of our models. To me this makes sense since the model used is OLS, and the Lasso belongs to this family of regressors.
# 
# The Bonferroni corrected test is included in Statsmodels, so we'll use it here. Note that this takes a very long time to run, so I'll simply post the code but it is up to you to run it.

# In[ ]:


import statsmodels.api as sm

y = train_unskew['SalePrice']
X = train_unskew.drop(['SalePrice','Id'], axis = 1)model = sm.OLS(y,X)results = model.fit()bonf_test = results.outlier_test()['bonf(p)']bonf_outlier = list(bonf_test[bonf_test<1e-3].index)
print(bonf_test[bonf_test<1e-3])
# I chose a p-value of less than 0.001 on the results that MeiChengShih got in this [thread](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/23409).
# 
# Running the above gave me the following outliers:

# In[ ]:


bonf_outliers = [88,462,523,588,632,968,1298,1324]


# Thankfully the two points highlighted above also exist in the above list, thus further validating this as an approach to discovering outliers.
# 
# Removing the above points gives a very similar result (0.1216x) to the one obtained by manually determining and removing the two outliers. Whilst it is tempted to be disappointed by this, I see this as encouraging as it gives us an analytical way of achieving the same results without needing to rely on lady luck to present us the offending points.

# ## Overfitting columns amongst other things

# At this point I averaged the result from my XGBoost and Lasso and climbed into the top 18%. I also tried a whole litany of different data engineering steps to improve my predictions to no avail.
# 
# I therefore thought it was time I checked out some other kernels to see what they were doing right. A [post](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/36280) I made was helpfully responded to by olivier, who pointed me in the direction of removing columns and overall looking at ways to improve the way my data generalised.
# 
# As a result, I discovered the incredibly comprehensive kernel of [Human Analog](https://www.kaggle.com/humananalog/xgboost-lasso). Immediately I noticed a couple things I'd been doing wrongly, including the presence of nulls in the test set.
# 
# Therefore I decided to borrow some of the ideas, as well as one of my own, which is to use the medians derived across both the test and train set.

# ### A note on overfitting

# What is important to note however is that of all the things I borrowed from Human Analog, the most important was removing two columns which were causing overfit, namely MSSubClass_160 and MSZoning_C (all).
# 
# As to how I'd find these in reality I'm not sure. A few ideas were discussed on the aforementioned thread between myself and olivier, but it boils down to the fact that there isn't enough data in this set, which is why these features cause overfit on CV.

# ### The final steps

# I am now going to print the final data engineering steps below which got me the data which formed the basis of my top 10% entry.

# ### Combine the two sets for medians

# ### 0s

# In[ ]:


train_test_raw = train.append(test)


# In[ ]:


lot_frontage_by_neighborhood_all = train_test_raw["LotFrontage"].groupby(train_test_raw["Neighborhood"])


# ### Train Set

# In[ ]:


has_rank = [col for col in train if 'TA' in list(train[col])]


# In[ ]:


dic_num = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}


# In[ ]:


cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]


# In[ ]:


cat_hasnull.remove('Electrical')


# In[ ]:


train_c2n = train.copy()


# In[ ]:


for key,group in lot_frontage_by_neighborhood_all:
    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)
    train_c2n.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()


# In[ ]:


train_c2n = train_c2n.drop(['Street','Utilities','Condition2','RoofMatl','Heating'], axis=1)


# In[ ]:


train_c2n['MSSubClass'] = train_c2n['MSSubClass'].astype('category')


# In[ ]:


for col in cat_hasnull:
    null_idx = train_c2n[col].isnull()
    train_c2n.loc[null_idx, col] = 'None'


# In[ ]:


null_idx_el = train_c2n['Electrical'].isnull()
train_c2n.loc[null_idx_el, 'Electrical'] = 'SBrkr'


# In[ ]:


for col in has_rank:
    train_c2n[col+'_2num'] = train_c2n[col].map(dic_num)


# In[ ]:


train_c2n = pd.get_dummies(train_c2n)


# In[ ]:


train_cols = train_c2n.select_dtypes(include=['number']).columns
train_c2n = train_c2n[train_cols]


# ### Test Set

# In[ ]:


test_c2n = test.copy()


# In[ ]:


# See Human Analog
test_c2n.loc[666, "GarageQual"] = "TA"
test_c2n.loc[666, "GarageCond"] = "TA"
test_c2n.loc[666, "GarageFinish"] = "Unf"
test_c2n.loc[666, "GarageYrBlt"] = 1980

test_c2n.loc[1116,'GarageType'] = np.nan


# In[ ]:


for key,group in lot_frontage_by_neighborhood_all:
    lot_f_nulls_nei = test['LotFrontage'].isnull() & (test['Neighborhood'] == key)
    test_c2n.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()


# In[ ]:


test_c2n = test_c2n.drop(['Street','Utilities','Condition2','RoofMatl','Heating'], axis=1)


# In[ ]:


test_c2n['MSSubClass'] = test_c2n['MSSubClass'].astype('category')


# In[ ]:


for col in cat_hasnull:
    null_idx = test_c2n[col].isnull()
    test_c2n.loc[null_idx, col] = 'None'


# In[ ]:


null_idx_el = test_c2n['Electrical'].isnull()
test_c2n.loc[null_idx_el, 'Electrical'] = 'SBrkr'


# In[ ]:


for col in has_rank:
    test_c2n[col+'_2num'] = test_c2n[col].map(dic_num)


# In[ ]:


test_c2n = pd.get_dummies(test_c2n)


# In[ ]:


test_c2n['SalePrice'] = 0


# In[ ]:


for col in train_cols:
    if col not in test_c2n:
        train_c2n = train_c2n.drop(col,axis=1)


# In[ ]:


test_c2n = test_c2n.drop('MSSubClass_150', axis = 1)


# In[ ]:


final_cols = test_c2n.select_dtypes(include=['number']).columns
test_c2n = test_c2n[final_cols]


# In[ ]:


test_c2n = test_c2n[train_c2n.columns]


# ### Fill with medians

# In[ ]:


train_test_combo = train_c2n.append(test_c2n)
train_test_raw = train.append(test)


# In[ ]:


train_test_combo = train_test_combo.fillna(train_test_combo.median())


# In[ ]:


train_med = train_test_combo[:1460]
test_med = train_test_combo[1460:]


# ### Unskewing

# In[ ]:


cols = [col for col in train_med if '_2num' in col or '_' not in col]
skew = [abs(stats.skew(train_med[col])) for col in train_med if '_2num' in col or '_' not in col]


# In[ ]:


skews = pd.DataFrame()
skews['Columns'] = cols
skews['Skew_Magnintudes'] = skew


# In[ ]:


cols_unskew = skews[skews.Skew_Magnintudes > 1].Columns


# In[ ]:


train_unskew2 = train_med.copy()
test_unskew2 = test_med.copy()


# In[ ]:


for col in cols_unskew:
    train_unskew2[col] = np.log1p(train_med[col])
    
for col in cols_unskew:
    test_unskew2[col] = np.log1p(test_med[col])


# ### Removing outliers in sample

# In[ ]:


bonf_outlier = [88,462,523,588,632,968,1298,1324]


# In[ ]:


train_unskew3 = train_unskew2.drop(bonf_outlier)


# ### 'Overfit' Columns

# In[ ]:


drop_cols = ["MSSubClass_160", "MSZoning_C (all)"]


# In[ ]:


train_unskew3 = train_unskew3.drop(drop_cols, axis = 1)
test_unskew2 = test_unskew2.drop(drop_cols, axis = 1)


# ### Final DFs

# In[ ]:


X_train = train_unskew3.drop(['Id','SalePrice'],axis = 1)
y_train = train_unskew3['SalePrice']


# In[ ]:


X_test = test_unskew2.drop(['Id','SalePrice'],axis=1)


# # 6. Final Modelling

# We've been doing bits and pieces of modelling throughout to verify our engineering steps, but let's consolidate all of this. Given that we've settled on our final feature set, let's tighten this up and make our final submission.

# ## Lasso

# Very easy, see before.

# In[ ]:


scaler = StandardScaler()
LCV = LassoCV()
scale_LCV = Pipeline([('scaler',scaler),('LCV',LCV)])

cv_score = cross_val_score(scale_LCV, X_train, y_train, cv = 5, n_jobs=-1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cv_score)))


# ## XGBoost

# ### Overfitting... on a seed?

# XGBoost requires a seed, as this affects how the trees are built. This can however result in issues. For example, if we were to keep the seed constant and tune hyperparameters using cross validation, we may overfit to that score/seed, resulting in a model which won't generalise onto new points. The way around this is to validate each improved hyperparameter choice with a few different seeds to ensure that the new hyperparameter really offers better generalisation.
# 
# Once you take this approach, the best final step is to average the final outputs from several different XGBoost models with the same optimal hyperparameters, but different seeds.

# ### The final model

# With the above in mind, let's create the final XGBoost model.

# In[ ]:


from sklearn.base import BaseEstimator, RegressorMixin


# In[ ]:


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append((regressor.predict(X).ravel()))
        return (np.mean(self.predictions_, axis=0))


# The above was helpfully taken and modified from Oleg Panichev's "Ensemble of 4 models with CV [LB: 0.11489]" [kernel](https://www.kaggle.com/opanichev/ensemble-of-4-models-with-cv-lb-0-11489).

# In[ ]:


xgb1 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200
                )

xgb2 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200,
                seed = 1234
                )

xgb3 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200,
                seed = 1337
                )


# The above hyperparameters were found with a manual gridsearch.

# In[ ]:


xgb_ens = CustomEnsembleRegressor([xgb1,xgb2,xgb3])


# Running the CV for the ensemble regressor crashes it; get an indication on one regressor.

# In[ ]:


cvscore = cross_val_score(cv=5,estimator=xgb1,X = X_train,y = y_train, n_jobs = -1)


# In[ ]:


print('CV Score is: '+ str(np.mean(cvscore)))


# ## The results?

# Our R^2 scores look great, and our models are finely tuned so drum roll please for the final results...
# 
# Lasso scores: 0.11912
# 
# XGBoost scores: 0.11974
# 
# Pretty astounding, we're in the top 17% with just a simple Lasso model. If you told me that before the competition I would have laughed at you!

# ### Take the average of the two for the top 10%!

# In the same way we improved our XGBoost by averaging different models, we can do the same for Lasso and XGBoost. This makes sense as these models are built in completely different ways, so any errors (under/over estimates) should be cancelled out by averaging with the other model, assuming these errors are a function of the modelling types, not the data.
# 
# As proof of this, we take our final step to glory by achieveing a score of 0.11549!

# # 7. Afterword

# ## Conclusions

# 1. Don't trust your CV score blindly in the face of limited data, find a balance between the LB score and CV
#     1. Crazily enough, CV can overfit!!!
# 2. Baseline your scores so you don't make suboptimal choices early on
# 3. Always reseed your seed-based algorithms when tuning hyperparameters
# 4. With regression the smallest things can make a huge difference, especially on LB scores:
#     1. Outliers
#     2. Overfitting to dummy features

# ## Things I tried out but didn't work:

# 1. Stacking
# 2. Different models and ensembling them in various ways:
#     1. KRR/SVR
#     2. kNN Regression
#     3. Ridge
#     4. ElasticNet
#     5. ExtraTrees
#     6. LARs/LassoLARs
#     7. Passive Aggressive Regression
#     8. Huber
# 3. A whole list of data engineering steps:
#     1. Making Year and Month categorical
#     2. Playing around with GarageYrBuilt
#     3. Adding in external data (interest rates)
#     4. etc etc
#     
# If anyone has tips on stacking that would be greatly appreciated! I'd love it for someone to take the data from here and use it to beat my score with a stacked model :)

# # Appendix

# ## Sumbit Code

# In[ ]:


xgb_ens.fit(X_train, y_train);
scale_LCV.fit(X_train,y_train);


# In[ ]:


preds_x = np.expm1(xgb_ens.predict(X_test));
preds_l = np.expm1(scale_LCV.predict(X_test));
preds = (preds_x+preds_l)/2
out_preds = pd.DataFrame()
out_preds['Id'] = test['Id']
out_preds['SalePrice'] = preds
out_preds.to_csv('output.csv', index=False)

