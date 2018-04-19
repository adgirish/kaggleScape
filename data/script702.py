
# coding: utf-8

# ###This is my first notebook! And rather than going the XGBoost way, I decided it's about time to go back to basics.
# 
# We'll look into 
# ###1. basic feature engineering and extraction
# ###2. use the Lasso model and give our newly developed data a test run!
# ###3. Some (minor) visualizations

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.stats import skew, skewtest
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read in the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ### Proposed feature: '1stFlrSF' + '2ndFlrSF' to give us combined Floor Square Footage
# but first, let's check how well it fares!

# In[ ]:


feat_trial = (train['1stFlrSF'] + train['2ndFlrSF']).copy()
print("Skewness of the original intended feature:",skew(feat_trial))
print("Skewness of transformed feature", skew(np.log1p(feat_trial)))

# hence, we'll use the transformed feature thank you very much!
feat_trial = np.log1p(feat_trial)
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

# seaborn's regression plot (I liked it a lot. hence it found it's way here!)
sns.regplot(x=(feat_trial), y=np.log1p(train['SalePrice']), data=train, order=1);


# In[ ]:


# lets create the feature then
train['1stFlr_2ndFlr_Sf'] = np.log1p(train['1stFlrSF'] + train['2ndFlrSF'])
test['1stFlr_2ndFlr_Sf'] = np.log1p(test['1stFlrSF'] + test['2ndFlrSF'])


# ### Feature number 2 -> 1stflr+2ndflr+lowqualsf+GrLivArea = All_Liv_Area
# 
# let's see how this fares too

# In[ ]:


feat_trial = (train['1stFlr_2ndFlr_Sf'] + train['LowQualFinSF'] + train['GrLivArea']).copy()
print("Skewness of the original intended feature:",skew(feat_trial))
print("Skewness of transformed feature", skew(np.log1p(feat_trial)))

# hence, we'll use the transformed feature thank you very much!
feat_trial = np.log1p(feat_trial)
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

# seaborn's regression plot (I liked it a lot. hence it found it's way here!)
sns.regplot(x=(feat_trial), y=np.log1p(train['SalePrice']), data=train, order=1);


# In[ ]:


train['All_Liv_SF'] = np.log1p(train['1stFlr_2ndFlr_Sf'] + train['LowQualFinSF'] + train['GrLivArea'])
test['All_Liv_SF'] = np.log1p(test['1stFlr_2ndFlr_Sf'] + test['LowQualFinSF'] + test['GrLivArea'])


# ## Those were the two features I added. Let's move further to Step 2

# In[ ]:


# get all features except Id and SalePrice
feats = train.columns.difference(['Id','SalePrice'])

# the most hassle free way of working with data is to concatenate them
# since there are many features that contain nan/null values in the test set
# that the train set doesn't
all_data = pd.concat((train.loc[:,feats],
                      test.loc[:,feats]))


# ## The To-Do List
# 
# ### 1. Transform skewed numeric features using log(p+1) transformation making them more normal
# ### 2. Find dummy variables for categorical features
# ### 3. Replace nans/null values

# In[ ]:


# But first, we log transform the target: (reason well explained in Alexandru's AWESOME Notebook)
train["SalePrice"] = np.log1p(train["SalePrice"])


# ## 1. Transformations
# 
# ###PS: log(p+1) transformations are not the available transformations for reducing skewness. I have tried sqrt, which yields better results as compared to log(p+1) (for certain features only. log(1+p) is a better way to go for this dataset at least)*emphasized text*

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# ## 2. Let's get them dummies

# In[ ]:


# getting dummies for all features. You can go the LabelEncoder way, but this method
# is more sound (and easier!!!) in my opinion
all_data = pd.get_dummies(all_data)


# ## 3. Fill them nan's

# In[ ]:


# 3. filling NA's with the mean of the column:
all_data = all_data.fillna(all_data[:train.shape[0]].mean())


# In[ ]:


print(all_data.shape)
# creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# In[ ]:


# optional. Save these newly created matrices for later usage
# new_test = pd.DataFrame(X_test.copy())
# new_test['Id'] = test['Id'].copy()
# new_test.to_csv("../input/new_test.csv", index=False)

# new_train = pd.DataFrame(X_train.copy())
# new_train['SalePrice'] = y.copy()
# new_train.to_csv("../input/new_train.csv", index=False)


# ## All set. Moving on to incorporate this data
# ### But first, Let's devise a cross-validation methodology once and for all (thanks again, Alexandru)

# In[ ]:


from sklearn.cross_validation import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
    return(rmse)


# In[ ]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression


# ### Let's get to work using LassoCV

# In[ ]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], selection='random', max_iter=15000).fit(X_train, y)
res = rmse_cv(model_lasso)
print("Mean:",res.mean())
print("Min: ",res.min())


# ### The above model yields a lb score of 0.12102* (some results I've gained through kaggle nb's and my local system have been different)

# In[ ]:


coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[ ]:


# plotting feature importances!
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# ### Woah. The first feature we engineered did end up being pretty important!!!! Way To go!

# In[ ]:


# Let's make some predictions and submit it to the lb
test_preds = np.expm1(model_lasso.predict(X_test))
submission = pd.DataFrame()
submission['Id'] = test['Id']
submission["SalePrice"] = test_preds
submission.to_csv("lasso_by_Sarthak.csv", index=False)

