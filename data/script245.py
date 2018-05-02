
# coding: utf-8

# # Modeling for Dummies
# This notebook will contain very simple methodologies for transforming the data inputs so that decent results can be generated from the popular machine learning models.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ### Read in the Data
# Make sure we put the Id column in the index of the pandas data frame. It is of no value for model building

# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)


# ### Inspect the Data

# In[ ]:


train_df.head()


# # Combine Test and Train into a Single DataFrame
# Next, we will combine train and test into a single pandas DataFrame. This is so that we can apply all our transformations to both the train and test set in one step and that both the train and the test set will have identical column names. We must first store the SalePrice variable as our y and drop it from train_df. Since model, evaluation will be done on the log of SalePrice, we go ahead and transform the final home price. We need to remember to transform it back in order to submit.

# In[ ]:


y_train = np.log(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)


# In[ ]:


all_df.shape


# In[ ]:


y_train.head()


# # Variable transformations
# For this example, only minimal variable transformations will occur. To obtain a better fit for your data, a more detailed introspection of each variable would be necessary. Reading the data description file would be helpful here.
# 
# One thing that we notice is that the first variable MSSubClass is actually a categorical variable that is recorded as numeric. Lets change this variable to a string.

# In[ ]:


all_df['MSSubClass'].dtypes


# In[ ]:


all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)


# In[ ]:


all_df['MSSubClass'].value_counts()


# ### Indicator (dummy) Variables
# Since sklearn doesn't natively handle categorical predictor variables we must 0/1 encode them as a different column for each unique category for each variable. As you can see above, MSSubClass has about a dozen unique values. Each of these values will turn into a column.
# 
# ### Using pd.get_dummies
# Pandas has a handy function that will do all the encoding for you. Let's see an example of this using just one variable.

# In[ ]:


pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()


# ### Using pd.get_dummies on an entire DataFrame.
# get_dummies can also work on an entire dataframe. This is very nice and works by ignoring all numeric features and making 0/1 columns for all categorical features (those that have 'object') as its data type.

# In[ ]:


all_dummy_df = pd.get_dummies(all_df)
all_dummy_df.head()


# ### All numeric variable
# Only numeric data is acceptable for all the machine learning algorithms in sklearn. And get_dummies has nearly done this for us. 'Nearly' because there are still missing values. get_dummies actually takes care of missing values in the categorical variables for us by simply not labeling them (they get labeled 0 for each new column created).

# ### Missing Values
# Let's check the missing values first to see how many we have to deal with.

# In[ ]:


all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)


# ### Replacing missing values
# There are many valid ways to replace missing values. Many times, a missing value might mean a very specific thing. Here we will bypass further introspection and simply replace each missing value with the mean. This could potentially be a very stupid thing to do, but to proceed to model building we will just go ahead with the mean.

# In[ ]:


mean_cols = all_dummy_df.mean()
mean_cols.head(10)


# In[ ]:


all_dummy_df = all_dummy_df.fillna(mean_cols)


# ### Check that missing values are no more

# In[ ]:


all_dummy_df.isnull().sum().sum()


# ### Stage 1 Data Prep Complete
# Our data is finished (for now). This is a very crude and fast approach that gives availability of all variables for input into sklearn's models.

# ### More Data Prep
# Since we will be using penalized (ridge) regression - Ridge its best practice to standardize our inputs by subtracting the mean and dividing by the standard deviation so that they are all scaled similarly. We only want to do this to our non 0/1 variables. In pandas these were our original numeric variables. We will apply this standardization to both train and test data at the same time, which is technically data snooping but will proceed again for simplicity.
# 
#  Lets find the names of these first. 

# In[ ]:


numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_cols


# In[ ]:


numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std


# ### Check a histogram of a variable to see that the scaling worked
# Checking the variable **GrLivArea** we see that the scaling has centered it to 0. We also see some outliers here > 3 standard deviations. We could apply a log transformation (something you can think about) or investigate those large values.

# In[ ]:


all_dummy_df['GrLivArea'].hist();


# # Model Building
# Next we will explore a few popular models and use cross validation to get an estimate for what we are likely to see as our score for a submission.

# ### Splitting Data back to Train/test
# At the beginning of the notebook we combined all the train and test data. We will no separate it back out.

# In[ ]:


dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]


# In[ ]:


dummy_train_df.shape, dummy_test_df.shape


# ### Ridge Regression with Cross Validation
# We perform ridge regression for our first model, which allows us to use all the variables without too much worry about multicollinearity because of the penalty imposed.

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# In[ ]:


# Not completely necessary, just converts dataframe to numpy array
X_train = dummy_train_df.values
X_test = dummy_test_df.values


# ### Using cross_val_score 
# Sklearn has a nice function that computes the cross validation score for any chosen model. The code below loops through an array of alphas (the penalty term for ridge regression) and outputs the results of 10-fold cross validation. Sklearn uses the negative mean squared error as its scoring method so we must take the negative and the square root to get the same metric that kaggle is using.

# In[ ]:


alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))


# In[ ]:


import matplotlib.pyplot as plt


# ### Plotting the cross validation score
# Below, we plot alpha vs the cross validation score. It looks like a value of alpha between 10 and 20 give about the same score. We can also expect to get  a score around .135 (or likely a bit higher) if we submit.

# In[ ]:


plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error");


# ### Using random forest
# Lets do the same procedure for fitting a random forest to the data. The training is done on 200 trees for 5-fold cv. This will take quite some time.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))


# In[ ]:


plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error");


# # Make final prediction as combination of Ridge and Random Forest.
# We will now train two models using the best parameters from cross validation on all the data for both Ridge Regression and Random Forest and then take the average of those two predictions.

# In[ ]:


ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)


# In[ ]:


ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)


# In[ ]:


y_ridge = np.exp(ridge.predict(X_test))
y_rf = np.exp(rf.predict(X_test))


# In[ ]:


y_final = (y_ridge + y_rf) / 2


# In[ ]:


submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})


# In[ ]:


submission_df.head(10)


# # Make Submisison

# In[ ]:


submission_df.to_csv('submisison_rf_ridge.csv', index=False)

