
# coding: utf-8

# ## Introduction

# This notebook aims at getting a good insight in the data for the PorteSeguro competition. Besides that, it gives some tips and tricks to prepare your data for modeling. The notebook consists of the following main sections:
# 
# 1. [Visual inspection of your data](#visual_inspection)
# 2. [Defining the metadata](#metadata)
# 3. [Descriptive statistics](#descriptive_stats)
# 4. [Handling imbalanced classes](#imbalanced_data)
# 5. [Data quality checks](#data_quality)
# 6. [Exploratory data visualization](#eda)
# 7. [Feature engineering](#feat_engineering)
# 8. [Feature selection](#feat_selection)
# 9. [Feature scaling](#feat_scaling)

# ## Loading packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)


# ## Loading data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# <a class="anchor" id="visual_inspection"></a>

# ## Data at first sight

# Here is an excerpt of the the data description for the competition:
# * Features that belong to **similar groupings are tagged** as such in the feature names (e.g.,  ind, reg, car, calc).
# * Feature names include the postfix **bin** to indicate binary features and **cat** to  indicate categorical features. 
# * Features **without these designations are either continuous or ordinal**. 
# * Values of **-1**  indicate that the feature was **missing** from the observation. 
# * The **target** columns signifies whether or not a claim was filed for that policy holder.
# 
# Ok, that's important information to get us started. Let's have a quick look at the first and last rows to confirm all of this.

# In[ ]:


train.head()


# In[ ]:


train.tail()


# We indeed see the following
# * binary variables
# * categorical variables of which the category values are integers
# * other variables with integer or float values
# * variables with -1 representing missing values
# * the target variable and an ID variable

# Let's look at the number of rows and columns in the train data.

# In[ ]:


train.shape


# We have 59 variables and 595.212 rows. Let's see if we have the same number of variables in the test data.<br>
# Let's see if there are duplicate rows in the training data.

# In[ ]:


train.drop_duplicates()
train.shape


# No duplicate rows, so that's fine.

# In[ ]:


test.shape


# We are missing one variable in the test set, but this is the target variable. So that's fine.<br>
# Let's now invesigate how many variables of each type we have.

# So later on we can create dummy variables for the 14 categorical variables. The *bin* variables are already binary and do not need dummification.

# In[ ]:


train.info()


# Again, with the info() method we see that the data type is integer or float. No null values are present in the data set. That's normal because missing values are replaced by -1. We'll look into that later.

# <a class="anchor" id="metadata"></a>

# ## Metadata
# To facilitate the data management, we'll store meta-information about the variables in a DataFrame. This will be helpful when we want to select specific variables for analysis, visualization, modeling, ...
# 
# Concretely we will store:
# - **role**: input, ID, target
# - **level**: nominal, interval, ordinal, binary
# - **keep**: True or False
# - **dtype**: int, float, str

# In[ ]:


data = []
for f in train.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    dtype = train[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)


# In[ ]:


meta


# Example to extract all nominal variables that are not dropped

# In[ ]:


meta[(meta.level == 'nominal') & (meta.keep)].index


# Below the number of variables per role and level are displayed. 

# In[ ]:


pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()


# <a class="anchor" id="descriptive_stats"></a>

# ## Descriptive statistics

# We can also apply the *describe* method on the dataframe. However, it doesn't make much sense to calculate the mean, std, ... on categorical variables and the id variable. We'll explore the categorical variables visually later.
# 
# Thanks to our meta file we can easily select the variables on which we want to compute the descriptive statistics. To keep things clear, we'll do this per data type.

# ### Interval variables

# In[ ]:


v = meta[(meta.level == 'interval') & (meta.keep)].index
train[v].describe()


# #### reg variables
# - only ps_reg_03 has missing values
# - the range (min to max) differs between the variables. We could apply scaling (e.g. StandardScaler), but it depends on the classifier we will want to use.
# 
# #### car variables
# - ps_car_12 and ps_car_15 have missing values
# - again, the range differs and we could apply scaling.
# 
# #### calc variables
# - no missing values
# - this seems to be some kind of ratio as the maximum is 0.9
# - all three *_calc* variables have very similar distributions
# 
# 
# **Overall**, we can see that the range of the interval variables is rather small. Perhaps some transformation (e.g. log) is already applied in order to anonymize the data?
# 

# ### Ordinal variables

# In[ ]:


v = meta[(meta.level == 'ordinal') & (meta.keep)].index
train[v].describe()


# - Only one missing variable: ps_car_11
# - We could apply scaling to deal with the different ranges

# ### Binary variables

# In[ ]:


v = meta[(meta.level == 'binary') & (meta.keep)].index
train[v].describe()


# - A priori in the train data is 3.645%, which is **strongly imbalanced**. 
# - From the means we can conclude that for most variables the value is zero in most cases.

# <a class="anchor" id="imbalanced_data"></a>

# ## Handling imbalanced classes
# As we mentioned above the proportion of records with target=1 is far less than target=0. This can lead to a model that has great accuracy but does have any added value in practice. 
# Two possible strategies to deal with this problem are:
# * oversampling records with target=1 
# * undersampling records with target=0
# 
# There are many more strategies of course and MachineLearningMastery.com gives a [nice overview]((https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/).
# As we have a rather large training set, we can go for **undersampling**. 

# In[ ]:


desired_apriori=0.10

# Get the indices per target value
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train = train.loc[idx_list].reset_index(drop=True)


# <a class="anchor" id="data_quality"></a>

# ## Data Quality Checks

# ### Checking missing values
# Missings are represented as -1

# In[ ]:


vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# - **ps_car_03_cat and ps_car_05_cat** have a large proportion of  records with missing values. Remove these variables.
# - For the other categorical variables with missing values, we can leave the missing value -1 as such.
# - **ps_reg_03** (continuous) has missing values for 18% of all records. Replace by the mean.
# - **ps_car_11** (ordinal) has only 5 records with misisng values. Replace by the mode.
# - **ps_car_12** (continuous) has only 1 records with missing value. Replace by the mean.
# - **ps_car_14** (continuous) has missing values for 7% of all records. Replace by the mean.

# In[ ]:


# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta

# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()


# ### Checking the cardinality of the categorical variables
# Cardinality refers to the number of different values in a variable. As we will create dummy variables from the categorical variables later on, we need to check whether there are variables with many distinct values. We should handle these variables differently as they would result in many dummy variables.

# In[ ]:


v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f, dist_values))


# Only **ps_car_11_cat** has many distinct values, although it is still reasonable. 
# 
# **EDIT:** [nickycan](https://www.kaggle.com/nickycan) made an excellent remark on the fact that my first solution could lead to data leakage. He also pointed me to another kernel made by [oliver](https://www.kaggle.com/ogrellier) which deals with that. I therefore replaced this part with the kernel of oliver. All credits go to him. It is so great what you can learn by participating in the Kaggle competitions :)

# In[ ]:


# Script by https://www.kaggle.com/ogrellier
# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


# In[ ]:


train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target=train.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)


# <a class="anchor" id="eda"></a>

# ## Exploratory Data Visualization

# ### Categorical variables
# Let's look into the categorical variables and the proportion of customers with target = 1

# In[ ]:


v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    plt.figure()
    fig, ax = plt.subplots(figsize=(20,10))
    # Calculate the percentage of target=1 per category value
    cat_perc = train[[f, 'target']].groupby([f],as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(f, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show();


# As we can see from the variables **with missing values**,  it is a good idea to keep the missing values as a separate category value, instead of replacing them by the mode for instance. The customers with a missing value appear to have a much higher (in some cases much lower) probability to ask for an insurance claim.

# ### Interval variables
# Checking the correlations between interval variables. A heatmap is a good way to visualize the correlation between variables. The code below is based on [an example by Michael Waskom](http://seaborn.pydata.org/examples/many_pairwise_correlations.html)

# In[ ]:


def corr_heatmap(v):
    correlations = train[v].corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();
    
v = meta[(meta.level == 'interval') & (meta.keep)].index
corr_heatmap(v)


# There are a strong correlations between the variables:
# - ps_reg_02 and ps_reg_03 (0.7)
# - ps_car_12 and ps_car13 (0.67)
# - ps_car_12 and ps_car14 (0.58)
# - ps_car_13 and ps_car15 (0.67)
# 
# Seaborn has some handy plots to visualize the (linear) relationship between variables. We could use a *pairplot* to visualize the relationship between the variables. But because the heatmap already showed the limited number of correlated variables, we'll look at each of the highly correlated variables separately.<br>
# **NOTE**: I take a sample of the train data to speed up the process. 

# In[ ]:


s = train.sample(frac=0.1)


# #### ps_reg_02 and ps_reg_03
# As the regression line shows, there is a linear relationship between these variables. Thanks to the *hue* parameter we can see that the regression lines for target=0 and target=1 are the same.

# In[ ]:


sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# #### ps_car_12 and ps_car_13

# In[ ]:


sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# #### ps_car_12 and ps_car_14

# In[ ]:


sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# #### ps_car_13 and ps_car_15

# In[ ]:


sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# Allright, so now what? How can we decide which of the correlated variables to keep? We could perform Principal Component Analysis (PCA) on the variables to reduce the dimensions. In the AllState Claims Severity Competition I made [this kernel](https://www.kaggle.com/bertcarremans/reducing-number-of-numerical-features-with-pca) to do that. But as the number of correlated variables is rather low, we will let the model do the heavy-lifting.

# ### Checking the correlations between ordinal variables

# In[ ]:


v = meta[(meta.level == 'ordinal') & (meta.keep)].index
corr_heatmap(v)


# For the ordinal variables we do not see many correlations. We could, on the other hand, look at how the distributions are when grouping by the target value.

# <a class="anchor" id="feat_engineering"></a>

# ## Feature engineering

# ### Creating dummy variables
# The values of the categorical variables do not represent any order or magnitude. For instance, category 2 is not twice the value of category 1. Therefore we can create dummy variables to deal with that. We drop the first dummy variable as this information can be derived from the other dummy variables generated for the categories of the original variable.

# In[ ]:


v = meta[(meta.level == 'nominal') & (meta.keep)].index
print('Before dummification we have {} variables in train'.format(train.shape[1]))
train = pd.get_dummies(train, columns=v, drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))


# So, creating dummy variables adds 52 variables to the training set.

# ### Creating interaction variables

# In[ ]:


v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
interactions.drop(v, axis=1, inplace=True)  # Remove the original columns
# Concat the interaction variables to the train data
print('Before creating interactions we have {} variables in train'.format(train.shape[1]))
train = pd.concat([train, interactions], axis=1)
print('After creating interactions we have {} variables in train'.format(train.shape[1]))


# This adds extra interaction variables to the train data. Thanks to the *get_feature_names* method we can assign column names to these 
# new variables.

# <a class="anchor" id="feat_selection"></a>

# ## Feature selection

# ### Removing features with low or zero variance

# Personally, I prefer to let the classifier algorithm chose which features to keep. But there is one thing that we can do ourselves. That is removing features with no or a very low variance. Sklearn has a handy method to do that: **VarianceThreshold**. By default it removes features with zero variance. This will not be applicable for this competition as we saw there are no zero-variance variables in the previous steps. But if we would remove features with less than 1% variance, we would remove 31 variables.

# In[ ]:


selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id', 'target'], axis=1)) # Fit to train without id and target variables

f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements

v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))


# We would lose rather many variables if we would select based on variance. But because we do not have so many variables, we'll let the classifier chose. For data sets with many more variables this could reduce the processing time.
# 
# Sklearn also comes with other [feature selection methods](http://scikit-learn.org/stable/modules/feature_selection.html). One of these methods is *SelectFromModel* in which you let another classifier select the best features and continue with these. Below I'll show you how to do that with a Random Forest.

# ### Selecting features with a Random Forest and SelectFromModel
# Here we'll base feature selection on the feature importances of a random forest. With Sklearn's SelectFromModel you can then specify how many variables you want to keep. You can set a threshold on the level of feature importance manually. But we'll simply select the top 50% best variables. 
# 
# > The code in the cell below is borrowed from the [GitHub repo of Sebastian Raschka](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch04/ch04.ipynb). This repo contains code samples of his book *Python Machine Learning*, which is an absolute must to read.

# In[ ]:


X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))


# With SelectFromModel we can specify which prefit classifier to use and what the threshold is for the feature importances. With the *get_support* method we can then limit the number of variables in the train data.

# In[ ]:


sfm = SelectFromModel(rf, threshold='median', prefit=True)
print('Number of features before selection: {}'.format(X_train.shape[1]))
n_features = sfm.transform(X_train).shape[1]
print('Number of features after selection: {}'.format(n_features))
selected_vars = list(feat_labels[sfm.get_support()])


# In[ ]:


train = train[selected_vars + ['target']]


# <a class="anchor" id="feat_scaling"></a>

# ## Feature scaling
# As mentioned before, we can apply standard scaling to the training data. Some classifiers perform better when this is done.

# In[ ]:


scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))


# ## Conclusion
# Hopefully this notebook helped you with some tips on how to start with this competition. Feel free to vote for it. And if you have questions, post a comment.
