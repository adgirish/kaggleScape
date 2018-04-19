
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
from scipy.stats import norm, skew
import numpy as np
import seaborn as sns


# # Assignment 1 - House Price Prediction
# ### Laurens ten Cate - MBD'18 - Machine Learning II
# 
# ###### If you decide to use some of my code/ideas for your own kaggle submissions/kernel submissions I would really appreciate you giving me some credit! Thanks!
# 
# Personally I was inspired with model stacking by Serigne's great notebook.
# (https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
# 
# **kaggle score notes**
# - Kaggle username: Laurenstc
# - Kaggle final best score: RMSLE = 0.11383
# - Kaggle final best score rank: 81
# 
# **local score notes**
# 
# Locally I managed when to obtain consistent RMSE's of below 0.10. However, this notebook does not include my best local RMSE score as this was due to a ton of overfitting.
# 
# **final delivery notes**
# 
# The experimental dataset we are going to use is the House Prices Dataset. It includes 79 explanatory variables of residential homes. For more details on the dataset and the competition see https://www.kaggle.com/c/house-prices-advanced-regression-techniques.
# 
# The workbook is structured as followed:
# 
# 1. Data Cleaning and Pre-processing
#     - Outliers
#     - Statistical transformations
# 2. Feature Engineering
#     - Concatenation
#     - NA's
#     - Incorrect values
#     - Factorization
#     - Further Statistical transformation
#     - Column removal
#     - Creating features
#     - Dummies
#     - In-depth outlier detection
#     - Overfit prevention
#     - Baseline model
# 3. Feature Selection
#     - Filter methods
#         - baseline coefficients
#     - Embedded methods
#         - L2: Ridge Regression
#         - L1: Lasso regression
#             - In-depth coefficient analysis
#         - Elasticnet
#         - XGBoost
#         - SVR
#         - LightGBM
# 3. Ensemble methods
#     - Stacked generalizations
#     - Averaging
#         - standard
#         - weighted
# 4. Prediction
# 
# 
# This notebook represents the data manipulation used for my final score on Kaggle (RMSLE = 0.11383). However, in the process of achieving this score a lot of different feature engineering tactics were employed. For the sake of brevity I left these out of the notebook though below is a quick overview of other things I tried that did not help my score. 
# 
# One thing that was used but was not included is GridsearchCV. Gridsearching helped me find ranges of Alphas and L1_ratios that I could reuse later. However, it became unfeasible to continuously gridserach for optimal parameters with each iteration of feature engineering. Thats why I decided to omit the code from the final delivery.
# 
# **feature engineering tries**
# - Recoding categoricals to keep ordering information (if data was really ordinal)
# - Binning date variables (yearbuilt etc)
# - simplify and recode neighborhood variable based on a groupby with SalePrice
# - create simplified quality variables (1-5 scale instead of 1-10)
# - create 2nd and 3rd order polynomials of top10 strongest correlating variables with SalePrice
# - create 2nd and 3rd order polynomials of all variables
# - create interaction variables by looking at individual interaction plots
# - use sklearns PolynomialPreprocessing for complete set of interaction and polynomial terms
# 
# **feature selection tries**
# - F-score selection 
# - Mutual information regression selection
# - Backwards stepwise selection (RFECV)
# - Forwards stepwise selection (LARS)
# 
# Besides feature selection and engineering a lot of time was spent on optimizing my ensemble of models. I believe some more gains can be made here specifically regarding my stacked generalization model.
# 
# In the end I believe the biggest gains in my score were achieved with a few things. OLS outlier removal, nuanced NA filling and averaging with a stacked generalization model.
# 

# In[ ]:


#Data reading

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


print("Train set size:", train.shape)
print("Test set size:", test.shape)


# # 1. Data Cleaning and Pre-processing
# ## Outliers
# According to the documentation of the dataset (http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt) there are outliers present that are recommended to be removed. Let's plot SalePrice vs GR LIV AREA to view these. 

# In[ ]:


plt.scatter(train.GrLivArea, train.SalePrice)


# These outliers are extremely clear. The documentation recommends removing all houses above 4000 sq ft living area. Trial and error showed that this led to a little bit underfitting. Better performance was above 4500 sq.

# In[ ]:


train = train[train.GrLivArea < 4500]
plt.scatter(train.GrLivArea, train.SalePrice)


# In[ ]:


print(len(np.unique(train['Id'])) == len(train))
len(np.unique(test['Id'])) == len(test)


# In[ ]:


len(train)


# So we can safely drop the Id columns.

# In[ ]:


train = train.drop(['Id'], axis=1)
test = test.drop(['Id'], axis=1)


# In[ ]:


print("Train set size:", train.shape)
print("Test set size:", test.shape)


# ## Statistical transformation
# Let's have a look at how the target variable is distributed.

# In[ ]:


df = pd.concat([train.SalePrice, np.log(train.SalePrice + 1).rename('LogSalePrice')], axis=1, names=['SalePrice', 'LogSalePrice'])
df.head()


# In[ ]:


plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice, kde=False, fit = norm)

plt.subplot(1, 2, 2)
sns.distplot(np.log(train.SalePrice + 1), kde=False, fit = norm)
plt.xlabel('Log SalePrice')


# There seems to be clear evidence of right-skewedness in the target variable. We can correct this with a simple log transformation.

# In[ ]:


train.SalePrice = np.log1p(train.SalePrice)


# # 2. Feature Engineering
# ## Concatenation
# To keep consistency between test and train features we concatenate the two sets while remembering the index so we can split it later again.

# In[ ]:


y = train.SalePrice.reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test


# In[ ]:


print(train_features.shape)
print(test_features.shape)


# In[ ]:


features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape


# ## NA's
# Let's figure out what NA's excist, sort them by categories and impute them in the best possible way.

# In[ ]:


nulls = np.sum(features.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = features.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
print(info)
print("There are", len(nullcols), "columns with missing values")


# Most of these can be filled with 'None'. Some exceptions though:
# 
# - Functional: The documentation says that we should assume "Typ", so lets impute that.
# - Electrical: The documentation doesn't give any information but obviously every house has this so let's impute the most common value: "SBrkr".
# - KitchenQual: Similar to Electrical, most common value: "TA".
# - Exterior 1 and Exterior 2: Let's use the most common one here. 
# - SaleType: Similar to electrical, let's use most common value.
# 

# In[ ]:


features['Functional'] = features['Functional'].fillna('Typ')
features['Electrical'] = features['Electrical'].fillna("SBrkr")
features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])


# Let's check some points individually to figure out the best imputation strategy

# In[ ]:


pd.set_option('max_columns', None)
features[features['PoolArea'] > 0 & features['PoolQC'].isnull()]


# There are three NaN's foor PoolQC that have a PoolArea. Let's impute them based on overall quality of the house.

# In[ ]:


features.loc[2418, 'PoolQC'] = 'Fa'
features.loc[2501, 'PoolQC'] = 'Gd'
features.loc[2597, 'PoolQC'] = 'Fa'


# In[ ]:


pd.set_option('max_columns', None)
features[(features['GarageType'] == 'Detchd') & features['GarageYrBlt'].isnull()]


# So there are houses with garages that are detached but that have NaN's for all other Garage variables. Let's impute these manually too.

# In[ ]:


features.loc[2124, 'GarageYrBlt'] = features['GarageYrBlt'].median()
features.loc[2574, 'GarageYrBlt'] = features['GarageYrBlt'].median()

features.loc[2124, 'GarageFinish'] = features['GarageFinish'].mode()[0]
features.loc[2574, 'GarageFinish'] = features['GarageFinish'].mode()[0]

features.loc[2574, 'GarageCars'] = features['GarageCars'].median()

features.loc[2124, 'GarageArea'] = features['GarageArea'].median()
features.loc[2574, 'GarageArea'] = features['GarageArea'].median()

features.loc[2124, 'GarageQual'] = features['GarageQual'].mode()[0]
features.loc[2574, 'GarageQual'] = features['GarageQual'].mode()[0]

features.loc[2124, 'GarageCond'] = features['GarageCond'].mode()[0]
features.loc[2574, 'GarageCond'] = features['GarageCond'].mode()[0]


# Let's look at the basements:
# - BsmtQual
# - BsmtCond
# - BsmtExposure
# - BsmtFinType1
# - BsmtFinType2
# - BsmtFinSF1
# - BsmtFinSF2
# - BsmtUnfSF
# - TotalBsmtSF

# In[ ]:


basement_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                   'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                   'TotalBsmtSF']

tempdf = features[basement_columns]
tempdfnulls = tempdf[tempdf.isnull().any(axis=1)]


# In[ ]:


#now select just the rows that have less then 5 NA's, 
# meaning there is incongruency in the row.
tempdfnulls[(tempdfnulls.isnull()).sum(axis=1) < 5]


# Let's impute all incongruencies with the most likely value.

# In[ ]:


features.loc[332, 'BsmtFinType2'] = 'ALQ' #since smaller than SF1
features.loc[947, 'BsmtExposure'] = 'No' 
features.loc[1485, 'BsmtExposure'] = 'No'
features.loc[2038, 'BsmtCond'] = 'TA'
features.loc[2183, 'BsmtCond'] = 'TA'
features.loc[2215, 'BsmtQual'] = 'Po' #v small basement so let's do Poor.
features.loc[2216, 'BsmtQual'] = 'Fa' #similar but a bit bigger.
features.loc[2346, 'BsmtExposure'] = 'No' #unfinished bsmt so prob not.
features.loc[2522, 'BsmtCond'] = 'Gd' #cause ALQ for bsmtfintype1


# Zoning is also interesting

# In[ ]:


subclass_group = features.groupby('MSSubClass')
Zoning_modes = subclass_group['MSZoning'].apply(lambda x : x.mode()[0])
Zoning_modes


# In[ ]:


features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))


# For the rest we will just use a loop to impute 'None' value. 

# In[ ]:


objects = []
for i in features.columns:
    if features[i].dtype == object:
        objects.append(i)

features.update(features[objects].fillna('None'))

nulls = np.sum(features.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = features.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
print(info)
print("There are", len(nullcols), "columns with missing values")


# Now let's think about imputing the missing values in the numerical features. Most of the time I will impute 0, but sometimes something else is needed.
# 
# - LotFrontage: This is linear feet of street connected to property. Let's impute with the median per neighborhood since I assume this is extremely linked to what kind of area you live in.

# In[ ]:


neighborhood_group = features.groupby('Neighborhood')
lot_medians = neighborhood_group['LotFrontage'].median()
lot_medians


# As expected the lotfrontage averages differ a lot per neighborhood so let's impute with the median per neighborhood.

# In[ ]:


features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# Let's also take a closer look at GarageYrBlt

# In[ ]:


pd.set_option('max_columns', None)
features[(features['GarageYrBlt'].isnull()) & features['GarageArea'] > 0]


# GarageYrBlt does not have any incongruencies. Let's also examine MasVnrArea.

# In[ ]:


pd.set_option('max_columns', None)
features[(features['MasVnrArea'].isnull())]


# No incongruencies here either.
# The rest can be safely imputed with 0 since this means that the property is not present in the house.

# In[ ]:


#Filling in the rest of the NA's

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes: 
        numerics.append(i)
        
features.update(features[numerics].fillna(0))

nulls = np.sum(features.isnull())
nullcols = nulls.loc[(nulls != 0)]
dtypes = features.dtypes
dtypes2 = dtypes.loc[(nulls != 0)]
info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
print(info)
print("There are", len(nullcols), "columns with missing values")


# ## Incorrect values
# Some values can be obviously wrong and this might impact our model. I used min and max values to check odd values in the data.

# In[ ]:


features.describe()


# Looking at the min and max of each variable there are some errors in the data.
# 
# - GarageYrBlt - the max value is 2207, this is obviously wrong since the data is only until 2010. 
# 
# The rest of the data looks fine. Let's inspect this row a bit more carefully and impute an approximate correct value.

# In[ ]:


features[features['GarageYrBlt'] == 2207]


# This particular datapoint has YearBuilt in 2006 and YearRemodAdd in 2007. 2207 most likely is a data input error that should have been 2007 when the remodel happened. Let's impute 2007.

# In[ ]:


features.loc[2590, 'GarageYrBlt'] = 2007


# ## Factorization
# There are features that are read in as numericals but are actually objects. Let's transform them.

# In[ ]:


#factors = ['MSSubClass', 'MoSold']
factors = ['MSSubClass']
 


for i in factors:
    features.update(features[i].astype('str'))


# ## Skew transformation features
# Let's check skew in our features and transform if necessary.

# In[ ]:


from scipy.stats import skew

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes: 
        numerics2.append(i)

skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
skews


# I use the boxcox1p transformation here because I tried the log transform first but a lot of skew remained in the data. I use boxcox1p over normal boxcox because boxcox can't handle zero values.

# In[ ]:


from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skew_features[skew_features > 0.5]
high_skew = high_skew
skew_index = high_skew.index

for i in skew_index:
    features[i]= boxcox1p(features[i], boxcox_normmax(features[i]+1))

        
skew_features2 = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews2 = pd.DataFrame({'skew':skew_features2})
skews2


# ## Incomplete cases
# Checking to see if levels of my variables in my train and test set match and if not or if the level distribution is very low whether it should be deleted.

# In[ ]:


objects3 = []
for i in features.columns:
    if features[i].dtype == object:
        objects3.append(i)


# In[ ]:


print("Training Set incomplete cases")

sums_features = features[objects3].apply(lambda x: len(np.unique(x)))
sums_features.sort_values(ascending=False)


# Let's take a closer look at some of these lower numbered variables.

# In[ ]:


print(features['Street'].value_counts())
print('-----')
print(features['Utilities'].value_counts())
print('-----')
print(features['CentralAir'].value_counts())
print('-----')
print(features['PavedDrive'].value_counts())


# I experimented a bunch with this and decided in the end that if a column has low amount of levels and most values are in the same class (>97%) I'd remove them.

# Let's delete Utilities because of how unbalanced it is.

# In[ ]:


#features = features.drop(['Utilities'], axis=1)
features = features.drop(['Utilities', 'Street'], axis=1)


# ## Creating features
# In this section I create some features that can be created from the current data. 
# 
# Size of the house. There are a few variables dealing with square footage, I don't use TotalBsmtSF as a proxy for the basement because I believe unfinished square feet in the basement area won't have a big impact on price as it needs money to make it 'livable' square footage, so I just use BsmtSF1 and BsmtSF2.
# - BsmtFinSF1
# - BsmtFinSF2
# - 1stFlrSF 
# - 2ndFlrSF
# 
# Another combined variable is the bathrooms in the house. I count fullbath for 1 and halfbath for 0.5.
# - FullBath
# - HalfBath
# - BsmtFullBath
# - BsmtHalfBath
# 
# Another combined variable is the total porch size.
# - OpenPorchSF
# - EnclosedPorch
# - 3SsnPorch
# - Screenporch
# - WoodDeckSF
# 
# Next to that I make some simplified features.
# - haspool
# - has2ndfloor
# - hasgarage
# - hasbsmt
# - hasfireplace

# In[ ]:


features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +
                                 features['1stFlrSF'] + features['2ndFlrSF'])

features['Total_Bathrooms'] = (features['FullBath'] + (0.5*features['HalfBath']) + 
                               features['BsmtFullBath'] + (0.5*features['BsmtHalfBath']))

features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +
                              features['EnclosedPorch'] + features['ScreenPorch'] +
                             features['WoodDeckSF'])


#simplified features
features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# ## Creating Dummies
# Since sklearn lm.fit() does not accept strings we have to convert our objects to dummy variables. 

# In[ ]:


features.shape


# In[ ]:


final_features = pd.get_dummies(features).reset_index(drop=True)
final_features.shape


# Now we resplit the model in test and train

# In[ ]:


y.shape


# In[ ]:


X = final_features.iloc[:len(y),:]
testing_features = final_features.iloc[len(X):,:]

print(X.shape)
print(testing_features.shape)


# ## Overfitting prevention
# 
# ### Outliers
# Let's do a little bit more in-depth and rigorous analysis first on outliers. I'll employ Leave-One-Out methodology with OLS to find which points have a significant effect on our model fit.  

# In[ ]:


import statsmodels.api as sm

#ols = sm.OLS(endog = y, exog = X)

#fit = ols.fit()
#test2 = fit.outlier_test()['bonf(p)']


# In[ ]:


outliers = list(test2[test2<1e-3].index) 

outliers

#print(test[test<1e-3])


# In[ ]:


outliers = [30, 88, 462, 631, 1322]


# So we find that these are outliers. Let's delete these.

# In[ ]:


X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])


# ### Dummy levels

# To prevent overfitting I'll also remove columns that have more than 97% 1 or 0 after doing pd.get_dummies.

# In[ ]:


overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 >99.94:
        overfit.append(i)


# In[ ]:


overfit = list(overfit)
overfit


# Let's drop these from 'X' and 'testing_features'. Let's also drop MSZoning_C (all). It has about 99.44% zeros but unlike others with that kind of percentage it's being included in my lasso/ridge/elasticnet models with quite strong coefficient sizes.

# In[ ]:


overfit.append('MSZoning_C (all)')


# In[ ]:


overfit


# In[ ]:


X.drop(overfit,axis=1,inplace=True)
testing_features.drop(overfit,axis=1,inplace=True)


# In[ ]:


print(X.shape)
print(testing_features.shape)


# ## Baseline model
# 
# ### Full Model w/ kfold cross validation
# 
# Let's build a baseline linear regression model to benchmark our feature selected models and advanced models on.
# 
# I decided not to do a manual train/test split but instead rely completely on 10-fold cross-validation for every model including our benchmark.
# 

# Our in-class benchmark has an RMSE of ~0.14 which is the goal to beat but I will rebuild a benchmark model in this notebook too.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

#Build our model method
lm = LinearRegression()

#Build our cross validation method
kfolds = KFold(n_splits=10, shuffle=True, random_state=23)

#build our model scoring function
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, 
                                   scoring="neg_mean_squared_error", 
                                   cv = kfolds))
    return(rmse)


#second scoring metric
def cv_rmsle(model):
    rmsle = np.sqrt(np.log(-cross_val_score(model, X, y,
                                           scoring = 'neg_mean_squared_error',
                                           cv=kfolds)))
    return(rmsle)


# Let's fit our first model

# In[ ]:


benchmark_model = make_pipeline(RobustScaler(),
                                lm).fit(X=X, y=y)
cv_rmse(benchmark_model).mean()


# ### Visualizing baseline model
# Let's see how the residuals and predictions vs actual values are distributed. Here I should note this looks absolutely ridiculous for a reason I can't figure out yet. Basically, for some reason, my baseline model gives an incredibly high residual error. I believe this is due to the fact that the dimensionality of my model is crazy high compared to the amount of data (~1500 rows vs ~320 columns). This gets reduced down with feature selection but the baseline model includes all which leads to a ton of multicollinearity causing high RMSE values. 

# # 3. Feature Selection
# 
# Before starting this section it should be noted that I will try to be extra careful not to create contamination during feature selection. Meaning that I will select features constrained per fold in my cross-validation to ensure no data leakage happens.
# 
# ## Filter methods
# 
# ### Coefficient importance
# 

# In[ ]:


coeffs = pd.DataFrame(list(zip(X.columns, benchmark_model.steps[1][1].coef_)), columns=['Predictors', 'Coefficients'])

coeffs.sort_values(by='Coefficients', ascending=False)


# ## Embedded methods
# 
# ### Ridge Regression (L2 penalty)
# 

# In[ ]:


from sklearn.linear_model import RidgeCV

def ridge_selector(k):
    ridge_model = make_pipeline(RobustScaler(),
                                RidgeCV(alphas = [k],
                                        cv=kfolds)).fit(X, y)
    
    ridge_rmse = cv_rmse(ridge_model).mean()
    return(ridge_rmse)


# In[ ]:


r_alphas = [.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 20, 30, 50, 60, 70, 80]

ridge_scores = []
for alpha in r_alphas:
    score = ridge_selector(alpha)
    ridge_scores.append(score)


# In[ ]:


plt.plot(r_alphas, ridge_scores, label='Ridge')
plt.legend('center')
plt.xlabel('alpha')
plt.ylabel('score')

ridge_score_table = pd.DataFrame(ridge_scores, r_alphas, columns=['RMSE'])
ridge_score_table


# In[ ]:


alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

ridge_model2 = make_pipeline(RobustScaler(),
                            RidgeCV(alphas = alphas_alt,
                                    cv=kfolds)).fit(X, y)

cv_rmse(ridge_model2).mean()


# In[ ]:


ridge_model2.steps[1][1].alpha_


# ### Lasso Regression (L1 penalty)

# In[ ]:


from sklearn.linear_model import LassoCV


alphas = [0.00005, 0.0001, 0.0003, 0.0005, 0.0007, 
          0.0009, 0.01]
alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
           0.0006, 0.0007, 0.0008]


lasso_model2 = make_pipeline(RobustScaler(),
                             LassoCV(max_iter=1e7,
                                    alphas = alphas2,
                                    random_state = 42)).fit(X, y)


# In[ ]:


scores = lasso_model2.steps[1][1].mse_path_

plt.plot(alphas2, scores, label='Lasso')
plt.legend(loc='center')
plt.xlabel('alpha')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()


# In[ ]:


lasso_model2.steps[1][1].alpha_


# In[ ]:


cv_rmse(lasso_model2).mean()


# In[ ]:


coeffs = pd.DataFrame(list(zip(X.columns, lasso_model2.steps[1][1].coef_)), columns=['Predictors', 'Coefficients'])


# In[ ]:


used_coeffs = coeffs[coeffs['Coefficients'] != 0].sort_values(by='Coefficients', ascending=False)
print(used_coeffs.shape)
print(used_coeffs)


# In[ ]:


used_coeffs_values = X[used_coeffs['Predictors']]
used_coeffs_values.shape


# In[ ]:


overfit_test2 = []
for i in used_coeffs_values.columns:
    counts2 = used_coeffs_values[i].value_counts()
    zeros2 = counts2.iloc[0]
    if zeros2 / len(used_coeffs_values) * 100 > 99.5:
        overfit_test2.append(i)
        
overfit_test2


# ### Elastic Net (L1 and L2 penalty)
# One of the issues with Lasso is that it's likely to pick, from correlated features, one at random. Elastic net would pick both. Its a bit of a mix between ridge and lasso. I decided to include it since R's implementation of ridge regression actually invovles some elasticNet properties. 

# In[ ]:


from sklearn.linear_model import ElasticNetCV

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elastic_cv = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                        cv=kfolds, l1_ratio=e_l1ratio))

elastic_model3 = elastic_cv.fit(X, y)


# In[ ]:


cv_rmse(elastic_model3).mean()


# In[ ]:


print(elastic_model3.steps[1][1].l1_ratio_)
print(elastic_model3.steps[1][1].alpha_)


# ### Xgboost
# The project I made this notebook for we weren't allowed to use more advanced algorithms than lasso, ridge, elasticnet. This was added later to see if I could improve my score.
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb
from xgboost import XGBRegressor


# Belows function was used to obtain the optimal boosting rounds. This is accomplished useing xgb.cv's early stopping. 

# In[ ]:


from sklearn.metrics import mean_squared_error

def modelfit(alg, dtrain, target, useTrainCV=True, 
             cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, 
                              label=y.values)
        
        print("\nGetting Cross-validation result..")
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,metrics='rmse', 
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval = True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    print("\nFitting algorithm to data...")
    alg.fit(dtrain, target, eval_metric='rmse')
        
    #Predict training set:
    print("\nPredicting from training data...")
    dtrain_predictions = alg.predict(dtrain)
        
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(mean_squared_error(target.values,
                                             dtrain_predictions)))



# Gridsearching gave me optimal parameters for XGBoost

# In[ ]:


xgb3 = XGBRegressor(learning_rate =0.01, n_estimators=3460, max_depth=3,
                     min_child_weight=0 ,gamma=0, subsample=0.7,
                     colsample_bytree=0.7,objective= 'reg:linear',
                     nthread=4,scale_pos_weight=1,seed=27, reg_alpha=0.00006)

xgb_fit = xgb3.fit(X, y)


# ### Support Vector Regression

# Gridsearching gave me optimal C and gamma for SVR.

# In[ ]:


from sklearn import svm
svr_opt = svm.SVR(C = 100000, gamma = 1e-08)

svr_fit = svr_opt.fit(X, y)


# ### LightGBM

# In[ ]:


from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[ ]:


cv_rmse(lgbm_model).mean()


# In[ ]:


lgbm_fit = lgbm_model.fit(X, y)


# ## Ensemble methods
# Let's see if I can get a better performance on the test data by employing ensemble methods. To stay in the constraints of the exercise I won't employ stronger models but instead combine three models.
# 
# - LassoCV
# - RidgeCV
# - Elasticnet
# 
# Experimenting with averaging cost a lot of time since local RMSE and kaggle RMSLE are disconnected at this point. Basically I am optimizing the tradeoff between under and over fitting.
# 
# First I'll build a meta-regressor through a process called stacking generalizations which trains a model on a part of the training set (it gets split first into a new training set and a holdout set). Then the algorithm test these models on the holdout set and uses these predictions (called out-of-fold predictions) as input for the 'meta model'. Below is a grahpical representation of the process.

# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url = "http://i.imgur.com/QBuDOjs.jpg")


# ### Ensemble 1 - Stacking Generalization
# To try to eek out more performance of our already decent rank let's try Stacking Generalization, I tried a few different options. Vecstack package from python seemed to be implementing it incorrectly, so instead I decided to use the mlxtend package.

# In[ ]:


from mlxtend.regressor import StackingCVRegressor
from sklearn.pipeline import make_pipeline

#setup models
ridge = make_pipeline(RobustScaler(), 
                      RidgeCV(alphas = alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7, alphas = alphas2,
                              random_state = 42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), 
                           ElasticNetCV(max_iter=1e7, alphas=e_alphas, 
                                        cv=kfolds, l1_ratio=e_l1ratio))

lightgbm = make_pipeline(RobustScaler(),
                        LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11))

xgboost = make_pipeline(RobustScaler(),
                        XGBRegressor(learning_rate =0.01, n_estimators=3460, 
                                     max_depth=3,min_child_weight=0 ,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective= 'reg:linear',nthread=4,
                                     scale_pos_weight=1,seed=27, 
                                     reg_alpha=0.00006))


#stack
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, 
                                            xgboost, lightgbm), 
                               meta_regressor=xgboost,
                               use_features_in_secondary=True)

#prepare dataframes
stackX = np.array(X)
stacky = np.array(y)


# In[ ]:


#scoring 

print("cross validated scores")

for model, label in zip([ridge, lasso, elasticnet, xgboost, lightgbm, stack_gen],
                     ['RidgeCV', 'LassoCV', 'ElasticNetCV', 'xgboost', 'lightgbm',
                      'StackingCVRegressor']):
    
    SG_scores = cross_val_score(model, stackX, stacky, cv=kfolds,
                               scoring='neg_mean_squared_error')
    print("RMSE", np.sqrt(-SG_scores.mean()), "SD", scores.std(), label)


# In[ ]:


stack_gen_model = stack_gen.fit(stackX, stacky)


# ### Ensemble 2 - averaging
# Final averaging weights are mostly trial and error as at this point my local scores were so completely detached from my real kaggle score. In the end I felt that SVR wasn't helping my score so it's not included in my final predictions.

# In[ ]:


em_preds = elastic_model3.predict(testing_features)
lasso_preds = lasso_model2.predict(testing_features)
ridge_preds = ridge_model2.predict(testing_features)
stack_gen_preds = stack_gen_model.predict(testing_features)
xgb_preds = xgb_fit.predict(testing_features)
svr_preds = svr_fit.predict(testing_features)
lgbm_preds = lgbm_fit.predict(testing_features)


# In[ ]:


stack_preds = ((0.2*em_preds) + (0.1*lasso_preds) + (0.1*ridge_preds) + 
               (0.2*xgb_preds) + (0.1*lgbm_preds) + (0.3*stack_gen_preds))


# ## Actual predictions for Kaggle
# I transform the predictions back to normal values because the model is trained with logSalePrice.

# In[ ]:


submission = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


submission.iloc[:,1] = np.expm1(stack_preds)


# In[ ]:


submission.to_csv("final_submission.csv", index=False)

