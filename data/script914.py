
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Ridge, Lasso, SGDRegressor
from sklearn.metrics import  make_scorer,  mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import skew, skewtest, norm
from xgboost.sklearn import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')



# In[ ]:


#Lets look at Sale Price which is the dependent variable here

sns.distplot(train_data['SalePrice'], fit=norm);


# In[ ]:


#This is not normally distributed and is skewed
print()
print("Skew is: %f" % train_data['SalePrice'].skew()) 


# In[ ]:


#Lets look at Price vs Living area - the bigger the house usually the more money its worth

plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], c = "blue", marker = "s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()



# In[ ]:


#Found a couple of outliers - low sale price and Large living area  -  Lets get rid of those

train_data = train_data[train_data['GrLivArea'] < 4500]



# In[ ]:


#We only want to look at "Normal" Sales

train_data = train_data[train_data['SaleCondition']== 'Normal']


# In[ ]:


#Lets look at Price vs Living area - the bigger the house usually the more money its worth

plt.scatter(train_data['GrLivArea'], train_data['SalePrice'], c = "blue", marker = "s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()


# In[ ]:


# Lets look at correlations here of the variables

corrmatrix = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix, vmax=.8, square=True);


# In[ ]:


#Correlation values

k = 10 #number of variables for heatmap
cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


#  Going to bin Neighborhood into quartiles by Median SalePrice 

Neighborhood = train_data.groupby('Neighborhood')
Neighborhood['SalePrice'].median()


# In[ ]:


# Based on the above we drop 4 items as well as ID since we dont need that - we will save it as we need it later

ID_train = train_data['Id']
ID_test = test_data['Id']


### These pop out - we are looking for variables that are telling us the same thing - multicollinearity

#TotalBsmntSF and 1stFlrSF are very highly correlated - basement sits below 1st floor
#GarageCars and GarageArea are veryhighly correlated - bigger the area more cars can fit in
#GarageYrBlt and YearBuilt are very highly correlated - usually build a garage same time as the house
#TotRmsAbvGrd and GrLivArea are very highly correlated - more area the more rooms

# Lets drop one of these from each pairing - how to decide which one look at correlation vs sale price and drop lower one

train_data.drop("Id", axis = 1, inplace = True)
test_data.drop("Id", axis = 1, inplace = True)
train_data.drop("TotRmsAbvGrd", axis = 1, inplace = True)
test_data.drop("TotRmsAbvGrd", axis = 1, inplace = True)
train_data.drop("GarageYrBlt", axis = 1, inplace = True)
test_data.drop("GarageYrBlt", axis = 1, inplace = True)
train_data.drop("GarageArea", axis = 1, inplace = True)
test_data.drop("GarageArea", axis = 1, inplace = True)
train_data.drop("1stFlrSF", axis = 1, inplace = True)
test_data.drop("1stFlrSF", axis = 1, inplace = True)






# In[ ]:


# Log transform the Sale price to make it more normally distributed and then drop it from the features

train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
y = train_data['SalePrice']

train_data.drop("SalePrice", axis = 1, inplace = True)


# In[ ]:


# Get number of records and set variable to identify split point once we combine them

print(train_data.shape)
print(test_data.shape)
ntrain = train_data.shape[0]
ntest = test_data.shape[0]
print(ntrain)


# In[ ]:


Combined_data = pd.concat([train_data,test_data]).reset_index(drop=True)


# In[ ]:


print("Combined size is : {}".format(Combined_data.shape))


# In[ ]:


#missing data - see what needs to be cleaned up

total = Combined_data.isnull().sum().sort_values(ascending=False)
percent = (Combined_data.isnull().sum()/Combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(40)


# In[ ]:


#Going to drop fields where most of the data is missing and is irrevelant in sale price of a home

# Missing too much data

Combined_data.drop("PoolQC", axis = 1, inplace = True)
Combined_data.drop("MiscFeature", axis = 1, inplace = True)
Combined_data.drop("Alley", axis = 1, inplace = True)


# In[ ]:


#Missing data handling

#Combined_data["GarageYrBlt"].value_counts(dropna=False)  this helps look at the distribution

# LotFrontage : NA most likely means no lot frontage
Combined_data["LotFrontage"].fillna(0, inplace=True) 

# Fence : NA means no fence
Combined_data["Fence"].fillna("None", inplace=True)

# FireplaceQu : NA means no fireplace
Combined_data["FireplaceQu"].fillna("None", inplace=True)

# GarageCond : NA means no garage
Combined_data["GarageCond"].fillna("None", inplace=True)

# GarageFinish : NA means no garage
Combined_data["GarageFinish"].fillna("None", inplace=True)

# GarageQual : NA means no garage
Combined_data["GarageQual"].fillna("None", inplace=True)

# GarageType : NA means no garage
Combined_data["GarageType"].fillna("None", inplace=True)


# BsmtFinType2 : NA means no basement
Combined_data["BsmtFinType2"].fillna("None", inplace=True)

# BsmtExposure : NA means no basement
Combined_data["BsmtExposure"].fillna("None", inplace=True)

# BsmtQual : NA means no basement
Combined_data["BsmtQual"].fillna("None", inplace=True)

# BsmtFinType1 : NA means no basement
Combined_data["BsmtFinType1"].fillna("None", inplace=True)

# BsmtCond: NA means no basement
Combined_data["BsmtCond"].fillna("None", inplace=True)

# MasVnrType: NA means none
Combined_data["MasVnrType"].fillna("None", inplace=True)

# MasVnrArea : NA most likely means 0
Combined_data["MasVnrArea"].fillna(0, inplace=True) 

# MasVnrArea : NA most likely means 0
Combined_data["Electrical"].fillna("SBrkr", inplace=True) 

# BsmtHalfBath : NA most likely means 0
Combined_data["BsmtHalfBath"].fillna(0, inplace=True)

# BsmtFullBath : NA most likely means 0
Combined_data["BsmtFullBath"].fillna(0, inplace=True)

# BsmtFinSF1 : NA most likely means 0
Combined_data["BsmtFinSF1"].fillna(0, inplace=True)

# BsmtFinSF2 : NA most likely means 0
Combined_data["BsmtFinSF2"].fillna(0, inplace=True)

# BsmtUnfSF : NA most likely means 0
Combined_data["BsmtUnfSF"].fillna(0, inplace=True)

# TotalBsmtSF: NA most likely means 0
Combined_data["TotalBsmtSF"].fillna(0, inplace=True)

# GarageCars : NA most likely means 0
Combined_data["GarageCars"].fillna(0, inplace=True)

## GarageArea : NA most likely means 0
#Combined_data["GarageArea"].fillna(0, inplace=True)

# BsmtCond: NA means no basement
Combined_data["Utilities"].fillna(0, inplace=True)

# BsmtCond: NA means no basement
Combined_data["Functional"].fillna(0, inplace=True)

# BsmtCond: NA means no basement
Combined_data["KitchenQual"].fillna(0, inplace=True)

#MSZoning (The general zoning classification) : 'RL' is by far the most common value.

Combined_data["MSZoning"].fillna("RL", inplace=True)

#SaleType : Fill in again with most frequent which is "WD"

Combined_data["SaleType"].fillna("WD", inplace=True)

#Exterior 1 and 2 : Fill in again with most frequent 

Combined_data['Exterior1st'] = Combined_data['Exterior1st'].fillna(Combined_data['Exterior1st'].mode()[0])
Combined_data['Exterior2nd'] = Combined_data['Exterior2nd'].fillna(Combined_data['Exterior2nd'].mode()[0])



# In[ ]:


# Some numerical features are actually really categories - switch them back

Combined_data = Combined_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                  7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                     })



# In[ ]:


# Encode some categorical features as ordered numbers when there is information in the order

Combined_data = Combined_data.replace({"BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageFinish" : {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )


# In[ ]:


#Adding new features

Combined_data['Total_Home_Quality'] = Combined_data['OverallQual'] + Combined_data['OverallCond']
Combined_data['Total_Basement_Quality'] = Combined_data['BsmtQual'] + Combined_data['BsmtCond']
Combined_data['Total_Basement_FinshedSqFt'] = Combined_data['BsmtFinSF1'] + Combined_data['BsmtFinSF2']
Combined_data['Total_Exterior_Quality'] = Combined_data['ExterQual'] + Combined_data['ExterCond']
Combined_data['Total_Garage_Quality'] = Combined_data['GarageCond'] + Combined_data['GarageQual'] + Combined_data['GarageFinish']
Combined_data['Total_Basement_FinshType'] = Combined_data['BsmtFinType1'] + Combined_data['BsmtFinType2']
Combined_data['Total_Garage_Quality'] = Combined_data['GarageCond'] + Combined_data['GarageQual'] + Combined_data['GarageFinish']
Combined_data['Total_Basement_FinshType'] = Combined_data['BsmtFinType1'] + Combined_data['BsmtFinType2']
Combined_data['Total_Bathrooms'] = Combined_data['BsmtFullBath'] + (Combined_data['BsmtHalfBath'] * 0.5) + Combined_data['FullBath'] + (Combined_data['HalfBath'] * 0.5)
Combined_data['Total_Land_Quality'] = Combined_data['LandSlope'] + Combined_data['LotShape']


#Drop the individual components from above to avoid multicolinearity

Combined_data.drop("OverallQual", axis = 1, inplace = True)
Combined_data.drop("OverallCond", axis = 1, inplace = True)
Combined_data.drop("BsmtQual", axis = 1, inplace = True)
Combined_data.drop("BsmtCond", axis = 1, inplace = True)
Combined_data.drop("BsmtFinSF1", axis = 1, inplace = True)
Combined_data.drop("BsmtFinSF2", axis = 1, inplace = True)
Combined_data.drop("ExterQual", axis = 1, inplace = True)
Combined_data.drop("ExterCond", axis = 1, inplace = True)
Combined_data.drop("GarageCond", axis = 1, inplace = True)
Combined_data.drop("GarageQual", axis = 1, inplace = True)
Combined_data.drop("GarageFinish", axis = 1, inplace = True)
Combined_data.drop("BsmtFinType1", axis = 1, inplace = True)
Combined_data.drop("BsmtFinType2", axis = 1, inplace = True)
Combined_data.drop("BsmtFullBath", axis = 1, inplace = True)
Combined_data.drop("BsmtHalfBath", axis = 1, inplace = True)
Combined_data.drop("FullBath", axis = 1, inplace = True)
Combined_data.drop("HalfBath", axis = 1, inplace = True)
Combined_data.drop("LandSlope", axis = 1, inplace = True)
Combined_data.drop("LotShape", axis = 1, inplace = True)

#also dropping LandContour variable as it is contained in LandSlope

Combined_data.drop("LandContour", axis = 1, inplace = True)


# In[ ]:


# Binning neighborhood into quartiles based on SalePrice

Combined_data = Combined_data.replace({"Neighborhood" : {
"MeadowV" : 0,
"IDOTRR" : 0,
"BrDale" : 0,
"OldTown" : 0,
"Edwards" : 0,
"BrkSide" : 0,
"Sawyer" : 0,
"Blueste" : 1,
"SWISU" : 1,
"NAmes" : 1,
"NPkVill" : 1,
"Mitchel" : 1,
"SawyerW" : 1,
"Gilbert" : 2,
"NWAmes" : 2,
"Blmngtn" : 2,
"CollgCr" : 2,
"ClearCr" : 2,
"Crawfor" : 2,
"Veenker" : 3,
"Somerst" : 3,
"Timber" : 3,
"StoneBr" : 3,
"NoRidge" : 3,
"NridgHt" : 3}})
    




# In[ ]:


#Check again to see that we are all cleaned up

new_total = Combined_data.isnull().sum().sort_values(ascending=False)
new_percent = (Combined_data.isnull().sum()/Combined_data.isnull().count()).sort_values(ascending=False)
new_missing_data = pd.concat([new_total, new_percent], axis=1, keys=['Total', 'Percent'])
new_missing_data.head(10)


# In[ ]:


#Use to check distributions

#sns.distplot(Combined_data['Total_Basement_FinshedSqFt']>0,fit=norm)


# In[ ]:


# Check pure numerical features (not ordinal) for skewed distributions and
# need to be normalized by taking the log. Value if skew >1 (indicates skewnewss)

Skewed_Feature_Check = ['LotArea','MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', 
                        'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 
                       'PoolArea', 'MiscVal', 'Total_Basement_FinshedSqFt']

#Skewed_Feature_Check = ['GrLivArea', 'LotArea', 'TotalBsmtSF', '1stFlrSF']

for feature in Skewed_Feature_Check:
    
    print((feature), skew(Combined_data[feature]), skewtest(Combined_data[feature]))
    
    from scipy.special import boxcox1p

    lam = 0.15
    
    
    Combined_data[feature] = boxcox1p(Combined_data[feature], lam)
    
    
    
    #Combined_data[feature] = np.log1p(Combined_data[feature])
         
        
    
#skewed = train_df_munged[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
#skewed = skewed[skewed > 0.75]
#skewed = skewed.index



# In[ ]:


# split into categorical and numberical features

categorical_features = Combined_data.select_dtypes(include = ["object"]).columns
numerical_features = Combined_data.select_dtypes(exclude = ["object"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))


# In[ ]:


# convert into data frames

Combined_data_numerical = Combined_data[numerical_features]
Combined_data_categorical = Combined_data[categorical_features]


# In[ ]:


corrmatrix_combined = Combined_data_numerical.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix_combined, vmax=.8, square=True);


# In[ ]:


#Fireplaces and Fireplace Quality highly correlated - examine


# In[ ]:


Combined_data['FireplaceQu'].hist()


# In[ ]:


Combined_data['Fireplaces'].hist()


# In[ ]:


# Lets drop Fireplace Quality as it is more or less dominated by no value making it correlated with # Fireplaces

Combined_data.drop("FireplaceQu", axis = 1, inplace = True)


# In[ ]:


# converting categorical to numeric values

Combined_data_categorical = pd.get_dummies(Combined_data_categorical,drop_first=True)


# In[ ]:


#Combine them back together

Combined_data = pd.concat([Combined_data_categorical, Combined_data_numerical], axis = 1)


# In[ ]:


# check shape again - we will have added a lot of features

print("Combined size is : {}".format(Combined_data.shape))


# In[ ]:


#resplit the data into training and test sets again

train_data = Combined_data[:ntrain]
test_data = Combined_data[ntrain:]
test_data = test_data.reset_index(drop=True)


# In[ ]:


# check that it matches original length

print(train_data.shape)
print(test_data.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size = 0.20, random_state = 1)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))


# In[ ]:


#Scale the data after it is split to avoid "leakage" into the test set
# Going to try to use 
# http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

scaler = RobustScaler()
#X_train.loc[:, numerical_features] = scaler.fit_transform(X_train.loc[:, numerical_features])
#X_test.loc[:, numerical_features] = scaler.transform(X_test.loc[:, numerical_features])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#scaler = RobustScaler().fit(X_train[i].values.reshape(-1, 1))
#X_train[i] = sc.transform(X_train[i].values.reshape(-1, 1))
#X_test[i] = sc.transform(X_test[i].values.reshape(-1, 1))




# In[ ]:


# Run Cross Val Score on basic model with no parameter tuning

for Model in [LinearRegression, Ridge, Lasso, XGBRegressor]:
    model = Model()
    print('%s: %s' % (Model.__name__,
                      np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()))
                      


# Not sure what is going on with LinearRegression as score is obviously not good
# 
# 

# In[ ]:


#Proceed with Ridge and Lasso and will fine tune alpha parameter

alphas = [.0001, .0003, .0005, .0007, .0009, .01, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50]



plt.figure(figsize=(5, 3))

for model in [Lasso, Ridge]:
  
    scores = [np.sqrt(-cross_val_score(model(alpha), X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()
              for alpha in alphas]
             
    plt.plot(alphas, scores, label=model.__name__)

    plt.legend(loc='center')
    plt.xlabel('alpha')
    plt.ylabel('cross validation score')
    plt.tight_layout()
    plt.show()


# In[ ]:


xgbreg = XGBRegressor(nthreads=-1, booster = 'gblinear') 
np.sqrt(-cross_val_score(xgbreg, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()


# In[ ]:


#XGBRegressor Tuning

params = {'learning_rate':[1,.01],
          #'gamma':[i/10.0 for i in range(3,6)], 
          'reg_alpha':[.09],
          #'reg_lambda':[i/10.0 for i in range(1,10)],
          'n_estimators': [1000] 
         }


xgbreg = XGBRegressor(nthreads=-1, booster = 'gblinear')  

from sklearn.model_selection import GridSearchCV

xgbreg_model = GridSearchCV(xgbreg, params, n_jobs=1, scoring="neg_mean_squared_error", cv=10)  
xgbreg_model.fit(X_train, y_train )    
  
xgbreg_model.best_estimator_


# In[ ]:


np.sqrt(-xgbreg_model.best_score_)


# Looks like a small alpha value for Lasso (.01) and for Ridge a value of 10

# In[ ]:


#Generating scores for training and test sets using tuned alpha parameters

Lasso_model = Lasso(alpha=.0045)
Ridge_model = Ridge(alpha=10)

print("Lasso Train")
print((np.sqrt(-cross_val_score(Lasso_model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()
print("Lasso Test")
print((np.sqrt(-cross_val_score(Lasso_model, X_test, y_test, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()
print("Ridge Train")
print((np.sqrt(-cross_val_score(Ridge_model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()
print("Ridge Test")
print((np.sqrt(-cross_val_score(Ridge_model, X_test, y_test, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()
print("XGBRegressor")
print((np.sqrt(-cross_val_score(xgbreg_model.best_estimator_, X_train, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()
print("XGBRegressorr")
print()
print((np.sqrt(-cross_val_score(xgbreg_model.best_estimator_, X_test, y_test, scoring="neg_mean_squared_error", cv = 10)).mean()))


# Results seem pretty good here. Not too much overfitting occuring at all.

# In[ ]:


#Fit Lasso model to train data and make predictions for both training and test sets and check residuals

Lasso_model.fit(X_train, y_train)

y_train_pred = Lasso_model.predict(X_train)
y_test_pred = Lasso_model.predict(X_test)
  
# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Lasso")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Lasso")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()  



    
        
    
        


# In[ ]:


# Plot important coefficients

coefs = pd.Series(Lasso_model.coef_, index = train_data.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(15),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()


# In[ ]:


imp_coefs


# In[ ]:


#Fit Ridge model to train data and make predictions for both training and test sets and check residuals

Ridge_model.fit(X_train, y_train)

y_train_pred = Ridge_model.predict(X_train)
y_test_pred = Ridge_model.predict(X_test)

# Plot residuals
plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Ridge")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Ridge")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()  


# Next step is to retrain the fitted models on the entire training set and then make predictions
# on the true Test set for submission.

# In[ ]:


#rescale the data again but this time based on the entire training set

scaler_final = RobustScaler()
train_data = scaler_final.fit_transform(train_data)


# In[ ]:


# intiate and fit the final Lasso model

Lasso_model_final= Lasso(alpha=.0045)
Lasso_model_final.fit(train_data, y)


# In[ ]:


# intiate and fit the final Ridge model

Ridge_model_final= Ridge(alpha=10)
Ridge_model_final.fit(train_data, y)


# In[ ]:


XGBRegressor_final = xgbreg_model.best_estimator_
XGBRegressor_final.fit(train_data, y)


# In[ ]:


# Scale the test numerical data using the scaler calculated on the entire training set

test_data = scaler_final.transform(test_data)


# In[ ]:


labels_lasso = np.expm1(Lasso_model_final.predict(test_data))
labels_ridge = np.expm1(Ridge_model_final.predict(test_data))
labels_xgbregressor = np.expm1(XGBRegressor_final.predict(test_data))


# In[ ]:


## Saving prediction file to CSV - basic model with parameter tuning

pd.DataFrame({'Id': ID_test, 'SalePrice': labels_lasso}).to_csv('LassoPredictions.csv', index =False) 
pd.DataFrame({'Id': ID_test, 'SalePrice': labels_ridge}).to_csv('RidgePredictions.csv', index =False) 
pd.DataFrame({'Id': ID_test, 'SalePrice': labels_xgbregressor}).to_csv('XgbregressorPredictions.csv', index =False) 


# In[ ]:


#Advanced feature selection testing


# In[ ]:


from sklearn.feature_selection import RFECV


# In[ ]:


# Create the RFE object and compute a cross-validated score - go back to the original training split and original tuned Lasso model

rfecv = RFECV(estimator=Lasso_model_final, step=1, cv=KFold(10),
              scoring='neg_mean_squared_error')

rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), np.sqrt(-rfecv.grid_scores_))
plt.show()



# In[ ]:


# Gets the index values of the features that you want to keep

Index = rfecv.get_support(indices=True)
Index


# In[ ]:


#build the new dataframes using those selected features

X_train_new = X_train[:,Index]
X_test_new = X_test[:,Index]
Test_data_new = test_data[:,Index]
Train_data_new = train_data[:,Index]


# In[ ]:


Train_data_new.shape


# In[ ]:


# retesting the Lasso model again using the same process but with reduced feature data set

print("Lasso Train")
print((np.sqrt(-cross_val_score(Lasso_model_final, X_train_new, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()
print("Lasso Test")
print((np.sqrt(-cross_val_score(Lasso_model_final, X_test_new, y_test, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()


# In[ ]:


# Scores are better using reduced features on their own as the dataset


# In[ ]:




print("Ridge Train")
print((np.sqrt(-cross_val_score(Ridge_model_final, X_train_new, y_train, scoring="neg_mean_squared_error", cv = 10)).mean()))
print()
print("Ridge Test")
print((np.sqrt(-cross_val_score(Ridge_model_final, X_test_new, y_test, scoring="neg_mean_squared_error", cv = 10)).mean()))


# In[ ]:


# Scores are actually worse using reduced features on their own as the dataset


# In[ ]:


#Fit the model to the entire training set using reduced features

Lasso_model_final.fit(Train_data_new, y)


# In[ ]:


Ridge_model_final.fit(Train_data_new, y)


# In[ ]:


Final_labels_Lasso = np.expm1(Lasso_model_final.predict(Test_data_new))
Final_labels_Ridge = np.expm1(Ridge_model_final.predict(Test_data_new))


# In[ ]:


## Saving prediction file to CSV - basic model with parameter tuning

pd.DataFrame({'Id': ID_test, 'SalePrice': Final_labels_Lasso}).to_csv('Lassorfecv.csv', index =False) 

pd.DataFrame({'Id': ID_test, 'SalePrice': Final_labels_Ridge}).to_csv('Ridgerfecv.csv', index =False) 


# In[ ]:


#with pd.option_context('display.max_columns', None):
 #   display(Combined_data.head())

