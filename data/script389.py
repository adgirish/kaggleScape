
# coding: utf-8

# ## Introduction
# 
# **_Poonam Ligade_**
# 
# *30th Jan 2017*
# 
# I am trying to predict for how much money each house can be sold??
# 
# In this mainly we will look at data exploration and visulisation part
# 
# EDA is often most tedious and boring job.
# 
# But the more time you spend here on understanding, cleaning and preparing data the better fruits your predictive model will bare!!
# 
# Lets start.
# 
# 1) **Introduction**
# 
#   1. Import Libraries
#   2. Load data
#   3. Variable Identification
#   4. Run Statistical summaries
#   5. Correlation with target variable
# 
#  
# 2) **Missing values imputation**
# 
#   1. Figure out missing value columns
#   2. Fill out missing values
# 
# 
# 3) **Visualisations**
# 
#  1. Univariate Analysis
#  2. Bivariate Analysis

# **Import libraries**
# ====================

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)


# **Load train & test data**
# ====================

# In[ ]:


houses=pd.read_csv("../input/train.csv")
houses.head()


# In[ ]:


houses_test = pd.read_csv("../input/test.csv")
#transpose
houses_test.head()
#note their is no "SalePrice" column here which is our target varible.


# In[ ]:


#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset
#(rows,columns)
houses.shape


# There are total **1460 samples** which we can use to train model and **80 features** and **1 target variable.**

# In[ ]:


houses_test.shape
#1 column less because target variable isn't there in test set!


# *Variable Identification*
# -----------------------

# In[ ]:


#info method provides information about dataset like 
#total values in each column, null/not null, datatype, memory occupied etc
houses.info()


# In[ ]:


#How many columns with different datatypes are there?
houses.get_dtype_counts()


# In[ ]:


##Describe gives statistical information about numerical columns in the dataset
houses.describe()


# **Correlation in Data**
# ====================

# In[ ]:


corr=houses.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]


# OverallQual ,GrLivArea ,GarageCars,GarageArea ,TotalBsmtSF, 1stFlrSF     ,FullBath,TotRmsAbvGrd,YearBuilt, YearRemodAdd have more than 0.5 correlation with SalePrice. 
# 
# EnclosedPorch and KitchenAbvGr  have little negative correlation with target variable.
# 
# These can prove to be important features to predict SalePrice.
# 

# In[ ]:


#plotting correlations
num_feat=houses.columns[houses.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(houses[col].values, houses.SalePrice.values)[0,1])
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t Sale Price");


# In[ ]:


correlations=houses.corr()
attrs = correlations.iloc[:-1,:-1] # all except target

threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0])     .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), 
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

unique_important_corrs


# This shows multicollinearity.
# In regression, "multicollinearity" refers to features that are correlated with other features.  Multicollinearity occurs when your model includes multiple factors that are correlated not just to your target variable, but also to each other.
# 
# Problem:
# 
# Multicollinearity increases the standard errors of the coefficients.
# That means, multicollinearity makes some variables statistically insignificant when they should be significant.
# 
# To avoid this we can do 3 things:
# 
# 1. Completely remove those variables
# 2. Make new feature by adding them or by some other operation.
# 3. Use PCA, which will reduce feature set to small number of non-collinear features.
# 
# Reference:http://blog.minitab.com/blog/understanding-statistics/handling-multicollinearity-in-regression-analysis

# **Heatmap**
# -----------

# In[ ]:


corrMatrix=houses[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');


# As we saw above there are few feature which shows high multicollinearity from heatmap.
# Lets focus on yellow squares on diagonal line and few on the sides.
# 
# SalePrice and OverallQual
# 
# GarageArea and GarageCars
# 
# TotalBsmtSF and 1stFlrSF
# 
# GrLiveArea and TotRmsAbvGrd
# 
# YearBulit and GarageYrBlt
# 
# We have to create a single feature from them before we use them as predictors.
# 

# *Pivotal Features*
# ------------------------

# In[ ]:


houses[['OverallQual','SalePrice']].groupby(['OverallQual'],
as_index=False).mean().sort_values(by='OverallQual', ascending=False)


# In[ ]:


houses[['GarageCars','SalePrice']].groupby(['GarageCars'],
as_index=False).mean().sort_values(by='GarageCars', ascending=False)


# In[ ]:


houses[['Fireplaces','SalePrice']].groupby(['Fireplaces'],
as_index=False).mean().sort_values(by='Fireplaces', ascending=False)


# *Visualising Target variable*
# -----------------

# *Univariate Analysis*
# --------------------
# 
# How 1 single variable is distributed in numeric range.
# What is statistical summary of it.
# Is it positively skewed or negatively.

# In[ ]:


sns.distplot(houses['SalePrice'], color="r", kde=False)
plt.title("Distribution of Sale Price")
plt.ylabel("Number of Occurences")
plt.xlabel("Sale Price");


# Prices are right skewed and  graph shows some peakedness.

# In[ ]:


#skewness  

houses['SalePrice'].skew()


# In[ ]:


#kurtosis

houses['SalePrice'].kurt()


# In[ ]:


#there are some outliers.lets remove them.
upperlimit = np.percentile(houses.SalePrice.values, 99.5)
houses['SalePrice'].ix[houses['SalePrice']>upperlimit] = upperlimit

plt.scatter(range(houses.shape[0]), houses["SalePrice"].values,color='orange')
plt.title("Distribution of Sale Price")
plt.xlabel("Number of Occurences")
plt.ylabel("Sale Price");


# **Missing Value Imputation**
# ====================
# 
# 
# Missing values in the training data set can affect prediction or classification of a model negatively.
# 
# Also some machine learning algorithms can't accept missing data eg. SVM.
# 
# But filling missing values with mean/median/mode or using another predictive model to predict missing values is also a prediction which may not be 100% accurate, instead you can use models like Decision Trees and Random Forest which handle missing values very well.
# 
# Some of this part is based on this kernel:
# https://www.kaggle.com/bisaria/house-prices-advanced-regression-techniques/handling-missing-data

# In[ ]:


#lets see if there are any columns with missing values 
null_columns=houses.columns[houses.isnull().any()]
houses[null_columns].isnull().sum()


# In[ ]:


labels = []
values = []
for col in null_columns:
    labels.append(col)
    values.append(houses[col].isnull().sum())
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,50))
rects = ax.barh(ind, np.array(values), color='violet')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values");


# *Multivariate Analysis*
# --------------------
# 
# When we understand how 3 or more variables behave according to each other.

# *LotFrontage*
# -------------
# 
# We can see if there is some correlation between LotArea and LotFrontage

# In[ ]:


houses['LotFrontage'].corr(houses['LotArea'])


# This is not great, we will try some polynomial expressions like squareroot 

# In[ ]:


houses['SqrtLotArea']=np.sqrt(houses['LotArea'])
houses['LotFrontage'].corr(houses['SqrtLotArea'])


# 0.60 looks good to go.

# In[ ]:


sns.jointplot(houses['LotFrontage'],houses['SqrtLotArea'],color='gold');


# In[ ]:


filter = houses['LotFrontage'].isnull()
houses.LotFrontage[filter]=houses.SqrtLotArea[filter]


# *MasVnrType and MasVnrArea*
# ===========================

# In[ ]:


plt.scatter(houses["MasVnrArea"],houses["SalePrice"])
plt.title("MasVnrArea Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Mas Vnr Area in sq feet");


# In[ ]:


sns.boxplot("MasVnrType","SalePrice",data=houses);


# In[ ]:


houses["MasVnrType"] = houses["MasVnrType"].fillna('None')
houses["MasVnrArea"] = houses["MasVnrArea"].fillna(0.0)


# *Bivariate Analysis*
# --------------------
# 
# When we try to figure out how 2 parameters in dataset are related to each other. in the sense when one decreases, other also decreases or when one increases other also increases i.e Positive Correlation 
# 
# And  when one increases , other decreases or vice versa i .e Negative correlation.

# *Electrical*
# ------------

# In[ ]:


sns.boxplot("Electrical","SalePrice",data=houses)
plt.title("Electrical Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Electrical");


# In[ ]:


#We can replace missing values with most frequent ones.
houses["Electrical"] = houses["Electrical"].fillna('SBrkr')


# *Alley*
# -------

# In[ ]:


sns.stripplot(x=houses["Alley"], y=houses["SalePrice"],jitter=True);


# All missing value indicate that particular house doesn't have an alley access.we can replace it with 'None'.

# In[ ]:


houses["Alley"] = houses["Alley"].fillna('None')


# *Basement Features*
# -------------------

# In[ ]:


plt.scatter(houses["TotalBsmtSF"],houses["SalePrice"])
plt.title("Total Basement area in Square Feet Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Total Basement area in Square Feet");


# In[ ]:


#there are few outliers in total basement area lets remove them
upperlimit = np.percentile(houses.TotalBsmtSF.values, 99.5)
houses['TotalBsmtSF'].ix[houses['TotalBsmtSF']>upperlimit] = upperlimit

plt.scatter(houses.TotalBsmtSF, houses["SalePrice"].values,color='orange')
plt.title("TotalBsmtSF Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Total Basement in sq feet");


# In[ ]:


basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
houses[basement_cols][houses['BsmtQual'].isnull()==True]


# All categorical variables contains NAN whereas continuous ones have 0.
# So that means there is no basement for those houses.
# we can replace it with 'None'.

# In[ ]:


for col in basement_cols:
    if 'FinSF'not in col:
        houses[col] = houses[col].fillna('None')


# *Fireplaces*
# ------------

# In[ ]:


sns.factorplot("Fireplaces","SalePrice",data=houses,hue="FireplaceQu");


# Having 2 fireplaces increases house price and fireplace of Excellent quality is a big plus. 

# In[ ]:


#If fireplace quality is missing that means that house doesn't have a fireplace
houses["FireplaceQu"] = houses["FireplaceQu"].fillna('None')
pd.crosstab(houses.Fireplaces, houses.FireplaceQu)


# *Garages*
# ---------

# In[ ]:


sns.distplot(houses["GarageArea"],color='r', kde=False);


# In[ ]:


#GarageArea has got some outliers lets remove them.
upperlimit = np.percentile(houses.GarageArea.values, 99.5)
houses['GarageArea'].ix[houses['GarageArea']>upperlimit] = upperlimit

plt.scatter(houses.GarageArea, houses["SalePrice"].values,color='violet')
plt.title("Garage Area Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Garage Area in sq feet");


# In[ ]:


sns.violinplot(houses["GarageCars"],houses["SalePrice"])
plt.title("Garage Cars Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Number of Garage cars");


# In[ ]:


garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
houses[garage_cols][houses['GarageType'].isnull()==True]


# All garage related features are missing values in same rows.
# that means we can replace categorical variables with None and continuous ones with 0.

# In[ ]:


for col in garage_cols:
    if houses[col].dtype==np.object:
        houses[col] = houses[col].fillna('None')
    else:
        houses[col] = houses[col].fillna(0)


# *Pool*
# -----------------------

# In[ ]:


#If PoolArea is 0, that means that house doesn't have a pool.
#So we can replace PoolQuality with None.
houses["PoolQC"] = houses["PoolQC"].fillna('None')
sns.factorplot("PoolArea","SalePrice",data=houses,hue="PoolQC",kind='bar')
plt.title("Pool Area , Pool quality and SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Pool Area in sq feet");


# *Fence*
# -----------------------

# In[ ]:


sns.violinplot(houses["Fence"],houses["SalePrice"])
plt.title("Fence wrt SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Type of Fence");


# Fence has got 1179 null values.
# We can safely assume that those houses doesn't have a Fence and replace those values with None.

# In[ ]:


houses["Fence"] = houses["Fence"].fillna('None')


# *MiscFeature*
# -----------------------

# In[ ]:


sns.barplot(houses["MiscFeature"],houses["SalePrice"])
plt.title("Miscelleneous Features  Vs SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Type of Miscelleneous Features");


# In[ ]:


#Some houses don't have miscellaneous features like shed, Tennis court etc..
houses["MiscFeature"] = houses["MiscFeature"].fillna('None')


# In[ ]:


#Let's confirm that we have removed all missing values
houses[null_columns].isnull().sum()


# **Visualizations**
# ==================

# *MSZoning*
# -----------

# In[ ]:



labels = houses["MSZoning"].unique()
sizes = houses["MSZoning"].value_counts().values
explode=[0.1,0,0,0,0]
parcent = 100.*sizes/sizes.sum()
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]

colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']
patches, texts= plt.pie(sizes, colors=colors,explode=explode,
                        shadow=True,startangle=90)
plt.legend(patches, labels, loc="best")

plt.title("Zoning Classification")
plt.show()



sns.violinplot(houses.MSZoning,houses["SalePrice"])
plt.title("MSZoning wrt Sale Price")
plt.xlabel("MSZoning")
plt.ylabel("Sale Price");


# *1st Floor in square feet*
# --------------------------

# In[ ]:


plt.scatter(houses["1stFlrSF"],houses.SalePrice, color='red')
plt.title("Sale Price wrt 1st floor")
plt.ylabel('Sale Price (in dollars)')
plt.xlabel("1st Floor in square feet");


# *Ground Living Area w.r.t SalePrice*
# --------------------

# In[ ]:


plt.scatter( houses["GrLivArea"],houses["SalePrice"],color='purple')
plt.title("Sale Price wrt Ground living area")
plt.ylabel('Sale Price')
plt.xlabel("Ground living area");


# *SalePrice per square foot*
# --------------------

# In[ ]:


houses['SalePriceSF'] = houses['SalePrice']/houses['GrLivArea']
plt.hist(houses['SalePriceSF'], bins=15,color="gold")
plt.title("Sale Price per Square Foot")
plt.ylabel('Number of Sales')
plt.xlabel('Price per square feet');


# In[ ]:


#Average Sale Price per square feet 
print("$",houses.SalePriceSF.mean())


# *Garage Area*
# -------------

# In[ ]:


plt.scatter(houses["GarageArea"],houses.SalePrice, color='green')
plt.title("Sale Price vs Garage Area")
plt.ylabel('Sale Price(in dollars)')
plt.xlabel("Garage Area in sq foot");


# *Building , remodelling years and age of house*
# ----------------------------------------

# In[ ]:


sns.distplot(houses["YearBuilt"],color='seagreen', kde=False);


# In[ ]:


sns.distplot(houses["YearRemodAdd"].astype(int),color='r', kde=False);


# In[ ]:


houses['ConstructionAge'] = houses['YrSold'] - houses['YearBuilt']
plt.scatter(houses['ConstructionAge'], houses['SalePriceSF'])
plt.ylabel('Price per square foot (in dollars)')
plt.xlabel("Construction Age of house");


# Price of house goes down with its age.

# *Heating and AC arrangements*
# -----------------------------

# In[ ]:


sns.stripplot(x="HeatingQC", y="SalePrice",data=houses,hue='CentralAir',jitter=True,split=True)
plt.title("Sale Price vs Heating Quality");


# Having AC definitely escalates price of house.

# *Bathrooms in house*
# --------------------------

# In[ ]:


sns.boxplot(houses["FullBath"],houses["SalePrice"])
plt.title("Sale Price vs Full Bathrooms");


# In[ ]:


sns.violinplot( houses["HalfBath"],houses["SalePrice"])
plt.title("Sale Price vs Half Bathrooms");


# *Total rooms above grade*
# -------------------------

# In[ ]:


sns.barplot(houses["TotRmsAbvGrd"],houses["SalePrice"],palette="Blues_d")
plt.title("Sale Price vs Number of rooms");


# *Kitchen Quality*
# =================

# In[ ]:


sns.factorplot("KitchenAbvGr","SalePrice",data=houses,hue="KitchenQual")
plt.title("Sale Price vs Kitchen");


# Having 1 Kitchen of Excellent quality hikes house price like anything.

# *Neighbourhood*
# --------------

# In[ ]:


plt.xticks(rotation=45) 
sns.barplot(houses["Neighborhood"],houses["SalePrice"])
plt.title("Sale Price vs Neighborhood");


# *Overall Quality*
# -----------------

# In[ ]:


plt.barh(houses["OverallQual"],width=houses["SalePrice"],color="r")
plt.title("Sale Price vs Overall Quality of house")
plt.ylabel("Overall Quality of house")
plt.xlabel("Sale Price");


# *2nd Floor with SalePrice*
# --------------------------

# In[ ]:


plt.scatter(houses["2ndFlrSF"],houses["SalePrice"],color="gold")
plt.title("Sale Price vs 2nd floor in sq feet");
plt.xlabel("2nd floor in sq feet")
plt.ylabel("Sale Price");


# *Street*
# --------

# In[ ]:


#most streets are paved lets visulalize it
sns.stripplot(x=houses["Street"], y=houses["SalePrice"],jitter=True)
plt.title("Sale Price vs Streets");


# More to come .. Watch this space.
