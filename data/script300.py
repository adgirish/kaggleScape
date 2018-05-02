
# coding: utf-8

# As the name says, all I am doing here is cleaning the data - more specifically, replacing missing values in the training data. The result should be the starting point for further exploration and/or feature engineering. In order to build the actual model, further steps will have to be taken.

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


# In[ ]:


houseprice=pd.read_csv('../input/train.csv')
houseprice.head()


# You can already see that there are NaNs in some columns. So let's see where exactly and how many

# In[ ]:


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing():
    missing = houseprice.columns[houseprice.isnull().any()].tolist()
    return missing


# In[ ]:


houseprice[show_missing()].isnull().sum()


# ## Data Cleaning Plan
# Let's look at these variables in the data dictionary:
# 
# * LotFrontage: Linear feet of street connected to property. I can't imagine that this would be 0 (as this would be a property without access), so either impute mean, or maybe see if there's a correlation with LotArea (like square root?).
# 
# * Alley: Type of alley access to property -> Many missing values, I would presume that these properties just don't have an alley access.
# 
# * MasVnrType/MasVnrArea -> both have 8 values missing, I presume they are the same ones. Either set as "None"/0 or use most frequent value/median.
# 
# * Bsmt... Variables: A  number of variables in connection with the basement. About the same number of missing values. However, there are two basement-related variables without missing values "BsmtFinSF1" and "BsmtFinSF2" - look at those and then decide what to do with the missing values.
# 
# * Electrical: Just one missing value - here just impute most frequent one.
# 
# * FireplaceQu: I assume the properties with missing values just don't have a fireplace. There's also the variable Fireplaces (without missing values) - check this and then decide.
# 
# * Garage ... Variables: 81 missing in these columns. However, there are some Garage-related variables without missing values: GarageCars, GarageArea - check these and then decide.
# 
# * PoolQC - probably no pool - but check against PoolArea (which has no missing values).
# 
# * Fence: Many missing values - probably no fence, just impute 'None'
# 
# * MiscFeature: Assuming none - probably no special features, just impute 'None'

# In[ ]:


# Looking at categorical values
def cat_exploration(column):
    return houseprice[column].value_counts()


# In[ ]:


# Imputing the missing values
def cat_imputation(column, value):
    houseprice.loc[houseprice[column].isnull(),column] = value


# ### LotFrontage/LotArea
# A number of values are missing and one possibility would be to just impute the mean. However, there should actually be a correlation with LotArea, which has no missing values.

# In[ ]:


# check correlation with LotArea
houseprice['LotFrontage'].corr(houseprice['LotArea'])


# Ok, that's not great. I we assume that most lots are rectangular, using the square root might be an improvement. 

# In[ ]:


# improvement - and good enough for now
houseprice['SqrtLotArea']=np.sqrt(houseprice['LotArea'])
houseprice['LotFrontage'].corr(houseprice['SqrtLotArea'])


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')
sns.pairplot(houseprice[['LotFrontage','SqrtLotArea']].dropna())


# In[ ]:


cond = houseprice['LotFrontage'].isnull()
houseprice.LotFrontage[cond]=houseprice.SqrtLotArea[cond]


# In[ ]:


# This column is not needed anymore
del houseprice['SqrtLotArea']


# ### Alley

# In[ ]:


cat_exploration('Alley')


# In[ ]:


# I assume empty fields here mean no alley access
cat_imputation('Alley','None')


# ### MasVnr

# In[ ]:


houseprice[['MasVnrType','MasVnrArea']][houseprice['MasVnrType'].isnull()==True]


# So the missing values for the "MasVnr..." Variables are in the same rows.

# In[ ]:


cat_exploration('MasVnrType')


# Since "None" is the most frequent value, I will impute "None" for the Type, and 0.0 for the area.

# In[ ]:


cat_imputation('MasVnrType', 'None')
cat_imputation('MasVnrArea', 0.0)


# ### Basement

# In[ ]:


basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
houseprice[basement_cols][houseprice['BsmtQual'].isnull()==True]


# So in the cases where the categorical variables are NaN, the numerical ones are 0. Which means there's no basement, so the categorical ones should also be set to "None".

# In[ ]:


for cols in basement_cols:
    if 'FinSF'not in cols:
        cat_imputation(cols,'None')


# ### Electrical

# In[ ]:


cat_exploration('Electrical')


# In[ ]:


# Impute most frequent value
cat_imputation('Electrical','SBrkr')


# ### Fireplace

# In[ ]:


cat_exploration('FireplaceQu')


# I would assume that the 690 just don't have a fireplace. Let's check:

# In[ ]:


houseprice['Fireplaces'][houseprice['FireplaceQu'].isnull()==True].describe()


# In[ ]:


cat_imputation('FireplaceQu','None')


# In[ ]:


pd.crosstab(houseprice.Fireplaces, houseprice.FireplaceQu)


# ### Garages

# In[ ]:


garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
houseprice[garage_cols][houseprice['GarageType'].isnull()==True]


# In[ ]:


#Garage Imputation
for cols in garage_cols:
    if houseprice[cols].dtype==np.object:
        cat_imputation(cols,'None')
    else:
        cat_imputation(cols, 0)


# ### Pool

# In[ ]:


cat_exploration('PoolQC')


# Many missing values - are they all without a pool?

# In[ ]:


houseprice['PoolArea'][houseprice['PoolQC'].isnull()==True].describe()


# Yes, seems like it - if PoolQC is empty, PoolArea is 0

# In[ ]:


cat_imputation('PoolQC', 'None')


# ### Fence

# In[ ]:


cat_imputation('Fence', 'None')


# ### MiscFeature

# In[ ]:


cat_imputation('MiscFeature', 'None')


# ### Are we done?

# In[ ]:


houseprice[show_missing()].isnull().sum()


# Yes, all missing values are gone!
