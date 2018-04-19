
# coding: utf-8

# ***Here is what was done on the first Zillow kernel*:**
# 
# 1. Look into the data (ofcourse!!)
# 2. Visualize the various independent variables visually (links for the same are provided)
# 3. Check for columns showing **non-variance**
# 4. Split columns based on their **data types**.
# 5. Carefully perform further analysis on **missing values** in those columns with respect to the values they can possibly hold; rather than assinging them to '0' blindly!
# 
# ***What we did later:***
# 1. Created new meaningful features (from 60 to 82 columns)
# 2. Visualizing the features.(to be updated)
# 
# ***What we did NOW:***
# 
# 1. Looked into **memory consumption** of the dataframe and **REDUCED** it !!
# 
# As always let's take a look into the data.

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv('../input/train_2016_v2.csv')
prop_df = pd.read_csv('../input/properties_2016.csv')
samp = pd.read_csv('../input/sample_submission.csv')

print (train_df.head())
print (prop_df.head())  


# In[ ]:


print(train_df.columns)
print(prop_df.columns)
print(train_df.shape)
print(prop_df.shape)


# I can see a **LOT** of missing values can you?
# 
# Let us see the number of rows and columns in each of the dataframes.

# First, let us merge both these columns based on their '**parcel_id**' and then we can perform our analysis.

# In[ ]:


train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
print(train_df.head())
print(train_df.shape)


# Let us visualize the distribution of the various independent variables in categorical form:
# 
# the independent variables (click both the links below to see what I mean):
# 
# 1. [Independent variables](https://github.com/JeruLuke/My-Kaggle-Participations/blob/master/Zillow_House/Inferences/coll_variables.JPG)
# 
# and
# 
# 2. [Rooms and various features](https://github.com/JeruLuke/My-Kaggle-Participations/blob/master/Zillow_House/Inferences/rooms%26features_2.JPG)
# 

# ***1. COLUMNS WITH NON-VARIANCE***
# 
# We have quite a lot of columns. Let us see if any of them have only **ONE distinct** value, so that they can be dropped.

# In[ ]:


count = 0
for c in list(train_df):
    if (len(train_df[c].unique()) == 1):
        print(c)
        count+=1
print(count)


# Let us see if there are any columns having less than 1% of unique values.

# In[ ]:


count = 0
low_var_cols = []
for c in list(train_df):
    if (len(train_df[c].unique()) < 907):
        print(c)
        low_var_cols.append(c)
        count+=1
print(count)


# There are 43 such columns. Out of this we can drop columns which have only two unique values as either of these values would be less than 1%, hence such columns would have less variance. (we can think about dropping them later)

# In[ ]:


len(low_var_cols)


# In[ ]:


count = 0
low_var_drop_cols = []
for c in low_var_cols:
    if (train_df[c].nunique() <= 2):
        print(c)
        low_var_drop_cols.append(c)
        count+=1
print(count)
len(low_var_drop_cols)


# The column `assessmentyear` alone has one unique variable throughout. All the other columns have some unique features which can considerably vary the cost. Hence, we will not drop those columns for now.

# Only **ONE** column '**assessmentyear**' has a single distinct value throughout. We can cross check that:

# In[ ]:


print(train_df['assessmentyear'].nunique())


# ***2. COLUMNS WITH MISSING VALUES***

# In[ ]:


#--- List of columns having Nan values and the number ---

missing_col = train_df.columns[train_df.isnull().any()].tolist()
print(missing_col)
print('There are {} missing columns'.format(len(missing_col)))


# ***3. COLUMNS WITH NO MISSING VALUES***

# In[ ]:


nonmissing_col = train_df.columns[~(train_df.isnull().any())].tolist()
print(nonmissing_col)
print('There are {} non-missing columns'.format(len(nonmissing_col)))


# There are 13 columns that have no missing values. Every other column has some missing values.
# 
# 
# ***4. CHECKING DATATYPE OF EACH COLUMN ***

# In[ ]:


#--- Data type of each column ---
print(train_df.dtypes)


# Let us see the split-up of the datatypes:

# In[ ]:


import seaborn as sns
#sns.barplot( x = train_df.dtypes.unique(), y = train_df.dtypes.value_counts(), data = train_df)
sns.barplot( x = ['float', 'object', 'int'], y = train_df.dtypes.value_counts(), data = prop_df)


# What is the exact count of all these datatypes?

# In[ ]:


print(train_df.dtypes.value_counts())


# Observations:
# 1. Only '**parcel_id**' is off **integer** type
# 2. '**hashottuborspa**', '**propertycountylandusecode**', '**propertyzoningdesc**', '**fireplaceflag**' and '**taxdelinquencyflag** are of type **object**
# 3. The remaining columns are of type **float**.

# ***5.1 Analyzing column of type `int` -> `parcelid`***

# In[ ]:


#--- Checking if all the parcelids are unique in both the dataframes ---

print (prop_df['parcelid'].nunique())
print (prop_df.shape[0])

print (train_df['parcelid'].nunique())
print (train_df.shape[0]) 


# Some of the parcelid in the merged dataframe have been repeated more than once.

# ***5.2 Analyzing columns of type `object`***
# 
# Let us get the unique elements in each of these columns (categorical variables) along with their count. Then we can convert them to numerical variables:

# In[ ]:


print(train_df['hashottuborspa'].nunique())
print(train_df['hashottuborspa'].unique())
print('\n')
print(train_df['propertycountylandusecode'].nunique())
print(train_df['propertycountylandusecode'].unique())
print('\n')
print(train_df['propertyzoningdesc'].nunique())
print(train_df['propertyzoningdesc'].unique())
print('\n')
print(train_df['fireplaceflag'].nunique())
print(train_df['fireplaceflag'].unique())
print('\n')
print(train_df['taxdelinquencyflag'].nunique())
print(train_df['taxdelinquencyflag'].unique()) 
print('\n') 
print(train_df['transactiondate'].nunique())


# Three of the columns mentioned above (**hashottuborspa**, **fireplaceflag** and **taxdelinquencyflag**) are merely **flags** indicating presence of the feature or not. Hence the `nan` values can be replaced with '0', while the other values can be replaced with '1'.

# In[ ]:


train_df['hashottuborspa'] = train_df['hashottuborspa'].fillna(0)
train_df['fireplaceflag'] = train_df['fireplaceflag'].fillna(0)
train_df['taxdelinquencyflag'] = train_df['taxdelinquencyflag'].fillna(0)

#---  replace the string 'True' and 'Y' with value '1' ---

train_df.hashottuborspa = train_df.hashottuborspa.astype(np.int8)
train_df.fireplaceflag = train_df.fireplaceflag.astype(np.int8)
train_df['taxdelinquencyflag'].replace( 'Y', 1, inplace=True)
train_df.taxdelinquencyflag = train_df.taxdelinquencyflag.astype(np.int8)


# Out of the remaining three columns `transactiondate` is a datetime object

# In[ ]:


train_df['transactiondate'] = pd.to_datetime(train_df['transactiondate'])

#--- Creating two additional columns each for the month and day ---
train_df['transaction_month'] = train_df.transactiondate.dt.month.astype(np.int64)
train_df['transaction_day'] = train_df.transactiondate.dt.weekday.astype(np.int64)

#--- Dropping the 'transactiondate' column now ---
train_df = train_df.drop('transactiondate', 1)


# The remaining two columns are random in nature, in terms of their values `propertycountylandusecode` and `propertyzoningdesc`

# In[ ]:


#--- Counting number of occurrences of Nan values in remaining two columns ---
print(train_df['propertycountylandusecode'].isnull().sum())
print(train_df['propertyzoningdesc'].isnull().sum())


# The occurrences of missing values for the second column (`propertyzoningdesc`) is so large for us to remove the observations. So we will replace them with a **random** value.

# In[ ]:


#--- Since there is only ONE missing value in this column we will replace it manually ---
train_df["propertycountylandusecode"].fillna('023A', inplace =True)
print(train_df['propertycountylandusecode'].isnull().sum())


# In[ ]:


train_df["propertyzoningdesc"].fillna('UNIQUE', inplace =True)
print(train_df['propertyzoningdesc'].isnull().sum())


# I have assigned same code `UNIQUE` for all the missing observations. If it does impact our modeling we can always change it.

# ***5.3 Analyzing columns of type `float`***
# 
#    ***5.3.1 Firstly, the target variable `logerror`. ***
#    
#    Let us see the statistics of this column and then plot it.

# In[ ]:


#--- Statistics of the target variable ---

print(train_df['logerror'].describe())


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(train_df['logerror'], train_df.logerror.values)
plt.xlabel('No of observations', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


# Most of the values lie in the range [**-1.75 - 3.75**] (approx)

# First I would like to take all columns of type `float` in a list

# In[ ]:


#--- putting all columns of 'float' type in a list ---
float_cols = list(train_df.select_dtypes(include=['float']).columns)
print('There are {} columns of type float having missing values'.format(len(float_cols)))
print('\n')
print(float_cols)


# Now collect all columns of type `float` and **missing** values in a list

# In[ ]:


#--- putting columns of type 'float' having missing values in a list ---
float_nan_col = []
for column in float_cols:
    if (train_df[column].isnull().sum() > 0):
        float_nan_col.append(column)

print('There are {} columns of type float having missing values'.format(len(float_nan_col)))
print('\n')
print(float_nan_col)


# ***5.3.2*** Analysis must be done on **THESE** 42 columns to impute missing values** with** care:
# 
# a. Columns **`regionidcity`, `regionidneighborhood`** and **`regionidzip`** can be given some random values based on the other values in the respective columns.
# 
# b. Missing values in column **`unitcnt`** must be assigned value **1**, signifying a single structural unit. Making it '0' would be absurd!!!
# 
# c. Column **`censustractandblock`** can be given random values to the missing observations present within the same column.
# 
# d. There are **756** missing values in the column **`yearbuilt`**, which cannot be imputed randomly AT ALL!! However, for the sake of starting off I will assign it random year values based in the same column.
# 
# e. The remaining columns can be safely assigned value '0', beacuse they all signify a presence of a particular ID or count.

# ***5.3.2.a*** Columns **`regionidcity`, `regionidneighborhood`** and **`regionidzip`**
# 
# Here I am imputing missing values by randomly assigning values already present in the respective columns. (If you have a better way to impute such values do mention it in the comments section!!)

# In[ ]:


cols = ['regionidcity', 'regionidneighborhood', 'regionidzip']
print(train_df['regionidcity'].isnull().sum())
print(train_df['regionidneighborhood'].isnull().sum())
print(train_df['regionidzip'].isnull().sum())

train_df["regionidcity"].fillna(lambda x: np.random(train_df[train_df["regionidcity"] != np.nan]), inplace =True)
train_df["regionidneighborhood"].fillna(lambda x: np.random(train_df[train_df["regionidneighborhood"] != np.nan]), inplace =True)
train_df["regionidzip"].fillna(lambda x : np.random(train_df["regionidzip"] != np.nan) , inplace =True)

#--- cross check whether nan values are present or not ---
print(train_df['regionidcity'].isnull().sum())
print(train_df['regionidneighborhood'].isnull().sum())
print(train_df['regionidzip'].isnull().sum())


# ***5.3.2.b*** Column **`unitcnt`**
# 
# Here we will replace missing values with the mostly occuring variable

# In[ ]:


#--- some analysis on the column values ---

print(train_df['unitcnt'].unique())
print(train_df['unitcnt'].value_counts())
sns.countplot(x = 'unitcnt', data = train_df)


# In[ ]:


#--- Replace the missing values with the maximum occurences ---
train_df['unitcnt'] = train_df['unitcnt'].fillna(train_df['unitcnt'].mode()[0])

#--- cross check for missing values ---
print(train_df['unitcnt'].isnull().sum())


# ***5.3.2.c*** Column **`censustractandblock`**
# 
# Let us see the correlation between `censustractandblock` and `rawcensustractandblock`

# In[ ]:


print(train_df['censustractandblock'].corr(train_df['rawcensustractandblock']))


# In[ ]:


print(train_df['censustractandblock'].nunique())
print(train_df['rawcensustractandblock'].nunique())


# The correlation between these columns is VERY high (~1). So filling missing values MUST be in relation to column `rawcensustractandblock`

# In[ ]:


'''  #--- to be continued ---
print(train_df['censustractandblock'].isnull().sum())
#print('\n')
#print(train_df['rawcensustractandblock'].nunique())

#train_df['censustractandblock'] = train_df['censustractandblock'].fillna()
pop = pd.DataFrame()
pop['censustractandblock'] = train_df['censustractandblock'] 
print(pop.shape[0])

a = 0
count = 0
for i in pop['censustractandblock']:
    if (np.isnan(i)):
        a = train_df.iloc[count]['rawcensustractandblock']
        #a.append(train_df['rawcensustractandblock'].iloc())
        for j in pop['censustractandblock']:
            if ((np.isfinite(j)) & ( )):
                
        count+=1
print(count)
#pop['censustractandblock'] = pop['censustractandblock'].fillna(pop['censustractandblock'] /
       # if )
print (a)    
''' 


# ***5.3.2.d*** Column **`yearbuilt`**
# 
# Arranging the `yearbuilt`column in ascending order.

# In[ ]:


print(train_df['yearbuilt'].sort_values().unique())


# Looking at the above sorted list of years, we have collection of houses built since 1885. Missing values are probably for the houses built in the year 2016 (I think !!!). So let us replace them:

# In[ ]:


train_df['yearbuilt'] = train_df['yearbuilt'].fillna(2016)

#--- cross check for missing values ---
print(train_df['yearbuilt'].isnull().sum())


# ***5.3.2.e***  ***Remaining columns***
# 
# The remaining columns can be safely assigned value '0', beacuse they all signify a presence of a particular ID or count.

# In[ ]:


#--- list of columns of type 'float' having missing values
#--- float_nan_col 

#--- list of columns of type 'float' after imputing missing values ---
float_filled_cols = ['regionidcity', 'regionidneighborhood', 'regionidzip', 'unitcnt', 'censustractandblock', 'yearbuilt']

count = 0
for i in float_nan_col:
    if i not in float_filled_cols:
        train_df[i] = train_df[i].fillna(0)
        count+=1
print(count)


# In[ ]:


print(len(float_nan_col))


# **Plotting columns `Latitude` and `Longitude`**
# 
# From the dictionary, this column specifies ' *Latitude and Longitude of the middle of the parcel multiplied by 10e6*'
# 

# In[ ]:


sns.regplot(x = 'latitude', y = 'longitude', data = train_df)


# In[ ]:


x = train_df.iloc[1]
#print(x)


# **CREATING NEW FEATURES**:
# 
# Let us create new features using the existing one in the dataframe

# In[ ]:


#--- how old is the house? ---
train_df['house_age'] = 2017 - train_df['yearbuilt']

#--- how many rooms are there? ---  
train_df['tot_rooms'] = train_df['bathroomcnt'] + train_df['bedroomcnt']

#--- does the house have A/C? ---
train_df['AC'] = np.where(train_df['airconditioningtypeid']>0, 1, 0)

#--- Does the house have a deck? ---
train_df['deck'] = np.where(train_df['decktypeid']>0, 1, 0)
train_df.drop('decktypeid', axis=1, inplace=True)

#--- does the house have a heating system? ---
train_df['heating_system'] = np.where(train_df['heatingorsystemtypeid']>0, 1, 0)

#--- does the house have a garage? ---
train_df['garage'] = np.where(train_df['garagecarcnt']>0, 1, 0)

#--- does the house come with a patio? ---
train_df['patio'] = np.where(train_df['yardbuildingsqft17']>0, 1, 0)

#--- does the house have a pool?
train_df['pooltypeid10'] = train_df.pooltypeid10.astype(np.int8)
train_df['pooltypeid7'] = train_df.pooltypeid7.astype(np.int8)
train_df['pooltypei2'] = train_df.pooltypeid2.astype(np.int8)
train_df['pool'] = train_df['pooltypeid10'] | train_df['pooltypeid7'] | train_df['pooltypeid2'] 

#--- does the house have all of these? -> spa/hot-tub/pool, A/C, heating system , garage, patio
train_df['exquisite'] = train_df['pool'] + train_df['patio'] + train_df['garage'] + train_df['heating_system'] + train_df['AC'] 

#--- Features based on location ---
train_df['x_loc'] = np.cos(train_df['latitude']) * np.cos(train_df['longitude'])
train_df['y_loc'] = np.cos(train_df['latitude']) * np.sin(train_df['longitude'])
train_df['z_loc'] = np.sin(train_df['latitude'])

print('DONE')


# When do people usually buy houses?

# In[ ]:


#train_df['transaction_year']
sns.countplot(x = 'transaction_month', data = train_df)


# Let us create a feature called 'season' to know in which season transactions are high.
# 
# According to [THIS WEBSITE](https://www.timeanddate.com/calendar/aboutseasons.html) the four seaons are categorized as:
# 1. Spring - from March 1 to May 31;
# 2. Summer - from June 1 to August 31;
# 3. Fall (autumn) - from September 1 to November 30; and,
# 4. Winter - from December 1 to February 28
# 

# In[ ]:


#--- create an additional feature called season ---
def seas(x):
    if 2 < x < 6:
        return 1        #--- Spring
    elif 5 < x < 9:
        return 2        #---Summer
    elif 8 < x < 12:
        return 3        #--- Fall (Autumn) 
    else:
        return 4        #--- Winter 

train_df['season'] = train_df['transaction_month'].apply(seas)


# In[ ]:


ax = sns.countplot(x = 'season', data = train_df)
ax.set(xlabel='Seasons', ylabel='Count')
season_list=['Spring','Summer','Fall','Winter']
plt.xticks(range(4), season_list, rotation=45)
plt.show()


# Most of the house transactions are done in the Spring and Summer seasons.

# Lets us see the distribution of the newly created exquisite features:

# In[ ]:


ax = sns.countplot(x = 'exquisite', data = train_df)
ax.set(xlabel='Exquisite features present', ylabel='Count')
plt.show()


# Most of the houses have ATLEAST ONE of the mentioned exquisite features.

# In[ ]:


ax = sns.countplot(x = 'transaction_day', data = train_df)
ax.set(xlabel='Transaction Days', ylabel='Count')
days_list=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.xticks(range(len(days_list)), days_list, rotation=45)
plt.show()


# **New feature**: Weekend/weekday transaction
# 
# Transactions are high during the weekdays. So we can create another categorical feature Weekdays/Weekends

# In[ ]:


#--- create an additional feature called weekday_trans ---
def weekday_transaction(x):
    if 4 < x <= 6:
        return 1        #--- Weekend
    else:
        return 2        #--- Weekday

train_df['weekday_trans'] = train_df['transaction_day'].apply(weekday_transaction)


# In[ ]:


ax = sns.countplot(x = 'weekday_trans', data = train_df)
ax.set(xlabel='Weekday/weekend', ylabel='Count')
weekend_day_list=['Weekend', 'Weekday']
plt.xticks(range(len(weekend_day_list)), weekend_day_list, rotation=45)
plt.show()


# **New features based on area** adapted from [THIS KERNEL](https://www.kaggle.com/nikunjm88/creating-additional-features?scriptVersionId=1379783)

# In[ ]:


#--- living area ---
train_df['LivingArea'] = train_df['calculatedfinishedsquarefeet']/train_df['lotsizesquarefeet']
train_df['LivingArea_2'] = train_df['finishedsquarefeet12']/train_df['finishedsquarefeet15']

#--- Extra space available
train_df['ExtraSpace'] = train_df['lotsizesquarefeet'] - train_df['calculatedfinishedsquarefeet'] 
train_df['ExtraSpace-2'] = train_df['finishedsquarefeet15'] - train_df['finishedsquarefeet12'] 


# **New features based on TAX**

# In[ ]:


#Ratio of tax of property over parcel
train_df['ValueRatio'] = train_df['taxvaluedollarcnt']/train_df['taxamount']

#TotalTaxScore
train_df['TaxScore'] = train_df['taxvaluedollarcnt']*train_df['taxamount']


# **New features based on the address**

# In[ ]:


#Number of properties in the zip
zip_count = train_df['regionidzip'].value_counts().to_dict()
train_df['zip_count'] = train_df['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = train_df['regionidcity'].value_counts().to_dict()
train_df['city_count'] = train_df['regionidcity'].map(city_count)

#Number of properties in the city
region_count = train_df['regionidcounty'].value_counts().to_dict()
train_df['county_count'] = train_df['regionidcounty'].map(region_count)


# In[ ]:


#--- Number of columns present in our dataframe now ---
a = train_df.columns.tolist()
print('Now there are {} columns in our dataframe'.format(len(a)))


# **VISUALIZATIONS**
# 
# **1. Target Variable**

# Obtaining the **absolute** error from the **`logerror`** column.

# In[ ]:


import math
p = pd.DataFrame()
p['val'] = np.exp(train_df['logerror'])
print(p.head())

plt.scatter(p['val'], p.val.values)
plt.xlabel('No of observations', fontsize=12)
plt.ylabel('vals', fontsize=12)
plt.show()


# We can clearly see 4 distinct outliers. 
# 
# Let us see the statistics of the column.

# In[ ]:


print(p.describe())


# Let us **remove** those outliers and visualize the plot and see the statistics again

# In[ ]:


p = p[p['val'] < 40]

plt.scatter(p['val'], p.val.values)
plt.xlabel('No of observations', fontsize=12)
plt.ylabel('vals', fontsize=12)
plt.show()

print(p.describe())


# The ***standard deviation*** and the ***maximum*** value have dropped by quite a large margin.

# In[ ]:


#plt.hist(np.log(train_df['trip_duration']+25), bins = 25)
plt.hist(train_df['logerror'], bins = 100)


# **2. Correlations**

# In[ ]:


corr = train_df.corr()
fig, ax = plt.subplots(figsize=(20, 20))
ax.matshow(corr)
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
plt.xticks(range(len(corr.columns)), corr.columns, fontsize = 15)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize = 15)


# **Number of houses built VS year**

# In[ ]:


alist = []
alist = train_df['yearbuilt'].unique()
alist.sort()


# In[ ]:


from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize=(20, 20))
ax = sns.countplot(x = 'yearbuilt', data = train_df)
ax.set(xlabel='Year Built', ylabel='Count')
#weekend_day_list=['Weekend', 'Weekday']
plt.xticks(range(len(alist)), alist, rotation=90)
plt.show()


# In[ ]:


'''
from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize=(20, 20))
ax = sns.countplot(x = 'yearbuilt', data = train_df)
ax.set(xlabel='Year Built', ylabel='Count')
#weekend_day_list=['Weekend', 'Weekday']
plt.xticks(range(len(alist)), alist, rotation=90)
plt.show()
''' 
'''
x = list(train_df['yearbuilt'])
y = train_df['logerror']
fig = plt.bar(x, y)
plt.show()
'''


# **MEMORY CONSUMPTION**
# 
# Let us look into the memory consumption of our dataframe and see if we can reduce it efficiently.

# In[ ]:


#--- Memory usage of entire dataframe ---
mem = train_df.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# In[ ]:


#--- Memory usage of each column ---
print(train_df.memory_usage()/ 1024**2)  #--- in MB ---


# We can reduce memory for columns only having type **`int`** and type **`float`**, or columns having **numeric** values.

# In[ ]:


#--- List of columns that cannot be reduced in terms of memory size ---
count = 0
for col in train_df.columns:
    if train_df[col].dtype == object:
        count+=1
        print (col)
print('There are {} columns that cannot be reduced'.format(count))        


# Reducing columns to type `int8` if possible

# In[ ]:


count = 0
for col in train_df.columns:
    if train_df[col].dtype != object:
        if ((train_df[col].max() < 255) & (train_df[col].min() > -255)):
            if((col != 'logerror')|(col != 'yearbuilt')|(col != 'xloc')|(col != 'yloc')|(col != 'zloc')):
                count+=1
                train_df[col] = train_df[col].astype(np.int8)
                print (col)
print(count)                
                


# In[ ]:


#--- Memory usage of reduced dataframe ---
mem = train_df.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# We just reduced the dataframe size **from** **~57MB to ~35MB**
# 
# *Splendid*
# 
# 

# In[ ]:


#--- Reducing memory of `float64` type columns to `float32` type columns

count = 0
for col in train_df.columns:
    if train_df[col].dtype != object:
        if train_df[col].dtype == float:
            train_df[col] = train_df[col].astype(np.float32)
            count+=1
print('There were {} such columns'.format(count))


# In[ ]:


#--- Let us check the memory consumed again ---
mem = train_df.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# WOW!!! Now reduced **from ~35MB to ~24MB !!!!**
# 
# Let us see what we can do with columns of type **`int64`**.

# In[ ]:


#print(train_df.dtypes)
#print(train_df.dtypes.value_counts())
col_int64 = []
for col in train_df.columns:
    if train_df[col].dtype == 'int64':
        print(col)
        col_int64.append(col)
print(col_int64)


# By checking the maximum and minimum values of these columns we can make sure which ones to convert to type **`int32`**.

# In[ ]:


for i in col_int64:
    print('{} - {} and {}'.format(i, max(train_df[i]), min(train_df[i])) )


# Clearly these three columns can be converted to type **`int32`**:
# 
# `zip_count`, `city_count` and `county_count`

# In[ ]:


train_df['zip_count'] = train_df['zip_count'].astype(np.int32)
train_df['city_count'] = train_df['city_count'].astype(np.int32)
train_df['county_count'] = train_df['county_count'].astype(np.int32)


# In[ ]:


#--- Let us check the memory consumed again ---
mem = train_df.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# KEY TAKEAWAYS FROM THE **PREVIOUS** KERNEL:
# 
# 1. There are so **many** **missing** values in this dataset.
# 2. Understanding the **target** variable.
# 3. Missing values in column `propertyzoningdesc` have been replaced with the **SAME** value throughout that column. If it does affect the performance at the time of modeling we must consider making more random values for them.
# 4. Columns of type `float` have been analysed in detail(except for `censustractandblock` column, which I will later).
# 5. We have lot of scope to create MANY new features!! (Houses along the coast/beach would cost more would'nt it?)
# 
# 
# KEY TAKEAWAYS FROM THIS **UPDATED** KERNEL:
# 
# 1. Reduced the memory of the dataframe.
# 
# 
# Since there are 2 months to the merger deadline. We have PLENTY of time to perform more analysis on the features!!!!
# 

# This Kernel is under regular update!
# 
# SO DO VISIT AGAIN!! 
# 
# Do give me suggestions on how to improve and ....
# 
# **UPVOTE  UPVOTE  UPVOTE ** (Ps. only if you think it deserves any.. :) )
