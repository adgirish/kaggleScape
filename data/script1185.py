
# coding: utf-8

# **Just planning to add the explorations that I am going to do for this competition. Happy Kaggling.!**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

color = sns.color_palette()

data_path = "../input/"
train_file = data_path + "train_ver2.csv"
test_file = data_path + "test_ver2.csv"


# **Dataset Size:**
# 
# First let us check the number of rows in train and test file

# In[ ]:


train = pd.read_csv(data_path+train_file, usecols=['ncodpers'])
test = pd.read_csv(data_path+test_file, usecols=['ncodpers'])
print("Number of rows in train : ", train.shape[0])
print("Number of rows in test : ", test.shape[0])


# **No of Customers:**
# 
# Now let us look at the number of unique customers in train data and test data and also the number of customers common between both

# In[ ]:


train_unique_customers = set(train.ncodpers.unique())
test_unique_customers = set(test.ncodpers.unique())
print("Number of customers in train : ", len(train_unique_customers))
print("Number of customers in test : ", len(test_unique_customers))
print("Number of common customers : ", len(train_unique_customers.intersection(test_unique_customers)))


# Let us see the count of occurrences of each of the customers in train set

# In[ ]:


num_occur = train.groupby('ncodpers').agg('size').value_counts()

plt.figure(figsize=(8,4))
sns.barplot(num_occur.index, num_occur.values, alpha=0.8, color=color[0])
plt.xlabel('Number of Occurrences of the customer', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.show()


# We have 17 months of data present in our train and we can clearly see that majority of the customers are present for all 17 months. There is also a small spike at '11 month' compared to other months.! 

# In[ ]:


del train_unique_customers
del test_unique_customers


# **Target Variables distribution:**
# 
# There are 24 target variables present in this dataset are as follows:
# 
# 1. ind_ahor_fin_ult1	  - Saving Account
# 
# 2. ind_aval_fin_ult1	  - Guarantees
# 
# 3. ind_cco_fin_ult1	  - Current Accounts
# 
# 4. ind_cder_fin_ult1	  - Derivada Account
# 
# 5. ind_cno_fin_ult1	  - Payroll Account
# 
# 6. ind_ctju_fin_ult1	  - Junior Account
# 
# 7. ind_ctma_fin_ult1 - Más particular Account
# 
# 8. ind_ctop_fin_ult1 - particular Account
# 
# 9. ind_ctpp_fin_ult1 - particular Plus Account
# 
# 10. ind_deco_fin_ult1 - Short-term deposits
# 
# 11. ind_deme_fin_ult1 - Medium-term deposits
# 
# 12. ind_dela_fin_ult1 - Long-term deposits
# 
# 13. ind_ecue_fin_ult1 - e-account
# 
# 14. ind_fond_fin_ult1 - Funds
# 
# 15. ind_hip_fin_ult1 - Mortgage
# 
# 16. ind_plan_fin_ult1 - Pensions
# 
# 17. ind_pres_fin_ult1 - Loans
# 
# 18. ind_reca_fin_ult1 - Taxes
# 
# 19. ind_tjcr_fin_ult1 - Credit Card
# 
# 20. ind_valo_fin_ult1 - Securities
# 
# 21. ind_viv_fin_ult1 - Home Account
# 
# 22. ind_nomina_ult1 - Payroll
# 
# 23. ind_nom_pens_ult1 - Pensions
# 
# 24. ind_recibo_ult1 - Direct Debit
# 
# Let us check the number of times the given product has been bought in the train dataset

# In[ ]:


train = pd.read_csv(data_path+"train_ver2.csv", dtype='float16', 
                    usecols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 
                             'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                             'ind_viv_fin_ult1', 'ind_nomina_ult1',
                             'ind_nom_pens_ult1', 'ind_recibo_ult1'])


# In[ ]:


target_counts = train.astype('float64').sum(axis=0)
#print(target_counts)
plt.figure(figsize=(8,4))
sns.barplot(target_counts.index, target_counts.values, alpha=0.8, color=color[0])
plt.xlabel('Product Name', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# Product "ind_cco_fin_ult1 " is the most bought one and "ind_aval_fin_ult1" is the least bought one.

# **Exploring Dates:**
# 
# Let us explore the dates now and see if there are any insights. There are 2 date fields present in the data.
# 
# 1. fecha_dato - The date of observation
# 2. fecha_alta - The date in which the customer became as the first holder of a contract in the bank

# In[ ]:


train = pd.read_csv(data_path+"train_ver2.csv", usecols=['fecha_dato', 'fecha_alta'], parse_dates=['fecha_dato', 'fecha_alta'])
train['fecha_dato_yearmonth'] = train['fecha_dato'].apply(lambda x: (100*x.year) + x.month)
yearmonth = train['fecha_dato_yearmonth'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(yearmonth.index, yearmonth.values, alpha=0.8, color=color[0])
plt.xlabel('Year and month of observation', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# For the first six months of the given train data, the number of customers / observations remain almost same and then there is a sudden spike in the number of customers / observations during July 2015.

# In[ ]:


train['fecha_alta_yearmonth'] = train['fecha_alta'].apply(lambda x: (100*x.year) + x.month)
yearmonth = train['fecha_alta_yearmonth'].value_counts()
print("Minimum value of fetcha_alta : ", min(yearmonth.index))
print("Maximum value of fetcha_alta : ", max(yearmonth.index))

plt.figure(figsize=(12,4))
sns.barplot(yearmonth.index, yearmonth.values, alpha=0.8, color=color[1])
plt.xlabel('Year and month of joining', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# So the first holder date starts from January 1995. But as we can see, the number is high during the recent years.! 
# 
# Also it seems there are some seasonal peaks in the data. Let us have a close look at them.!

# In[ ]:


year_month = yearmonth.sort_index().reset_index()
year_month = year_month.ix[185:]
year_month.columns = ['yearmonth', 'number_of_customers']

plt.figure(figsize=(12,4))
sns.barplot(year_month.yearmonth.astype('int'), year_month.number_of_customers, alpha=0.8, color=color[2])
plt.xlabel('Year and month of joining', fontsize=12)
plt.ylabel('Number of customers', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# From 2011, the number of customers becoming the first folder of a contract in **the second six months is much higher than the first six months** in a calendar year and it is across all years after that. Looks interesting to me from a business standpoint.!  
# 
# **Numerical variables exploration:**
# 
# Let us explore the 3 numerical variables present in the data.
# 
# 1. Age
# 2. Antiguedad - customer seniority
# 3. Renta
# 
# We can check the number of missing values, distribution of the data, distribution of the target variables based on the numerical variables in this notebook.
# 
# **Age:**

# In[ ]:


train = pd.read_csv(train_file, usecols=['age'])
train.head()


# In[ ]:


print(list(train.age.unique()))


# There are quite a few different formats for age (number, string with leading spaces, string). 
# 
# Also if we see, there is a **' NA'** value present in this field. So let us first take care of that by changing it to np.nan.

# In[ ]:


train['age'] = train['age'].replace(to_replace=[' NA'], value=np.nan)


# We can now convert the field to dtype 'float' and then get the counts

# In[ ]:


train['age'] = train['age'].astype('float64')

age_series = train.age.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(age_series.index.astype('int'), age_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# We could see that there is a very long tail at both the ends. So we can have min and max cap at some points respectively (I would use 20 and 86 from the graph). 

# In[ ]:


train.age.isnull().sum()


# In[ ]:


train.age.mean()


# We have 27734 missing values and the mean age is 40. We could probably do a mean imputation here. 
# 
# We could look at test set age distribution to confirm both train and test have same distribution.

# In[ ]:


test = pd.read_csv(test_file, usecols=['age'])
test['age'] = test['age'].replace(to_replace=[' NA'], value=np.nan)
test['age'] = test['age'].astype('float64')

age_series = test.age.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(age_series.index.astype('int'), age_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# Good to see that the distribution is similar between train and test.!
# 
# **ANTIGUEDAD:**
# 
# Customer seniority in months.

# In[ ]:


train = pd.read_csv(train_file, usecols=['antiguedad'])
train.head()


# In[ ]:


print(list(train.antiguedad.unique()))


# Here again we could see that there is a **'     NA'** value present in this field similar to age. Also we could see that there is a special value '-999999' present in the data. May be this special value also represent missing value?!
# 
# We shall first convert the NA value to np.nan value

# In[ ]:


train['antiguedad'] = train['antiguedad'].replace(to_replace=['     NA'], value=np.nan)
train.antiguedad.isnull().sum()


# So here again we have 27734 missing values.
# 
# We can convert the field to dtype 'float' and then check the count of special value -999999.

# In[ ]:


train['antiguedad'] = train['antiguedad'].astype('float64')
(train['antiguedad'] == -999999.0).sum()


# We have 38 special values. If we use a tree based model, we could probably leave it as such or if we use a linear model, we need to map it to mean or some value in the range of 0 to 256.
# 
# Now we can see the distribution plot of this variable.

# In[ ]:


col_series = train.antiguedad.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(col_series.index.astype('int'), col_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Customer Seniority', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# There are few peaks and troughs in the plot but there are no visible gaps or anything as such which is alarming (atleast to me.!)
# 
# So we can also see whether test follows a similar pattern and if it does then we are good.

# In[ ]:


test = pd.read_csv(test_file, usecols=['antiguedad'])
test['antiguedad'] = test['antiguedad'].replace(to_replace=[' NA'], value=np.nan)
test['antiguedad'] = test['antiguedad'].astype('float64')

col_series = test.antiguedad.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(col_series.index.astype('int'), col_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Customer Seniority', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# Peaks are comparatively bigger than the train set. Any implications?
# 
# **RENTA:**
# 
# Gross income of the household.
# 
# To update

# In[ ]:


train = pd.read_csv(train_file, usecols=['renta'])
train.head()


# In[ ]:


unique_values = np.sort(train.renta.unique())
plt.scatter(range(len(unique_values)), unique_values)
plt.show()


# It seems the distribution of rent is highly skewed. There are few very high valued customers present in the data.
# 
# Let us get the mean and median value for this field.

# In[ ]:


train.renta.mean()


# In[ ]:


train.renta.median()


# Now let us see the number of missing values in this field.

# In[ ]:


train.renta.isnull().sum()


# There are quite a few number of missing values present in this field.! We can do some form of imputation for the same. One very good idea is given by Alan in this [script][1].
# 
# We can check the quantile distribution to see how the value changes in the last percentile.
# 
# 
#   [1]: https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python

# In[ ]:


train.fillna(101850., inplace=True) #filling NA as median for now
quantile_series = train.renta.quantile(np.arange(0.99,1,0.001))
plt.figure(figsize=(12,4))
sns.barplot((quantile_series.index*100), quantile_series.values, alpha=0.8)
plt.ylabel('Rent value', fontsize=12)
plt.xlabel('Quantile value', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# As we can see there is a sudden increase in the rent value from 99.9% to 100%. So let us max cap the rent values at 99.9% and then get a box plot.

# In[ ]:


rent_max_cap = train.renta.quantile(0.999)
train['renta'][train['renta']>rent_max_cap] = 101850.0 # assigining median value 
sns.boxplot(train.renta.values)
plt.show()


# From the box plot, we can see that most of the rent values fall between 0 and 300,000.
# 
# Now we can see the distribution of rent in test data as well.

# In[ ]:


test = pd.read_csv(test_file, usecols=['renta'])
test['renta'] = test['renta'].replace(to_replace=['         NA'], value=np.nan).astype('float') # note that there is NA value in test
unique_values = np.sort(test.renta.unique())
plt.scatter(range(len(unique_values)), unique_values)
plt.show()


# *Please note that there is a new value '   NA' present in the test data set while it is not in train data.*
# 
# The distribution looks similar to train though.

# In[ ]:


test.renta.mean()


# In[ ]:


test.fillna(101850., inplace=True) #filling NA as median for now
quantile_series = test.renta.quantile(np.arange(0.99,1,0.001))
plt.figure(figsize=(12,4))
sns.barplot((quantile_series.index*100), quantile_series.values, alpha=0.8)
plt.ylabel('Rent value', fontsize=12)
plt.xlabel('Quantile value', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


test['renta'][test['renta']>rent_max_cap] = 101850.0 # assigining median value 
sns.boxplot(test.renta.values)
plt.show()


# So box and quantile plots are similar to that of the train dataset for rent.!
# 
# **Numerical variables Vs Target variables:**
# 
# Now let us see how the targets are distributed based on the numerical variables present in the data. Let us subset the first 100K rows for the same. 

# In[ ]:


train = pd.read_csv(data_path+"train_ver2.csv", nrows=100000)
target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                             'ind_viv_fin_ult1', 'ind_nomina_ult1',
                             'ind_nom_pens_ult1', 'ind_recibo_ult1']
train[target_cols] = (train[target_cols].fillna(0))
train["age"] = train['age'].map(str.strip).replace(['NA'], value=0).astype('float')
train["antiguedad"] = train["antiguedad"].map(str.strip)
train["antiguedad"] = train['antiguedad'].replace(['NA'], value=0).astype('float')
train["antiguedad"].ix[train["antiguedad"]>65] = 65 # there is one very high skewing the graph
train["renta"].ix[train["renta"]>1e6] = 1e6 # capping the higher values for better visualisation
train.fillna(-1, inplace=True)


# In[ ]:


fig = plt.figure(figsize=(16, 120))
numeric_cols = ['age', 'antiguedad', 'renta']
#for ind1, numeric_col in enumerate(numeric_cols):
plot_count = 0
for ind, target_col in enumerate(target_cols):
    for numeric_col in numeric_cols:
        plot_count += 1
        plt.subplot(22, 3, plot_count)
        sns.boxplot(x=target_col, y=numeric_col, data=train)
        plt.title(numeric_col+" Vs "+target_col)
plt.show()


# Seems all these numerical variables have some predictive power since they show some different behavior between 0's and 1's.
# 
# **Exploring categorical fields:**
# 
# Now let us look at the distribution of categorical fields present in the data by using the first 1 million rows.

# In[ ]:


cols = ["ind_empleado","pais_residencia","sexo","ind_nuevo","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","tipodom","cod_prov","nomprov","ind_actividad_cliente","segmento"]
for col in cols:
    train = pd.read_csv("../input/train_ver2.csv", usecols = ["ncodpers", col], nrows=1000000)
    train = train.fillna(-99)
    len_unique = len(train[col].unique())
    print("Number of unique values in ",col," : ",len_unique)
    if len_unique < 200:
        agg_df = train[col].value_counts()
        plt.figure(figsize=(12,6))
        sns.barplot(agg_df.index, np.log1p(agg_df.values), alpha=0.8, color=color[0])
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Log(Number of customers)', fontsize=12)
        plt.xticks(rotation='vertical')
        plt.show()
    print()
    
       


# Hope this one is helpful.!
