
# coding: utf-8

# # Introduction
# In this exploratory data analysis, I plan on analyzing the various data sets provided to us. I started with the training and test sets, sample submission, and store, item, and transaction metadata but plan on including the other data sets as well at a later point. Below, I will investigate the distribution of observations with respect to different features. I also inspected the distribution of the target variable, which is unit sales, for outliers. I found some interesting differences between the distribution of observations in the training and tests sets that are worth considering before training models. I also found that, within the store metadata, that store cluster is a subdivision of store type and that one state can have many different store types. Please upvote if you find this useful! Enjoy.

# In[ ]:


# import necessary modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc


# In[ ]:


# read data sets (this may take a few minutes)
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_sub = pd.read_csv("../input/sample_submission.csv")
df_stores = pd.read_csv("../input/stores.csv")
df_items = pd.read_csv("../input/items.csv")
df_trans = pd.read_csv("../input/transactions.csv")
df_oil = pd.read_csv("../input/oil.csv")
df_holiday = pd.read_csv("../input/holidays_events.csv")


# # Training set

# In[ ]:


# inspect training set
print(df_train.shape)
df_train.head()


# The training set has 125,497,040 rows and 6 columns: row id, date, store number, item number, unit sales (keep in mind that this can be an integer, float, or -1, which represents a returned item), and whether there was a promotion for a particular item. 

# ## Dates
# 
# Now, let's get a sense of the time range for which the data was collected.

# In[ ]:


# convert date to datetime
df_train["date"] =  pd.to_datetime(df_train["date"])


# In[ ]:


df_train["date"].dt.year.value_counts(sort = False).plot.bar()


# They have collected data from 2013 to 2017. There is an increase in the number of observations for each year except 2017 but this is probably because it is not yet over. Note that the training set is quite large so I will focus my analysis hereafter on 2016 data.

# In[ ]:


df_train_2016 = df_train[df_train["date"].dt.year == 2016]
del df_train; gc.collect() # free up some memory


# Let's take a look at how the data is distributed by month.

# In[ ]:


df_train_2016["date"].dt.month.value_counts(sort = False).plot.bar()


# The observations are almost uniformly distributed by month. The maximum occurs in December and the minimum in February. How about by day of the month?

# In[ ]:


df_train_2016["date"].dt.day.value_counts(sort = False).plot.bar()


# Again, the observations are almost uniformly distributed by day.
# 
# ## Stores
# Now, let's determine how many stores there are and the distribution of observations for each store.

# In[ ]:


df_train_2016["store_nbr"].unique()


# There is data for 54 stores in 2016. At a later point, I will analyze the stores data also provided. Now, how about the distibution of observations for each store?

# In[ ]:


df_train_2016["store_nbr"].value_counts(sort = False).plot.bar()


# The interpretation of this plot is as follows. The y-axis is the number of observations corresponding to a particular store. A store with more observations does not necessarily outperform a store with fewer observations. This is because some stores may offer a wider range of products but push less volume. This would inflate their numbers in this plot. We can interpret the y-axis as a measure of the store's variety.
# 
# ## Items
# Now, let's determine how many unique items there are overall and in each store.

# In[ ]:


df_train_2016["item_nbr"].unique().shape[0]


# There were 3886 different types of items either sold or returned during 2016.

# In[ ]:


stores = np.arange(1, 55)
items_store = np.zeros((54, ))
for i, store in enumerate(stores) :
    items_store[i] = df_train_2016["item_nbr"][df_train_2016["store_nbr"]                                                == store].unique().shape[0]
sns.barplot(stores, items_store)


# As expected, this is very similar to the last bar plot because it measures the variety of items in each store. Interestingly, store 52 has 0 unique items. This is because there is no store number 52.
# 
# ## Item sales
# This is our target variable so it is very important to get a sense of its distribution.

# In[ ]:


df_train_2016["unit_sales"].describe()


# It is probably a good thing that the mean and median for unit sales is positive, otherwise the company would be losing money. Suprisingly, on one day, 4673 items were returned. I wonder if this corresponds to some sort of outbreak or health concern for a particular product. On the other hand, on another day, 89440 items were purchased. Perhaps this was before some sort of natural disaster (*e.g.* a hurricane).
# 
# ## Promotions
# Now, let's find out how many items were purchased by coupon clippers.

# In[ ]:


df_train_2016["onpromotion"].value_counts()


# In[ ]:


3514584 / (3514584 + 31715287) * 100


# About 10% of items are purchased on promotion.
# 
# ## Missing data and outliers
# A less exciting but very important step in this analysis is to determine if any data is missing and if there are any outliers in the target data.

# In[ ]:


df_train_2016.isnull().sum()


# Yay! There is no missing data in the training set. How about outliers in the target variable?

# In[ ]:


unit_sales = df_train_2016["unit_sales"].values
del df_train_2016; gc.collect()


# Again, I had to remove some data to free up memory for plots.

# In[ ]:


plt.scatter(x = range(unit_sales.shape[0]), y = np.sort(unit_sales))


# This plot isn't super informative but does show us that quite a few outliers exist on the sales side.

# In[ ]:


del unit_sales; gc.collect()


# # Test set

# In[ ]:


# inspect test set
print(df_test.shape)
df_test.head()


# There are 3,370,464 rows in the test set, which is approximately 1.5 orders of magnitude smaller than the training set. Since the training set is so large, we can afford to reserve more of this data for validation, which will promote more robust predictions. The columns in the test set are similar to those in the training set except the unit sales column is missing. Now, let's determine the range of dates present in the test set and compare to the training set.

# In[ ]:


# convert date to datetime
df_test["date"] =  pd.to_datetime(df_test["date"])
df_test["date"].dt.year.value_counts(sort = False).plot.bar()


# In[ ]:


df_test["date"].dt.month.value_counts(sort = False).plot.bar()


# In[ ]:


df_test["date"].dt.day.value_counts(sort = False).plot.bar()


# The test set only samples observations from July 16-31, 2017.
# 
# ## Stores
# Now, let's see if the same store numbers appear in the test set as do in the training set.

# In[ ]:


df_test["store_nbr"].value_counts(sort = False).plot.bar()


# The stores are sampled uniformly and, unlike the training set, store 52 has observations.
# 
# ## Items
# It's also useful to know how many unique items appear in the test set.

# In[ ]:


df_test["item_nbr"].unique().shape[0]


# There are actually more unique items in the test set (3901) than in the training set (3886). This makes sense because the supermarket adapts over times to meet the diverse needs of its customers.

# In[ ]:


stores = np.arange(1, 55)
items_store = np.zeros((54, ))
for i, store in enumerate(stores) :
    items_store[i] = df_test["item_nbr"][df_test["store_nbr"]                                          == store].unique().shape[0]
sns.barplot(stores, items_store)


# For the test set, the number of unique items in each store (a measure of that stores variety) are exactly the same. This is suprising because we would expect that each store caters to slightly different communities and their needs and therefore the variety should not be uniform. However, it is also possible that their itention was to provide a test set with balanced observations.
# 
# ## Promotions
# Last for the test set, let's see if the percentage of items on sale matches that of the training set.

# In[ ]:


df_test["onpromotion"].value_counts()


# In[ ]:


198597/(198597 + 3171867) * 100


# The percentage of items on sale for the test set is a little more than half of that for the training set ($\approx$10%). From this analysis of the test set, we find that while it has a similar structure to that of the training set, it's observations correspond to a time period not sampled by the training set. Additionally, the number of unique items, each stores' variety of items, and the number of observations corresponding to promotional purchases are different than the training set. Therefore, it may be beneficial to sample the training set in a way that more closely matches the distribution of items in the test set.

# # Sample submission
# Next, let's take a look at what they would like us to submit.

# In[ ]:


print(df_sub.shape)
df_sub.head()


# Just like the test set, there are 3,370,464 rows in the sample submission. The submission file has two columns, the row id and the unit sales. Let's see if the row ids in the sample submission match that of the test set.

# In[ ]:


(df_sub["id"] - df_test["id"]).sum()


# Yes, they do! So, at the very least, our task boils down to predicting the number of unit sales given the date, store number, item number, and promotional status. But we are also provided with other information about the stores, items, transactions, economic health, and holidays. Let's clear some memory before looking at this metadata.

# In[ ]:


del df_test, df_sub; gc.collect()


# # Stores
# First, let's inspect the contents of the data set.

# In[ ]:


print(df_stores.shape)
df_stores.head()


# Store metadata includes the store number, the city in which the store is located, the state in which the city is located, type, and cluster. A cluster is a similar grouping of stores perhaps in the type of products they sell, their size, or their locations. It is not clear how the different types are defined or how they differ from clusters. Are there any missing values in the stores data set?

# In[ ]:


df_stores.isnull().sum()


# Nope!
# 
# ## Cities

# In[ ]:


df_stores["city"].unique().shape[0]


# In[ ]:


df_stores["city"].value_counts(sort = False).plot.bar()


# The 54 stores are located in 22 different cities. The city with the most stores is Quinto and there are a number of cities that only have one store.

# ## State

# In[ ]:


print(df_stores["state"].unique().shape[0])
df_stores["state"].value_counts(sort = False).plot.bar()


# Of the 24 provinces in Ecuador (https://en.wikipedia.org/wiki/Provinces_of_Ecuador), 16 of them have supermarkets.
# 
# ## Type and cluster
# Since we don't know what type means and the cluster definition is a little bit vague, let's compare them.

# In[ ]:


print(df_stores["type"].unique().shape[0])
df_stores["type"].value_counts(sort = False).plot.bar()


# In[ ]:


print(df_stores["cluster"].unique().shape[0])
df_stores["cluster"].value_counts(sort = False).plot.bar()


# From these plots, we can see a few differences between type and cluster. First, there are fewer types (5, A through E) than clusters (1-17). Second, for the most part, there are more stores per type than stores per cluster. Let's see how each cluster is represented by type.

# In[ ]:


df_stores.groupby(["type", "cluster"]).size()


# For the most part, stores in a particular cluster have one type. However, there is one exception. Cluster 10 can be type B, D, and E. How about state by type?

# In[ ]:


df_stores.groupby(["type", "state"]).size()


# There are many states that have different types of stores. For example, Guayas has stores of type A, B, D, and E. This tells us that clusters are, for the most part, a subdivision of store type.

# In[ ]:


del df_stores; gc.collect()


# # Items
# Now let's take a look at the items metadata.

# In[ ]:


print(df_items.shape)
df_items.head()


# The items data has 4100 rows and 4 columns: item number, family, class, and whether or not the item is perishable. Now, let's see how many different item families there are.

# In[ ]:


print(df_items["family"].unique().shape[0])
df_items["family"].value_counts(sort = False).plot.bar()


# There are 33 differenet item families ranging from school and office supplies to poulty. The most popular family is called "grocery 1". This must be a generic groceries column. How about item classes?

# In[ ]:


print(df_items["class"].unique().shape[0])
print(df_items["class"].value_counts()[0:5])
df_items["class"].plot.hist(bins = 50)


# There are 337 different item classes and the number of items in each class is plotted above. There is an abundance of items in classes 1016 (133), 1040 (110), 1124 (100), 1034 (98), and 1122 (81). Finally, let's find out the percentage of items that are perishable.

# In[ ]:


df_items["perishable"].value_counts()


# In[ ]:


986 / (986 + 3114) * 100


# 24% of the of the items are perishable.

# In[ ]:


del df_items; gc.collect()


# # Transactions
# We're getting there only a few more data sets to analyze. Next up is the transactions metadata.

# In[ ]:


print(df_trans.shape)
print(df_trans.head())
df_trans.isnull().sum()


# There are 83488 columns in the transactions data and three columns: date, store number, and the number of transactions. This data tells us how many transactions were made in each store on each business day since they started collecting data. There is also no missing data.

# In[ ]:


df_trans["date"] =  pd.to_datetime(df_trans["date"])


# In[ ]:


df_trans["date"].dt.year.value_counts(sort = False).plot.bar()


# There is data from 2013 to 2017 with an increase in the number of observations each year except for 2017, which is not yet complete. This is similar to the training data. This tells us that the number of stores is increasing each year and perhaps they are now open on more days.

# In[ ]:


df_trans["date"].dt.month.value_counts(sort = False).plot.bar()


# In general, there are more observations for January through July than the colder months. I'm not sure why this is but perhaps there are more store closures during this month for holiday or weather reasons.

# In[ ]:


df_trans["date"].dt.day.value_counts(sort = False).plot.bar()


# The distribution of observations by day of the month is nearly uniform with minima at the 1st, 25th, and 31st days of the month. The dips at the 1st and 25th are likely due to New Years and Christmas whereas the dip at the 31st is likely due to the fact that not all months have 31 days. Now, let's take a look at the distribution of observations for each store.

# In[ ]:


df_trans["store_nbr"].value_counts(sort = False).plot.bar()


# There are a lot of stores with > 1600 observations. These were all probably open prior to 2013. Stores with < 1600 observations were all probably opened after 2013, the stores with the fewest observations being opened most recently.

# In[ ]:


df_trans["transactions"].plot.hist(bins = 100)


# In[ ]:


del df_trans; gc.collect()


# This is a very informative plot as it tells us that the total number of transactions is approximately Poisson distributed. Perhaps the number of transactions for particular items are also Poisson distributed.
# 
# # Conclusion
# We looked at how the observations in both the training and test sets are distributed by date, store, item, and promotions. We also got more familiar with the distribution of the target value, unit sales, which may require processing to remove outliers. We noticed that there are some differences between the training and test sets with regards to the time period sampled, the number of unique items, each stores' variety, and percentage of sale items. We also discovered that store cluster is a subdivision of store type and that one state can have many different store types.
# 
# # Future directions
# In the future, I plan on exploring the other data sets that were provided. I hope to update this kernel soon! Thanks for reading.
