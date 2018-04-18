
# coding: utf-8

# "***Feature Engineering is an art***". Someone once said to spend more time on deriving more and **meaningful** features. Why did I bold meaningful? Because infomation fed to the model must make sense in whatever form it is given. Creating loads of features having no sense is of no use at all. To add more features, an added advantage would be understanding the real world situation at hand or holding sme (subject ,atter experience).
# 
# On this second kernel of mine let's see how we can garner more features from the existing ones. This kernel did not run properly upon execution. So I have decided to incorporate memory reduction techniques adopted from my previous kernel mentioned below.
# 
# To see my first kernel on how to reduce memory effectively [SEE THIS KERNEL](https://www.kaggle.com/jeru666/memory-reduction-and-data-insights/notebook)
# 
# ## Loading libraries and data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

#df_members = pd.read_csv('../input/members_v3.csv')
df_transactions = pd.read_csv('../input/transactions.csv')


# Have a quick look at the head to see what features we can create!

# In[ ]:


print(df_transactions.shape)
df_transactions.head()


# ## Memory Reduction
# As stated earlier, let us first reduce memory wherever  possible. But first how much memory does df_train dataframe consume?

# In[ ]:


mem = df_transactions.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# The following functions check whether a column's datatype can be reduced based on the maximum and minimum value present in that column

# In[ ]:


def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

change_datatype(df_transactions)

def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(df_transactions)

mem = df_transactions.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# We have reduced memory of **transactions** dataframe from 1.4 GB to ~500 MB. Now performing the same for **members** dataframe.

# In[ ]:


#--- Members dataframe
mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")

change_datatype(df_members)
change_datatype_float(df_members)

#--- Recheck memory of Members dataframe
mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# We have reduced memory almost by 50%. Let us see the column types to notice the changes:

# In[ ]:


print(df_transactions.dtypes, '\n')
print(df_members.dtypes)


# ## Transactions dataframe
# 
# Now let us create new features!!
# 
# Before creating features let us keep a count of the number of columns we have at the moment:

# In[ ]:


len(df_transactions.columns)


# ## Feature 1 : ***discount***
# We can create a **discount** column to see how much discount was offered to the customer.

# In[ ]:


df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']

df_transactions['discount'].unique()


# ## Feature 2 : ***is_discount***
# Let us create another column **is_column** to check whether the customer has availed any discount or not. 
# 
# Why this feature? Oh come on you now why!

# In[ ]:


df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
print(df_transactions['is_discount'].head())
print(df_transactions['is_discount'].unique())


# ## Feature 3 : ***amount_per_day***
# A new column featuring amount per-day can be added.

# In[ ]:


df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']
df_transactions['amt_per_day'].head()


# Now we have two date columns :
# * transaction_date	
# * membership_expire_date
# 
# Let us see if we can extract some features from them!

# In[ ]:


date_cols = ['transaction_date', 'membership_expire_date']
print(df_transactions[date_cols].dtypes)


# Both the date columns are of **integer** type. We have to convert them to type **datetime**.
# 

# Converting date columns from integer to datetime:

# In[ ]:


for col in date_cols:
    df_transactions[col] = pd.to_datetime(df_transactions[col], format='%Y%m%d')
    
df_transactions.head()


# ## Feature 4 : ***membership_duration***
# 
# The difference between **transaction_date** and	**membership_expire_date** would give us membership duration.
# 
# Here we  find the differnce between these two columns in terms of days and months and later preserve the result as type integer.

# In[ ]:


#--- difference in days ---
df_transactions['membership_duration'] = df_transactions.membership_expire_date - df_transactions.transaction_date
df_transactions['membership_duration'] = df_transactions['membership_duration'] / np.timedelta64(1, 'D')
df_transactions['membership_duration'] = df_transactions['membership_duration'].astype(int)

 
#---difference in months ---
#df_transactions['membership_duration_M'] = (df_transactions.membership_expire_date - df_transactions.transaction_date)/ np.timedelta64(1, 'M')
#df_transactions['membership_duration_M'] = round(df_transactions['membership_duration_M']).astype(int)
#df_transactions['membership_duration_M'].head()


# Let us check the number of columns now:

# In[ ]:


len(df_transactions.columns)


# Now that we have created 5 more columns, we have increased the memory consumption as well. So we will run the previous functions again to keep the memory in check.

# In[ ]:


change_datatype(df_transactions)
change_datatype_float(df_transactions)


# ## Members dataframe
# Now let us see the members.csv file

# In[ ]:


df_members.head()


# In[ ]:


#--- Number of columns 
len(df_members.columns)


# We will have to convert the date columns as before:

# In[ ]:


date_cols = ['registration_init_time', 'expiration_date']

for col in date_cols:
    df_members[col] = pd.to_datetime(df_members[col], format='%Y%m%d')


# ## Feature 5 : ***registration_duration***

# In[ ]:


#--- difference in days ---
df_members['registration_duration'] = df_members.expiration_date - df_members.registration_init_time
df_members['registration_duration'] = df_members['registration_duration'] / np.timedelta64(1, 'D')
df_members['registration_duration'] = df_members['registration_duration'].astype(int)

#---difference in months ---
#df_members['registration_duration_M'] = (df_members.expiration_date - df_members.registration_init_time)/ np.timedelta64(1, 'M')
#df_members['registration_duration_M'] = round(df_members['registration_duration_M']).astype(int)


# In[ ]:


#--- Reducing and checking memory again ---
change_datatype(df_members)
change_datatype_float(df_members)

#--- Recheck memory of Members dataframe
mem = df_members.memory_usage(index=True).sum()
print(mem/ 1024**2," MB")


# ## Merging dataframes (*transactions* and *members*)
# Now let us combine both both the dataframes(**transactions** and **members**) based on **msno** and see if we can create anything more.

# In[ ]:


#-- merging the two dataframes---
df_comb = pd.merge(df_transactions, df_members, on='msno', how='inner')

#--- deleting the dataframes to save memory
del df_transactions
del df_members

df_comb.head()


# In[ ]:


#df_comb = df_comb.drop('msno', 1)
mem = df_comb.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))


# ## Feature 6 : ***reg_mem_duration***
# 
# After merging both the dataframes, we observe that the  average **registration_duration** is much longer than that of **membership_duration**.
# 
# So it is highly likely that customers renew their membership on a monthly or quarterly basis.
# 
# We can create another column stating the difference between the registration and membership duration. I don't know if it makes sense, but let's create it.

# In[ ]:


df_comb['reg_mem_duration'] = df_comb['registration_duration'] - df_comb['membership_duration']
#df_comb['reg_mem_duration_M'] = df_comb['registration_duration_M'] - df_comb['membership_duration_M']
df_comb.head()


# ## Feature 7 : ***autorenew_&_not_cancel***
# 
# A binary feature to see whether mebers have** auto renewed** and **not cancelled** at the same time:
# * **auto_renew** = 1 and
# * **is_cancel** = 0

# In[ ]:


df_comb['autorenew_&_not_cancel'] = ((df_comb.is_auto_renew == 1) == (df_comb.is_cancel == 0)).astype(np.int8)
df_comb['autorenew_&_not_cancel'].unique()


# ## Feature 8 : ***notAutorenew_&_cancel***
# Binary feature to predict possible churning if 
# * **auto_renew** = 0 and
# * **is_cancel** = 1

# In[ ]:


df_comb['notAutorenew_&_cancel'] = ((df_comb.is_auto_renew == 0) == (df_comb.is_cancel == 1)).astype(np.int8)
df_comb['notAutorenew_&_cancel'].unique()


# ## Feature 9 : *long_time_user*
# 
# A binary feature to check whether user has been registered for more than a year. This can prompt the company to offer the user some discount.

# In[ ]:


df_comb['long_time_user'] = (((df_comb['registration_duration'] / 365).astype(int)) > 1).astype(int)


# ## Important Note:
# I have noticed that columns of type **datetime64[ns]** consume a lot of memory. So after having extracted features from these columns they can be dropped to reduce memory!!

# In[ ]:


datetime_cols = list(df_comb.select_dtypes(include=['datetime64[ns]']).columns)


# Dropping columns of type **datetime64[ns]**.

# In[ ]:


df_comb = df_comb.drop([datetime_cols], 1)


# You can also include duration periods in terms of months and years to get insight analysis.
# 
# ## Share your ideas as well!!
# 
# ## Do upvote if you find it useful!!
