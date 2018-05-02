
# coding: utf-8

# #### Last Update: 2017-10-21

# # Introduction
# ***
# On this Kernel, we will explore the data given to us by the Favorita Corporation, and how this data impacts on the sales.
# 
# We will be seing some graphics and try to understand what's happening below the surface.<br>
# 
# ## Tidy Notebook
# ***
# The idea for this notebook, is to also try to keep it as tidy as possible, in order to make it easy to understand for the reader. This includes trying to write every analysis we do on the markdown cells, and explain the code as much as possible on the code comments. ** COMMENTS ARE EVERY PROGRAMMERS FRIENDS **
# 
# ## About me
# ***
# I've been a developer for more than 12 years and recently, just a few months ago, I got captivated by the magic of machine learning, and how easy it got for us programmers to get into it. My main motivation is learning and grow in the Machine Learning area, but in the process, I'd love to help anyone who's not been on the programming field, to make their career path easier. <br>
# > Finally we get to make AI!
# 
# I'm from Buenos Aires, Argentina, 33 years old.<br>
# If you want; you can reach me on:
# * Linked In: https://www.linkedin.com/in/juanuhaedo/
# * Twitter: https://www.twitter.com/juanumusic/
# * GitHub: https://github.com/HarkDev/
# 
# 
# # First Steps
# ***
# First we will take the mandatory steps of importing the required libaries as well as reading the CSVs into Pandas DataFrames:
# * Import required libraries
# * Read datasets

# **Import Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read CSVs**
# ***
# We will read the csv files, explicitly passing the datatypes of each column for memory performance. This is done with the **dtype** parameter of the `read_csv` method.<br>
# For dates, we will pass the `parse_dates` parameters so that the read_csv turns it's type into `datetime` type

# In[ ]:


# For simplicity, we will create a variable that indicates the location of our input files.
INPUT_PATH = '../input/'

# Read the CSV files
# Holiday Events Dataset
df_holiday_events = pd.read_csv(INPUT_PATH + 'holidays_events.csv', parse_dates=['date'])

# Items dataset
df_items = pd.read_csv(INPUT_PATH + 'items.csv', dtype={'item_nbr':np.uint32, 'class': np.uint16, 'perishable': np.bool})

# Oil dataset
df_oil = pd.read_csv(INPUT_PATH + 'oil.csv', parse_dates=['date'], dtype={'dcoilwtico':np.float16})

# Stores dataset
df_stores = pd.read_csv(INPUT_PATH + 'stores.csv', dtype={'store_nbr':np.uint8, 'cluster':np.uint8})

# Transactions dataset
df_transactions = pd.read_csv(INPUT_PATH + 'transactions.csv', parse_dates=['date'], dtype={'store_nbr':np.uint8, 'transactions':np.uint16})

# Train dataset
df_train = pd.read_csv(INPUT_PATH + 'train.csv', parse_dates=['date'], dtype={'id':np.uint32, 'store_nbr':np.uint8, 'item_nbr': np.uint32, 'onpromotion': np.bool, 'unit_sales': np.float32})

# Test dataset
df_test = pd.read_csv(INPUT_PATH + 'test.csv', parse_dates=['date'], dtype={'id':np.uint32, 'store_nbr':np.uint8, 'item_nbr': np.uint32, 'onpromotion': np.bool})

# Sample Submission dataset
df_sample_submission = pd.read_csv(INPUT_PATH + 'sample_submission.csv', dtype={'id':np.uint32, 'unit_sales':np.float32})


# # Exploratory Analysis
# ***
# Time to get into the data world and see what we can find. We will be exploring each dataset on its own, and then we will do some cross exploration between a lot of them...
# 
# 

# ## Train
# ***
# The train dataset contains the raw data we need to use to train our model. We will first explore this dataset in order to gain some information about how are their numbers achieved.<br>

# ### Unit sales by Date
# ***
# We can see a clear yearly growth on unit sales. This could be because of the company's growth. We'll see this more in detail on the next plots.<br>
# Also, an because we plotted the unit sales for products on promotion and not on promotion, we discover that not always product were on promotion, or at least this wasn't recorded. **Unit sales of Products on promotion started on about the second quarter of year 2014**

# In[ ]:


df_plot = df_train.groupby(by=['date']).agg({'unit_sales':'sum'}).reset_index()
# turn date into datetime type
df_plot.date = pd.to_datetime(df_plot.date)

# Unit Sales by Date
fig, ax = plt.subplots(1)
fig.autofmt_xdate()
#Set the plot figure
fig.set_figheight(10)
fig.set_figwidth(16)
# Create the plot title
plt.title('Unit Sales by Date (Total, On promotion and Not on promotion)', fontsize=20)
# Plot total unit sales
plt.plot(df_plot.date, df_plot.unit_sales)

# plot unit sales on promotion
df_plot = df_train.loc[df_train['onpromotion'] == True].groupby(by=['date']).agg({'unit_sales':'sum'}).reset_index()
plt.plot(df_plot.date, df_plot.unit_sales)

# plot unit sales not on promotion
df_plot = df_train.loc[df_train['onpromotion'] == False].groupby(by=['date']).agg({'unit_sales':'sum'}).reset_index()
plt.plot(df_plot.date, df_plot.unit_sales)

# Set the legend
plt.legend(['Total unit sales','On Promotion','Not On Promotion'], fontsize=14)

# Set the labels
plt.xlabel('Date',fontsize=14)
plt.ylabel('Unit Sales',fontsize=14)


# ### Number of records by "onpromotion."
# ***
# We can see that there are some NaN values for on promotion. Could those be the dates where promotions were not implemented?

# In[ ]:


# Group the data by the "onpromotion" field. Make a count of the values and reset the index.
# reset_index converts the index created (the by parameter) back into a column
df_plot = df_train.fillna(-1).groupby(by=['onpromotion']).agg({'store_nbr':'count'}).reset_index()

# Replace the numeric values with text (for plot readability)
df_plot.onpromotion = df_plot.onpromotion.replace({0:'Not on promotion', 1:'On promotion', -1: 'Missing'})

# Set the plot size
plt.figure(figsize=(10,5))

# Create the barplot
sns.barplot(data=df_plot, x='onpromotion', y='store_nbr')

# Set the plot title
plt.title('# of records by On Promotion', fontsize=16)

# Set the axes labels
plt.xlabel('On Promotion', fontsize=14)
plt.ylabel('# of records', fontsize=14)


# ### Number of nan "onpromotion" per year
# ***
# Plotting the onpromotion values per year, we can see that the MISSING values of "onpromotion" correspond to the time when promotions weren't implemented. **This validates the first plot of Unit Sales by date**

# In[ ]:


# Group the data by the "onpromotion" and date year fields. Make a count of the values and reset the index.
# reset_index converts the index created (the by parameter) back into a column
df_plot = df_train.fillna(-1).groupby(by=[df_train.date.dt.year, 'onpromotion']).agg({'store_nbr':'count'}).reset_index()

# Replace the numeric values with text (for plot readability)
df_plot.onpromotion = df_plot.onpromotion.replace({0:'Not on promotion', 1:'On promotion', -1: 'Missing'})

plt.figure(figsize=(16,6))
sns.barplot(data=df_plot, x='date', y='store_nbr', hue='onpromotion')
plt.title('# of records by On Promotion by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('# of records', fontsize=14)


# ### Stores by Year
# ***
# By looking at the number of stores per year, we can see that there was an increae of stores from the beggining to the last registered date, so we can infer that the increase in sales is also linked to the increase of stores.

# In[ ]:


# Group by year and get the count of unique values (nunique) of store_nbr
df_plot = df_train.groupby(df_train.date.dt.year)['store_nbr'].nunique().reset_index()
#Set the plot figure
plt.figure(figsize=(12,6))
# Create the plot title
plt.title('Number of Stores per Year', fontsize=20)
sns.barplot(data=df_plot, x='date', y='store_nbr')
plt.xlabel('Year', fontsize=14)
plt.ylabel('# of Stores', fontsize=14)


# ## Items
# ***
# The items datas|et contains information about each item. Let's take a look at the data.
# 
# ### Family Count
# ***
# We can see that the family with most items is *GROCERY I*.<br>

# In[ ]:


#Set the plot figure
plt.figure(figsize=(15,6))
# Create the plot title
plt.title('Item Family Distribution', fontsize=20)
# Plot the data
sns.countplot(data=df_items, x='family')
# Rotate the name of the states
plt.xticks(rotation=75, fontsize=12)


# ### Perishable Families
# ***
# Now, let's take a look at the families and how many perishable products each contain.<br>
# As we can see, there are NO families that contain BOTH perishable and non-perishable items<br> 
# This means that the perishable feature is also a feature of the family, and not only of the product.

# In[ ]:


# Get the data grouping by family an6d perishable, and count each occurrency
df_plot = df_items.groupby(['family','perishable']).agg({'item_nbr':'count'}).reset_index()
#Set the plot figure
plt.figure(figsize=(15,6))
# Create the plot title
plt.title('Perishable Items by Family', fontsize=20)
# Plot the data
sns.barplot(data=df_plot, y='item_nbr', x='family', hue='perishable')
# Rotate the name of the states
plt.xticks(rotation=75, fontsize=12)


# ### Ammount of perishable items and non perishable items
# ***
# Let's check the balance between the perishable items and non perishable items.<br>
# By also trying to plot the missings as a 3rd category (Missing), we automatically verify that this column doesn't have missing values! 

# In[ ]:


# Group the data by the "perishable" field. Make a count of the values and reset the index.
# reset_index converts the index created (the "by" parameter) back into a column
df_plot = df_items.fillna(-1).groupby(by=['perishable']).agg({'item_nbr':'count'}).reset_index()

# Replace the numeric values with text (for plot readability)
df_plot.perishable = df_plot.perishable.replace({0:'Non perishable', 1:'Perishable', -1: 'Missing'})

# Set the plot size
plt.figure(figsize=(10,5))

# Create the barplot
sns.barplot(data=df_plot, x='perishable', y='item_nbr')

# Set the plot title
plt.title('# of perishables and non-perishables', fontsize=16)

# Set the axes labels
plt.xlabel('Is perishable?', fontsize=14)
plt.ylabel('# of items', fontsize=14)


# ## Oil
# ***
# This is an interesting dataset. It contains the daily oil price, since ecuador is an oil dependent, and we are trying to understand about sales, which has to do something with economics.<br>
# 

# In[ ]:


# set the plot size
plt.figure(figsize=(14,6))
# Create the plot title
plt.title('Oil Price by Date', fontsize=20)
# Plot total unit sales
plt.plot(df_oil.date, df_oil.dcoilwtico)

# Set the labels
plt.xlabel('Date',fontsize=14)
plt.ylabel('Oil Price',fontsize=14)


# ### A key discovery
# The promotions were implemented almost at the same time that the oil started to drop! 

#  ## Stores
#  ***
#  Let's see what's going on with stores...
#  
#  ### Stores by state
#  ***
#  How many stores by state are there?<br>
#  Pichincha and Guayas are the leaders in number of Stores!

# In[ ]:


# Group by state column and count on the store_nbr column
df_plot = df_stores.groupby('state').agg({'store_nbr':'count'}).reset_index()

# Set the plot size
plt.figure(figsize=(16,8))
# Create the Plot 
sns.barplot(data=df_plot, x='state',y='store_nbr')
# Set the plot title
plt.title('Stores by State', fontsize=18)
#Set the axis text
plt.xlabel('States',fontsize=14)
plt.ylabel('# of stores', fontsize=14)

# Rotate the name of the states
plt.xticks(rotation=75, fontsize=12)


#  ### Stores by city
#  ***
#  How many stores by city are there?<br>
#  Quito and Guayaquil are the leaders in number of Stores by city!

# In[ ]:


# Group by state column and count on the store_nbr column
df_plot = df_stores.groupby('city').agg({'store_nbr':'count'}).reset_index()

# Set the plot size
plt.figure(figsize=(16,8))
# Create the Plot 
sns.barplot(data=df_plot, x='city',y='store_nbr')
# Set the plot title
plt.title('Stores by City', fontsize=18)
#Set the axis text
plt.xlabel('Cities',fontsize=14)
plt.ylabel('# of stores', fontsize=14)

# Rotate the name of the states
plt.xticks(rotation=75, fontsize=12)


# # Holiday Events
# ***
# Let's take a look at the holiday events dataset.
# 
# ## How many holidays?
# *** 
# An overview of the total number of holidays by Locale.<br>
# By the looks of it, we can say that most holidays, happen on a local scale, and also, this is the only time that they happen on a regional scale.
# 
# The rest of the holiday event types happen eithe

# In[ ]:


# Set the plot size
plt.figure(figsize=(10,5))

# Create the barplot
sns.countplot(data=df_holiday_events, x='type', hue='locale')

# Set the plot title
plt.title('Ammount of holiday events per type and locale', fontsize=16)

# Set the axes labels
plt.xlabel('Holiday EventType', fontsize=14)
plt.ylabel('# of Holiday Events', fontsize=14)


# # Cross Analysis
# ***
# Let's merge some datasets and see what we can find!
# 
# ## Merge Train with Items
# ***
# Let's merge the items and train datasets

# In[ ]:


# Merge train and items and store it on the train variable
df_train = pd.merge(df_train, df_items, on='item_nbr', how='left')


# ## Unit Sales by Date (perishable  and non perishable)
# ***
# We will now plot the unit sales separating perishable and non perishable items to see how those move.

# In[ ]:


df_plot = df_train.groupby(by=['date','perishable']).agg({'unit_sales':'sum'}).reset_index()
# turn date into datetime type
df_plot.date = pd.to_datetime(df_plot.date)

# Unit Sales by Date
fig, ax = plt.subplots(1)
fig.autofmt_xdate()
#Set the plot figure
fig.set_figheight(10)
fig.set_figwidth(16)
# Create the plot title
plt.title('Unit Sales by Date (Perishable and non perishable)', fontsize=20)

# Plot total perishable unit sales
plt.plot(df_plot.loc[df_plot['perishable'] == True,'date'], df_plot.loc[df_plot['perishable'] == True,'unit_sales'])
# Plot total non-perishable unit sales
plt.plot(df_plot.loc[df_plot['perishable'] == False,'date'], df_plot.loc[df_plot['perishable'] == False,'unit_sales'])

# Set the legend
plt.legend(['Perishable','Non Perishable'], fontsize=14)

# Set the labels
plt.xlabel('Date',fontsize=14)
plt.ylabel('Unit Sales',fontsize=14)


# ## Unit Sales by Item Family (perishable and non perishable)
# ***
# How do sales variate on each product family?
# 
# Now this is a really messy plot, but we can see some information about the life of some item families.<br>
# **Perishables**:
# * The `PRODUCE` family was sold at certain points, and after that it started to be sold continuously.
# * The `DAIRY` family is the most sold product after `PRODUCE`
# 
# **Non-Perishables**:
# * As we saw earlier, the `GROCERY I` family is the one sold the most.

# In[ ]:


df_plot = df_train.groupby(by=['date','family','perishable']).agg({'unit_sales':'sum'}).reset_index()
# turn date into datetime type
df_plot.date = pd.to_datetime(df_plot.date)

# Unit Sales by Date
fig, ax = plt.subplots(1)
fig.autofmt_xdate()
#Set the plot figure
fig.set_figheight(20)
fig.set_figwidth(16)

# Set the first subplot
plt.subplot(2,1,1)

# Create the plot title
plt.title('Unit Sales by Item Family (Perishables)', fontsize=20)

# Get each item family.
families = df_plot.loc[df_plot['perishable'] == True].family.unique()

# For each family
for family in families:
    
    # Filter the dataframe by the family
    df_curr_plot = df_plot.loc[(df_plot['family'] == family)]
    
    # Plot unit sales of the current item family
    plt.plot(df_curr_plot.date, df_curr_plot.unit_sales, label=family)

# Set the legends
plt.legend(families, fontsize=8)

# Set the labels
plt.xlabel('Date',fontsize=14)
plt.ylabel('Unit Sales',fontsize=14)

# Set the second subplot
plt.subplot(2,1,2)

# Create the plot title
plt.title('Unit Sales by Item Family (Non Perishables)', fontsize=20)

# Get each item family.
families = df_plot.loc[df_plot['perishable'] == False].family.unique()

# For each family
for family in families:
    
    # Filter the dataframe by the family
    df_curr_plot = df_plot.loc[(df_plot['family'] == family)]
    
    # Plot unit sales of the current item family
    plt.plot(df_curr_plot.date, df_curr_plot.unit_sales, label=family)

# Set the legends
plt.legend(fontsize=10)#, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Set the labels
plt.xlabel('Date',fontsize=14)
plt.ylabel('Unit Sales',fontsize=14)

plt.tight_layout()


# # To be continued...
# ***
# We still have a lot more data and datsets to explore, so keep coming back for new insights!
