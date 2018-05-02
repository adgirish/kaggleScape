
# coding: utf-8

# # Introduction
# This is an Exploratory Data Analysis of crime in Vancouver from 2003 to 2017.
# 
# 
# <br>
# # Importing the Data Analysis and Visualization packages

# In[1]:


# Import data manipulation packages
import numpy as np
import pandas as pd

# Import data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Processing and Transforming the data

# In[2]:


# Import CSV file as a pandas df (Data Frame)
df = pd.read_csv('../input/crime.csv')

# Take a look at the first entries
df.head()


# ### Remove unnecessary columns
# 
# Column MINUTE can be deleted as we don't need to go to the minute level.
# Columns regarding location (X, Y, Neighbourhood, Hundred block) won't be used in this analysis, but they will be used in the Tableau dashboard (link in the end of this section), so let's keep them for now.

# In[3]:


# Dropping column. Use axis = 1 to indicate columns and inplace = True to 'commit' the transaction
df.drop(['MINUTE'], axis = 1, inplace=True)


# ### Missing Values

# In[4]:


# Let's take a look into our data to check for missing values and data types
df.info()


# We can see that we have 530,652 entries, but some columns (HOUR, HUNDRED_BLOCK and NEIGHBOURHOOD) have less, which means that there are missing values. Let's fill them.

# In[5]:


# As HOUR is a float data type, I'm filling with a dummy value of '99'. For others, filling with 'N/A'
df['HOUR'].fillna(99, inplace = True)
df['NEIGHBOURHOOD'].fillna('N/A', inplace = True)
df['HUNDRED_BLOCK'].fillna('N/A', inplace = True)


# ### Transforming the DATE column
# The date is separated in different columns (YEAR, MONTH, DAY) , let's combine them into a single column and add it as a new column called 'DATE'

# In[6]:


# Use pandas function to_datetime to convert it to a datetime data type
df['DATE'] = pd.to_datetime({'year':df['YEAR'], 'month':df['MONTH'], 'day':df['DAY']})


# It might be useful to have the day of the week...

# In[7]:


# Let's use padas dt.dayofweek (Monday=0 to Sunday=6) and add it as a column called 'DAY_OF_WEEK'
df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek


# When working with time series data, using the date as the index is helpful.

# In[8]:


# Change the index to the colum 'DATE'
df.index = pd.DatetimeIndex(df['DATE'])


# This dataset was extracted in 2017-07-18 and it contains partial data for this month. I'm excluding them to keep full months only.

# In[9]:


# Filtering the data to exclude month of July 2017
df = df[df['DATE'] < '2017-07-01']


# ### Creating Categories
# Let's see the type of crimes that we have in our data and categorize them.

# In[10]:


# Using pandas value_counts function to aggregate types
df['TYPE'].value_counts().sort_index()


# From the types above, I'm creating the following categories: Break and Enter, Theft, Vehicle Collision, Others

# In[11]:


# Create a function to categorize types, using an 'if' statement.
def category(crime_type):
    if 'Theft' in crime_type:
        return 'Theft'
    elif 'Break' in crime_type:
        return 'Break and Enter'
    elif 'Collision' in crime_type:
        return 'Vehicle Collision'
    else:
        return 'Others'


# In[12]:


# Apply the function and add it as CATEGORY column
df['CATEGORY'] = df['TYPE'].apply(category)


# Although 'vehicle collision' is included in this data, I'll separate it apart because I believe it is a different kind of crime.

# In[13]:


vehicle_collision = df[df['CATEGORY'] == 'Vehicle Collision']
crimes = df[df['CATEGORY'] != 'Vehicle Collision']


# Now we have our data frame named __*crimes*__ ready to analyze.

# # Exploratory Data Analysis
# ---
# ### What's the distribution of crimes per day?
# Let's start with a histogram of crimes per day.

# In[14]:


# Using resample('D') to group it by day and size() to return the count
plt.figure(figsize=(15,6))
plt.title('Distribution of Crimes per day', fontsize=16)
plt.tick_params(labelsize=14)
sns.distplot(crimes.resample('D').size(), bins=60);


# * We can see that the distribution looks like a *normal distribution* with a mean around *95 crimes per day*.
# 
# * There is one outlier over 600. Let's find out which day it is.
# 
# <br>
# ### Outlier

# In[15]:


# Using idxmax() to find out the index of the max value
crimes.resample('D').size().idxmax()


# So the day was 2011-06-15. 
# 
# Let's make a *time series graph* with crimes per day.

# In[16]:


# Create a Upper Control Limit (UCL) and a Lower Control Limit (LCL) without the outlier
crimes_daily = pd.DataFrame(crimes[crimes['DATE'] != '2011-06-15'].resample('D').size())
crimes_daily['MEAN'] = crimes[crimes['DATE'] != '2011-06-15'].resample('D').size().mean()
crimes_daily['STD'] = crimes[crimes['DATE'] != '2011-06-15'].resample('D').size().std()
UCL = crimes_daily['MEAN'] + 3 * crimes_daily['STD']
LCL = crimes_daily['MEAN'] - 3 * crimes_daily['STD']

# Plot Total crimes per day, UCL, LCL, Moving-average
plt.figure(figsize=(15,6))
crimes.resample('D').size().plot(label='Crimes per day')
UCL.plot(color='red', ls='--', linewidth=1.5, label='UCL')
LCL.plot(color='red', ls='--', linewidth=1.5, label='LCL')
crimes_daily['MEAN'].plot(color='red', linewidth=2, label='Average')
plt.title('Total crimes per day', fontsize=16)
plt.xlabel('Day')
plt.ylabel('Number of crimes')
plt.tick_params(labelsize=14)
plt.legend(prop={'size':16});


# We can see some days over the Control Limits, indicating *signals*. Also, the period of 2003 to 2008 is above the average. Maybe a different Control Limit could be done for that period, but that's ok for now.
# 
# Let's focus on the day 2011-06-15 which is way above. Is that an error on the data?
# 
# Let's *drill down* and find out...

# In[17]:


# Find out how many crimes by getting the length
len(crimes['2011-06-15'])


# In[18]:


# Check how many crimes per type
crimes['2011-06-15']['TYPE'].value_counts().head(5)


# In[19]:


# Check how many crimes per neighbourhood
crimes['2011-06-15']['NEIGHBOURHOOD'].value_counts().head(5)


# In[20]:


# Check how many crimes per hour
crimes['2011-06-15']['HOUR'].value_counts().head(5)


# There are 647 occurrences, mostly mischief type, in the Central Business District, around the same hour but no exactly. They don't really seem to be duplicated entries.
# 
# 
# <br>
# ### The Stanley Cup Riot
# 
# I moved to Canada in 2016 and I had no idea if something happened that day, so a Google search showed me it was the Stanley Cup's final game, Boston Bruins vs Vancouver Canucks. There were 155,000 people watching it in the downtown area. Before the game was over, as the Vancouver Canucks was about to loose it, a big riot started. It seems that it was ugly... If you want to know more about it, there is a <a href="http://www2.gov.bc.ca/assets/gov/law-crime-and-justice/criminal-justice/prosecution-service/reports-publications/stanley-cup-riot-prosecutions.pdf" target="_blank">21-pages report</a> by the BC Ministry of Justice.
# 
# So that day wasn't an error on the data and it showed me something that happened in Vancouver that I wasn't expecting at all. Interesting...
# 
# 
# <br>
# ### Which days have the highest and lowest average number crimes?

# In[21]:


# Create a pivot table with day and month; another that counts the number of years that each day had; and the average. 
crimes_pivot_table = crimes[(crimes['DATE'] != '2011-06-15')].pivot_table(values='YEAR', index='DAY', columns='MONTH', aggfunc=len)
crimes_pivot_table_year_count = crimes[(crimes['DATE'] != '2011-06-15')].pivot_table(values='YEAR', index='DAY', columns='MONTH', aggfunc=lambda x: len(x.unique()))
crimes_average = crimes_pivot_table/crimes_pivot_table_year_count
crimes_average.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Using seaborn heatmap
plt.figure(figsize=(7,9))
plt.title('Average Number of Crime per Day and Month', fontsize=14)
sns.heatmap(crimes_average.round(), cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f");


# Blue means good days. Red bad days. White average days.
# * The calmest day of crime is Christmas Day, December 25 (30% below average). Criminals also celebrate with families...
# * The worst day is New Year's Day, January 1 (30% above average). And then after Christmas celebration, criminals take advantage of drunk people at parties?
# * The first day of the month is a busy day for all months.
# * Halloween (October 30,31 and November 1) are also dangerous days.
# * The second week of summer months are usually the most dangerous.
# * BC Day (August 7) long weekend have high averages.
# 
# <br>
# ### Is crime descreasing or increasing?
# Now let's plot the number of crimes per month and a moving average:

# In[22]:


# Using resample 'M' and rolling window 12
plt.figure(figsize=(15,6))
crimes.resample('M').size().plot(label='Total per month')
crimes.resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-months Moving Average')

plt.title('Crimes per month', fontsize=16)
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# * Average number of crimes per month decreased from 4,000 crimes per month to around 2,400 in the period of 2003 to 2011. That's really good!
#   * Vancouver hosted the 2010 Winter Olympics and the election for host city happened in 2003. So maybe the decrease of crimes is related to this event?
# 
# 
# * From 2011 to 2014, the moving average was around the same.
# * From 2014, the average has increased and 2016 reached similar leves of 2008.
# 
# <br>
# ### Is this trend the same for all categories?
# Let's redo the plot but with categories.

# In[23]:


# Using pivot_table to groub by date and category, resample 'M' and rolling window 12
crimes.pivot_table(values='TYPE', index='DATE', columns='CATEGORY', aggfunc=len).resample('M').sum().rolling(window=12).mean().plot(figsize=(15,6), linewidth=4)
plt.title('Moving Average of Crimes per month by Category', fontsize=16)
plt.xlabel('')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# * Theft is the major category. The decrease and increase that we saw in the average number of crimes per month was mainly because of the variation in this category.
# 
# <br>
# ### Is there any trend within the year?
# Let's make a heatmap with months and categories

# In[24]:


# Create a pivot table with month and category. 
crimes_pivot_table = crimes.pivot_table(values='TYPE', index='CATEGORY', columns='MONTH', aggfunc=len)

# To compare categories, I'm scaling each category by diving by the max value of each one
crimes_scaled = pd.DataFrame(crimes_pivot_table.iloc[0] / crimes_pivot_table.iloc[0].max())

# Using a for loop to scale others
for i in [2,1]:
    crimes_scaled[crimes_pivot_table.index[i]] =  pd.DataFrame(crimes_pivot_table.iloc[i] / crimes_pivot_table.iloc[i].max())
                    
# Using seaborn heatmap
plt.figure(figsize=(4,4))
plt.title('Month and Category heatmap', fontsize=14)
plt.tick_params(labelsize=12)
sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);


# * Break and Enter has most incidents in January.
# * Theft and Others are more frequent during summer months. Maybe because people go out more and there is an increased number of tourists?
# * December is not a "good month" for Theft and Others. Or are thieves busy with Breaking and Entering people's home while they are travelling?
# 
# <br>
# ### What about Day of the Week?

# In[25]:


# Create a pivot table with day of the week and category. 
crimes_pivot_table = crimes.pivot_table(values='TYPE', index='CATEGORY', columns='DAY_OF_WEEK', aggfunc=len)

# To compare categories, I'm scaling each category by diving by the max value of each one
crimes_scaled = pd.DataFrame(crimes_pivot_table.iloc[0] / crimes_pivot_table.iloc[0].max())

# Using a for loop to scale row
for i in [2,1]:
    crimes_scaled[crimes_pivot_table.index[i]] = crimes_pivot_table.iloc[i] / crimes_pivot_table.iloc[i].max()
                    
crimes_scaled.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# Using seaborn heatmap
plt.figure(figsize=(4,4))
plt.title('Day of the Week and Category heatmap', fontsize=14)
plt.tick_params(labelsize=12)
sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);


# * Break and Enter is more frequent on week days.
# * Theft and Others on weekends.
# 
# <br>
# ### What hours do crimes happen?

# In[26]:


# Create a pivot table with hour and category. 
crimes_pivot_table = crimes.pivot_table(values='TYPE', index='CATEGORY', columns='HOUR', aggfunc=len)

# To compare categories, I'm scaling each category by diving by the max value of each one
crimes_scaled = pd.DataFrame(crimes_pivot_table.iloc[0] / crimes_pivot_table.iloc[0].max())

# Using a for loop to scale row
for i in [2,1]:
    crimes_scaled[crimes_pivot_table.index[i]] =  pd.DataFrame(crimes_pivot_table.iloc[i] / crimes_pivot_table.iloc[i].max())
                    
# Using seaborn heatmap
plt.figure(figsize=(5,5))
plt.title('Hour and Category heatmap', fontsize=14)
plt.tick_params(labelsize=12)
sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);


# * Most crimes happen between 17:00 and 20:00.
# * Break and Enter has some activity between 3 and 5 am.
# * Theft doesn't occur much in those hours between 3 and 5 as expected, because most people are sleeping.
# * Category Others doesn't have Hour in most of the entries (they are 'offset to protect privacy').
#  
#  
#  <br>
# ### Do crimes happen in the same hour for each day of the week?

# In[27]:


# Create a pivot table with hour and day of week. 
crimes_pivot_table = crimes[crimes['HOUR'] != 99].pivot_table(values='TYPE', index='DAY_OF_WEEK', columns='HOUR', aggfunc=len)

# To compare categories, I'm scaling each category by diving by the max value of each one
crimes_scaled = pd.DataFrame(crimes_pivot_table.loc[0] / crimes_pivot_table.loc[0].max())

# Using a for loop to scale each day
for i in [1,2,3,4,5,6]:
    crimes_scaled[i] = crimes_pivot_table.loc[i] / crimes_pivot_table.loc[i].max()

# Rename days of week
crimes_scaled.columns = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

# Using seaborn heatmap
plt.figure(figsize=(6,6))
plt.title('Hour and Day of the Week heatmap', fontsize=14)
plt.tick_params(labelsize=12)
sns.heatmap(crimes_scaled, cmap='seismic', linecolor='grey',linewidths=0.1, cbar=False);


# * Here we can notice that week days have a different pattern than weekends.  It's like there is an additional *"crime working hours"* on weekends.
# 
# * It looks like it makes sense that the City wanted to close Granville street clubs one hour earlier... 
# 
#   See: <a href="http://www.cbc.ca/news/canada/british-columbia/granville-street-clubs-will-close-their-doors-1-hour-earlier-this-summer-1.4163233" target="_blank">Granville Street clubs will close their doors 1 hour earlier this summer.</a>
#   
#   
# <br>
# ### Exploring Category "Theft"
# We saw before that most crimes are in the category "Theft". Let's explore it better

# In[28]:


# Let's check what types of theft we have and how many
crimes[crimes['CATEGORY'] == 'Theft']['TYPE'].value_counts()


# *Theft from Vehicle* is the major type. Let's view each type over time.

# In[29]:


# Initiate the figure and define size
plt.figure(1)
plt.figure(figsize=(15,8))

# Using a for loop to plot each type of crime with a moving average
i = 221
for crime_type in crimes[crimes['CATEGORY'] == 'Theft']['TYPE'].unique():    
    plt.subplot(i);
    crimes[crimes['TYPE'] == crime_type].resample('M').size().plot(label='Total per month')
    crimes[crimes['TYPE'] == crime_type].resample('M').size().rolling(window=12).mean().plot(color='red', linewidth=5, label='12-months Moving Average')
    plt.title(crime_type, fontsize=14)
    plt.xlabel('')
    plt.legend(prop={'size':12})
    plt.tick_params(labelsize=12)
    i = i + 1


# * First, let me start with "Theft of Vehicle:
#   * It had a major decrease, from an average of around 520 crimes per month in 2003 to around 100 in 2012. That's impressive!
#   * Although the average has been increasing in the past years, it's way below 2003.
#   * In 2002, the <a href="http://www.baitcar.com/about-impact-bait-car-program" target="_blank">"Bait Car"</a> program was launched and in 2003 the IMPACT group was formed in response to this peak in theft. It looks like they've been doing a great job!
#   * Side note: I wonder if this decrease of around 80% in the number of theft had any impact on insurance policies prices...
# 
# 
# * Second, about "Other Theft":
#   * On the opposite trend, other theft has been increasing, from around 200 to almost 500 crimes per month.
#   * Is it because stealing a car became too risky, but thieves still need to "make a living"?
#   
#   
# * About "Theft from Vehicle":
#   * It is the most frequent type.
#   * It decreased along with "Theft of Vehicle" until 2012, but then it increased significantly.
#   
#   
# * Finally, about "Theft of Bicycle":
#   * We can see a clear trend within the year. It has peaks during summer months, which is expected.
#   * The average has also been increasing.
#   
#  
#  <br>
# ### That's it for now!
# 
# I've also created a Tableau dashboard to visualize this data. You can check it: <a href="https://wosaku.github.io/crime-vancouver-dashboard.html">here</a>
