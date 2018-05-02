
# coding: utf-8

# # Where it Pays to Attend College: An Exploration
# ## https://www.kaggle.com/wsj/college-salaries
# ### First step, import libraries.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt


# ### Import the datasets.

# In[ ]:


df1 = pd.read_csv('../input/degrees-that-pay-back.csv')    #by major (50)              -- starting, median, percentile salaries
df2 = pd.read_csv('../input/salaries-by-college-type.csv') #by uni (269) / school type -- starting, median, percentile salaries
df3 = pd.read_csv('../input/salaries-by-region.csv')       #by uni (320) / region      -- starting, median, percentile salaries


# ### Let's see what we are working with.

# In[ ]:


df1.head()


# ### Rename columns to be easier to work with.

# In[ ]:


df1.columns = ['major','bgn_p50','mid_p50','delta_bgn_mid','mid_p10','mid_p25','mid_p75','mid_p90']
df1.head()


# ### We notice that the the Dollar Sign values are actually strings.

# In[ ]:


type(df1['bgn_p50'][1])


# ### We'll convert all the Dollar Sign values from strings to numbers.

# In[ ]:


dollar_cols = ['bgn_p50','mid_p50','mid_p10','mid_p25','mid_p75','mid_p90']

for x in dollar_cols:
    df1[x] = df1[x].str.replace("$","")
    df1[x] = df1[x].str.replace(",","")
    df1[x] = pd.to_numeric(df1[x])

df1.head()


# ### Do we have a number now? Yes.

# In[ ]:


df1['bgn_p50'].mean()


# ### Let's take a look at what we have.
# ##### 
# Analysis: Looks like most majors coming out of school start at salaries around $41,000, with most people making between $37,000 and $50,000. 
# By mid career, average salaries are around $72,000, with most people making between $60,000 and $90,000.
# 
# If you are a top performer (75th percentile), your salary, might be around $100,000, or, between $83,000 and $120,000.
# 
# If you are a very top performer (90th percentile), your salary, might be around $146,000, or, between $124,000 and $162,000.
# 
# Of course, these numbers will vary depending on major. There can be a lot of variability depending on the college major.

# In[ ]:


df1.describe()


# ### Let's look at the dataset again.

# In[ ]:


df1.head()


# ### Let's sort values by starting median salary, so that our graph will be in order.
# ##### Analysis: Major with highest starting salary = Physicians Assistant

# In[ ]:


df1.sort_values(by = 'bgn_p50', ascending = False, inplace=True)
df1.head()


# ### Let's declare a new index, to graph our values by.

# In[ ]:


df1 = df1.reset_index()
df1.head(10)


# ## Initial Graph
# ### Let's see what we have.

# In[ ]:


x = df1.index
y = df1['bgn_p50']
labels = df1.index

plt.scatter(x,y, color='g', label = 'Starting Median Salary')
plt.xticks(x, labels) 

plt.xlabel('Major')
plt.ylabel('US Dollars')
plt.title('Starting Median Salary by Major')
plt.legend()
plt.show()


# ### Let's add the major names along the x-axis.

# In[ ]:


x = df1.index
y = df1['bgn_p50']
labels = df1['major']
#labels = df1.index

plt.scatter(x,y, color='g', label = 'Starting Median Salary')
plt.xticks(x, labels, rotation = 'vertical') #rotation = 'vertical'

plt.xlabel('Major')
plt.ylabel('US Dollars')
plt.title('Starting Median Salary by Major')
plt.legend()
plt.show()


# ### Let's flip the x and y axes.

# In[ ]:


x = df1['bgn_p50'] #switch x and y labels
y = df1.index
labels = df1['major']
#labels = df1.index

plt.scatter(x, y, color='g', label = 'Starting Median Salary') 
plt.yticks(y, labels)

plt.xlabel('US $')
plt.ylabel('') #hide label
plt.title('Starting Median Salary by Major')
plt.legend()
plt.show()


# ### Let's change the index so the values go from high to low.

# In[ ]:


x = df1['bgn_p50']
y = len(df1.index) - df1.index #swap high and low
labels = df1['major']

plt.scatter(x, y, color='g', label = 'Starting Median Salary')
plt.yticks(y, labels)

plt.xlabel('US $')
plt.ylabel('')
plt.title('Starting Median Salary by Major')
plt.legend()
plt.show()


# ## Graph 1: Starting Median Salary by Major
# ### Let's make the figure bigger. Nice! We have our first valuable graph.
# ##### 
# Analysis: Physician's Assistants had very high starting salaries, followed by engineering degrees. Nursing also seems higher than one might expect. Next come majors such as Business Management, Political Science, Marketing. At the bottom, we have the art majors: Music, Drama, Art, followed by Education, Religion, and Spanish at the bottom.

# In[ ]:


fig = plt.figure(figsize=(8,12))

x = df1['bgn_p50']
y = len(df1.index) - df1.index
labels = df1['major']

plt.scatter(x, y, color='g', label = 'Starting Median Salary')
plt.yticks(y, labels)

plt.xlabel('US $')
plt.ylabel('')
plt.title('Starting Median Salary by Major')
plt.legend(loc=2) #move the legend
plt.show()


# ### But wait, that was just the (median) starting salary. Let's add the (median) mid-career salary, which will be more important over the long-haul.
# ##### 
# Analysis: When we plot the mid-career salaries, we see that majors carry generally the same pattern, but there are some shifts as well. Physicians Assistants, Nursing, and Nutrition majors all seem to have some decreases in the rankings.

# In[ ]:


fig = plt.figure(figsize=(8,12))

x = df1['bgn_p50']
y = len(df1.index) - df1.index
labels = df1['major']

plt.scatter(x, y, color='#d6d6d6', label = 'Median Starting Salary')
plt.yticks(y, labels)

x3 = df1['mid_p50']
plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')

plt.xlabel('US $')
plt.ylabel('')
plt.title('Salary Information by Major')
plt.legend(loc=2) #move the legend
plt.show()


# ## Graph 2: Starting & Mid Career Median Salary by Major
# ### Now we'll sort by the (median) mid-career salary, since we think that is more important to us. We will also have the median starting salary as a benchmark of potential interest.
# ##### 
# Analysis: We see again that the Physicians Assistants, Nursing, and Nutrition majors have fallen in the rankings. Top money makers (by mid-career salaries) are the engineering majors, with Chemical Engineering at the top.

# In[ ]:


df2 = df1.sort_values(by = 'mid_p50', ascending = False)
df2 = df2.reset_index()

fig = plt.figure(figsize=(8,12))

x = df2['bgn_p50']
y = len(df2.index) - df2.index
labels = df2['major']

plt.scatter(x, y, color='#d6d6d6', label = 'Median Starting Salary')
plt.yticks(y, labels)

x3 = df2['mid_p50']
plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')

plt.xlabel('US $')
plt.ylabel('')
plt.title('Salary Information by Major')
plt.legend(loc=2) #move the legend
plt.show()


# ### Let's erase the starting salary now. We'll be looking at the mid-career salaries, since we think those are more important. We'll also plot the 25th and 75th percentiles of mid-career salaries for the given majors. 

# In[ ]:


df2 = df1.sort_values(by = 'mid_p50', ascending = False)
df2 = df2.reset_index()

fig = plt.figure(figsize=(8,12))

x = df2['bgn_p50']
y = len(df2.index) - df2.index
labels = df2['major']

#plt.scatter(x, y, color='b', label = 'Median Starting Salary')
plt.yticks(y, labels)

x2 = df2['mid_p25']
plt.scatter(x2, y, color='#ecc833', label = '25th pct. Mid Career Salary')

x3 = df2['mid_p50']
plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')

x4 = df2['mid_p75']
plt.scatter(x4, y, color='#2b5bde', label = '75th pct. Mid Career Salary')

plt.xlabel('US $')
plt.ylabel('')
plt.title('Salary Information by Major')
plt.legend(loc=2) #move the legend
plt.show()


# ### Let's add the 10th and 90th percentiles. So now we have a graph that gives, by major, the 10th, 25th, 50th (median), 75th, and 90th, mid-career salaries.

# In[ ]:


df2 = df1.sort_values(by = 'mid_p50', ascending = False)
df2 = df2.reset_index()

fig = plt.figure(figsize=(8,12))

x = df2['bgn_p50']
y = len(df2.index) - df2.index + 1
labels = df2['major']

#plt.scatter(x, y, color='b', label = 'Median Starting Salary')
plt.yticks(y, labels)

x1 = df2['mid_p10']
plt.scatter(x1, y, color='#f7e9ad', label = '10th pct. Mid Career Salary')

x2 = df2['mid_p25']
plt.scatter(x2, y, color='#ecc833', label = '25th pct. Mid Career Salary')

x3 = df2['mid_p50']
plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')

x4 = df2['mid_p75']
plt.scatter(x4, y, color='#2b5bde', label = '75th pct. Mid Career Salary')

x5 = df2['mid_p90']
plt.scatter(x5, y, color='#a1b6f0', label = '90th pct. Mid Career Salary')

plt.xlabel('US $')
plt.ylabel('')
plt.title('Salary Information by Major')
plt.legend(loc='upper right', bbox_to_anchor=(1.42,.98)) #move the legend

plt.show()


# ## Graph 3: Mid-Career Salary Percentiles by Major (w/ Grid)
# ### We should add some gridlines. Our previous graph was pretty, but not super useful to someone who is really analyzing the chart. Now we have a graph that is both pretty and useful -- Nice!
# ##### 
# Analysis: Again, we see the Engineering Majors as the top money makers. 
# 
# On a general scale, looking at the medians, it seems that most people will make between $50K and $100K annual salary mid-way through their careers. For the top performers (75th pct), that range would be between $75K and $125K. Very top performers (90th pct.) will bring home between $100K and $200K yearly.
# 
# It is interesting that some majors have pretty conservative ranges of salary -- Physician's Assistants and Nursing majors.
# 
# In some the arts majors of Music and Drama, the very top performers do very well -- the top 10% performers (musicians, actors) do very well, making even more money than the top 25% of performers in the engineering/math/science majors.
# 
# The people who make the very most money, regarding the top 10% of performers, are Economics and Finance majors. That makes sense as their job is working with and analyzing money. Another strong performer, again, is Chemical Engineering. 

# In[ ]:


df2 = df1.sort_values(by = 'mid_p50', ascending = False)
df2 = df2.reset_index()

fig = plt.figure(figsize=(8,12))
matplotlib.rc('grid', alpha = .5, color = '#e3dfdf')   #color the grid lines
matplotlib.rc('axes', edgecolor = '#67746A')           #color the graph edge
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)  #this will reset default params if you need

x = df2['bgn_p50']
y = len(df2.index) - df2.index + 1
labels = df2['major']

#plt.scatter(x, y, color='b', label = 'Median Starting Salary')
plt.yticks(y, labels)

x1 = df2['mid_p10']
plt.scatter(x1, y, color='#f7e9ad', label = '10th pct. Mid Career Salary')

x2 = df2['mid_p25']
plt.scatter(x2, y, color='#ecc833', label = '25th pct. Mid Career Salary')

x3 = df2['mid_p50']
plt.scatter(x3, y, color='#54d45d', label = 'Median Mid Career Salary')

x4 = df2['mid_p75']
plt.scatter(x4, y, color='#2b5bde', label = '75th pct. Mid Career Salary')

x5 = df2['mid_p90']
plt.scatter(x5, y, color='#a1b6f0', label = '90th pct. Mid Career Salary')

plt.xlabel('US $')
plt.ylabel('')
plt.title('Salary Information by Major')
plt.legend(loc='upper right', bbox_to_anchor=(1.42,.98))

plt.grid(True) #turn grid on

plt.show()


# ##### 
# Thanks! 
# - Chris
