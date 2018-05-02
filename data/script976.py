
# coding: utf-8

# # Beginners Tutorial: Analyze guns deaths in the US w/ Python
# ## Analyzation Gun Deaths in the US: 2012-2014

# This analyzation inspects gun-death in the US in the years 2012-2014. The data originated from the CDC. I came across this thanks to FiveThirtyEight's [Gun Deaths in America][1] project. The data can be found [here][2].
# 
#   [1]: https://fivethirtyeight.com/features/gun-deaths/
#   [2]: https://github.com/fivethirtyeight/guns-data
# 
# This notebook was made for learning purposes. It definitely served its purpuse and I hope others can make use of it. I believe a beginner can learn a lot from it. 
# 
# **If you're here for the visualizations: skip to the "Visualization" part. 

# ## 1. Importing, cleaning and getting familiar with the data

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


guns = pd.read_csv('../input/guns.csv', index_col=0)
print(guns.shape)
guns.head()


# Our data has almost 101,000 rows (gun death incidents) and 10 columns (categories).
# 
# Here's an explanation of each column:
# 
# - **' '**: this is an identifier column, which contains the row number. It's common in CSV files to include a unique identifier for each row, but we can ignore it in this analysis.
# - **year**: the year in which the fatality occurred.
# - **month**: the month in which the fatality occurred.
# - **intent**: the intent of the perpetrator of the crime. This can be Suicide, Accidental, NA, Homicide, or Undetermined.
# - **police**: whether a police officer was involved with the shooting. Either 0 (false) or 1 (true).
# - **sex**: the gender of the victim. Either M or F.
# - **age**: the age of the victim.
# - **race**: the race of the victim. Either Asian/Pacific Islander, Native American/Native Alaskan, Black, Hispanic, or White.
# - **hispanic**: a code indicating the Hispanic origin of the victim.
# - **place**: where the shooting occurred. Has several categories, which you're encouraged to explore on your own.
# - **education**: educational status of the victim. Can be one of the following:
#     + **1**: Less than High School
#     + **2**: Graduated from High School or equivalent
#     + **3**: Some College
#     + **4**: At least graduated from College
#     + **5**: Not available
# 
# It's good practice to get to know our data set before begining to analyze. 

# In[ ]:


guns.index.name = 'Index'
guns.head()


# In[ ]:


# for readability and concistency - capitalizing column names
guns.columns = map(str.capitalize, guns.columns)
guns.columns


# In[ ]:


guns.dtypes


# The float values in the education column could indicate there are NaN valuse, since every category is a whole number [1,2,3,4,5].
# 
# **Incompeteness/ Completeness**: checking to see how much of the data is NaN is important in order to know if the column is usefull/ useless.

# In[ ]:


guns.notnull().sum()


# In[ ]:


# In order to see the percentage of valid data:
guns.notnull().sum() * 100.0/guns.shape[0]


# It seems that most of the columns have a at least 98.6% of the values - which means the data is close to complete (unlike real world data). That means we could probably delete all rows with NaN values and still not loose that much of our potential insights.
# 
# Let's try to substitute as many NaN's as we can with real value, and then delete the rows we couldn't fill.

# In[ ]:


# Organizing the data by a column value: first by the year, then by month:
guns.sort_values(['Year', 'Month'], inplace=True)
guns.head(10)


# For me, the most interesting column is 'Intent'. This is what I would want to predict.

# ## 2. Exploring and analyzing the data

# In[ ]:


guns.Intent.value_counts(ascending=False)


# In[ ]:


# Looking at the normalized values makes the picture clearer.
# Note: 'normalize=False' excludes the 'NaN's where here it includes them
guns.Intent.value_counts(ascending=False, dropna=False, normalize=True)


# It is interesting that while suicide does not sound like the most common gun death in the US, it amounted to almost 2/3 of the gun deaths in 2012-2014. Yet the media tend to give more coverage to homicides.
# Off topic: If this interest you, I recommend listening to The suicide paradox on the amazing podcast Freakonomics.
# The describe() method can give us an overview of numerical columns. Since most columns are categorical, we can use it on the 'age' and 'education' columns.

# In[ ]:


cols = ['Education', 'Age']
for col in cols:
    print(col + ':')
    print(guns[col].describe())
    print('-' * 20 + '\n')


# In[ ]:


percentiles = np.arange(0.1,1.1,0.1)
for col in cols:
    print(col + ':')
    print(guns[col][guns[col].notnull()].describe(percentiles=percentiles))
    print('-' * 20 + '\n')


# Cleaning 'Education' column: Notice that the minimum age is 0. Let's check how many of the gun incidents resulted in the deaths of children under the age of 16:

# In[ ]:


guns[guns['Age'] < 16].shape


# In[ ]:


# We can see more info if we filter only those cases:
guns[guns['Age'] < 16].head()


# Converting NaN values in 'education' column: Since we have only 11 entries with NaN, we could just throw them away, but for practice - let's try to fill in real data in its place. Note, that if it was a larger proportion (say >5%) and you were using for Machine Learning you might want to figure out a way to fill in the information in some manner.
# Notice that in the education columns - a lot of these children have 'NaN' or the value 5.0 (= Not available). Let's assume all the children under 16 had education 1.0 (= Less than high school), and fill in this data accordingly:

# In[ ]:


guns[(guns['Age'] < 16) & ((guns['Education'].isnull()) | (guns['Education'] == 5.0))].head()


# In[ ]:


index_temp = guns[(guns['Age'] < 16) & 
                  ((guns['Education'].isnull()) | (guns['Education'] == 5.0))].index
guns.loc[index_temp, 'Education'] = 1.0
guns[guns.Education.isnull()].shape


# In fact, having a 2 year old catagorized under the same education level as a 16 year old (1.0= Under high schools) does not make sence. Let's catagorize children under 5 as 0.0 (= Less than elementary school).

# In[ ]:


index_temp = guns[(guns['Age'] < 5)].index
guns.loc[index_temp, 'Education'] = 0.0
guns.Education.describe()


# In[ ]:


# Let's get rid of rows that has '5.0' (Not available) and NaN in the 'education' column:
# subset = can include a list of column names
guns.dropna(inplace=True)
guns = guns[guns.Education != 5.0]


# In[ ]:


guns.Education.value_counts()


# Another way to get insights about the DataFrame is to look at the unique() values in the columns. I chose to leave out the columns 'age' and index:

# In[ ]:


for col in guns.columns:
    if col not in ['Age', '']:
        print(col, ': ', guns[col].unique())


# ### Gender numeric comparison
# This will be easier to see in the visualizations, but here's a first look at the gender distribution of our data.

# In[ ]:


guns.Sex.value_counts()


# We can use this column to split the percentage of Male/Female death cases:

# In[ ]:


guns.Sex.value_counts(normalize=True)


# The normalized values and the count values seem similar since there are close to 100,000 rows in the data set.
# 
# ### Here are a few questions to practice basic pandas and numpy methods:
# 
# **Question:** Compare yearly number of deaths: rising or decending?

# In[ ]:


guns.Year.value_counts(sort=False)


# In[ ]:


# evaluating the percentage change between years
n2012 = guns[2012 == guns['Year']].shape[0]
(guns.Year.value_counts(sort=False) - n2012) * 100./ n2012


# **Answer:** Seems like it stays about the same. The change does not look significant (<0.3%) by looking at the values.
# 
# **Question:** Are there certain months with significally more/less gun deaths than others?

# In[ ]:


guns.Month.value_counts(sort=True)


# In[ ]:


nexpected_month = guns.shape[0]/12.
(guns.Month.value_counts(sort=True) - nexpected_month) * 100./nexpected_month


# **Answer:** It seems February has around 15% less gun deaths than the expected (Is it its length, the weather, a different reason or a combination of all? If we want to further investigate this - we can to look at the 'location' column to see if most gun deaths in certain months occured outside/inside).
# 
# Also, we see July has 7% more than expected year-wide, way above anything month.
# This is not accurate since we're not taking in account the number of days in each month. We could devide it by number of days in each month, but 2012 was a leap year which changes the number for february.
# 
# The best practice is to convert the month values to datetime objects.
# 
# **Season/Month analysis:** Since we're dealing with dates, let's arrange the data by year and month:

# In[ ]:


guns.sort_values(['Year', 'Month'], inplace=True)
guns.head()


# A date object implies year, month and day. We cannot create a datetime object without a day, but for the sake of practice - let's combine the 2 date values (of type int64) into one datetime object, and assign '1' in the day value.

# In[ ]:


import datetime
# The purpose of *10000 and the *100 are to convert 2012, 01, 01 into 20120101 for readability
guns['Date'] = pd.to_datetime((guns.Year * 10000 + guns.Month * 100 + 1).apply(str),format='%Y%m%d')
guns.dtypes.tail(1)


# In[ ]:


# now that we're done with these columns:
del guns['Year']
del guns['Month']
guns.head()


# In[ ]:


import calendar
monthly_rates = pd.DataFrame(guns.groupby('Date').size(), columns=['Count'])
monthly_rates.index.to_datetime
print(monthly_rates.index.dtype)
print(monthly_rates.shape)
monthly_rates.head()


# In[ ]:


days_per_month = []
for val in monthly_rates.index:
    days_per_month.append(calendar.monthrange(val.year, val.month)[1])
monthly_rates['Days_per_month'] = days_per_month
monthly_rates.head()


# Note: another way to do this is to set the 'day' in each date as the last day of each month [instead of the first day]. This way we have the length of the month in the day field and don't need 'monthrange'. However, having a '01' in each of the datetime objects make it look more consistent and less random, in my view. As long as we remember that the values refer to the whole month.

# In[ ]:


monthly_rates['Average_per_day'] = monthly_rates['Count']*1./monthly_rates['Days_per_month']
print(monthly_rates.shape)
monthly_rates.tail()


# In[ ]:


month_rate_dict = {}
for i in range(1,13):
    bool_temp = monthly_rates.index.month == i
    month_average = (sum(monthly_rates.loc[bool_temp, 'Average_per_day']))/3.
    month_rate_dict[i] = month_average
# avg_month_rate = pd.DataFrame.from_dict(month_rate_dict)
# python 2: 
# avg_month_rate = pd.DataFrame(month_rate_dict.items(), columns=['Month', 'Value'])
avg_month_rate = pd.DataFrame.from_dict(list(month_rate_dict.items()))
avg_month_rate.columns = ['Month', 'Value']
avg_month_rate


# In[ ]:


# calculating the expected cases for each day [+1. becuase 2012 was a leap year]
nexpected_day = guns.shape[0]/(365*3 + 1.)
nexpected_day


# In[ ]:


avg_month_rate['Percent_change'] = (avg_month_rate.Value - nexpected_day) * 100./ nexpected_day
print(avg_month_rate.sort('Percent_change'))


# Now that we have the daily average, we can say with more certainty that July (~(+5.3%)) and June (~(+4.6%)) have higher gun death rate than the rest of the months, while February (~(-9%)) has significantly lower amount of guns deaths (but not 15% lower as the 'rough analysis' from before showed).
# 
# **Question:** What percentage of cases were police officers involved in?

# In[ ]:


100 * guns.Police.value_counts(normalize=True)


# Seeing this kind of distribution means this column does not give us any additional information about the cases and we can remove it all together.

# In[ ]:


del guns['Police']
print(guns.shape)
guns.head()


# **Question:** Which race appears the most in the df and which appears the least?

# In[ ]:


guns.Race.value_counts(sort=True, normalize=True)


# **Answer:** We can not conclude anything by those numbers unless we take in account the distribution of races in the US population.
# 
# **Question:** How do you sample a dataframe?
# 
# **Answer:**

# In[ ]:


# a sample of about 10% of the data may look like this:
sample_guns = guns.sample(n=10000)
sample_guns.head()


# In[ ]:


# we can look at the M/F distribution, in order to make sure it's similar to the original data
sample_guns.Sex.value_counts(normalize=True)


# **Question:** How do you define a categorical columns/pd.Series? E.g please order guns['intent'] by this order: 'Homicide','Suicide','Accidental','Undetermined'

# In[ ]:


list_ordered = ['Homicide','Suicide','Accidental','Undetermined']
guns['Intent'] = guns['Intent'].astype('category')
guns.Intent.cat.set_categories(list_ordered, inplace=True)
guns.sort_values(['Intent']).head()


# We can treat 'Undetermined' values in the 'intent' column as NaN and drop those rows, since these values do not give us any additional info and they are less than 1% of the rows (798/ about 100,000).

# In[ ]:


guns.Intent.value_counts()


# Let's take those lines also out of our df.
# Note: my assumption in dropping these values is that my prediction model would try to predict the 'intent' column, and 'Undetermined' cases have no value for that process. If you're predicting a different column - you may want to leave these values in the df.

# In[ ]:


guns = guns[guns.Intent != 'Undetermined']
guns.Intent.value_counts()


# Note: Converting NaNs highly depends on context:
# It Might (rarely) contain information and hence it's own category
# Sometimes you want to take it out totally when either plottting or training in machine learning
# Other times, when machine learning with sparse data you might want to fill the values. That, of course depends on context
# Question: Why does 'Undetermined' still appears in the value_counts() with 0 count?
# Answer: We defined labels for the 'intent' column earlier. We need to remove 'Undetermined' from 'list_ordered' for the intent column. Otherwise pandas will keep listing it as if it is part of our DF with 0 counts.

# In[ ]:


# removing last value in list ordered - which is 'Undetermined'
list_ordered = list_ordered[:-1]
guns.Intent.cat.set_categories(list_ordered, inplace=True)
guns.Intent.value_counts()


# **Question:** Given a Series which contains strings, how do you find the length of each of the strings?

# In[ ]:


guns.Race.str.len().unique()


# In[ ]:


guns.Race.unique()


# **Question:** For the same series, how do you know if any given entry contains a string segment. E.g: Which entries int the 'intent' column contain the segment 'cide'?

# In[ ]:


guns.Intent.str.contains('cide').sum()


# **Suggestion for further analysis:** Use census data to get the precentahge of each race name in the population in order to normalize the data.

# ## 3. Visualizing the data 
# 
# We can quickly find patterns in the visual information we encounter. However, our ability to quickly process symbolic values (like numbers and words) is very poor. Data visualization focuses on transforming data from table representations visual ones. By noticing visual patterns, we may have a good indication about correlations before moving to our prediction part of the analyzation.
# 
# Let's import the pyplot library, some extra styles and show the plots inline the Jupyter notebook.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Line Charts / Time analysis
# 
# **Line charts** work best when there is a logical connection between adjacent points. When dealing with dates, line charts are an appropriate choise for visualizing because the rows had a natural ordering to it. Each row reflected information about an event that occurred after the previous row (if the df is ordered by date).
# 
# To emphasize how the visual representation of the line chart helps us observe trends easily, let's look at the same 36 data points from 2012 to the end of 2014 as a line chart.
# 
# To create a line chart of the unemployment data from 2012, we need:
# - the x-axis to range from 2012-01-01 to 2014-12-01
# - the y-axis to range from 2357 to 3079 (which correspond to the minimum and maximum death incident values)
# We don't have to specify the values. we pass in the list of x-values as the first parameter and the list of y-values as the second parameter to plot().
# 
# Let's begin with a yearly plot:

# In[ ]:


# 2012
plt.plot(monthly_rates.index[:12], monthly_rates['Count'][:12], 
         linestyle='--', linewidth=3., alpha=0.6)
plt.xticks(rotation=70)
plt.tick_params(axis='both', which='both',length=0)
plt.show()


# In[ ]:


# notice the y column in the previous plot begins at 2200; 
# Let's look at the real picture from 0 
plt.plot(monthly_rates.index[:12], monthly_rates['Count'][:12],
        linestyle='--', linewidth=3., alpha=0.6)
plt.xticks(rotation=70)
plt.ylim(ymin=0, ymax=3500)
plt.tick_params(axis='both', which='both',length=0)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Gun Deaths\ncount', fontsize=14)
plt.title('Monthly Gun Death Count in the US, 2012', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# In[ ]:


# year 2013:
plt.plot(monthly_rates.index[12:24], monthly_rates['Count'][12:24],
        linestyle='--', linewidth=3., alpha=0.6, color='r')
plt.xticks(rotation=70)
plt.ylim(ymin=0, ymax=3500)
plt.tick_params(axis='both', which='both',length=0)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Gun Death\ncount', fontsize=14)
plt.title('Monthly Gun Death Count in the US, 2013', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# In[ ]:


# year 2014:
plt.plot(monthly_rates.index[24:], monthly_rates['Count'][24:],
        linestyle='--', linewidth=3., alpha=0.6, color='g')
plt.xticks(rotation=70)
plt.ylim(ymin=0, ymax=3500)
plt.tick_params(axis='both', which='both',length=0)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Gun Death\nCount', fontsize=14)
plt.title('Monthly Gun Death Count in the US, 2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# Let's print them on the same plot in order to compare:

# In[ ]:


# years 2012 - 2014
# Changing linestyle to a constant line = seeing intersections more clearly
fig = plt.figure()
plt.plot(monthly_rates.index.month[0:12], monthly_rates['Count'][0:12], label='2012',
        linestyle='-', linewidth=2., alpha=0.6)
plt.plot(monthly_rates.index.month[12:24], monthly_rates['Count'][12:24], label='2013',
        linestyle='-', linewidth=2., alpha=0.6, color='r')
plt.plot(monthly_rates.index.month[24:36], monthly_rates['Count'][24:36], label='2014',
        linestyle='-', linewidth=2., alpha=0.6, color='g')
plt.xlim(xmin=1, xmax=12)
plt.ylim(ymax=max(monthly_rates['Count'])+100)
plt.tick_params(axis='both', which='both',length=0)
plt.xticks(np.arange(1, 13, 1))
plt.legend(loc='upper left', frameon=False)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Gun Death\nCount', fontsize=14)
plt.title('Monthly Gun Death Count in the US: 2012-2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# But this is a distorted look on the data. It's important our axis starts from 0. Also, let's not repeat ourselves and enlarge the plot:

# In[ ]:


fig = plt.figure(figsize=(10,6))

colors = ['b', 'r', 'g']
labels = ['2012', '2013', '2014']

for i in range(len(labels)):
    start_index = i*12
    end_index = (i+1)*12
    subset = monthly_rates[start_index:end_index]
    plt.plot(subset.index.month, subset['Count'], color=colors[i], label=labels[i],
            linestyle='-', linewidth=2., alpha=0.6)

plt.xlim(xmin=1, xmax=12)
plt.ylim(ymin=0, ymax=max(monthly_rates['Count'])+100)
plt.tick_params(axis='both', which='both',length=0)
plt.xticks(np.arange(1, 13, 1))
plt.legend(loc='center right', frameon=False)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Gun Death Count', fontsize=14)
plt.title('Monthly Gun Death Count in the US: 2012-2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# The trend of deaths does not seem to vary significantly over the years. There is a recurring phenomenon of very low number of incidents during february. July has the highest rate of gun deaths in 2012 and 2013, but it is not the case in 2014.
# 
# ### Bar Plots
# 
# When we need visualization that scales graphical objects to the quantitative values we're interested in comparing - we can use a **bar plot**.
# 
# Let's look at the 'intent' division with inner gender ['sex'] division.

# In[ ]:


intent_sex = guns.groupby(['Intent', 'Sex'])['Intent'].count().unstack('Sex')
ax = intent_sex.plot(kind='bar', stacked=True, alpha=0.7)
ax.set_xlabel('Intent', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.xticks(rotation=0)
plt.tick_params(axis='both', which='both',length=0)
ax.legend(labels=['Female', 'Male'], frameon=False, loc=0)
plt.title('Gender distribution\nGun Deaths US: 2012-2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# There are far more male incidents than female. From this visual we can infer that it will be hard to learn from 'Accidental' cases since there is so little of them.
# 
# We can look at a similar split to get a sence of the education of the victims:

# In[ ]:


intent_edu = guns.groupby(['Intent', 'Education'])['Intent'].count().unstack('Education')
# creating a range of 5 colors - from light to dark
edu_legend_labels = ['Less than\nElementry school','Less than \nHigh School', 'Graduated from\nHigh School\nor equivalent', 
                 'Some College', 'At least\ngraduated\nfrom College']
colors = plt.cm.GnBu(np.linspace(0, 1, 5))
ax = intent_edu.plot(kind='bar', stacked=True, color=colors, width=0.5, alpha=0.6)
plt.xticks(rotation=0)
ax.set_xlabel('Intent', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
plt.tick_params(axis='both', which='both',length=0)
ax.legend(edu_legend_labels, ncol=1, frameon=False, prop={'size':10}, loc=0)
plt.ylim(ymin=0, ymax=90000)
plt.title('Education distribution\n in Gun Deaths US: 2012-2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# But this is too crowded, so let's make it horizontal and spread it out a bit:

# In[ ]:


intent_edu = guns.groupby(['Intent', 'Education'])['Intent'].count().unstack('Education')
ax = intent_edu.plot(kind='barh', figsize=(15,6), stacked=True, color=colors, alpha=0.6)
ax.set_xlabel('Count', fontsize=20)
ax.set_ylabel('Intent', fontsize=20)
ax.legend(edu_legend_labels, loc=0,  prop={'size':12}, frameon=False)
plt.xlim(xmin=0, xmax=80000)
plt.tick_params(axis='both', which='both',length=0)
plt.title('Education distribution\nin Gun Deaths US: 2012-2014', fontsize=20, fontweight='bold')
sns.despine()
plt.show()


# In[ ]:


1# the percentage visual is more informative
education = pd.crosstab(guns.Education, guns.Intent)
education.div(education.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, alpha=0.6)
plt.title('Intent Percentage by Education')
plt.xlabel('Education level')
plt.ylabel('Percentage')
plt.legend(loc='upper center', bbox_to_anchor=(1.1,0.9))
sns.despine()


# Using this split visualization we can see that there is a larger percentage of people with some college education and higher in the suicide gun deaths than in the Homicide incidents. The dark blues are more than a third of the Suicide cases, but only about a quarter of the Homicide cases. This is indication that **'education' could be a helpful variable for our 'intent' prediction.**
# 
# Again, the Accidental bar does not seem to give us any additional info. It may be useless for our prediction, and we're better off going with binary: 'Suicide' vs. 'Homicide'.
# 
# Moving on to distribution of location:

# In[ ]:


intent_place = guns.groupby(['Intent', 'Place'])['Intent'].count().unstack('Place')

colors = plt.cm.GnBu(np.linspace(0, 2, 20))
ax = intent_place.plot(kind='barh', stacked=True, color=colors, alpha=0.8)
ax.set_xlabel('Count', fontsize=14)
ax.set_ylabel('Intent', fontsize=14)
plt.tick_params(axis='both', which='both', length=0)
ax.legend(loc=0, ncol=2, prop={'size':10}, frameon=False)
plt.title('Location distribution\nin Gun Deaths US: 2012-2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# We can see that most cases of Suicide happen at home. This means that **location could be an important variable when we want to predict intent**. The Accidental column, again, seems too dense to give us any valuble info. 
# 
# It's almost impossible to conclude things about this visualization since there are a lot of values (i.e colors), and this distribution isn't that useful for us. Let's make it better by merging some of the values. 

# In[ ]:


guns.Place.value_counts()


# I'm not sure if the values 'other specified' and 'other unspecified' give us any info. We may want to drop this column all together, or chose: Home, Street and other as our 3 values for this.

# In[ ]:


#These are too many categories and it's hard to arrive to conclusions
# let's merge 'street' with 'trade/service area' and the rest to 'Other'
index_temp = guns[(guns['Place'] == 'Trade/service area') | (guns.Place == 'Industrial/construction')].index
guns.loc[index_temp, 'Place'] = 'Street'
index_temp = guns[(guns['Place'] != 'Street') & (guns.Place != 'Home')].index
guns.loc[index_temp, 'Place'] = 'Other'

guns.Place.value_counts()


# In[ ]:


# Let's take another look:
intent_place = guns.groupby(['Intent', 'Place'])['Intent'].count().unstack('Place')
colors = plt.cm.GnBu(np.linspace(0,2,6))
ax = intent_place.plot(kind='barh', stacked=True, color=colors, alpha=0.6)
ax.set_xlabel('Count', fontsize=14)
ax.set_ylabel('Intent', fontsize=14)
plt.tick_params(axis='both', which='both',length=0)
ax.legend(loc='upper right', prop={'size':10}, frameon=False)
plt.title('Location distribution\nin Gun Deaths US: 2012-2014', fontsize=14, fontweight='bold')
sns.despine()
plt.show()


# In[ ]:


# the percentage visual is more informative
place_died = pd.crosstab(guns.Place, guns.Intent)
place_died.div(place_died.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, alpha=0.6)
plt.title('Intent Percentage by Place')
plt.xlabel('Place of death')
plt.ylabel('Percentage')
plt.legend(loc='upper center', bbox_to_anchor=(1.1,0.9))
sns.despine()


# This is far less dense and better for understanding our data. It is not surprising that the minoroty of suicide cases were located in the street, and the majority - at home. Homicide seems to be split 3 ways pretty evenly.

# In[ ]:


# barplot of gender grouped by intent 
pd.crosstab(guns.Sex, guns.Intent).plot(kind='bar', alpha=0.6)
plt.title('Gender Distribution by Intent')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.legend(loc=0)
sns.despine()


# In[ ]:


# barplot of education grouped by intent 
pd.crosstab(guns.Education, guns.Intent).plot(kind='bar', alpha=0.6)
plt.title('Education Distribution by Intent')
plt.xlabel('Education')
plt.ylabel('Frequency')
sns.despine()


# Let's use a stacked barplot to look at the percentage intent by place of death.

# ## Histograms
# 
# We use value counts and **sort_index()** in order to organize the age values according to frequency. 

# In[ ]:


age_freq = guns.Age.value_counts()
sorted_age_freq = age_freq.sort_index()
sorted_age_freq.head()
plt.hist(guns['Age'], range=(0,107), alpha=0.4)
plt.tick_params(axis='both', which='both',length=0)
plt.xlim(xmin=0, xmax=110)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Age distribution', fontsize=14, fontweight='bold')
sns.despine(bottom=True, left=True)
plt.show()


# Let's look at the age histogram for the suicide deaths vs. the homicide deaths, and check if there are any evident differences.

# In[ ]:


fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

suicide = guns[guns['Intent'] == 'Suicide']
homicide = guns[guns['Intent'] == 'Homicide']

ax1.hist(suicide.Age, 20, alpha=0.4)
ax1.set_title('Suicide gun deaths\nAge Distribution', fontsize=14, fontweight='bold')
ax2.hist(homicide.Age, 20, alpha=0.4)
ax2.set_title('Homicide gun deaths\nAge Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Age', fontsize=14)
ax2.set_xlabel('Age', fontsize=14)
ax1.set_ylabel('Frequency', fontsize=14)
ax2.set_ylabel('Frequency', fontsize=14)
ax1.tick_params(axis='both', which='both',length=0)
ax2.tick_params(axis='both', which='both',length=0)
ax1.set_xlim(xmin=0, xmax=110)
ax2.set_xlim(xmin=0, xmax=110)
sns.despine(bottom=True, left=True)
plt.show()


# By looking at this we can see that most of the Homicide deaths occured around the age of 20-21, while most suicide cases are circled just under the age of 55-58 (there is also a noticable peak around the age of 20). If we wanted more accurate numbers - we could take a look at the mean and median of these variables. Again, since we have so little incidents of accidental gun deaths, it's hard to infer anything from that part of the data [red dots].

# In[ ]:


g = sns.FacetGrid(suicide, col='Sex')  
g.map(sns.distplot, 'Age')
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.05))
g.fig.suptitle('Suicide ages: Gender comparison', fontsize=14, fontweight='bold')
g = sns.FacetGrid(homicide, col='Sex') 
g.map(sns.distplot, 'Age')
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.05), xlabel='Age', ylabel='Percentage', )
g.fig.suptitle('Homicide ages: Gender comparison', fontsize=14, fontweight='bold')


# In[ ]:


g = sns.FacetGrid(suicide, col='Race')  
g.map(sns.distplot, 'Age')
g.set(xlim=(0, None))
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.06), xlabel='Age')
g.fig.suptitle('Suicide ages: Race comparison', fontsize=14, fontweight='bold')
g = sns.FacetGrid(homicide, col='Race') 
g.map(sns.distplot, 'Age')
g.set(xlim=(0, None))
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.06), xlabel='Age')
g.fig.suptitle('Homicide ages: Race comparison', fontsize=14, fontweight='bold')


# In[ ]:


# in order to get in in the same order for better comparison:
race_ordered = ['Black', 'White', 'Hispanic', 'Asian/Pacific Islander', 'Native American/Native Alaskan']
guns['Race'] = guns['Race'].astype('category')
guns.Race.cat.set_categories(race_ordered, inplace=True)

suicide = guns[guns['Intent'] == 'Suicide']
homicide = guns[guns['Intent'] == 'Homicide']

g = sns.FacetGrid(suicide, col='Race')  
g.map(sns.distplot, 'Age')
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.06), xlabel='Age')
g.fig.suptitle('Suicide ages: Race comparison', fontsize=16, fontweight='bold')
g = sns.FacetGrid(homicide, col='Race') 
g.map(sns.distplot, 'Age')
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.06), xlabel='Age')
g.fig.suptitle('Homicide ages: Race comparison', fontsize=16, fontweight='bold')


# It's very apperant that the peak in most races in both suicide incidents is around age 20, while the peak in the race 'white' is much higher - around 55 in suicide cases. There is a difference in the homicide cases of this race as well.

# In[ ]:


# we can ignore education = 0 - since these are all very young ages
g = sns.FacetGrid(suicide[suicide.Education > 0], col='Education')
g.map(sns.distplot, 'Age')
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.06), xlabel='Age')
g.fig.suptitle('Suicide ages: Education comparison', fontsize=16, fontweight='bold')
g = sns.FacetGrid(homicide[homicide.Education > 0], col='Education') 
g.map(sns.distplot, 'Age')
plt.subplots_adjust(top=0.8)
g.set(xlim=(0, 110), ylim=(0, 0.06), xlabel='Age')
g.fig.suptitle('Homicide ages: Education comparison', fontsize=16, fontweight='bold')


# The peaks show an interesting picture: All of the homicides plot peaks are in the area of 20 years old, and this is similar for suicides in cases where the education was low. However, when the victim was of college university [3.0-4.0] the peak is also around age 60.
# This may mean that age can also help us predict intent.
# 
# ## KDE Plot
# 
# This plot is useful for looking at univariate relations. It creates and visualizes a kernel density estimate of the underlying feature.
# 
# Let's look at the age values:
# 
# This view will make it look like there are negative ages in our dataset, but that is not the case. The reason for that is that KDE smoothes the lines and thus manipulates the truth. We can limit the x axis.

# In[ ]:


# limit the x-axis
sns.FacetGrid(guns, hue='Intent', size=4).map(sns.kdeplot, 'Age')
plt.legend(loc=9, frameon=False)
plt.xlim(xmin=0)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Density', fontsize=14)
sns.despine(left=True)
plt.title('Age distribution\nHomicide vs. Suicide', fontsize=14, fontweight='bold')


# This is very similar to the plot we saw before, only seeing it on the same grid makes it easier to see differences.

# In[ ]:


sns.FacetGrid(guns, hue='Sex', size=4).map(sns.kdeplot, 'Age').add_legend()
sns.despine(left=True)
plt.xlim(xmin=0)
plt.title('Age distribution\nMale vs. Female', fontsize=14, fontweight='bold')


# This is also very useful in seeing differences between the genders. Let's make different KDE plots for each intent:

# In[ ]:


sns.FacetGrid(suicide, hue='Sex', size=4).map(sns.kdeplot, 'Age').add_legend()
plt.xlabel('Age', fontsize=14)
sns.despine(left=True)
plt.title('Suicide ages: Gender comparison', fontsize=14, fontweight='bold')
sns.FacetGrid(homicide, hue='Sex', size=4).map(sns.kdeplot, 'Age').add_legend()
plt.xlabel('Age', fontsize=14)
sns.despine(left=True)
plt.xlim(xmin=0)
plt.title('Homicide ages: Gender comparison', fontsize=14, fontweight='bold')


# # Box Plots
# A box plot consists of box-and-whisker diagrams, which represents the different quartiles in a visual way. It is useful to show differences between different groups in our data.

# In[ ]:


fig, ax = plt.subplots()
data_to_plot = [suicide.Age, homicide.Age]
plt.xlim(xmin=0, xmax=110)
plt.boxplot(data_to_plot)
plt.ylim(ymin=-1, ymax=110)
plt.xticks([1, 2, 3], ['Suicide', 'Homicide'], fontsize=14)
plt.tick_params(axis='both', which='both',length=0)
plt.ylabel('Age', fontsize=14)
plt.title('Ages in Suicide vs. Homicide',
          fontsize=14, fontweight='bold')
sns.despine(bottom=True)
plt.show()


# Another useful tool is the ability to split column by their value. Here we are doing a gender comparison of age in each 'intent' value:

# In[ ]:


#sns.set(style='ticks')
sns.boxplot(x='Intent', y='Age', hue='Sex', data=guns, palette='PRGn', width=0.6)
sns.despine(bottom=True)


# Again we see that Accidental does not give us any additional info.
# We see the ages in suicides among women varies more than men. However, there is a smaller variation in the ages of females that are murderd in homicide cases.
# 
# ## Violin plots

# In[ ]:


sns.violinplot(x='Intent', y='Age', hue='Sex', split=True, data=guns, size=4, inner='quart')
sns.despine(bottom=True)


# This visualization of the variance in each gender help us to better understand the numbers: Again, the accidental column does not give us much info since the distribution and quartiles are very similar between genders. However, we see the distribution between male and female in homicide differs in quartiles and in the range of the quarters. This may mean that combining age and gender can be a helpful predictor.
# 
# **Note:** Since most of our variables are categorical, **scatter plots** would not give us much information (believe me, I tried). The same applies to **scatter matrix plot**.
# 
# ## Main Takeaways
# - Age, education, place and age~gender may be helpful in predicting intent.
# - Place variable should be used carefuly since some of the values were combined.
# - There is a large number of gun deaths during July and June, and a smaller number on February.
# - To do: Prediction!
# 
# Would love to get your feedback!
