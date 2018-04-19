
# coding: utf-8

# # A Bird's-Eye View of Bird Strikes
# 
# _May Shen, j.shen33@gmail.com_
# 
# This is my first Kaggle submission. Any suggestions or comments will be greatly appreciated!
# 
# # Table of contents
# 1. [Introduction](#introduction)
# 
# 2. [Descriptive Statistics](#descriptive)
#     1. [What types of aircrafts are involved in bird strikes?](#descriptive1)
#     2. [What kinds of birds are involved in bird strikes?](#descriptive2)
#     3. [What are the flight statuses during bird strikes?](#descriptive3)
#     4. [What are the geological locations of bird strikes?](#descriptive4)
#     5. [What times do bird strikes occur?](#descriptive5)
#     6. [What are the consequences of bird strikes?](#descriptive6)
# 
# 3. [Inferential Statistics](#inferential)
#     1. [Logistic Regression](#inferential1)
#     2. [Support Vector Machines](#inferential12)
#     3. [Random Forests](#inferential13)
#     4. [K-Nearest Neighbors](#inferential14)
#     5. [Gaussian Naive Bayes](#inferential15)
#     6. [Model Summary](#inferential16)
#     7. [Correlation Coefficients](#inferential17)
# 4. [Conclusions and Suggestions](#conclusion)
#     
# ## Introduction <a name="introduction"></a>
# 
# In this report, I analyze a data set on bird strikes, focusing on six aspects in the data, including aircraft information, bird information, flight information, time, location, and outcome. The goal is to better understand the causes of bird strikes and propose actionable recommendations to prevent such events.
# 
# ## Descriptive Statistics <a name="descriptive"></a>
# 
# In this section, a series of descriptive analyses will be performed with the data to better understand the seven aforementioned aspects of bird strikes. 
# 
# ### Data Import
# 
# The bird strikes data set is imported into Python. The first five rows of data are shown.

# In[1]:


# import libraries
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from collections import Counter
from sklearn.metrics import mean_squared_error
from pandas import concat
from pandas import Series, DataFrame
import statsmodels.api as sm

# machine learning
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', 100)


# In[2]:


# import in the bird strikes data set
bird = pd.read_csv("../input/Bird Strikes Test.csv", low_memory=False, thousands=',')
# only drop rows that are all NA:
bird = bird.dropna(how='all')


# In[3]:


# take a look at the first 5 rows of data
bird.head()


# Information on each variable in the data is shown below. From the below information, we can see that this is a relatively large data set with 65610 entries, with tons of missing data. There are 6 major components in the data, including aircraft information, bird information, flight information, time, location, and outcome. The list below outlines the variables that will be examined for each component.
# 
# 1. aircraft information (including _Aircraft: Make/Model_, _Aircraft: Number of engines_, and _Aircraft: Airline/Operator_)
# 2. bird information (including _Wildlife: Size_, and _Wildlife: Species_)
# 3. flight information (including _When: Phase of flight_, _Miles from airport_, _Feet above ground_, and _Speed (IAS) in knots_)
# 4. time (_FlightDate_ and _When: Time (HHMM)_)
# 5. location (_Airport: Name_)
# 6. outcome (_Cost: Total \$_ and _Effect: Indicated Damage_) 
# 
# Information on the number of records and data types of all variables are shown below.

# In[4]:


# check the number of entries and data type for each variable
bird.info()


# In[5]:


# get a quick description of non-null numeric values in the data
bird.drop(['Record ID'], axis=1).describe()


# The description of non-null numeric values shows that this data set is fairly unbalanced. For example, the median of total cost is 0, while the maximum is 37.95 million. This makes sense because most of the bird strikes would not cause any damage to the airplane. Only few bird strikes would lead to accidents, sometimes with human casualties. 
# 
# Given the unbalanceness in the data, each descriptive statistic will be performed with both the whole data set and the subset with any damage. This issue of unbalanced data will also be addressed before modeling work.
# 
# Next, specific descriptive questions will be asked about seven aspects of bird strikes. Abnormal data record will be processed along the way.

# In[6]:


# subset the data with any damage or negative impact to the flight
bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 
                    (bird['Cost: Total $'] > 0) ]


# ### What types of aircrafts are involved in bird strikes? <a name="descriptive1"></a>
# 
# Several variables related to aircraft information will be used to answer this question, including _Aircraft: Make/Model_, _Aircraft: Number of engines_, and _Aircraft: Airline/Operator_. 
# 
# First, a table of counts over aircraft types and number of engines are shown:

# In[7]:


# get a table of number of strikes across aircraft type and aircraft engine numbers
count_air_type = DataFrame({'count' : bird.groupby( ['Aircraft: Number of engines?'] ).size()}).reset_index()
count_air_type.sort_values(['count'], ascending=0)


# The entry "S" may be data entry errors as "S" does not make sense as engine number. This entry is marked as NA. Afterwards, a plot of counts over aircraft type and number of engines is generated:

# In[8]:


# set abnormal entries for Aircraft: Number of engines? to be NaN
bird.loc[(bird['Aircraft: Number of engines?'] == 'S'),'Aircraft: Number of engines?'] = np.nan

# update bird_dmg as well
bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 
                    (bird['Cost: Total $'] > 0) ]

# re-generate count table
count_air_type = DataFrame({'count' : bird.groupby( ['Aircraft: Number of engines?'] ).size()}).reset_index()
# plot the frequency of Aircraft: Number of engines?
fig_air_type = sns.barplot(x=u'Aircraft: Number of engines?', y='count', data=count_air_type)
fig_air_type.set(xlabel='Aircraft: Number of Engines', ylabel='Counts - All Strikes');
fig_air_type.set_title('The Frequency of All Strikes Over Aircraft Number of Engines');


# The bar plot above shows that bird strikes occur to airplanes with 2 engines the most frequently. However, this is merely a frequency description, by no means suggesting that such aircraft type is more prone to bird strikes. The reason could simply be that such aircraft is the most popular aircraft in the sky.
# 
# The bar plot below with only damaging strikes rather than all strikes shows that airplanes with 2 engines still is the most frequently struck. However, among all aircrafts struck with birds, one-engine aircrafts are damaged disproportionately. This makes sense because without a back-up engine, one-engine aircrafts are indeed more prone to damages once struck.

# In[9]:


# table for damaging stikes
count_air_type0 = DataFrame({'count' : bird_dmg.groupby( [ 'Aircraft: Number of engines?'] ).size()}).reset_index()
count_air_type0['All Strikes Counts'] = count_air_type['count']
count_air_type0['Damage Rate'] = count_air_type0['count']/count_air_type0['All Strikes Counts']
# plot the frequency of Aircraft: Number of engines?
fig_air_type0 = sns.barplot(x=u'Aircraft: Number of engines?', y='count', data=count_air_type0)
fig_air_type0.set(xlabel='Aircraft: Number of Engines', ylabel='Counts - Damaging Strikes');
fig_air_type0.set_title('The Frequency of Damaging Strikes Over \n Aircraft Number of Engines');


# In[10]:


# plot the damage rate of Aircraft: Number of engines?
fig_air_type01 = sns.barplot(x=u'Aircraft: Number of engines?', y='Damage Rate', data=count_air_type0)
fig_air_type01.set(xlabel='Aircraft: Number of Engines', ylabel='Damage Rate');
fig_air_type01.set_title('Damage Rate Over Aircraft Number of Engines');


# The above bar plot shows the rate of any damage once an aircraft is struck. It suggests that one-engine airplanes are the most prone to damage once struck.
# 
# Two more pieces of information on aircraft are provided below. First, this table shows the top ten most frequent aircraft operators. Among airlines or operators, military operated aircrafts are the most frequently struck. 

# In[11]:


count_air_n_eng = DataFrame({'count' : bird.groupby( ['Aircraft: Airline/Operator'] ).size()}).reset_index()
count_air_n_eng.sort_values(['count'], ascending=0).head(10)


# In[12]:


count_air_n_eng0 = DataFrame({'count' : bird_dmg.groupby( ['Aircraft: Airline/Operator'] ).size()}).reset_index()
count_air_n_eng0.sort_values(['count'], ascending=0).head(10)


# Military operated aircrafts are still high on the list, but business operated aircrafts climbs to the top of aircrafts struck if we focus only on damaging strikes.
# 
# Second, the table below shows the top ten aircraft makes/models. Among all aircraft makes or models, Boeing 737 300 series are the most frequently struck. Again, no inference is made at this point, since the frequency could simply be a result of the popularity of these makes/models.

# In[13]:


count_air_make = DataFrame({'count' : bird.groupby( ['Aircraft: Make/Model'] ).size()}).reset_index()
count_air_make.sort_values(['count'], ascending=0).head(10)


# A similar table with damaging strikes:

# In[14]:


count_air_make0 = DataFrame({'count' : bird_dmg.groupby( ['Aircraft: Make/Model'] ).size()}).reset_index()
count_air_make0.sort_values(['count'], ascending=0).head(10)


# Together, the analyses show that two-engine aircrafts are most frequently struck. But one-engine aircrafts are more prone to damages once struck.

# ### What kinds of birds are involved in bird strikes? <a name="descriptive2"></a>
# 
# This section focuses on variables including _Wildlife: Size_, and _Wildlife: Species_ to describe the birds involved.
# 
# The following table shows the top 10 bird species involved in all strikes:

# In[15]:


# top 10 bird species - all strikes
count_species = DataFrame({'count' : bird.groupby( ['Wildlife: Species'] ).size()}).reset_index()
count_species.sort_values(['count'], ascending=0).head(10)


# A lot of birds are missing species information, but luckily many have size information available in the data. A table of top 10 birds causing damages is shown below.

# In[16]:


# top 10 bird species causing damages
count_species0 = DataFrame({'count' : bird_dmg.groupby( ['Wildlife: Species'] ).size()}).reset_index()
count_species0.sort_values(['count'], ascending=0).head(10)


# The list above looks different from the list of birds from all strikes. Particularly, large birds like vulture, hawk, and goose now appear on the top 10 list. This suggests that large birds cause damage more frequently once the bird hit the aircraft.
# 
# The bar plot below shows the frequency of strikes over the bird size information.

# In[17]:


# count of strikes by bird size 
count_bird = DataFrame({'count' : bird.groupby( ['Wildlife: Size'] ).size()}).reset_index()
# plot the frequency of all strikes over Wildlife: Size
fig_bird = sns.barplot(x=u'Wildlife: Size', y='count', data=count_bird)
fig_bird.set(ylabel='Count - All Strikes',xlabel='Wildlife Size');
fig_bird.set_title('The Frequency of All Strikes over Wildlife Size');


# A similar bar plot is shown below for only damaging strikes:

# In[18]:


# count of strikes by bird size
count_bird0 = DataFrame({'count' : bird_dmg.groupby( ['Wildlife: Size'] ).size()}).reset_index()
count_bird0['All Strikes Counts'] = count_bird['count']
count_bird0['Damage Rate'] = count_bird0['count']/count_bird0['All Strikes Counts']

# plot the frequency of damaging strikes Wildlife: Number struck and Wildlife: Size
fig_bird0 = sns.barplot(x=u'Wildlife: Size', y='count', data=count_bird0)
fig_bird0.set(ylabel='Count - Damaging Strikes',xlabel='Wildlife Size');
fig_bird0.set_title('The Frequency of Damaging Strikes over Wildlife Size');


# The two bar plots above suggest that large birds are more likely to cause damage. This is confirmed in the bar plot below showing damage rate over bird size.

# In[19]:


# plot damage rate over Wildlife: Size
fig_bird01 = sns.barplot(x=u'Wildlife: Size', y=u'Damage Rate', data=count_bird0)
fig_bird01.set(xlabel='Wildlife: Size', ylabel='Damage Rate');
fig_bird01.set_title('Damage Rate over Wildlife Size');


# In sum, small birds are the most frequently involved in bird strikes, but medium and large birds cause damages more often. 

# ### What are the flight statuses during bird strikes? <a name="descriptive3"></a>
# 
# Several variables in the data set can be used to answer this question, including the altitude of the aircraft (_Altitude bin_ and _Feet above ground_), the phase of the flight (_When: Phase of flight_), the speed of the flight (_Speed (IAS) in knots_), and the distance of the aircraft from the airport (_Miles from airport_).
# 
# First, altitude information of all strikes is listed in a table.

# In[20]:


# There are a lot of missing data in these variables, but since the phase of the flight is available 
# some remedy is done here by filling in reasonalble values
bird.loc[ (bird['Miles from airport'].isnull()) & ( (bird['When: Phase of flight'] == 'Take-off run') |
          (bird['When: Phase of flight'] == 'Parked') | (bird['When: Phase of flight'] == 'Taxi') |
          (bird['When: Phase of flight'] == 'Landing Roll') ),'Miles from airport'] = 0
bird.loc[ (bird['Feet above ground'].isnull()) & ( (bird['When: Phase of flight'] == 'Take-off run') |
          (bird['When: Phase of flight'] == 'Parked') | (bird['When: Phase of flight'] == 'Taxi') |
          (bird['When: Phase of flight'] == 'Landing Roll') ),'Feet above ground'] = 0


# In[21]:


DataFrame({'count' : bird.groupby( ['Altitude bin'] ).size()}).reset_index()


# The table below shows altitude information of damaging strikes:

# In[22]:


DataFrame({'count' : bird_dmg.groupby( ['Altitude bin'] ).size()}).reset_index()


# Both tables above show that more than 2/3 of strikes (damaging and non-damaging) occur below 1000 feet. Below is a histogram of aircraft altitude for all bird strikes.

# In[23]:


# histogram of aircraft altitude information
hist_altitude = sns.distplot(bird['Feet above ground'].dropna(),kde=False);
hist_altitude.set_title('The Frequency of All Strikes over Aircraft Altitude');
hist_altitude.set(ylabel='Count - All Strikes');


# A histogram of aircraft altitude is shown below for only damaging strikes:

# In[24]:


# histogram of aircraft altitude information
hist_altitude0 = sns.distplot(bird_dmg['Feet above ground'].dropna(),kde=False);
hist_altitude0.set_title('The Frequency of Damaging Strikes over Aircraft Altitude');
hist_altitude0.set(ylabel='Count - Damaging Strikes');


# In[25]:


# rate of aircraft below 1000 and 5000 ft for all strikes
rate_1000 = len( bird.loc[bird['Altitude bin']=='< 1000 ft','Altitude bin'] ) / len( bird.loc[(bird['Altitude bin']=='< 1000 ft') | (bird['Altitude bin']=='> 1000 ft'),'Altitude bin'] )
rate_5000 = len( bird.loc[bird['Feet above ground']<5000,'Feet above ground'] ) / len( bird.loc[~(bird['Feet above ground'].isnull()),'Feet above ground'] )
# rate of aircraft below 1000 and 5000 ft for damaging strikes
rate_1000 = len( bird_dmg.loc[bird_dmg['Altitude bin']=='< 1000 ft','Altitude bin'] ) / len( bird_dmg.loc[(bird_dmg['Altitude bin']=='< 1000 ft') | (bird_dmg['Altitude bin']=='> 1000 ft'),'Altitude bin'] )
rate_5000 = len( bird_dmg.loc[bird_dmg['Feet above ground']<5000,'Feet above ground'] ) / len( bird_dmg.loc[~(bird_dmg['Feet above ground'].isnull()),'Feet above ground'] )


# The above data and plots show that most of the bird strikes happen at a low altitude, with 78.48% below 1000 feet and 95.21% below 5000 feet for all strikes, and 66.96% below 1000 feet and 92.04% below 5000 feet for damaging strikes. 

# In[26]:


count_phase = bird['When: Phase of flight'].value_counts()
fig_count = sns.barplot(x=count_phase.index, y=count_phase)
fig_count.set_xticklabels(labels=count_phase.index,rotation=30);
fig_count.set(xlabel='Phase of Flight', ylabel='Counts - All Strikes');
fig_count.set_title('The Frequency of All Strikes over Flight Status');


# The above bar plot nicely breaks down the timing of bird strikes. A similar plot is shown for damaging strikes.

# In[27]:


count_phase0 = bird_dmg['When: Phase of flight'].value_counts()
fig_count0 = sns.barplot(x=count_phase0.index, y=count_phase0)
fig_count0.set_xticklabels(labels=count_phase0.index,rotation=30);
fig_count0.set(xlabel='Phase of Flight', ylabel='Counts - Damaging Strikes');
fig_count0.set_title('The Frequency of Damaging Strikes over Flight Status');


# The two bar plots above further show that most of the strikes occur during take-off and landing, especially the _Approach_ phase. 
# 
# It is intriguing that the frequency during _Approach_ is higher than any other phases. It could be that the planes are still some distance away from the airport, where some countermeasures of bird strikes are in place. To check if this is the case, I plot the altitude of the plane and the distance from the airport over the flight phase in the two boxplots below. 

# In[28]:


flight_altitude = sns.boxplot(x="When: Phase of flight", y="Feet above ground", data=bird)
flight_altitude.set_xticklabels(flight_altitude.get_xticklabels(), rotation=30);
flight_altitude.set_title('Flight Altitude across Flight Status among All Strikes');


# In[29]:


# one point stands out as the aircraft being 1200 miles from the airport in the Approach phase
# which is unlikely and could be a data entry error, the 'Miles from airport' in this row is thus
# replaced with NA, the boxplot is redrawn after the replacement
bird.loc[bird['Miles from airport'] > 1200,'Miles from airport'] = np.nan

# update bird_dmg as well
bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 
                    (bird['Cost: Total $'] > 0) ]

# re-draw the box plot
flight_miles1 = sns.boxplot(x="When: Phase of flight", y="Miles from airport", data=bird) 
flight_miles1.set_xticklabels(labels=flight_miles1.get_xticklabels(),rotation=30);
flight_miles1.set_title('Airplane-Airport Distance across Flight Status among All Strikes');


# The two box plots suggest that overall, aircrafts during the _Approach_ phase are indeed further away from the airport and higher in altitude than those in all other flight phases except the _En Route_ and _Descent_ phases.
# 
# A scatter plot of aircraft altitude over distance from airport is shown below for all strikes:

# In[30]:


bird1 = bird.loc[(~bird['Miles from airport'].isnull()) &
                        (~bird['Feet above ground'].isnull()) ]
plt.scatter(x='Miles from airport', y='Feet above ground', 
              data= bird1);
plt.xlabel('Miles from Airport');
plt.ylabel('Feet above Ground');
plt.title('Flight Altitude over Airplane-Airport Distance among All Strikes');


# A similar scatter plot of aircraft altitude over distance from airport is shown below for damaging strikes:

# In[31]:


bird10 = bird_dmg.loc[(~bird_dmg['Miles from airport'].isnull()) &
                        (~bird_dmg['Feet above ground'].isnull()) ]
plt.scatter(x='Miles from airport', y='Feet above ground', 
              data= bird10);
plt.xlabel('Miles from Airport');
plt.ylabel('Feet above Ground');
plt.title('Flight Altitude over Airplane-Airport Distance among Damaging Strikes');


# The two scatter plots above further confirm that bird strikes happen most often during takeoff or landing, and during low altitude flights. 
# 
# Another metric regarding flight status is the speed of the aircraft. Below is a histogram of the speed for all strikes:

# In[32]:


# histogram of speed
# the current record of airplane is 6082.834 knots, any entry higher than that is set as NA
bird.loc[bird['Speed (IAS) in knots'] > 6100,'Speed (IAS) in knots'] = np.nan
speed = sns.distplot(bird['Speed (IAS) in knots'].dropna(),kde=False);
speed.set(xlabel='Speed in Knots', ylabel='Counts - All Strikes');
speed.set_title('The Frequency of All Strikes over Flight Speed');


# Below is a histogram of the speed for damaging strikes. The speed range is much smaller compared to that of all strikes.

# In[33]:


# histogram of speed
speed0 = sns.distplot(bird_dmg['Speed (IAS) in knots'].dropna(),kde=False);
speed0.set(xlabel='Speed in Knots', ylabel='Counts - Damaging Strikes');
speed.set_title('The Frequency of Damaging Strikes over Flight Speed ');


# Overall, bird strikes happen most often during takeoff or landing, especially approaching phase, and during low altitude and low speed flights. 

# ### What are the geological locations of bird strikes? <a name="descriptive5"></a>
# 
# _Airport: Name_ will be analyzed for this question. The table below shows the top 10 airports of all bird strikes.  

# In[34]:


# top 10 airports among all strikes
df_location = pd.DataFrame({'count' : bird.groupby( ['Airport: Name'] ).size()}).reset_index()
df_location.sort_values(['count'], ascending=False).head(10)


# A similar list of top 10 airports among all damaging strikes is show below:

# In[35]:


# top 10 airports among all damaging strikes
df_airport0 = pd.DataFrame({'count' : bird_dmg.groupby( ['Airport: Name'] ).size()}).reset_index()
df_airport0.sort_values(['count'], ascending=False).head(10)


# There are discrepancies between the two list. It would be good to examine the reasons behind the differences, e.g., why does Sacramento airport have such a high rate of damaging strikes? Is it because the airport happens to be in the route of migration for some medium and large-size bird species, or because the airport is not doing a good job controlling bird population, or simply because there are more air traffic in that state?
# 
# ### What times do bird strikes occur? <a name="descriptive6"></a>
# 
# Three variables, including _FlightDate_, _When: Time (HHMM)_, and _When: Time of day_, are used to answer this question. First, the flight date variable is engineered into two features: the year of the flight and the month of the flight. Year information can be used to understand the trend of bird strikes over time, while the month information along with migration input can be used to understand whether bird migration plays a key role in bird strikes.
# 
# First, a heatmap of strike frequency over flight month and flight year is shown for all strikes:

# In[36]:


# month variable
bird['Flight Month'] = pd.DatetimeIndex(bird['FlightDate']).month
# year variable
bird['Flight Year'] = pd.DatetimeIndex(bird['FlightDate']).year

# subset the data with any damage or negative impact to the flight
bird_dmg = bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 
                    (bird['Cost: Total $'] > 0) ]


# In[37]:


# count over flight month and year
count_time = DataFrame({'count' : bird.groupby( ['Flight Month', 'Flight Year'] ).size()}).reset_index()
# reshape frame
count_time_p=count_time.pivot("Flight Month", "Flight Year", "count")
# plot the frequency over month and year in a heat map
plt.figure(figsize=(8, 7))
heat_time = sns.heatmap(count_time_p);
heat_time.set_title('The Frequency of All Strikes Over Flight Year and Month');


# Then a heatmap of strike frequency over flight month and flight year is shown for damaging strikes only:

# In[38]:


# count over flight month and year
count_time0 = DataFrame({'count' : bird_dmg.groupby( ['Flight Month', 'Flight Year'] ).size()}).reset_index()
# reshape frame
count_time_p0=count_time0.pivot("Flight Month", "Flight Year", "count")
# plot the frequency over month and year in a heat map
plt.figure(figsize=(8, 7))
heat_time0 = sns.heatmap(count_time_p0);
heat_time0.set_title('The Frequency of Damaging Strikes Over Flight Year and Month');


# Together, the above analyses show that bird strikes happen mostly between July and October, with an increasing trend from year 2000 to year 2011. The increasing trend over the years could be due to increasing air traffic. Damaging strikes can also happen between March and May, with a relatively stable trend over the years.
# 
# A histogram below shows the frequency of strikes over the time of the day in HHMM format:

# In[39]:


# histogram of time information
fig_time = sns.distplot(bird['When: Time (HHMM)'].dropna(),kde=False);
fig_time.set(ylabel='Counts - All Strikes');
fig_time.set_title('The Frequency of All Strikes Over Time of the Day');


# A similar plot below shows a count of damaging strikes over time of the day in HHMM format:

# In[40]:


# histogram of time information
fig_time0 = sns.distplot(bird_dmg['When: Time (HHMM)'].dropna(),kde=False);
fig_time0.set(ylabel='Counts - Damaging Strikes');
fig_time.set_title('The Frequency of Damaging Strikes Over Time of the Day');


# The two heatmaps above suggest that bird strikes happen year-round, mostly between July and October, increasing in frequency from year 2000 to year 2011. Damaging strikes happen between March and May, as well as between July and October, with a relatively stable trend over the years, suggesting that countermeasures take effect over the years.
# 
# As to time of the day, histograms suggest that both damaging and non-damaging strikes occur mostly between 5am and 1am of the next day.

# ### What are the consequences of bird strikes? <a name="descriptive7"></a>
# 
# Several aspects of the consequences are considered, including financial cost (_Cost: Total \$_) and damage (_Effect: Indicated Damage_).
# 
# First, a histogram of total financial cost is shown below for all strikes. Note that the x axis is log-transformed so that the distribution is not too sparse toward the high end of cost. Only strikes with costs greater than $0 are included.

# In[41]:


# cost histogram
cost = sns.distplot(np.log10(bird.loc[bird['Cost: Total $']>0,'Cost: Total $']),kde=False);
cost.set(xlabel='Log 10 of Total Cost in Dollar', ylabel='Counts - All Strikes');
cost.set_title('The Frequency of All Strikes Over Log Cost');


# A similar histogram is shown below for damaging strikes:

# In[42]:


# cost histogram
cost0 = sns.distplot(np.log10(bird_dmg.loc[bird_dmg['Cost: Total $']>0,'Cost: Total $']),kde=False);
cost0.set(xlabel='Log 10 of Total Cost in Dollar', ylabel='Counts - Damaging Strikes');
cost0.set_title('The Frequency of Damaging Strikes Over Log Cost');


# The two histograms above are fairly similar, since most damaging strikes involve some sort of financial cost and vice versa. The table below shows a count of damage vs. no-damage strikes.

# In[43]:


# damage count table
DataFrame({'count' : bird.groupby( ['Effect: Indicated Damage'] ).size()}).reset_index()


# Together, these statistics show that most of the strikes do not cause damages (92.41%). However, bird strikes can cost money and even human lives at times.

# ## Inferential Statistics <a name="inferential"></a>
# 
# In this section, damaging/non-damaging bird strikes will be classified using five models including Logistic Regression, Support Vector Machines, Random Forests, K-Nearest Neighbors, and Gaussian Naive Bayes. Training accuracy, testing accuracy, and cross-validation accuracy will be used to check the fit of the model.
# 
# First, I will resample the data such that we have a balanced dataset. Given the large amount of missing values and the large sample size, entries with missing values are removed.

# In[44]:


# add damage index
bird['Damage'] = 0
bird.loc[(bird['Effect: Indicated Damage'] != 'No damage') | 
                    (bird['Cost: Total $'] > 0) ,'Damage'] = 1

# define independent and dependent variables
X = ['Aircraft: Number of engines?',
     'Wildlife: Size',
     'When: Phase of flight','Feet above ground','Miles from airport','Speed (IAS) in knots',
     'Flight Month','Flight Year','When: Time (HHMM)',
     'Pilot warned of birds or wildlife?']
Y = ['Damage']

# clean missing data, keep those with values on key metrics
bird_keep = bird[np.concatenate((X,Y))].dropna(how='any')


# In[45]:


# list of damage indices
damage_index = np.array(bird_keep[bird_keep["Damage"]==1].index)

# getting the list of normal indices from the full dataset
normal_index = bird_keep[bird_keep["Damage"]==0].index

No_of_damage = len(bird_keep[bird_keep["Damage"]==1])

# choosing random normal indices equal to the number of damaging strikes
normal_indices = np.array( np.random.choice(normal_index, No_of_damage, replace= False) )

# concatenate damaging index and normal index to create a list of indices
undersampled_indices = np.concatenate([damage_index, normal_indices])


# In[46]:


# add dummy variables for categorical variables
wildlife_dummies = pd.get_dummies(bird_keep['Wildlife: Size'])
bird_keep = bird_keep.join(wildlife_dummies)

phase_dummies = pd.get_dummies(bird_keep['When: Phase of flight'])
bird_keep = bird_keep.join(phase_dummies)

warn_dummies = pd.get_dummies(bird_keep['Pilot warned of birds or wildlife?'])
bird_keep = bird_keep.join(warn_dummies)

#  convert engine number to numeric
bird_keep['Aircraft: Number of engines?'] = pd.to_numeric(bird_keep['Aircraft: Number of engines?'])

# scale variables before fitting our model to our dataset
# flight year scaled by subtracting the minimum year
bird_keep["Flight Year"] = bird_keep["Flight Year"] - min(bird_keep["Flight Year"])
# scale time by dividing 100 and center to the noon
bird_keep["When: Time (HHMM)"] = bird_keep["When: Time (HHMM)"]/100-12
# scale speed
bird_keep["Speed (IAS) in knots"] = scale( bird_keep["Speed (IAS) in knots"], axis=0, with_mean=True, with_std=True, copy=False )


# In[47]:


# use the undersampled indices to build the undersampled_data dataframe
undersampled_bird = bird_keep.loc[undersampled_indices, :]

# drop original values after dummy variables added
bird_use = undersampled_bird.drop(['Wildlife: Size','When: Phase of flight',
     'Pilot warned of birds or wildlife?'],axis=1)


# The clean data set is separated to train and test data sets randomly. Below is a peek of the top 5 rows of the training data set.

# In[48]:


# scale the X_train and X_test
X_use = bird_use.drop("Damage",axis=1)
standard_scaler = StandardScaler().fit(X_use)
X_use1 = standard_scaler.transform(X_use) # Xs is the scaled matrix but has lost the featuren names
X_use2 = pd.DataFrame(X_use1, columns=X_use.columns) # Add feature names


# In[49]:


# define training and testing sets
# choosing random indices equal to the number of damaging strikes
train_indices = np.array( np.random.choice(X_use2.index, int((X_use2.shape[0]/2)), replace= False) )
test_indices = np.array([item for item in X_use2.index if item not in train_indices])

# choosing random indices equal to the number of damaging strikes
bird_use = bird_use.reset_index()
X_train = X_use2.loc[train_indices,]
Y_train = bird_use.loc[train_indices,'Damage']
X_test = X_use2.loc[test_indices,]
Y_test = bird_use.loc[test_indices,'Damage']


# ### Logistic Regression<a name="inferential1"></a>

# In[50]:


# Logistic Regression using Scikit-learn
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print('Training Accuracy: %1.3f.' % logreg.score(X_train, Y_train))


# In[51]:


# generate evaluation metrics
logreg_t = metrics.accuracy_score(Y_test, Y_pred)
print('Testing Accuracy: %1.3f.' % logreg_t)


# In[72]:


# evaluate the model using 10-fold cross-validation
scores_lr = cross_val_score(logreg, X_train, Y_train, scoring='accuracy', cv=10)
print('Cross-Validation Accuracy: %1.3f.' % scores_lr.mean())


# In[53]:


# ROC AUC on train set
Y_prob_train = logreg.predict_proba(X_train)
lr_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])
print ("ROC AUC on train set: %1.3f." % lr_auc_train)

# Predict on validation set
Y_prob_test = logreg.predict_proba(X_test)
lr_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])
print ("ROC AUC on validation set: %1.3f." % lr_auc_test)


# In[54]:


# logistic regression using statsmodels
logit = sm.Logit(bird_use["Damage"].reset_index(drop=True), X_use2)
result = logit.fit()
result.summary()


# In[55]:


# check significance of the features
features_coefs = result.params.sort_values(ascending=False)
selectSignificant = result.pvalues[result.pvalues <= 0.05].index
selectSignificant


# ### Support Vector Machines <a name="inferential2"></a>

# In[76]:


# Support Vector Machines
svc = SVC(class_weight='balanced',probability=True)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print('Training Accuracy: %1.3f.' % svc.score(X_train, Y_train))


# In[75]:


# generate evaluation metrics
svc_t = metrics.accuracy_score(Y_test, Y_pred)
print('Testing Accuracy: %1.3f.' % svc_t)


# In[77]:


# evaluate the model using 10-fold cross-validation
scores_svc = cross_val_score(svc, X_train, Y_train, scoring='accuracy', cv=10)
print('Cross-Validation Accuracy: %1.3f.' % scores_svc.mean())


# In[79]:


# ROC AUC on train set
Y_prob_train = svc.predict_proba(X_train)
svc_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])
print ("ROC AUC on train set: %1.3f." % svc_auc_train)

# Predict on validation set
Y_prob_test = svc.predict_proba(X_test)
svc_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])
print ("ROC AUC on validation set: %1.3f." % svc_auc_test)


# ### Random Forests <a name="inferential3"></a>

# In[80]:


# Random Forests
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
print('Training Accuracy: %1.3f.' % rf.score(X_train, Y_train))


# In[81]:


# generate evaluation metrics
rf_t = metrics.accuracy_score(Y_test, Y_pred)
print('Testing Accuracy: %1.3f.' % rf_t)


# In[82]:


# evaluate the model using 10-fold cross-validation
scores_rf = cross_val_score(rf, X_train, Y_train, scoring='accuracy', cv=10)
print('Cross-Validation Accuracy: %1.3f.' % scores_rf.mean())


# In[83]:


# ROC AUC on train set
Y_prob_train = rf.predict_proba(X_train)
rf_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])
print ("ROC AUC on train set: %1.3f." % rf_auc_train)

# Predict on validation set
Y_prob_test = rf.predict_proba(X_test)
rf_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])
print ("ROC AUC on validation set: %1.3f." % rf_auc_test)


# ### K-Nearest Neighbors Classifier <a name="inferential4"></a>

# In[62]:


# KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print('Training Accuracy:')
knn.score(X_train, Y_train)


# In[63]:


# generate evaluation metrics
knn_t = metrics.accuracy_score(Y_test, Y_pred)
print('Testing Accuracy:')
knn_t


# In[64]:


# evaluate the model using 10-fold cross-validation
scores_knn = cross_val_score(knn, X_train, Y_train, scoring='accuracy', cv=10)
print('Cross-Validation Accuracy:')
print(scores_knn.mean())


# In[84]:


# ROC AUC on train set
Y_prob_train = knn.predict_proba(X_train)
knn_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])
print ("ROC AUC on train set: %1.3f." % knn_auc_train)

# Predict on validation set
Y_prob_test = knn.predict_proba(X_test)
knn_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])
print ("ROC AUC on validation set: %1.3f." % knn_auc_test)


# ### Gaussian Naive Bayes <a name="inferential5"></a>

# In[65]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
print('Training Accuracy:')
gaussian.score(X_train, Y_train)


# In[66]:


# generate evaluation metrics
gaussian_t = metrics.accuracy_score(Y_test, Y_pred)
print('Testing Accuracy:')
gaussian_t


# In[67]:


# evaluate the model using 10-fold cross-validation
scores_gaussian = cross_val_score(gaussian, X_train, Y_train, scoring='accuracy', cv=10)
print('Cross-Validation Accuracy:')
print (scores_gaussian.mean())


# In[85]:


# ROC AUC on train set
Y_prob_train = gaussian.predict_proba(X_train)
gaussian_auc_train = metrics.roc_auc_score(Y_train, Y_prob_train[:, 1])
print ("ROC AUC on train set: %1.3f." % gaussian_auc_train)

# Predict on validation set
Y_prob_test = gaussian.predict_proba(X_test)
gaussian_auc_test = metrics.roc_auc_score(Y_test, Y_prob_test[:, 1])
print ("ROC AUC on validation set: %1.3f." % gaussian_auc_test)


# ### Model Summary <a name="inferential6"></a>

# In[87]:


train_acc = [logreg.score(X_train, Y_train), svc.score(X_train, Y_train), rf.score(X_train, Y_train),
             knn.score(X_train, Y_train), gaussian.score(X_train, Y_train)]
test_acc = [logreg_t, svc_t, rf_t, knn_t, gaussian_t]
cross_val_acc = [scores_lr.mean(), scores_svc.mean(), scores_rf.mean(), scores_knn.mean(), scores_gaussian.mean()]
train_auc = [lr_auc_train, svc_auc_train, rf_auc_train, knn_auc_train, gaussian_auc_train]
test_auc = [lr_auc_test, svc_auc_test, rf_auc_test, knn_auc_test, gaussian_auc_test]
models = DataFrame({'Training Accuracy': train_acc, 'Testing Accuracy': test_acc, 
                    "Cross-Validation Accuracy": cross_val_acc,'Training AUC': train_auc,
                   'Testing AUC': test_auc})
models.index = ['Logistic Regression','Support Vector Machines ','Random Forests','K-Nearest Neighbors','Gaussian Naive Bayes']
models


# In[88]:


models1 = DataFrame({'Accuracy' : models.unstack()}).reset_index()
# plot accuracies
plt.figure(figsize=(8, 7))
fig_models = sns.barplot(x='level_0', y='Accuracy', hue='level_1', data=models1);
fig_models.set(xlabel='Accuracy Metric', ylabel='Accuracy');
fig_models.set_title('The Accuracy of All Models Over Five Metrics');


# Based on the cross-validation and testing accuracies, the two most important metrics in model prediction among the three considered here, the Logistic Regression model yields the best performance. 
# 
# It is noteworthy that the Random Forests model performs slightly worse than the Logistic Regression on the metrics of cross-validation and testing accuracies. However, the Random Forests model has the highest training accuracy among all models. This suggests a potential model overfit.
# 
# In summary, the logistic regression model outperforms other models and can be chosen as a model for prediction and warning system for pilots and operators.

# ### Correlation Coefficients <a name="inferential7"></a>
# 
# The table below shows the coefficients, sorted by their absolute value in descending order.

# In[70]:


x=zip(X_train.columns, np.transpose(logreg.coef_))
x1=pd.DataFrame(list(x))
x1.head()


# In[71]:


# get Correlation Coefficient for each feature using Logistic Regression
logreg_df = pd.DataFrame(list(zip(X_train.columns, np.transpose(logreg.coef_))))
logreg_df.columns = ['Features','Coefficient Estimate']
logreg_df['sort'] = logreg_df['Coefficient Estimate'].abs()
logreg_df.sort_values(['sort'],ascending=0).drop('sort',axis=1).head(10)


# The table of top 10 most influential coefficient estimates suggest that bird size, population, weather, aircraft type, as well as the flight status all play roles in making bird strikes damaging events. 

# ## Conclusions and Suggestions<a name="conclusion"></a>
# 
# 1. Two-engine aircrafts are most frequently struck, but one-engine aircrafts are more prone to damage once struck. Such aircrafts should be equipped and operated with caution on this regard.
# 2. Small birds are the most frequently involved in bird strikes, but medium and large birds are more damaging once involved. Airports built near migration route of medium to large-size birds should have more countermeasures in place.
# 3. Bird strikes happen most often during takeoff or landing, especially the approach phase (when few countermeasures are in place), and during low altitude and low speed flights. Pilots should be informed with bird information during the approach phase if possible.
# 4. Sacramento International Airport stands out as the airport with most damaging bird strikes. More information on the airport would be needed to understand the cause, e.g. bird migration routes, airport traffic, bird control measures etc.
# 5. Bird strikes happen mostly between March to May, and July to October, between 5am and 1am of the next day. If resources are limited, detection and countermeasures should be optimized during such times to prevent bird strikes as much as possible.
# 6. Most of the strikes do not cause damages. However, bird strikes can cost money and even human lives at times.
# 7. Among the models tested, logistic regression model outperforms other models and can be chosen as a model for prediction and warning system for pilots and other operators. However, missing data is a huge issue with this data set, it would be useful to have the records as complete as possible so that a higher prediction accuracy can be achieved.
