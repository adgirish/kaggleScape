
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Set figure aesthetics
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)


# I wanted to take a look at the user data we have for this competition so I made this little notebook to share my findings and discuss about those. At the moment I've started with the basic user data, I'll take a look at sessions and the other *csv* files later on this month.
# 
# Please, feel free to comment with anything you think it can be improved or fixed. I am not a professional in this field and there will be mistakes or things that can be *improved*. This is the flow I took and there are some plots not really interesting but I thought on keeping it in case someone see something interesting.
# 
# Let's see the data!

# ## Data Exploration

# Generally, when I start with a Data Science project I'm looking to answer the following questions:
# 
# - Is there any mistakes in the data?
# - Does the data have peculiar behavior?
# - Do I need to fix or remove any of the data to be more realistic?

# In[ ]:


# Load the data into DataFrames
train_users = pd.read_csv('../input/train_users.csv')
test_users = pd.read_csv('../input/test_users.csv')


# In[ ]:


print("We have", train_users.shape[0], "users in the training set and", 
      test_users.shape[0], "in the test set.")
print("In total we have", train_users.shape[0] + test_users.shape[0], "users.")


# Let's get those together so we can work with all the data.

# In[ ]:


# Merge train and test users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Remove ID's since now we are not interested in making predictions
users.drop('id',axis=1, inplace=True)

users.head()


# The data seems to be in an ussable format so the next important thing is to take a look at the missing data.

# ### Missing Data

# Usually the missing data comes in the way of *NaN*, but if we take a look at the DataFrame printed above we can see at the `gender` column some values being `-unknown-`. We will need to transform those values into *NaN* first:

# In[ ]:


users.gender.replace('-unknown-', np.nan, inplace=True)


# Now let's see how much data we are missing. For this purpose let's compute the NaN percentage of each feature.

# In[ ]:


users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')


# We have quite a lot of *NaN* in the `age` and `gender` wich will yield in lesser performance of the classifiers we will build. The feature `date_first_booking` has a 58% of NaN values because this feature is not present at the tests users, and therefore, we won't need it at the *modeling* part.

# In[ ]:


print("Just for the sake of curiosity; we have", 
      int((train_users.date_first_booking.isnull().sum() / train_users.shape[0]) * 100), 
      "% of missing values at date_first_booking in the training data")


# The other feature with a high rate of *NaN* was `age`. Let's see:

# In[ ]:


users.age.describe()


# There is some inconsistency in the age of some users as we can see above. It could be because the `age` inpout field was not sanitized or there was some mistakes handlig the data.

# In[ ]:


print(sum(users.age > 122))
print(sum(users.age < 18))


# So far, do we have 801 users with [the longest confirmed human lifespan record](https://en.wikipedia.org/wiki/Jeanne_Calment) and 176 little *gangsters* breaking the [Aribnb Eligibility Terms](https://www.airbnb.com/terms)?

# In[ ]:


users[users.age > 122]['age'].describe()


# It's seems that the weird values are caused by the appearance of 2014. I didn't figured why, but I supose that might be related with a wrong input being added with the new users.

# In[ ]:


users[users.age < 18]['age'].describe()


# The young users seems to be under an acceptable range being the 50% of those users above 16 years old. 
# We will need to hande the outliers. The simple thing that came to my mind it's to set an acceptance range and put those out of it to NaN.

# In[ ]:


users.loc[users.age > 95, 'age'] = np.nan
users.loc[users.age < 13, 'age'] = np.nan


# ### Data Types

# Let's treat each feature as what they are. This means we need to transform into categorical those features that we treas as categories and the same with the dates:

# In[ ]:


categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')


# In[ ]:


users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')


# ### Visualizing the Data

# Usually, looking at tables, percentiles, means, and other several measures at this state is rarely useful unless you know very well your data.
# 
# For me, it's usually better to visualize the data in some way. Visualization makes me see the outliers and errors immediately!

# #### Gender

# In[ ]:


users.gender.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Gender')
sns.despine()


# As we've seen before at this plot we can see the ammount of missing data in perspective. Also, notice that there is a slight difference between user gender.
# 
# Next thing it might be interesting to see if there is any gender preferences when travelling:

# In[ ]:


women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')

female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100

# Bar width
width = 0.4

male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()


# There are no big differences between the 2 main genders, so this plot it's not really ussefull except to know the relative destination frecuency of the countries. Let's see it clear here:

# In[ ]:


destination_percentage = users.country_destination.value_counts() / users.shape[0] * 100
destination_percentage.plot(kind='bar',color='#FD5C64', rot=0)
# Using seaborn can also be plotted
# sns.countplot(x="country_destination", data=users, order=list(users.country_destination.value_counts().keys()))
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()


# The first thing we can see that if there is a reservation, it's likely to be inside the US. But there is a 45% of people that never did a reservation.

# #### Age

# Now that I know there is no difference between male and female reservations at first sight I'll dig into the age.

# In[ ]:


sns.distplot(users.age.dropna(), color='#FD5C64')
plt.xlabel('Age')
sns.despine()


# As expected, the common age to travel is between 25 and 40. Let's see if, for example, older people travel in a different way. Let's pick an arbitrary age to split into two groups. Maybe 45?

# In[ ]:


age = 45

younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())

younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100
older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100

younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Youngers', rot=0)
older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Olders', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()


# We can see that the young people tends to stay in the US, and the older people choose to travel outside the country. Of vourse, there are no big differences between them and we must remember that we do not have the 42% of the ages. 
# 
# The first thing I thought when reading the problem was the importance of the native lenguage when choosing the destination country. So let's see how manny users use english as main language:

# In[ ]:


print((sum(users.language == 'en') / users.shape[0])*100)


# With the 96% of users using English as their language, it is understandable that a lot of people stay in the US. Someone maybe thinking, if the language is important, why not travel to GB? We need to remember that there is also a lot of factor we are not acounting so making assumpions or predictions like that might be dangerous.

# #### Dates

# To see the dates of our users and the timespan of them, let's plot the number of accounts created by time:

# In[ ]:


sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=1.1)
users.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')


# It's appreciable how fast Airbnb has grown over the last 3 years. Does this correlate with the date when the user was active for the first time? It should be very similar, so doing this is a way to check the data!

# In[ ]:


users.date_first_active.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')


# We can se that's almost the same as `date_account_created`, and also, notice the small peaks. We can, either smooth the graph or dig into those peaks. Let's dig in:

# In[ ]:


users_2013 = users[users['date_first_active'] > pd.to_datetime(20130101, format='%Y%m%d')]
users_2013 = users_2013[users_2013['date_first_active'] < pd.to_datetime(20140101, format='%Y%m%d')]
users_2013.date_first_active.value_counts().plot(kind='line', linewidth=2, color='#FD5C64')
plt.show()


# At first sight we can see a small pattern, there are some peaks at the same distance. Looking more closely:

# In[ ]:


weekdays = []
for date in users.date_account_created:
    weekdays.append(date.weekday())
weekdays = pd.Series(weekdays)


# In[ ]:


sns.barplot(x = weekdays.value_counts().index, y=weekdays.value_counts().values, order=range(0,7))
plt.xlabel('Week Day')
sns.despine()


# The local minimums where the Sundays(where the people use less *the Internet*), and it's usually to hit a maximum at Tuesdays!
# 
# The last date related plot I want to see is the next:

# In[ ]:


date = pd.to_datetime(20140101, format='%Y%m%d')

before = sum(users.loc[users['date_first_active'] < date, 'country_destination'].value_counts())
after = sum(users.loc[users['date_first_active'] > date, 'country_destination'].value_counts())
before_destinations = users.loc[users['date_first_active'] < date, 
                                'country_destination'].value_counts() / before * 100
after_destinations = users.loc[users['date_first_active'] > date, 
                               'country_destination'].value_counts() / after * 100
before_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Before 2014', rot=0)
after_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='After 2014', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()


# It's a clean comparision of usual destinations then and now, where we can see how the new users, register more and book less, and when they book they stay at the US.

# I'll make more plots about the devices and singup methods/flow later this week. I hope you all have enjoyed this little analysis that despine not being very rellevant to make the predictions, it is to understand the problem and the user behaviour. 
# 
# Again, criticism is welcomed!
# 
#                                                                                 David Gasquez
