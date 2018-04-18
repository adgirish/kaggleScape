
# coding: utf-8

# In[ ]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read in the data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
sessions = pd.read_csv('../input/sessions.csv')


# ## Preliminary examination of the data
# The training data consists of 213,451 rows x 16 columns while the test data has 62,096 rows x 15 columns with our target feature 'country_destination' being the additional column in the training set. We can immediately see that we have a substantial number of missing values for age.

# In[ ]:


# training data
df_train.info()


# In[ ]:


# test data
df_test.info()


# ## Cleaning age data
# The age data is a bit messy - there are many values in the order of thousands. We will try to make some reasonable assumptions and use this to clean it up. We establish a valid range of ages as (14, 100) and assume that any values between 1919 and 1995 are birth years. Any ages outside the valid range are set to NaN.

# In[ ]:


df_train[df_train.age > 1000].age.hist(bins=5)


# In[ ]:


av = df_train.age.values
df_train['age'] = np.where(np.logical_and(av>1919, av<1995), 2015-av, av)
df_train['age'] = np.where(np.logical_or(av<14, av>100), np.nan, av)


# The age data also has a significant number of missing values. The plots below show the country destination proportions for users who have provided their age and for those where age is missing. Only 25% of users with missing ages book while 55% of users with age provided book. This suggests there is value to identifying users who have not provided age if, for example, we decide to impute missing ages. It could be that taking the step of providing age is a sign a user is more serious about making a booking. 

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 4))
complete_age = df_train[df_train.age.notnull()].country_destination.value_counts()
missing_age = df_train[df_train.age.isnull()].country_destination.value_counts()
complete_age.div(complete_age.sum()).plot(kind='bar',title='Country Destination Proportion (Age completed)',ax=axes[0])
missing_age.div(missing_age.sum()).plot(kind='bar',title='Country Destination Proportion (Age missing)',ax=axes[1])


# ## Data Exploration
# We begin by looking at the frequency of each class of our target feature, Country Destination. We know from the competition description that there should be 12 classes.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(15, 4))
country_counts = df_train.country_destination.value_counts()
country_counts.plot(kind='bar',title='Country Destination Count',ax=axes[0])
ax = country_counts.div(country_counts.sum()).plot(kind='bar',title='Country Destination %',ax=axes[1] )
ax.set_yticklabels(['{:3.1f}%'.format(x*100) for x in ax.get_yticks()])
booked_count = df_train[df_train.country_destination != 'NDF'].country_destination.value_counts()
ax1 = booked_count.div(booked_count.sum()).plot(kind='bar',title='Country Destination % excl NDF',ax=axes[2] )
ax1.set_yticklabels(['{:3.1f}%'.format(x*100) for x in ax1.get_yticks()])


# All 12 classes are represented in the training data and we can see that nearly 60% of users do not make a booking. Of the users that do book, about 70% do so in the US. We have rather imbalanced classes and will need to keep this in mind when training our model. 

# ### Age
# 
# The age data shows some interesting differentiation by country destination that could be useful. Users who book trips to Spain and Portugal tend to be younger while those that book trips to Great Britain tend to be older.

# In[ ]:


import seaborn as sns
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='country_destination', y='age', data=df_train, palette="muted", ax =ax)
ax.set_ylim([10, 60])


# ### Categorical variables
# 
# There are a lot of categorical variable so this chart is a bit crowded. Below are some quick comments about each.
# 
# - Starting with gender, it appears users with 'unknown' gender book less frequently than those with a known one while users with gender 'other' book more frequently
# - Users with the 'google' signup_method book less frequently than 'basic' or 'facebook'
# - Users with signup_flow 3 book more frequently than any other category while several have nearly 100% 'NDF'
# - Users with affiliate_channel 'content' book less frequently than other categories
# - Users with affiliate_provider 'craigslist', direct', and 'google' book more frequently than other categories (this begs the question as to why the google affiliate channel is more effective than the google sign up method)
# - Users with first_affiliate_tracked 'local ops' book less frequently than other categories
# - Users with signup_app 'Web' booked the most frequently, while those with 'Android' booked the least
# - Users with first_device_type 'Mac_Desktop' booked the most frequently, while those with 'Android Phone' booked the least
# - The chart on first_browser highlights the large number used above all else; it is difficult to glean any meaningful insights beyond that some obscure browsers that are not likely widely used have very high or very low booking frequencies.
# - The chart on language is somewhat surprising given that all the users were from the US - there are a large number of languages represented and this may warrant further investigation

# In[ ]:


bar_order = ['NDF','US','other','FR','IT','GB','ES','CA','DE','NL','AU','PT']
cat_vars = ['gender', 'signup_method', 'signup_flow', 'affiliate_channel', 'affiliate_provider', 
            'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'language']
from matplotlib.colors import Colormap
fig, ax4 = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
def pltCatVar(var,axis,ax_num):
    ctab = pd.crosstab([df_train[var]], df_train.country_destination).apply(lambda x: x/x.sum(), axis=1)
    ctab[bar_order].plot(kind='bar', stacked=True, ax=axis.reshape(-1)[ax_num],legend=False, 
                         colormap='terrain')
for i,var in enumerate(cat_vars[:8]):
    pltCatVar(var,ax4,i)
plt.tight_layout()
fig, ax5 = plt.subplots(nrows=2, ncols=1, figsize=(16, 8), sharey=True)
for i,var in enumerate(cat_vars[8:]):
    pltCatVar(var,ax5,i)
box = ax5[0].get_position()
ax5[0].set_position([box.x0, box.y0 + box.height * 0.4, box.width, box.height * 0.6])
ax5[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6)


# ### Dates

# In[ ]:


df_train['date_account_created'] = pd.to_datetime(df_train['date_account_created'])
df_train['date_first_active'] = pd.to_datetime((df_train.timestamp_first_active // 1000000), format='%Y%m%d')
df_train['date_first_booking'] = pd.to_datetime(df_train['date_first_booking'])
time_btw_dac_dfb = df_train['date_first_booking'] - df_train['date_account_created']
time_btw_tfa_dfb = df_train['date_first_booking'] - df_train['date_first_active']
print('---Time between Date Account Created and First Booking---')
print(time_btw_dac_dfb.describe())
print('---Time between Date First Active and First Booking---')
print(time_btw_tfa_dfb.describe())


# After cleaning up the 3 date variables, we can take a look at the distribution of time between 1) 'Date Account Created' 2) 'Date First Active' and 'Date First Booking.' We can see both average 44 days but have a very skewed distribution. 50% of people take 3 days and 75% take 29 with the average pulled up by a few very large outliers.

# We next take a look at a plot of the count of 'Date First Booking' over time. Note the 3 US summer holidays (Memorial Day, Independence Day, Labor Day) are marked with a 'O' each year (not working here).

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
df_train['date_first_booking'].value_counts().plot(kind='line', ax=axes)
'''import holidays # this is code to plot the 3 major US summer holidays - the package is not available here
holidays_tuples = holidays.US(years=[2010,2011,2012,2013])
popular_holidays = ['Independence Day', 'Labor Day', 'Memorial Day']
holidays_tuples = {k:v for (k,v) in holidays_tuples.items() if v in popular_holidays}
us_holidays = pd.to_datetime([i[0] for i in np.array(holidays_tuples.items())])
for date in us_holidays:
    axes.annotate('O', (date, df_train[df_train.date_first_booking == date]['date_first_booking'].value_counts()), xytext=(-35, 145), 
                textcoords='offset points', arrowprops=dict(arrowstyle='wedge'))'''
fig.autofmt_xdate()
plt.show()


# What immediately stands out is the seasonality of bookings. Bookings reach a peak around Labor Day each year and then decline into year end before starting to pick up again after New Year's. This suggests the month in the date variables could be useful to separate. Additionally, there appear to be small spikes in bookings right around the summer holidays so perhaps this could be useful, as well. (The spike in early 2014 is possibly the Super Bowl).
