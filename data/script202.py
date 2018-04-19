
# coding: utf-8

# # Data Exploration and Feature Engeenering Starter
# ## Two Sigma Connect   Renthop Competition
# ## Sergei Neviadomski

# In[ ]:


### Necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
### Seaborn style
sns.set_style("whitegrid")


# In[ ]:


### Let's import our data
train_data = pd.read_json('../input/train.json')
### and test if everything OK
train_data.head()


# In[ ]:


### ... check for NAs
train_data.isnull().sum()


# In[ ]:


### Target variable exploration
sns.countplot(train_data.interest_level, order=['low', 'medium', 'high']);
plt.xlabel('Interest Level');
plt.ylabel('Number of occurrences');


# In[ ]:


### Quantitative substitute of Interest Level
train_data['interest'] = np.where(train_data.interest_level=='low', 0,
                                  np.where(train_data.interest_level=='medium', 1, 2))


# In[ ]:


### Bathrooms graphs
fig = plt.figure(figsize=(12,12))
### Number of occurrences
sns.countplot(train_data.bathrooms, ax = plt.subplot(221));
plt.xlabel('Number of Bathrooms');
plt.ylabel('Number of occurrences');
### Average number of Bathrooms per Interest Level
sns.barplot(x='interest_level', y='bathrooms', data=train_data, order=['low', 'medium', 'high'],
            ax = plt.subplot(222));
plt.xlabel('Interest Level');
plt.ylabel('Average Number of Bathrooms');
### Average interest for every number of bathrooms
sns.pointplot(x="bathrooms", y="interest", data=train_data, ax = plt.subplot(212));
plt.xlabel('Number of Bathrooms');
plt.ylabel('Average Interest');


# In[ ]:


### Bedrooms graphs
fig = plt.figure(figsize=(12,12))
### Number of occurrences
sns.countplot(train_data.bedrooms, ax = plt.subplot(221));
plt.xlabel('Number of Bedrooms');
plt.ylabel('Number of occurrences');
### Average number of Bedrooms per Interest Level
sns.barplot(x='interest_level', y='bedrooms', data=train_data, order=['low', 'medium', 'high'],
            ax = plt.subplot(222));
plt.xlabel('Interest Level');
plt.ylabel('Average Number of Bedrooms');
### Average interest for every number of bedrooms
sns.pointplot(x="bedrooms", y="interest", data=train_data, ax = plt.subplot(212));
plt.xlabel('Number of Bedrooms');
plt.ylabel('Average Interest');


# In[ ]:


### Most advertised buildings
train_data.building_id.value_counts().nlargest(10)


# In[ ]:


### Convertion to Python Date
train_data.created = pd.to_datetime(train_data.created, format='%Y-%m-%d %H:%M:%S')


# In[ ]:


### New Month, Day of Week and Hour Features
train_data['month'] = train_data.created.dt.month
train_data['day_of_week'] = train_data.created.dt.weekday_name
train_data['hour'] = train_data.created.dt.hour


# In[ ]:


### First date in DataFrame
print('First advert created at {}'.format(train_data.created.nsmallest(1).values[0]))


# In[ ]:


### Last date in DataFrame
print('Last advert created at {}'.format(train_data.created.nlargest(1).values[0]))


# In[ ]:


### Iterest per month
fig = plt.figure(figsize=(12,6))
ax = sns.countplot(x="month", hue="interest_level", hue_order=['low', 'medium', 'high'],
                   data=train_data);
plt.xlabel('Month');
plt.ylabel('Number of occurrences')

### Adding percents over bars
height = [p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):    
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 50,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center") 


# In[ ]:


### Iterest per Day of Week
fig = plt.figure(figsize=(12,6))
ax = sns.countplot(x="day_of_week", hue="interest_level",
                   hue_order=['low', 'medium', 'high'], data=train_data,
                   order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']);
plt.xlabel('Day of Week');
plt.ylabel('Number of occurrences');

### Adding percents over bars
height = [p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):    
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 50,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center") 


# In[ ]:


### Iterest per Day of Week
fig = plt.figure(figsize=(12,6))
sns.countplot(x="hour", hue="interest_level", hue_order=['low', 'medium', 'high'], data=train_data);
plt.xlabel('Hour');
plt.ylabel('Number of occurrences');


# In[ ]:


### Number of unique Display Addresses
print('Number of Unique Display Addresses is {}'.format(train_data.display_address.value_counts().shape[0]))


# In[ ]:


### 15 most popular Display Addresses
train_data.display_address.value_counts().nlargest(15)


# In[ ]:


### Top 20 northernmost points
train_data.latitude.nlargest(20)


# In[ ]:


### Top 20 southernmost points
train_data.latitude.nsmallest(20)


# In[ ]:


### Top 20 easternmost points
train_data.longitude.nlargest(20)


# In[ ]:


### Top 20 westernmost points
train_data.longitude.nsmallest(20)


# In[ ]:


### Rent interest graph of New-York
sns.lmplot(x="longitude", y="latitude", fit_reg=False, hue='interest_level',
           hue_order=['low', 'medium', 'high'], size=9, scatter_kws={'alpha':0.4,'s':30},
           data=train_data[(train_data.longitude>train_data.longitude.quantile(0.005))
                           &(train_data.longitude<train_data.longitude.quantile(0.995))
                           &(train_data.latitude>train_data.latitude.quantile(0.005))                           
                           &(train_data.latitude<train_data.latitude.quantile(0.995))]);
plt.xlabel('Longitude');
plt.ylabel('Latitude');


# In[ ]:


### Let's get a list of top 10 managers
top10managers = train_data.manager_id.value_counts().nlargest(10).index.tolist()
### ...and plot number of different Interest Level rental adverts for each of them
fig = plt.figure(figsize=(12,6))
ax = sns.countplot(x="manager_id", hue="interest_level",
                   data=train_data[train_data.manager_id.isin(top10managers)]);
plt.xlabel('Manager');
plt.ylabel('Number of advert occurrences');
### Manager_ids are too long. Let's remove them
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off');

### Adding percents over bars
height = [0 if np.isnan(p.get_height()) else p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):    
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 20,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center")


# In[ ]:


### Getting number of photos
train_data['photos_number'] = train_data.photos.str.len()


# In[ ]:


### Number of photos graphs
fig = plt.figure(figsize=(12,12))
### Average number of Photos per Interest Level
sns.barplot(x="interest_level", y="photos_number", order=['low', 'medium', 'high'],
            data=train_data, ax=plt.subplot(221));
plt.xlabel('Interest Level');
plt.ylabel('Average Number of Photos');
### Average interest for every number of photos
sns.barplot(x="photos_number", y="interest", data=train_data, ax=plt.subplot(222));
plt.xlabel('Number of Photos');
plt.ylabel('Average Interest');
### Number of occurrences
sns.countplot(x='photos_number', hue='interest_level', hue_order=['low', 'medium', 'high'],
              data=train_data, ax=plt.subplot(212));
plt.xlabel('Number of Photos');
plt.ylabel('Number of occurrences');


# In[ ]:


### Price exploration
fig = plt.figure(figsize=(12,12))
### Price distribution
sns.distplot(train_data.price[train_data.price<=train_data.price.quantile(0.99)], ax=plt.subplot(211));
plt.xlabel('Price');
plt.ylabel('Density');
### Average Price per Interest Level
sns.barplot(x="interest_level", y="price", order=['low', 'medium', 'high'],
            data=train_data, ax=plt.subplot(223));
plt.xlabel('Interest Level');
plt.ylabel('Price');
### Violinplot of price for every Interest Level
sns.violinplot(x="interest_level", y="price", order=['low', 'medium', 'high'],
               data=train_data[train_data.price<=train_data.price.quantile(0.99)],
               ax=plt.subplot(224));
plt.xlabel('Interest Level');
plt.ylabel('Price');

