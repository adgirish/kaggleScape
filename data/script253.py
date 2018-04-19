
# coding: utf-8

# # <center> Relevent Member Data from Churn Competition </center>

# I feel this kernal was inevitable, so I decided to help everyone out by creating the dataset of relevant information from the Churn Competition. I have not read anywhere that we could not use this data for this competition, but use this data at your own risk. 

# In[ ]:


# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
get_ipython().run_line_magic('matplotlib', 'inline')

churn_data_path = '../input/kkbox-churn-prediction-challenge/'
recommend_data_path = '../input/kkbox-music-recommendation-challenge/'


# Grabbing the list of members in the Music Recommendation challenge for creating the subset of relevent user_logs data.

# In[ ]:


df_members = pd.read_csv(recommend_data_path + 'members.csv')
members = pd.DataFrame(df_members['msno'])


# Due to memory constraints, the user_log file must be read in by chunks and handled chunk by chunk. Created a dataframe of all the data related to the members in the Music Recommendation competition.

# In[ ]:


user_data = pd.DataFrame()
for chunk in pd.read_csv(churn_data_path + 'user_logs.csv', chunksize=500000):
    merged = members.merge(chunk, on='msno', how='inner')
    user_data = pd.concat([user_data, merged])


# Almost all of the members in the Music Recommendation challenge now now have additional information:

# In[ ]:


# Almost all members have additional information now
print (str(len(members['msno'].unique())) + " unique members in Music Recommendation Challenge")
print (str(len(user_data['msno'].unique())) + " users now have additional information")


# If you are curious, this reduced the size of the user_logs file from ~ 30GB down to about 500MB! <br>
# I'll output the data here into a csv file in case you disagree with the upcoming pre-processing.

# In[ ]:


user_data.to_csv('user_logs2.csv', index=False)


# A preview of the relevant members in the user_logs file:

# In[ ]:


print (user_data.head())


# There are some strange outliers in the total_secs column ( values < 0 ). <br>
# Since its only a fraction of the data, I'll just remove those rows.

# In[ ]:


for col in user_data.columns[1:]:
    outlier_count = user_data['msno'][user_data[col] < 0].count()
    print (str(outlier_count) + " outliers in column " + col)
user_data = user_data[user_data['total_secs'] >= 0]
print (user_data['msno'][user_data['total_secs'] < 0].count())


# I think the most logical thing to do next is to group the data by member id and then sum the columns corresponding to each member. In addition, the number of days a user listened to songs might be useful (the frequency count of each member), so this was added as well. The date column becomes useless if we do this, so it will be removed first.

# In[ ]:


del user_data['date']

print (str(np.shape(user_data)) + " -- Size of data large due to repeated msno")
counts = user_data.groupby('msno')['total_secs'].count().reset_index()
counts.columns = ['msno', 'days_listened']
sums = user_data.groupby('msno').sum().reset_index()
user_data = sums.merge(counts, how='inner', on='msno')

print (str(np.shape(user_data)) + " -- New size of data matches unique member count")
print (user_data.head())


# To get an idea of the effect of each new feature on the target, I have plotted the probabilty of a user repeating a song vs the new features:

# In[ ]:


df_train = pd.read_csv(recommend_data_path + 'train.csv')
train = df_train.merge(user_data, how='left', on='msno')

def repeat_chance_plot(groups, col, plot=False):
    x_axis = [] # Sort by type
    repeat = [] # % of time repeated
    for name, group in groups:
        count0 = float(group[group.target == 0][col].count())
        count1 = float(group[group.target == 1][col].count())
        percentage = count1/(count0 + count1)
        x_axis = np.append(x_axis, name)
        repeat = np.append(repeat, percentage)
    plt.figure()
    plt.title(col)
    sbn.barplot(x_axis, repeat)

for col in user_data.columns[1:]:
    tmp = pd.DataFrame(pd.qcut(train[col], 15, labels=False))
    tmp['target'] = train['target']
    groups = tmp.groupby(col)
    repeat_chance_plot(groups, col)


# Logically, it would seem like these columns would be pretty heavily correlated since they all relate to how many songs a user has listened to over a set amount of time. To see if this is true, a correlation heatmap proves pretty useful:

# In[ ]:


corrmat = user_data[user_data.columns[1:]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sbn.heatmap(corrmat, vmax=1, cbar=True, annot=True, square=True);
plt.show()


# From this map, almost everything seems to be pretty correlated. However a couple column pairs jump out; (num_75, num_50) and (num_unq, num_100) are the most heavily correlated and so I will remove one from each pair.

# In[ ]:


del user_data['num_75']
del user_data['num_unq']


# Lastly, I will look at the distribution of data in each column. From having done so already, I know that the distributions are heavily skewed so I will log transform the data in attempt to create normally distributed data and plot them both for you to see. In addition, I have normalized the data (std of 1, mean of 0) using the sklearn StandardScaler.

# In[ ]:


from sklearn.preprocessing import StandardScaler

cols = user_data.columns[1:]
log_user_data = user_data.copy()
log_user_data[cols] = np.log1p(user_data[cols])
ss = StandardScaler()
log_user_data[cols] = ss.fit_transform(log_user_data[cols])

for col in cols:
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    sbn.distplot(user_data[col].dropna())
    plt.subplot(1,2,2)
    sbn.distplot(log_user_data[col].dropna())
    plt.figure()


# In[ ]:


log_user_data.to_csv('user_logs_final.csv', index=False)


# I am new to the Machine Learning world and  would greatly appreciate any comments you may have.
