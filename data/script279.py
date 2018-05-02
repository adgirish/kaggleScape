
# coding: utf-8

# Based on the discussion here: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/52374
# I did some analysis on IP assignment.
# 
# **It looks very much like the organizers used different patterns in assigning dummy IPs in TEST and TRAIN sets.******
# 
# In the Train set, there is a strong correlation to assign IPs with higher frequencies to lower numbers.  The pattern **DOES NOT hold in Test**.
# 
# To copy my comment from discussion:
# 
# " Based on train_sample.csv and my own subsamples (i haven't been able to run a test on full train set), it appears that lower number IPs are strongly associated with higher number of clicks. i.e. numbers in the range 1 through 125000 have substantially more clicks than IPs in the range 300000 and up. It's almost as if when the ip values were generated to mask real ips, the data was pre-sorted by the number of clicks per IP, in a few major chunks. (So they gathered most frequent group, and assigned numbers from 1 to 125000, than next group and another bulk of numbers, than next, etc). I see about 4 bands of frequencies.
# 
# However, based on a few test subsamples I ran, the pattern does not repeat in test data. The ips over test seem to be mapped truly randomly, and if anything have consistent click density.
# 
# Also, (and somebody please check these calculations!) there appear to be the following distribution of IPs:
# 
# **Overall the number of IPs (test OR train): 333168 <br>
# Number of IPs that are in both (test AND train): 38164 <br>
# Number of IPs that are in Train and NOT in Test: 239232<br> Number of IPs that are in Test and NOT in Train: 55772**
# 
# That means that way over half of IPs in test do not follow the mapping rules of train data.
# 
# Hense I think need to be careful in validation. The pattern of IP assignment in Train does not mimic the one in test. If you go by using IP value as a signal in train, your final test results will be off substantially."
# 
# ** see below for visualization based on Train subsample and full Test data**
# 

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import gc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


input_path = '../input/'


# ### TRAIN SAMPLE

# In[3]:


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

train = pd.read_csv(input_path+'train_sample.csv', dtype=dtypes)
train.head()


# In[12]:


#convert to date/time
train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])

#extract hour as a feature
train['click_hour']=train['click_time'].dt.hour


# In[28]:


def plotStrip(x, y, hue, figsize = (14, 9)):
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y,              hue = hue, jitter = 0.4, marker = '.',              size = 4, palette = colours)
        ax.set_xlabel('')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['col1', 'col2'], bbox_to_anchor=(1, 1),                loc=2, borderaxespad=0, fontsize = 16);
    return ax


# In[18]:


X = train
Y = X['is_attributed']


# In[19]:


ax = plotStrip(X.click_hour, X.ip, Y)
ax.set_ylabel('ip', size = 16)
ax.set_title('IP (vertical), by HOUR(horizontal), split by converted or not(color)', size = 20);


# In[42]:


del train
gc.collect()


# ### TEST SUBSAMPLE

# In[44]:


total_rows = 18790470
sample_size = total_rows//120

def get_skiprows(total_rows, sample_size):
    inc = total_rows // sample_size
    return [row for row in range(1, total_rows) if row % inc != 0]

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        }

test = pd.read_csv(input_path+'test.csv',
                 skiprows=get_skiprows(total_rows,sample_size), dtype=dtypes)
test.head()


# In[45]:


#convert to date/time
test['click_time'] = pd.to_datetime(test['click_time'])

#extract hour as a feature
test['click_hour']=test['click_time'].dt.hour


# In[46]:


#dummy variable for hour color bands in test
test['band'] = np.where(test['click_hour']<=6, 0,                         np.where(test['click_hour']<=11, 1,                                 np.where(test['click_hour']<=15, 2, 3)))


# In[47]:


print(len(test))
X = test
Y = test['band']


# In[48]:


ax = plotStrip(X.click_hour, X.ip, Y)
ax.set_ylabel('ip', size = 16)
ax.set_title('IP (vertical), by HOUR(horizontal), split by hour band', size = 20);

