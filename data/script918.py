
# coding: utf-8

# # The Mobile phone activity

# The Mobile phone activity dataset is a part of the Telecom Italia Big Data Challenge 2014, which is a rich and open multi-source aggregation of telecommunications, weather, news, social networks and electricity data from the city of Milan and the Province of Trentino (Italy).  
# 
# The original å has been created by Telecom Italia in association with EIT ICT Labs, SpazioDati, MIT Media Lab, Northeastern University, Polytechnic University of Milan, Fondazione Bruno Kessler, University of Trento and Trento RISE.
# 
# 
# In order to make it easy-to-use, here we provide a subset of telecommunications data that allows researchers to design algorithms able to exploit an enormous number of behavioral and social indicators. The complete version of the dataset is available at the following link: http://go.nature.com/2fz4AFr
# 

# ## Download the data
# From the [original paper](http://go.nature.com/2fz4AFr), in the "Data Citations" section, you find all the urls to download the data. Particularly, [Data citation n.4](http://dx.doi.org/10.7910/dvn/QLCABU) is related to this Kaggle aggregation.
# 
# You can download the first 7 days of november, to replicate the dataset we presented in Kaggle

# In[ ]:


import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

sns.set_style("ticks")
sns.set_context("paper")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Plots

# ### Data read

# In[ ]:


df_cdrs = pd.DataFrame({})
for i in range(1,8):
    df = pd.read_csv('../input/sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['datetime'])
    df_cdrs = df_cdrs.append(df)
    
df_cdrs=df_cdrs.fillna(0)
df_cdrs['sms'] = df_cdrs['smsin'] + df_cdrs['smsout']
df_cdrs['calls'] = df_cdrs['callin'] + df_cdrs['callout']
df_cdrs.head()


# ### Internet activity
# We select three areas and we plot the Internet activity of people
# 
# * Duomo (downtown)
# * Bocconi (university)
# * Navigli (night life!)

# In[ ]:


df_cdrs_internet = df_cdrs[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()
df_cdrs_internet['hour'] = df_cdrs_internet.datetime.dt.hour+24*(df_cdrs_internet.datetime.dt.day-1)
df_cdrs_internet = df_cdrs_internet.set_index(['hour']).sort_index()


# In[ ]:


f = plt.figure()

ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')
df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')
df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(ax=ax, label='Navigli')
plt.xlabel("Weekly hour")
plt.ylabel("Number of connections")
sns.despine()

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)


# ### Weekly boxPlot

# In[ ]:


boxplots = {
    'calls': "Calls",
    'sms': "SMS",
    "internet": "Internet CDRs"
}

df_cdrs_internet['weekday'] = df_cdrs_internet.datetime.dt.weekday

f, axs = plt.subplots(len(boxplots.keys()), sharex=True, sharey=False)
f.subplots_adjust(hspace=.35,wspace=0.1)
i = 0
plt.suptitle("")
for k,v in boxplots.items():
    ax = df_cdrs_internet.reset_index().boxplot(column=k, by='weekday', grid=False, sym='', ax =axs[i])
    axs[i].set_title(v)
    axs[i].set_xlabel("")
    sns.despine()
    i += 1
    
plt.xlabel("Weekday (0=Monday, 6=Sunday)")
f.text(0, 0.5, "Number of events", rotation="vertical", va="center")

