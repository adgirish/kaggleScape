
# coding: utf-8

# **Simple Notebook for exploring the data ...**

# In[ ]:


import numpy as np 
import pandas as pd 
from scipy import signal
import matplotlib.pyplot as plt

import kagglegym


# In[ ]:


# This part is going to be for explorind the dataset ...
# so we want the entire dataset ..
with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# In[ ]:


dfId = df[['id', 'timestamp', 'y']].groupby('id').agg([
                    np.min, np.max, len, 
                lambda m: (list(m)[0] - list(m)[-1])/np.abs(np.mean(list(m))) ]  ).reset_index()
dfId.sort_values([('timestamp', 'amax')], inplace=True, ascending=False)
print(dfId.head())
print(dfId['y'].columns)


# When were the stocks bought and sole? This is interesting. It appears that the stocks are bought at regular intervals, but dropped at random times ...

# In[ ]:


plt.plot(dfId[('timestamp', 'amin')], dfId['id'], '.', mfc='green', mec='None', label='bought')
plt.plot(dfId[('timestamp', 'amax')], dfId['id'], '.', mfc='red',   mec='None', label='sold')
plt.xlabel('timestamp')
plt.ylabel('stock number')
plt.legend()


# Why are stocks sold at random points? Is it because the stock prices drop significantly? Let's look at the difference in the price between the start and end of a stock value when they are dropped. Are there any temporal trends in the data?

# In[ ]:


plt.scatter( dfId[('timestamp', 'amax')], 
             dfId['id'], 
             c    = dfId[('y', '<lambda>')], 
             s    = dfId[('timestamp', 'len')]/10,
             cmap = plt.cm.BrBG, vmin=-40, vmax=40).set_alpha(0.6)
plt.colorbar()


# It does not appear as if the number to stocks that are sold is is due to the value of the stock taking a dive. 

# In[ ]:


import seaborn as sns
dfStats = df[['y', 'id']].groupby('id').agg([np.median, np.std, np.min, np.max, np.mean]).reset_index()
dfStats.sort_values( ('y', 'median'), inplace=True )
print( dfStats.head() )
print( dfStats['y'].apply(np.median) )


sns.violinplot( dfStats[('y',  'amin')]   , color='orange')
sns.violinplot( dfStats[('y',  'median')] , color='teal')
sns.violinplot( dfStats[('y',  'amax')]   , color='indianred')


plt.figure()
temp = sns.kdeplot(dfStats[('y', 'amin')]  )
temp = sns.kdeplot(dfStats[('y', 'median')])
temp = sns.kdeplot(dfStats[('y', 'mean')]  )
temp = sns.kdeplot(dfStats[('y', 'amax')]  )
plt.yscale('log')

plt.figure()
temp = sns.kdeplot(dfStats[('y', 'std')])


# In[ ]:


plt.figure()
temp = sns.kdeplot(df['y'])
plt.yscale('log')


# In[ ]:


# Finding distributions of the result. 
# This is an entire portfolio. It will 
# be good to see how each variable changes 
# independent of each other ...
# -------------------------------------------

for i, (idVal, dfG) in enumerate(df[['id', 'timestamp', 'y']].groupby('id')):
    if i> 100: break
    df1 = dfG[['timestamp', 'y']].groupby('timestamp').agg(np.mean).reset_index()
    plt.plot(df1['timestamp'], np.cumsum(df1['y']),label='%d'%idVal)


# In[ ]:


# Finding distributions of the result. 
# This is an entire portfolio. It will 
# be good to see how each variable changes 
# independent of each other ...
# -------------------------------------------

for i, (idVal, dfG) in enumerate(df[['id', 'timestamp', 'y']].groupby('id')):
    if i> 100: break
    #df1 = dfG[['timestamp', 'y']].groupby('timestamp').agg(np.mean).reset_index()
    #plt.plot(df1['timestamp'], np.cumsum(df1['y']),label='%d'%idVal)
    dfG.head()


# So this 'asset' is made up of other different assets. Looks like there are several assets that "track each other." One thing to do would be to try and understand which one's do and which one's don't ...
# 
# Will look more into this after work ...

# In[ ]:


df2 = df[['id', 'timestamp', 'y']].pivot_table(values='y', index='timestamp', columns='id').reset_index(False)
df2.head()


# Now, lets find the autocorrelations for the different id's

# In[ ]:


cols = [ c for c in df2.columns if str(c) != 'timestamp']
lags = [1]
aCorrs = []
for i, c in enumerate(cols):
    try:
        aCorrs.append((c , max([(df2[c].autocorr(lag)) for lag in lags])))
    except:
        pass
    
aCorrs = pd.DataFrame(aCorrs, columns=['id', 'maxAcorr']).sort_values('maxAcorr', ascending=False)
print(aCorrs.head())


# Lets now plot the ones that have the highest autocorrelations ...

# In[ ]:


lags = range(1, 15)
for c in list(aCorrs.id)[:10]:
    plt.figure()
    plt.plot(list(df2[c])[:-1], list(df2[c])[1:], 's')
    plt.title(str(c))


# In[ ]:


cols = [ c for c in df2.columns if str(c) != 'timestamp']
corrs = df2[cols].corr()


# In[ ]:


corrs


# In[ ]:


temp = np.where(np.triu(corrs) < -0.9)
temp = [sorted(a) for a in zip(temp[0], temp[1]) if a[0]!=a[1]]


# Lets plot the first few that are highly correlated ...

# In[ ]:


prevId = -1
for i, (a, b) in enumerate(temp):
    
    if a != prevId:
        plt.figure()
        prevId = a
        plt.plot(np.cumsum(df2.ix[:, a]), label='id=%d'%a)
    plt.plot(np.cumsum(df2.ix[:, b]), label='id=%d'%b)
    plt.legend()
    
    if i > 5: break
    


# In[ ]:


list(df.columns)


# It appears that the portfolio is maintained by some form of advanced "pairs trading" platform. We are supposed to be predicting the result of the *result* of the entire system. 
# 
# Because of the "hedging strategy" that it is applying, I believe that the cumsum of the `y` variable over time is so fairly constant. If we can figure out that, we should be able to do something interesting. 
# 
# The problem just got infinitely more challenging!
