
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import functools
df = pd.read_csv('../input/consolidated_coin_data.csv')
df.Date=pd.to_datetime(df.Date)
df = df[::-1]
df.head()


# In[ ]:


crypto_coin = []
crypto_list = []
for i in df.Currency.unique():
    # we only take crypto coin that got more than 200 rows
    if df[df.Currency==i][['Date','Close']].shape[0] > 200:
        crypto_coin.append(df[df.Currency==i][['Date','Close']])
        crypto_list.append(i)


# In[ ]:


concatenated=functools.reduce(lambda left,right: pd.merge(left,right,on='Date',how='left' if left.shape[0] > right.shape[0] else 'right'), crypto_coin)


# In[ ]:


print(concatenated.shape)
concatenated.head()


# In[ ]:


correlation_set=pd.DataFrame(MinMaxScaler().fit_transform(concatenated.iloc[:,1:].dropna()), columns = crypto_list)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(correlation_set.corr())
plt.show()


# ## Let we check bitcoin Top 20 correlated with what coins

# In[ ]:


bitcoin_correlation = correlation_set.corr().iloc[:,crypto_list.index('bitcoin')].values
ind = (-bitcoin_correlation).argsort()[:20]
for i in ind:
    print(crypto_list[i],bitcoin_correlation[i])


# In[ ]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,20))
ind = (-bitcoin_correlation).argsort()[:6]
for i in ind:
    plt.plot(correlation_set.iloc[:, i])
plt.legend()
x_range_date = np.arange(correlation_set.shape[0])
plt.xticks(x_range_date[::60], concatenated.Date.iloc[::60])
plt.title('top 6 most trending with bitcoin')
plt.show()


# ## How about Top 20 least correlated with bitcoin

# In[ ]:


ind = (bitcoin_correlation).argsort()[:20]
for i in ind:
    print(crypto_list[i],bitcoin_correlation[i])


# In[ ]:


plt.figure(figsize=(20,20))
ind = (bitcoin_correlation).argsort()[:6]
for i in ind:
    plt.plot(correlation_set.iloc[:, i])
plt.legend()
x_range_date = np.arange(correlation_set.shape[0])
plt.xticks(x_range_date[::60], concatenated.Date.iloc[::60])
plt.title('top 6 least trending with bitcoin')
plt.show()


# ## Let we check Ethereum, top 20 most and least

# In[ ]:


e_correlation = correlation_set.corr().iloc[:,crypto_list.index('ethereum')].values
ind = (-e_correlation).argsort()[:20]
for i in ind:
    print(crypto_list[i],e_correlation[i])


# In[ ]:


plt.figure(figsize=(20,20))
ind = (-e_correlation).argsort()[:6]
for i in ind:
    plt.plot(correlation_set.iloc[:, i])
plt.legend()
x_range_date = np.arange(correlation_set.shape[0])
plt.xticks(x_range_date[::60], concatenated.Date.iloc[::60])
plt.title('top 6 most trending with Ethereum')
plt.show()


# In[ ]:


ind = (e_correlation).argsort()[:20]
for i in ind:
    print(crypto_list[i],e_correlation[i])


# In[ ]:


plt.figure(figsize=(20,20))
ind = (e_correlation).argsort()[:6]
for i in ind:
    plt.plot(correlation_set.iloc[:, i])
plt.legend()
x_range_date = np.arange(correlation_set.shape[0])
plt.xticks(x_range_date[::60], concatenated.Date.iloc[::60])
plt.title('top 6 least trending with ethereum')
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
sns.jointplot("bitcoin", "monacoin", 
              data=correlation_set, kind="reg",color="r", size=15)
plt.title('bitcoin with most correlated')
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
sns.jointplot("bitcoin", "iconomi", 
              data=correlation_set, kind="reg",color="r", size=15)
plt.title('bitcoin with least correlated')
plt.show()


# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()


# In[ ]:


plt.figure(figsize=(20,10))
ethereum=df[df.Currency=='ethereum'][['Date','Close']]
ethereum_shifted = ethereum - ethereum.shift(31)
mean_month=pd.rolling_mean(ethereum.Close, window=31).values
regr.fit(np.arange(mean_month[31:].shape[0]).reshape((-1,1)), mean_month[31:].reshape((-1,1)))
future_linear=regr.predict(np.arange(mean_month.shape[0]+1000).reshape((-1,1)))[:,0]
plt.plot(ethereum.Close.values, label = 'normal close')
plt.plot(ethereum_shifted.Close.values, label = 'minus month shifted')
plt.plot(ethereum.shift(31).Close.values, label = 'month shifted')
plt.plot(mean_month, label = 'mean every month')
plt.plot(future_linear, label='linear future moving mean')
plt.xticks(np.arange(ethereum.shape[0])[::280], ethereum.Date.iloc[::280])
plt.legend()
plt.title('ethereum analysis')
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
bitcoin=df[df.Currency=='bitcoin'][['Date','Close']]
bitcoin_shifted = bitcoin - bitcoin.shift(31)
mean_month=pd.rolling_mean(bitcoin.Close, window=31).values
regr.fit(np.arange(mean_month[31:].shape[0]).reshape((-1,1)), mean_month[31:].reshape((-1,1)))
future_linear=regr.predict(np.arange(mean_month.shape[0]+1000).reshape((-1,1)))[:,0]
plt.plot(bitcoin.Close.values, label = 'normal close')
plt.plot(bitcoin_shifted.Close.values, label = 'minus month shifted')
plt.plot(bitcoin.shift(31).Close.values, label = 'month shifted')
plt.plot(mean_month, label = 'mean every month')
plt.plot(future_linear, label='linear future moving mean')
plt.xticks(np.arange(bitcoin.shape[0])[::380], bitcoin.Date.iloc[::380])
plt.legend()
plt.title('bitcoin analysis')
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
ripple=df[df.Currency=='ripple'][['Date','Close']]
ripple_shifted = ripple - ripple.shift(31)
mean_month=pd.rolling_mean(ripple.Close, window=31).values
regr.fit(np.arange(mean_month[31:].shape[0]).reshape((-1,1)), mean_month[31:].reshape((-1,1)))
future_linear=regr.predict(np.arange(mean_month.shape[0]+1000).reshape((-1,1)))[:,0]
plt.plot(ripple.Close.values, label = 'normal close')
plt.plot(ripple_shifted.Close.values, label = 'minus month shifted')
plt.plot(ripple.shift(31).Close.values, label = 'month shifted')
plt.plot(mean_month, label = 'mean every month')
plt.plot(future_linear, label='linear future moving mean')
plt.xticks(np.arange(ripple.shape[0])[::380], ripple.Date.iloc[::380])
plt.legend()
plt.title('ripple analysis')
plt.show()

