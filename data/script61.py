
# coding: utf-8

# # Crypto-Correlations
# 
# The goal of this analysis is to create a correlation matrix for these crypto-currencies. 

# In[ ]:


import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
import sklearn as sk
from sklearn import preprocessing as pr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import correlate
from scipy.stats.mstats import spearmanr
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_pacf


# ## Data Handling
#  
#  ### Data representation and a quick clean

# In[ ]:


crypto = {}

crypto['bitcoin'] = pd.read_csv('../input/cryptocurrencypricehistory/bitcoin_price.csv')
crypto['bitcoin_cash'] = pd.read_csv("../input/cryptocurrencypricehistory/bitcoin_cash_price.csv")
crypto['dash'] = pd.read_csv("../input/cryptocurrencypricehistory/dash_price.csv")
crypto['ethereum'] = pd.read_csv("../input/cryptocurrencypricehistory/ethereum_price.csv")
crypto['iota'] = pd.read_csv("../input/cryptocurrencypricehistory/iota_price.csv")
crypto['litecoin'] = pd.read_csv("../input/cryptocurrencypricehistory/litecoin_price.csv")
crypto['monero'] = pd.read_csv("../input/cryptocurrencypricehistory/monero_price.csv")
crypto['nem'] = pd.read_csv("../input/cryptocurrencypricehistory/nem_price.csv")
crypto['neo'] = pd.read_csv("../input/cryptocurrencypricehistory/neo_price.csv")
crypto['numeraire'] = pd.read_csv("../input/cryptocurrencypricehistory/numeraire_price.csv")
crypto['ripple'] = pd.read_csv("../input/cryptocurrencypricehistory/ripple_price.csv")
crypto['stratis'] = pd.read_csv("../input/cryptocurrencypricehistory/stratis_price.csv")
crypto['waves'] = pd.read_csv("../input/cryptocurrencypricehistory/waves_price.csv")


# In[ ]:


# For this analysis I will only be looking at closing price to make things more manageable
for coin in crypto:
    for column in crypto[coin].columns:
        if column not in ['Date', 'Close']:
            crypto[coin] = crypto[coin].drop(column, 1)
    # Make date the datetime type and reindex
    crypto[coin]['Date'] = pd.to_datetime(crypto[coin]['Date'])
    crypto[coin] = crypto[coin].sort_values('Date')
    crypto[coin] = crypto[coin].set_index(crypto[coin]['Date'])
    crypto[coin] = crypto[coin].drop('Date', 1)


# In[ ]:


for coin in crypto:
    print(coin, len(crypto[coin]))


# ### Note: 
# The coins numeraire, iota, and bitcoin_cash all are relatively young and therefore do not have many data points. I will omit these currencies and for the time being consider only the most recent 350 data points for the remaining currencies.

# In[ ]:


del crypto['bitcoin_cash'], crypto['numeraire'], crypto['iota']


# In[ ]:


cryptoAll = {} # for later on

for coin in crypto:
    cryptoAll[coin] = crypto[coin]
    crypto[coin] = crypto[coin][-350:]


# ## Goal:
# 
#  As previously stated, the goal of this analysis is to create a correlation matrix for these currencies. One way to find correlation between timeseries is to look at *cross-correlation* of the timeseries. Cross-correlation is computed between two timeseries using a lag, so when creating the correlation matrix I will specify the correlation as well as the lag.
#  
#  Before computing the cross correlation, it is important to have wide-sense station (often just called stationary) data. There are a few ways to make data stationary-- one of which is through differencing. But even after this it is famously difficult to avoid spurious correlations between timeseries data that are often caused by autocorrelation. See this article for an in depth analysis of how spurious correlations arise and how to avoid them: https://link.springer.com/article/10.3758/s13428-015-0611-2.
#  
#  For now I employ daily differencing (as it is not seasonal) and test for stationarity to prepare for cross correlation testing.

# In[ ]:


# Differencing
for coin in crypto:
    crypto[coin]['CloseDiff'] = crypto[coin]['Close'].diff().fillna(0)


# ### Graphing
# 
# Now lets take a preliminary look at how our graph looks. Further steps may have to be taken to make the data stationary.

# In[ ]:


for coin in crypto:
    plt.plot(crypto[coin]['CloseDiff'], label=coin)
plt.legend(loc=2)
plt.title('Daily Differenced Closing Prices')
plt.show()


# ### Note:
# Here we see that one of the coins (bitcoin) has much larger spikes than the other coins. While this may still have given us stationarity, it may be useful to also look at the percentage change per day of the timeseries.

# In[ ]:


# Percent Change
for coin in crypto:
    crypto[coin]['ClosePctChg'] = crypto[coin]['Close'].pct_change().fillna(0)
    


# In[ ]:


for coin in crypto:
    plt.plot(crypto[coin]['ClosePctChg'], label=coin)
plt.legend(loc=2)
plt.title('Daily Percent Change of Closing Price')
plt.show()


# ### Note:
# As before, we still have some very large peaks, but overall the data looks more contained than previously. Most importantly, we do not have a single coin dominating the others.
# 
# Focus on one particular part of the graph to get an idea of any correlation going on.

# In[ ]:


for coin in crypto:
    plt.plot(crypto[coin]['ClosePctChg'][-30:], label=coin)
plt.legend(loc=2)
plt.title('Daily Percent Change of Closing Price')
plt.show()


# ### Note:
# Looks here as if we do in fact have some correlation going on, which is what we were hoping for.
# 
# It is also important to note that a number of other types of differencing or normalizations could have been applied. As this is only a preliminary analysis, this may not end up being the best way to prepare the data.

# ## Stationarity
# 
# We can test for stationarity by using *unit root tests*. One of which is the Augmented Dickey-Fuller Test. Dickey Fuller utilizes the following regression.
# 
# $$ Y'_t \space = \space \phi Y_{t-1} \space + \space b_1 Y'_{t-1} \space + \space b_2 Y'_{t-2} \space +...+ \space b_p Y'_{t-p} $$
# $$ $$
# $$ Y'_t \space = \space Y_t \space - \space Y_{t-1} $$
# 
# Using the Augmented Dickey Fuller test, we look at the following statistic.
# 
# $$ DF_t \space = \space \frac{\hat{\phi}}{SE(\hat{\phi}}) $$
# 
# Then this statistic is compared to a table given by Dickey Fuller. Given the number of samples, we can guess with a % certainty whether or not our data is stationary.
# 
# $$ H_{0} \space : data \space is \space nonstationary $$
# $$ H_{A} \space : data \space is \space stationary $$
# 
# To check these hypotheses, we look at the p-value of our given statistic using table (web.sgh.waw.pl/~mrubas/EP/TabliceStatystyczneDF.doc). On the table we look at model 2 with 250 < n < 500. Form here we can see that in order to know with 5% certainty whether or not our data is stationary, we can compare our $ DF_t $ statistic to the values 3.46 and 3.44.

# In[ ]:


for coin in crypto:
    print('\n',coin)
    adf = adfuller(crypto[coin]['ClosePctChg'][1:])
    print(coin, 'ADF Statistic: %f' % adf[0])
    print(coin, 'p-value: %f' % adf[1])
    print(coin, 'Critical Values', adf[4]['1%'])
    print(adf)


# In[ ]:


for coin in crypto:
    print('\n',coin)
    adf = adfuller(crypto[coin]['CloseDiff'][1:])
    print(coin, 'ADF Statistic: %f' % adf[0])
    print(coin, 'p-value: %f' % adf[1])
    print(coin, 'Critical Values', adf[4]['1%'])
    print(adf)


# ### Note:
# Here we see that  our data is very stationary! This is clear because of the extremely low p-values.. 
# 
# It is important here to note there are other wasy to detrend other than looking at differenced data or percent change. However some of these methods would not have proven fruitful for this data set. Take for example using the residuals of this data based on a simple linear regression. This can be easily done using scikit learn's linear regression tool.

# In[ ]:


for coin in crypto:
    model = LinearRegression()
    model.fit(np.arange(350).reshape(-1,1), crypto[coin]['Close'].values)
    trend = model.predict(np.arange(350).reshape(-1,1))
    plt.subplot(1, 2, 1)
    plt.plot(trend, label='trend')
    plt.plot(crypto[coin]['Close'].values)
    plt.title(coin)
    
    plt.subplot(1, 2, 2)
    plt.plot(crypto[coin]['Close'].values - trend, label='residuals')
    plt.title(coin)
    
    plt.show()


# ### Note:
# We are getting poor results, as many of these currencies only started gaining traction recently, this shows that the preferred method was what was done originally.

# ## Correlations
# 
# Now we will look at the cross correlations between the different series. To do this scipy's correlate function will be used. The cross-correlation will tell us if we should lag one of the series. Cross-correlation is often used in signal process to match signals.

# In[ ]:


corrBitcoin = {}
corrDF = pd.DataFrame()

for coin in crypto: 
    corrBitcoin[coin] = correlate(crypto[coin]['ClosePctChg'], crypto['bitcoin']['ClosePctChg'])
    lag = np.argmax(corrBitcoin[coin])
    laggedCoin = np.roll(crypto[coin]['ClosePctChg'], shift=int(np.ceil(lag)))
    corrDF[coin] = laggedCoin
    
    plt.figure(figsize=(15,10))
    plt.plot(laggedCoin)
    plt.plot(crypto['bitcoin']['ClosePctChg'].values)
    title = coin + '/bitcoin PctChg lag: ' + str(lag-349)
    plt.title(title)

    plt.show()


# Now that we have done that we will look at the correlations among these currencies. 
# We will compute the correlations using three different methods: pearson, spearman, and kendall.

# In[ ]:


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }

plt.matshow(corrDF.corr(method='pearson'))
plt.xticks(range(10), corrDF.columns.values, rotation='vertical')
plt.yticks(range(10), corrDF.columns.values)
plt.xlabel('Pearson Correlation', fontdict=font)
plt.show()
corrDF.corr(method='pearson')


# In[ ]:


plt.matshow(corrDF.corr(method='spearman'))
plt.xticks(range(10), corrDF.columns.values, rotation='vertical')
plt.yticks(range(10), corrDF.columns.values)
plt.xlabel('Spearman Correlation', fontdict=font)
plt.show()
corrDF.corr(method='spearman')


# In[ ]:


plt.matshow(corrDF.corr(method='kendall'))
plt.xticks(range(10), corrDF.columns.values, rotation='vertical')
plt.yticks(range(10), corrDF.columns.values)
plt.xlabel('Kendall Correlation', fontdict = font)
plt.show()
corrDF.corr(method='kendall')


# ### Note:
# We see here that with all of these correlation methods we get about the same results, but with slightly different magnitudes.
# Also we should note that there are *no* correlations greater than .5
# This is contrary to what may be found if we were to for example take the correlation of the nonstationary datasets. This leads me to believe that I have avoided spurious correlations between currencies. Also note that only two of the currencies showed to have better correlations with lagged data. This makes sense as these currencies have shown to be very responsive to media in the recent past.
# 
# Thanks for reading I hope you enjoyed this notebook. If you have any suggestions or if I missed anything please let me know in the comments!
