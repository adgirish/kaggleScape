
# coding: utf-8

# # Exploring BNP Data Distributions
# 
# Hopefully this will run on Kaggle servers.  You should see a lot of plots (I can only see one right now, before pressing view HTML output).  
# If it doesn't I guess you'll have to run the code on your own machine. 

# In[ ]:


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read the Data
train = pd.read_csv("../input/train.csv")
train = train.drop(['ID'],axis=1)
test = pd.read_csv("../input/test.csv")
test = test.drop(['ID'],axis=1)
target = train.target
featureNames = train.columns.values


# In[ ]:


# Function to convert to hexavigesimal base
def az_to_int(az,nanVal=None):
    if az==az:  #catch NaN
        hv = 0
        for i in range(len(az)):
            hv += (ord(az[i].lower())-ord('a')+1)*26**(len(az)-1-i)
        return hv
    else:
        if nanVal is not None:
            return nanVal
        else:
            return az


# In[ ]:


# Prepare the data: combine, process, split
test['target'] = -999
all_data = train.append(test)

# convert v22 to hexavigesimal
all_data.v22 = all_data.v22.apply(az_to_int)

for c in all_data.columns.values:
    if all_data[c].dtype=='object':
        all_data[c], tmpItter = all_data[c].factorize()

# replace all NA's with -1
all_data.fillna(-1, inplace=True)

# split the data
train = all_data[all_data['target']>-999]
test = all_data[all_data['target']==-999]
test = test.drop(['target'],axis=1)


# ## Plot Descriptions
# 
# ### Histogram Plots on the the left:
# * Blue:  All of the train data (normalized)
# * Red:  Train Data where the target variable is one (again normalized)
# * Na's are -1, so the first column is usually large
# 
# ### CDF Plots on the right:
# * Blue and red as before
# * Black line is the difference in the CDF's (x10 + 0.5 for visualization)
# 
# ### A few interesting insights:
# * It's easy to see why v50 is such a powerful predictor
# * Somewhat counterintuitive, most of the features have more NA's when the target is true.  This is indicated both by the first red bar on the left being higher than the blue and by the cdf difference line being negative at the start.  Perhaps it's the presence of certain information, not the lack of it, that prevents fast-track processing.
# * With v22 coded in hexavigesimal, there is some large scale structure in the pdf, and possibly some structure in the CDF difference plot

# In[ ]:


plt.rcParams['figure.max_open_warning']=300
nbins=20
for c in  featureNames:
    if train[c].dtype != 'object' and c != 'target':
        if c=='v22':
            hbins = 100
        else:
            hbins = nbins
        fig=plt.figure(figsize=(14,4))
        ax1 = fig.add_subplot(1,2,1) 
        
        dataset1 = train[c][~np.isnan(train[c])]
        dataset2 = train[c][~np.isnan(train[c]) & train.target]
        
        # left plot
        hd = ax1.hist((dataset1, dataset2), bins=hbins, histtype='bar',normed=True,
                        color=["blue", "red"],label=['all','target=1'])
        ax1.set_xlabel('Feature: '+c)
        ax1.set_xlim((-1,max(train[c])))
        
        binwidth = hd[1][1]-hd[1][0]
        midpts = (hd[1][:-1]+hd[1][1:])/2
        cdf_all= np.cumsum(hd[0][0])*binwidth
        cdf_ones = np.cumsum(hd[0][1])*binwidth

        # right plot
        ax2 = fig.add_subplot(1,2,2) 
        ax2.set_ylim((0,1))
        ax2.set_xlim((0,nbins))
        ax2.plot(midpts,cdf_all,color='b')
        ax2.plot(midpts,cdf_ones,color='r')
        ax2.plot(midpts,0.5+10*(cdf_all-cdf_ones),color='k')
        ax2.grid()
        ax2.set_xlim((-1,max(train[c])))
        ax2.set_xlabel('cdfs plus cdf_diff*10+0.5')
        ax2.axhline(0.5,color='gray',linestyle='--')

