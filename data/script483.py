
# coding: utf-8

# This script calculates the mean for each categorical value. It also bins the continuous features and then plots the mean loss per binned continuous feature.

# In[ ]:


# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import pylab as p

train = pd.read_csv("../input/train.csv")
#plt.rcParams['figure.figsize'] = 8, 6 #[6.0, 4.0]


# In[ ]:


features = train.columns
cats = [feature for feature in features if feature.startswith('cat')]
for feat in cats:
    train[feat] = pd.factorize(train[feat], sort=True)[0]


# In[ ]:


def plot_feature_loss(input_df,feature_name = 'cont1',num_bins = 50):
    if feature_name.startswith('cont'):
        bins = np.linspace(0,1.0,num_bins)
        feature_name_binned = feature_name + '_binned'
        input_df[feature_name_binned] = np.digitize(input_df[feature_name],bins=bins,right=True)
        input_df[feature_name_binned] = input_df[feature_name_binned] / num_bins
        temp_dict = input_df.groupby(feature_name_binned)['loss'].mean().to_dict()
        temp_err_dict = input_df.groupby(feature_name_binned)['loss'].sem().to_dict()
    else:
        temp_dict = input_df.groupby(feature_name)['loss'].mean().to_dict()
        temp_err_dict = input_df.groupby(feature_name)['loss'].sem().to_dict()

    lists = sorted(temp_dict.items())
    x, y = zip(*lists)
    lists_err = sorted(temp_err_dict.items())
    x_err, y_error = zip(*lists_err)

    p.figure()
    plt.errorbar(x,y,fmt = 'o',yerr = y_error,label = feature_name)
    p.xlabel(feature_name,fontsize=20)
    p.ylabel('loss',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    p.legend(prop={'size':20},numpoints=1,loc=(0.05,0.8))
    p.xlim([input_df[feature_name].min() - 0.02, input_df[feature_name].max() + 0.02 ])
    plt.grid()
    ax = plt.gca()

    plt.tick_params(axis='both', which='major', labelsize=15)
    ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax.xaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))


for name in train.columns:
    if name.startswith('cont'):
        plot_feature_loss(train,feature_name = name)
    if name.startswith('cat'):
        #limit number of pics made because of script limit on output files
        if int(name[3:]) >= 100: 
            plot_feature_loss(train,feature_name = name)

