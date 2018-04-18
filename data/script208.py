
# coding: utf-8

# # Data Exploration Starter
# ## Data Exploration Porto Seguro's Safe Driver Prediction
# ## Sergei Neviadomski

# First thing I want to mention is that this notebook is intended to provide first glance on data and variables. You won't find here any analysis of features interrelation or feature engineering. I intentionally provide only charts that show feature related information (distribution, bar plots and etc) and feature - label relationship.
# 
# I appreciate any constructive criticism and hope my notebook will help you understand data little bit more.
# 
# Thanks.
# 
# P.S. I think the best way to use my notebook is to quickly read through it and then come back as soon as you need additional information related to any feature.

# In[ ]:


### Necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Seaborn style
sns.set_style("whitegrid")


# In[ ]:


### Let's import our data
train_data = pd.read_csv('../input/train.csv',index_col='id')
### and test if everything OK
train_data.head()


# In[ ]:


### ... check for NAs in sense Pandas understands them
train_data.isnull().sum()


# In[ ]:


### ... check for -1 (NAs in our data)
(train_data==-1).sum()


# In[ ]:


### Now let's prepare lists of numeric, categorical and binary columns
# All features
all_features = train_data.columns.tolist()
all_features.remove('target')
# Numeric Features
numeric_features = [x for x in all_features if x[-3:] not in ['bin', 'cat']]
# Categorical Features
categorical_features = [x for x in all_features if x[-3:]=='cat']
# Binary Features
binary_features = [x for x in all_features if x[-3:]=='bin']


# In[ ]:


### Adding new column with beautiful target names
train_data['target_name'] = train_data['target'].map({0: 'Not Filed', 1: 'Filed'})


# ### Target Exploration

# In[ ]:


### Target variable exploration
sns.countplot(train_data.target_name);
plt.xlabel('Is Filed Claim?');
plt.ylabel('Number of occurrences');
plt.show()


# ### Corralation between features (variables)

# In[ ]:


### Corralation matrix heatmap
# Getting correlation matrix
cor_matrix = train_data[numeric_features].corr().round(2)
# Plotting heatmap 
fig = plt.figure(figsize=(18,18));
sns.heatmap(cor_matrix, annot=True, center=0, cmap = sns.diverging_palette(250, 10, as_cmap=True), ax=plt.subplot(111));
plt.show()


# ### Numeric Features Exploration

# In[ ]:


### Plotting Numeric Features
# Looping through and Plotting Numeric features
for column in numeric_features:    
    # Figure initiation
    fig = plt.figure(figsize=(18,12))
    
    ### Distribution plot
    sns.distplot(train_data[column], ax=plt.subplot(221));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Density', fontsize=14);
    # Adding Super Title (One for a whole figure)
    plt.suptitle('Plots for '+column, fontsize=18);
    
    ### Distribution per Claim Value
    # Claim Not Filed hist
    sns.distplot(train_data.loc[train_data.target==0, column], color='red', label='Claim not filed', ax=plt.subplot(222));
    # Claim Filed hist
    sns.distplot(train_data.loc[train_data.target==1, column], color='blue', label='Claim filed', ax=plt.subplot(222));
    # Adding Legend
    plt.legend(loc='best')
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Density per Claim Value', fontsize=14);
    
    ### Average Column value per Claim Value
    sns.barplot(x="target_name", y=column, data=train_data, ax=plt.subplot(223));
    # X-axis Label
    plt.xlabel('Is Filed Claim?', fontsize=14);
    # Y-axis Label
    plt.ylabel('Average ' + column, fontsize=14);
    
    ### Boxplot of Column per Claim Value
    sns.boxplot(x="target_name", y=column, data=train_data, ax=plt.subplot(224));
    # X-axis Label
    plt.xlabel('Is Filed Claim?', fontsize=14);
    # Y-axis Label
    plt.ylabel(column, fontsize=14);
    # Printing Chart
    plt.show()


# ### Categorical Features Exploration

# In[ ]:


### Plotting Categorical Features
# Looping through and Plotting Categorical features
for column in categorical_features:
    # Figure initiation
    fig = plt.figure(figsize=(18,12))
    
    ### Number of occurrences per categoty - target pair
    ax = sns.countplot(x=column, hue="target_name", data=train_data, ax = plt.subplot(211));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Number of occurrences', fontsize=14)
    # Adding Super Title (One for a whole figure)
    plt.suptitle('Plots for '+column, fontsize=18);
    
    ### Adding percents over bars
    # Getting heights of our bars
    height = [p.get_height() for p in ax.patches]
    # Counting number of bar groups 
    ncol = int(len(height)/2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    # Looping through bars
    for i, p in enumerate(ax.patches):    
        # Adding percentages
        ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 1000,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 

    
    ### Filed Claims percentage for every value of feature
    sns.pointplot(x=column, y='target', data=train_data, ax = plt.subplot(212));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Filed Claims Percentage', fontsize=14);
    # Printing Chart
    plt.show()


# ### Binary Features Exploration

# In[ ]:


### Plotting Binary Features
# Looping through and Plotting Binary features
for column in binary_features:
    ### Figure initiation 
    fig = plt.figure(figsize=(18,12))
    
    ### Number of occurrences per binary value - target pair
    ax = sns.countplot(x=column, hue="target_name", data=train_data, ax = plt.subplot(211));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Number of occurrences', fontsize=14)
    # Adding Super Title (One for a whole figure)
    plt.suptitle('Plots for '+column, fontsize=18);
    
    ### Adding percents over bars
    # Getting heights of our bars
    height = [p.get_height() for p in ax.patches]
    # Counting number of bar groups 
    ncol = int(len(height)/2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    # Looping through bars
    for i, p in enumerate(ax.patches):    
        # Adding percentages
        ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 1000,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 
        

    ### Filed Claims percentage for every value of feature
    sns.pointplot(x=column, y='target', data=train_data, ax = plt.subplot(212));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Filed Claims Percentage', fontsize=14);
    # Printing Chart
    plt.show()

