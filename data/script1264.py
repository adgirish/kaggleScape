
# coding: utf-8

# Let us do some univariate analysis in this notebook and build simple regression models.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model as lm
import kagglegym

get_ipython().run_line_magic('matplotlib', 'inline')


# Read the train file from Kaggle gym.

# In[ ]:


# Create environment
env = kagglegym.make()

# Get first observation
observation = env.reset()

# Get the train dataframe
train = observation.train


# In[ ]:


train.shape


# In[ ]:


mean_values = train.mean(axis=0)
train.fillna(mean_values, inplace=True)
train.head()


# **Correlation coefficient plot:**
# 
# Let us look at the correlation of each of the variables with the target variables to get some important variables to be used for our next steps.

# In[ ]:


# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train.columns if col not in ['id','timestamp','y']]

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train[col].values, train.y.values)[0,1])
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(values), color='y')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient")
#autolabel(rects)
plt.show()


# As expected, the correlation coefficient values are very low and the maximum value is around 0.016 (in both positive and negative) as seen from the plot above.
# 
# Let us take the top 4 variables from the plot above and do some more analysis on them alone.
# 
#  - technical_30
#  - technical_20
#  - fundamental_11
#  - technical_19
# 
# As a first step, let us get the correlation coefficient in between these variables. 

# In[ ]:


cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

temp_df = train[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# There is some negative correlation between 'technical_30' and 'technical_20'. 
# 
# As the next step, let us build simple linear regression models using these variables alone and see how they perform.
# 
# Let us first build our models.

# In[ ]:


models_dict = {}
for col in cols_to_use:
    model = lm.LinearRegression()
    model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)
    models_dict[col] = model


# So we have built 4 univariate models using the train data.
# 
# **Technical_30:**
# 
# So we will start predicting with the model using 'technical_30' variable.

# In[ ]:


col = 'technical_30'
model = models_dict[col]
while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = model.predict(test_x)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
info


# We are getting a public score of 0.011 using this variable.
# 
# **Technical_20:**
# 
# Now let us predict the test using our second univariate model which we have built.

# In[ ]:


# Get first observation
env = kagglegym.make()
observation = env.reset()

col = 'technical_20'
model = models_dict[col]
while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = model.predict(test_x)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
info


# Using 'technical_20' as input variable, we are getting a public score of 0.0169 which is slightly better than the previous one.
# 
# Submitting this model to the LB gave me a score of 0.006. I have exported the above script into a kernel and it can be accessed [here][1].  
# 
# Let us do the same for our last two variables as well.
# 
# **Fundamental_11:**
# 
# 
#   [1]: https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/univariate-model

# In[ ]:


# Get first observation
env = kagglegym.make()
observation = env.reset()

col = 'fundamental_11'
model = models_dict[col]
while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = model.predict(test_x)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
info


# **Technical_19:**

# In[ ]:


# Get first observation
env = kagglegym.make()
observation = env.reset()

col = 'technical_19'
model = models_dict[col]
while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = model.predict(test_x)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
info


# **Regression using all 4 variables:**
# 
# Now let us build multiple regression model using all these 4 variables.

# In[ ]:


cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

# Get first observation
env = kagglegym.make()
observation = env.reset()
train = observation.train
train.fillna(mean_values, inplace=True)

model = lm.LinearRegression()
model.fit(np.array(train[cols_to_use]), train.y.values)

while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[cols_to_use])
    observation.target.y = model.predict(test_x)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
info


# This multiple regression gave a score of 0.019 which is better than all univariate models. So probably submitting this model might give a better LB score.
# 
# **Model with Clipping:**
# 
# As we can see from this [script][1] which gives the best public LB score of 0.00911, clipping the 'y' values help. 
# 
# So let us dig a little deeper to see why the public LB score increased from 0.006 to 0.009 when we clip the 'y' values.
# 
#   [1]: https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189/code

# In[ ]:


print("Max y value in train : ",train.y.max())
print("Min y value in train : ",train.y.min())


# Let us now do the clipping and see the number of rows that will be discarded from the training. 

# In[ ]:


low_y_cut = -0.086093
high_y_cut = 0.093497

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
y_is_within_cut.value_counts()


# So there are 9418 rows in the training set that lie between (-0.086093 and -0.0860941) and (0.093497 and 0.0934978) in the training set. So many values in such a small range.
# 
# As we can see from [anokas script][1], the distribution of 'y' values have two small spikes at both the ends. Probably values which are higher than these values are clipped in the training data and so not using these rows in our model building might be a good idea.
# 
# 
# 
# Now let us re-train our model (using technical_20) by excluding these rows from the training.
# 
# 
#   [1]: https://www.kaggle.com/anokas/two-sigma-financial-modeling/two-sigma-time-travel-eda

# In[ ]:


# Get first observation
env = kagglegym.make()
observation = env.reset()

col = 'technical_20'
model = lm.LinearRegression()
model.fit(np.array(train.loc[y_is_within_cut, col].values).reshape(-1,1), train.loc[y_is_within_cut, 'y'])

while True:
    observation.features.fillna(mean_values, inplace=True)
    test_x = np.array(observation.features[col].values).reshape(-1,1)
    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)
    #observation.target.fillna(0, inplace=True)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))
        
    observation, reward, done, info = env.step(target)
    if done:
        break
info


# So we got almost same public score of 0.0169 with clip.
# 
# But on the leaderboard, we are getting some improvement in the score from 0.006 to 0.009. 
# 
# Hope this gives a good starting point for building models. Happy Kaggling under new environment.!
