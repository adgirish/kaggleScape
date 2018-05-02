
# coding: utf-8

# **Porto Competition**
# 
# So I've been reading a lot about this Kaggel competitions, and tried to execute a couple of kernels myself, some with good results, others total failures... So I finally decided to join this competition and see how well it goes, and I also decided to stop using my personal laptop and give a try to this kaggle kernels and see how they perform. I will be using this notebook as reference (https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial).

# Anyways, if I find something nice on this kernel I will publish it later (try to get away from novice level!), if not at least I will try to do some feature engineering using this, eventually I will need to execute some portion of the code either in a dedicated kernell or rent some time on AWS.
# 
# I have three major intentions with this tutorial: (sorry about the typos I will fix them at some point in the future)
# 
# **1. Data validation Check.** Validation if there is any null, -1 or Nan.
# 
# **2. Feature Inspection. **Correlation plots, inspect the data.
# 
# **3. Feature importance** and analysis for implementing the classificaton methods. 

# Importing the useful functions, packages and others.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)

# Try ploty libraries
import plotly.tools as tls
import warnings

import seaborn as sns
plt.style.use('fivethirtyeight')

from collections import Counter
warnings.filterwarnings('ignore')

import plotly.graph_objs as go
import plotly.tools as tls
import plotly.plotly as plpl

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# Some data visualization, first see what we got and then we can start cleaning up the dataset.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(20)


# In[ ]:


test.head()


# I like to see some statistical information about the dataset. Since we have a lot of features, it's going to be a lot of information, but if at some point I will use feature engineering I would need to go back here and think about something.
#   
#  

# In[ ]:


# train.shape 
pd.set_option('precision',3)
train.describe()


# **Part One: Data validation Checks**
# 
# We can run a simple validation from the dataset just checking if there is any null.****

# In[ ]:


# Check if there is any null information anywhere
train.isnull().any().any()


# If there is any -1, according to the data description, it indicates the feature was missing from the observation.  So let's change it for NaN in a copy of our train.

# In[ ]:


train_cp = train
train_cp = train_cp.replace(-1, np.NaN)

data = train


# The following code will allow us to see how many NaN we have in the dataset, I'm taking a paranohic approach but, this should be useful, I guess this should at least give me the name of the colums for further pre-processing.

# In[ ]:


colwithnan = train_cp.columns[train_cp.isnull().any()].tolist()
print("Just a reminder this dataset has %s Rows. \n" % (train_cp.shape[0]))
for col in colwithnan:
    print("Column: %s has %s NaN" % (col, train_cp[col].isnull().sum()))


# According to the data description, anything ending in ind, reg, car and calc are simply similar groupings. cat corresponde categorial features and anything that do not have any designations they are continous or ordinal. 
# 
# You can see clearly they don't like to include **ps_car_03_cat** and **ps_car_05_cat** on the dataset, maybe it's a formality or an optional information when you fill your assurance information, who knows at this point. Another colum with lot of missing information is **ps_reg_03**, we will need to see how the prediction goes and the feature importance for the algorithms before taking any step further.
# 
# You should have 3 types of features:
# 
# **Categorical: **Where it has two or more categories and each value in the feature can be categorised by them, for example Color, Marrital status, gender. 
# 
# **Ordinal:** It's kind of the same than categorical values, but this kind of features can be ordered or sorted between the values, for example Size, Height, range. 
# 
# **Continous: **Those features can take any value between a minimun and a maximum, for example Age, Salary,
# 
# 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('target',data=train,ax=ax[1])
ax[1].set_title('target')
plt.show()


# You can see a big inbalance in the target, there are only a few amount of people the claim was filed. 

# In[ ]:


train_float = train.select_dtypes(include=['float64'])
train_int = train.select_dtypes(include=['int64'])
Counter(train.dtypes.values)


# **Part two: Feature Inspection**
# 
# At this point, I already have all the features split by data type.

# I would like to see some histograms to understand better the features, individually: (TODO)

# I would like to see some correlation plots about the different features, I will start with the float features we captured in the last step from previous sections:

# In[ ]:


colormap = plt.cm.jet
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# I chossed this colors because you have right away good insights of the data, if you see green, the features are no correlated, but if you see something not green, you will identify some correlation. 
# 
# They are somehow correlated ps_reg_01 with 02 and 03:
# * ps_reg_01, ps_reg_03 = 0.64
# * ps_reg_02, ps_reg_03 = 0.53
# * ps_reg_01, ps_reg_02 = 0.47
# 
# You will think since, 12 and 15 are related with 13 you will see some form of correlation between them, but is close to zero as you can see in the chart. 
# * ps_car_13, ps_car_12 = 0.67
# * ps_car_13 and ps_car_15 = 0.53
# 
# Since the features ps_calc_03, 02 and 01 are not correlated to anything, I will retire them to have a more condensed graphic. 

# In[ ]:


colormap = plt.cm.jet
cotrain_float = train_float.drop(['ps_calc_03', 'ps_calc_02', 'ps_calc_01'], axis=1)
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(cotrain_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# Now this is a really good condensed chart where you can see right away all the features correlation!

# In[ ]:


colormap = plt.cm.jet
plt.figure(figsize=(21,16))
plt.title('Pearson correlation of categorical features', y=1.05, size=15)
sns.heatmap(train_int.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)


# The chart is telling us the following: "Everything is pale green is not correlated, either negative or possitive. If we already know they don't correlate we should take as many features as possible from this correlation chart to spend quality time on the features that are correlated. For example ps_cal_04 to ps_calc_20_bin you can just take them away. (It would be a good idea to retire id and target).

# In[ ]:


colormap = plt.cm.jet
cotrain = train_int.drop(['id','target','ps_car_11', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin'], axis=1)
plt.figure(figsize=(21,16))
plt.title('Pearson correlation of int features withot ps_calc', y=1.05, size=12)
sns.heatmap(cotrain.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)


# We can see with more detail the features that are correlated and negative correlated:
# 
# * ps_ind_06_bin and 07 are ngative corrleated -0.47
# * ps_ind_11_bin ps_ind_14 =  0.56
# * ps_ind_12_bin with ps_ind_14 = 0.89
# * ps_ind_13_bin with ps_ind_14 = 0.43 (just because 14 is related to the others)
# * ps_ind_17_bin with ps_ind_16_bin -0.52
# * ps_ind_18_bin with ps_ind_16_bin -0.59
# * ps_ind_18_bin with ps_ind_15 -0.4
# 
# And finally just as to have the check in the final feature map and correlations (taking away id and target)
# 

# In[ ]:


colormap = plt.cm.jet
# train = train.drop(['id', 'target'], axis=1)
plt.figure(figsize=(25,25))
plt.title('Pearson correlation of All the features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)


# Let's dig into the continous features, and see if we find something interesting.

# In[ ]:


train_float.describe()


# In[ ]:


train_float.plot(kind='box', subplots=True, layout=(2,5), sharey=False, figsize=(18,18))
plt.show()


# We have real meaningfull information about the float features, in some cases the data is spread out, in some others not that much, I guess standarization will help a little bit for some particulars algorithms, but I guess is still early stages to say that.

# In[ ]:


#train_int = train_int.drop(['id', 'target'], axis=1)
#train_int.describe()


# I took me 14 version of this notebook to notice there are some categorical features (_cat) that really are binary... I will try to spot them and remove from the analisys before ploting the boxes.

# In[ ]:


# This section of code takes forever to execute!!
#train_int.plot(kind='box', subplots=True, layout=(10,5), sharey=False, figsize=(18,90))
#plt.show()


# This one is the most important output of this notebook, you can see right away there are some features really spread out, and others with high values, I maybe goig to produce another graphic where I kind of standarize the outputs, or I can classify them by the Y (range).  I mean you can see there are features between 0 and 1 but, you can also see there are features between 6-10, or 1-3, 8-10, etc etc. 
# 
# It's almost sure that I will go back and do more with this features before the final model.

# Binary Features

# In[ ]:


# Check the binary features
bin_col = [col for col in train.columns if '_bin' in col]
zeros = []
ones = []
for col in bin_col:
    zeros.append((train[col]==0).sum())
    ones.append((train[col]==1).sum())
    


# In[ ]:


trace1 = go.Bar(
    x=bin_col,
    y=zeros ,
    name='Zero count'
)
trace2 = go.Bar(
    x=bin_col,
    y=ones,
    name='One count'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Count of 1 and 0 in binary variables'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# I tried to produce the binary plot with plot.ly and with matplot lib, you can see right away the difference between both packages. I didn't want to copy that portion but, they are better graphics! I will transform all the graphics untl now, at some point.
# 
# This graph is basically telling, features ps_ind_10_bin, 11, 12 and 13 are basically useless. I will remove them from the feature correlaton map and redraw it, to see if I can spot something.

# In[ ]:


train_int = train_int.drop(['id', 'target'], axis=1)
train_int = train_int.drop(bin_col, axis=1)
some_bin = train_int.describe()
some_bin


# In[ ]:


cat_asbin = []
for col in some_bin:
    #print(some_bin[col]['max'])
    if (some_bin[col]['max']==1):
        if ((some_bin[col]['min']==0) or (some_bin[col]['min']==-1)):
            cat_asbin.append(col)
cat_asbin
    


# In[ ]:


cat_zeros = []
cat_ones = []
for col in cat_asbin:
    cat_zeros.append((train[col]==0).sum())
    cat_ones.append((train[col]==1).sum())


# In[ ]:


trace1 = go.Bar(
    x=cat_asbin,
    y=cat_zeros ,
    name='Zero count'
)
trace2 = go.Bar(
    x=cat_asbin,
    y=cat_ones,
    name='One count'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Count of 1 and 0 in binary variables'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# ps_car_07_cat is totally unbalanced as well as 08, I think 03 and 05 will be more balanced if we took away the NaN.
# 
# Remember:
#     
#     Column: ps_car_02_cat has 5 NaN
#     Column: ps_car_03_cat has 411231 NaN
#     Column: ps_car_05_cat has 266551 NaN
#     Column: ps_car_07_cat has 11489 NaN
#    
#  I would also exclude 07 and even 02 and 08 from some of the algorithms, in order to save some features. we will see what's next.
# 

# In[ ]:


colormap = plt.cm.jet
cotrainnb = cotrain.drop(['ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin', 'ps_ind_13_bin'], axis=1)
plt.figure(figsize=(21,16))
plt.title('Taking away some binary data', y=1.05, size=12)
sns.heatmap(cotrainnb.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=False)


# In[ ]:


cat_col = [col for col in train.columns if '_cat' in col]
catds = train[cat_col]
ncatds = catds.drop(cat_asbin, axis=1)
ncatds.describe()


# You can see here some interesting information:
# * Features 05 and 04 pretty much zeros. (75% - 0)
# * Feature 10 is pretty much ones.
# 
# Let's include some histograms to see the feature distribution.
# 

# In[ ]:


#ncatds.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
#plt.show()

from plotly import tools
hist_cat = []

for col in ncatds:
    hist_cat.append(go.Histogram(x=ncatds[col], opacity=0.75, name =col))

fig = tools.make_subplots(rows=len(hist_cat), cols=1)

for i in range(0,len(hist_cat),1):
    fig.append_trace(hist_cat[i], i+1, 1)
    
fig['layout'].update(height=1500, width=750, title='Cat Features Histogram')
py.iplot(fig, filename='cat-histogram')


# In[ ]:


no_cat = some_bin.drop(ncatds, axis=1)
no_cat.describe()


# **3. Feature Importance**
# 
# At this point, let's create a baseline of performance on this problem and spot-check a number of different algorithms:
# 
# * Linear Algorithms: 
#     1. Logistic Regression (LR)
#     2. Linear Discriminant Analysis (LDA).
# * Nonlinear Algorithms: 
#     1. Classification and Regression Trees (CART)
#     2. Gaussian Naive Bayes (NB)
#     3. k-Nearest Neighbors (KNN).
# 
# And then I will evaluate some others ensembles like:
# * Boosting Methods: 
#     1. AdaBoost (AB) 
#     2. Gradient Boosting (GBM).
# * Bagging Methods: 
#     1. Random Forests (RF)
#     2. Extra Trees (ET).
# 

# Some useful global information:
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Test options and evaluation metric
num_folds = 10
seed = 8
scoring = 'accuracy'

X = train.drop(['id','target'], axis=1)
Y = train.target

validation_size = 0.3
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Generating a first impression with the Linear and Non linear algorithms (now it's time to get some coffee, even 3 or 4 cups). Most of them are relatively fast to train, but the KNN it takes forever to do their job. 

# In[ ]:


models = [('LR', LogisticRegression()), 
          ('LDA', LinearDiscriminantAnalysis()),
          #('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('NB', GaussianNB())]
results = []
names = []
for name, model in models:
    print("Training model %s" %(name))
    model.fit(X_train, Y_train)
    result = model.score(X_validation, Y_validation)
    #kfold = KFold(n_splits=num_folds, random_state=seed)
    #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #results.append(cv_results)
    #names.append(name)
    msg = "Classifier score %s: %f" % (name, result)
    print(msg)
print("----- Training Done -----")


# All the algorithms showed good results up to 90%, but you can see the linear algorithms perform better with this dataset.

#     Training model LR
#     LR: 0.963379 (0.000790)
#     Training model LDA
#     LDA: 0.963374 (0.000794)
#     Training model KNN
#     KNN: 0.962825 (0.000789)
#     Training model CART
#     CART: 0.918605 (0.001355)
#     Training model NB
#     NB: 0.907404 (0.001143)
#     ----- Training Done -----

# Plotting the results for a visual comparision

# What about if we standarize Data, do you think we will have an improvement? I think we will, at least in some of the non linear algorithms. To do that I will use pipelines, just because it's crazy simple to do it.

# In[ ]:


pipelines = [('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])),
             ('ScaledLDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])),
             # ('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])),
             ('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])),
             ('ScaledNB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())]))]
results = []
names = []
for name, model in pipelines:
    print("Training model %s" %(name))
    model.fit(X_train, Y_train)
    result = model.score(X_validation, Y_validation)
    #kfold = KFold(n_splits=num_folds, random_state=seed)
    #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #results.append(cv_results)
    #names.append(name)
    msg = "Classifier score %s: %f" % (name, result)
    print(msg)
print("----- Training Done -----")


# In[ ]:


# ensembles
ensembles = [('ABC', AdaBoostClassifier()), 
             ('GBM', GradientBoostingClassifier()),
             ('RFC', RandomForestClassifier()),
             ('ETC', ExtraTreesClassifier())]
results = []
names = []

for name, model in ensembles:
    print("Training model %s" %(name))
    model.fit(X_train, Y_train)
    result = model.score(X_validation, Y_validation)
    #kfold = KFold(n_splits=num_folds, random_state=seed)
    #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #results.append(cv_results)
    #names.append(name)
    msg = "Classifier score %s: %f" % (name, result)
    print(msg)
print("----- Training Done -----")



#     AB: 0.963384 (0.000789)
#     GBM: 0.963365 (0.000793)
#     RF: 0.963341 (0.000788)
#     ET: 0.963382 (0.000805)
#     ----- Training Done -----

# I wanted to see in a single plot all the feature importance, so take a look to this piece of code! :) most of the ideas I took it from this  [link](https://plot.ly/python/bar-charts/)

# In[ ]:


toplot = []
for name, model in ensembles:
    trace = go.Bar(x=model.feature_importances_,
                   y=X_validation.columns,
                   orientation='h',
                   textposition = 'auto',
                   name=name
                  )
    toplot.append(trace)

layout = dict(
        title = 'Barplot of features importance',
        width = 900, height = 2000,
        barmode='group')

fig = go.Figure(data=toplot, layout=layout)
py.iplot(fig, filename='features-figure')
    


# At this point I think you have a lot of information to process, you know how the algorithms behave, you know the feature importance (for a couple of really important algorithms) and well, at this point you can start thinking what your next steps are going to be. 
# 
# Next iteration will include boosting for two or three methods, and I will start stacking.
