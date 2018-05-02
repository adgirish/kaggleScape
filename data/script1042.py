
# coding: utf-8

# # Python for Padawans
# 
# This tutorial will go throughthe basic data wrangling workflow I'm sure you all love to hate, in Python! 
# FYI: I come from a R background (aka I'm not a proper programmer) so if you see any formatting issues please cut me a bit of slack. 
# 
# **The aim for this post is to show people how to easily move their R workflows to Python (especially pandas/scikit)**
# 
# One thing I especially like is how consistent all the functions are. You don't need to switch up style like you have to when you move from base R to dplyr etc. 
# |
# And also, it's apparently much easier to push code to production using Python than R. So there's that. 
# 
# ### 1. Reading in libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math


# #### Don't forget that %matplotlib function. Otherwise your graphs will pop up in separate windows and stop the execution of further cells. And nobody got time for that.
# 
# ### 2. Reading in data

# In[ ]:


data = pd.read_csv('../input/loan.csv', low_memory=False)
data.drop(['id', 'member_id', 'emp_title'], axis=1, inplace=True)

data.replace('n/a', np.nan,inplace=True)
data.emp_length.fillna(value=0,inplace=True)

data['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
data['emp_length'] = data['emp_length'].astype(int)

data['term'] = data['term'].apply(lambda x: x.lstrip())


# ### 3. Basic plotting using Seaborn
# 
# Now let's make some pretty graphs. Coming from R I definitely prefer ggplot2 but the more I use Seaborn, the more I like it. If you kinda forget about adding "+" to your graphs and instead use the dot operator, it does essentially the same stuff.
# 
# **And I've just found out that you can create your own style sheets to make life easier. Wahoo!**
# 
# But anyway, below I'll show you how to format a decent looking Seaborn graph, as well as how to summarise a given dataframe.

# In[ ]:


import seaborn as sns
import matplotlib

s = pd.value_counts(data['emp_length']).to_frame().reset_index()
s.columns = ['type', 'count']

def emp_dur_graph(graph_title):

    sns.set_style("whitegrid")
    ax = sns.barplot(y = "count", x = 'type', data=s)
    ax.set(xlabel = '', ylabel = '', title = graph_title)
    ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
emp_dur_graph('Distribution of employment length for issued loans')


# ### 4. Using Seaborn stylesheets
# 
# Now before we move on, we'll look at using style sheets to customize our graphs nice and quickly.

# In[ ]:


import seaborn as sns
import matplotlib

print (plt.style.available)


# Now you can see that we've got quite a few to play with. I'm going to focus on the following styles:
# 
# - fivethirtyeight (because it's my fav website)
# - seaborn-notebook
# - ggplot
# - classic

# In[ ]:


import seaborn as sns
import matplotlib

plt.style.use('fivethirtyeight')
ax = emp_dur_graph('Fivethirty eight style')


# In[ ]:


plt.style.use('seaborn-notebook')
ax = emp_dur_graph('Seaborn-notebook style')


# In[ ]:


plt.style.use('ggplot')
ax = emp_dur_graph('ggplot style')


# In[ ]:


plt.style.use('classic')
ax = emp_dur_graph('classic style')


# ### 5. Working with dates
# 
# Now we want to looking at datetimes. Dates can be quite difficult to manipulate but it's worth the wait. Once they're formatted correctly life becomes much easier

# In[ ]:


import datetime

data.issue_d.fillna(value=np.nan,inplace=True)
issue_d_todate = pd.to_datetime(data.issue_d)
data.issue_d = pd.Series(data.issue_d).str.replace('-2015', '')
data.emp_length.fillna(value=np.nan,inplace=True)

data.drop(['loan_status'],1, inplace=True)

data.drop(['pymnt_plan','url','desc','title' ],1, inplace=True)

data.earliest_cr_line = pd.to_datetime(data.earliest_cr_line)
import datetime as dt
data['earliest_cr_line_year'] = data['earliest_cr_line'].dt.year


# ### 6. Making faceted graphs using Seaborn
# 
# Now I'll show you how you can build on the above data frame summaries as well as make some facet graphs.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

s = pd.value_counts(data['earliest_cr_line']).to_frame().reset_index()
s.columns = ['date', 'count']

s['year'] = s['date'].dt.year
s['month'] = s['date'].dt.month

d = s[s['year'] > 2008]

plt.rcParams.update(plt.rcParamsDefault)
sns.set_style("whitegrid")

g = sns.FacetGrid(d, col="year")
g = g.map(sns.pointplot, "month", "count")
g.set(xlabel = 'Month', ylabel = '')
axes = plt.gca()
_ = axes.set_ylim([0, d.year.max()])
plt.tight_layout()


# Now I want to show you how to easily drop columns that match a given pattern. Let's drop any column that includes "mths" in it.

# In[ ]:


mths = [s for s in data.columns.values if "mths" in s]
mths

data.drop(mths, axis=1, inplace=True)


# ### 7. Using groupby to create summary graphs

# In[ ]:


group = data.groupby('grade').agg([np.mean])
loan_amt_mean = group['loan_amnt'].reset_index()

import seaborn as sns
import matplotlib

plt.style.use('fivethirtyeight')

sns.set_style("whitegrid")
ax = sns.barplot(y = "mean", x = 'grade', data=loan_amt_mean)
ax.set(xlabel = '', ylabel = '', title = 'Average amount loaned, by loan grade')
ax.get_yaxis().set_major_formatter(
matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=0)


# ### 8. More advanced groupby statements visualised with faceted graphs

# In[ ]:


filtered  = data[data['earliest_cr_line_year'] > 2008]
group = filtered.groupby(['grade', 'earliest_cr_line_year']).agg([np.mean])

graph_df = group['int_rate'].reset_index()

import seaborn as sns
import matplotlib

plt.style.use('fivethirtyeight')
plt.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

sns.set_style("whitegrid")
g = sns.FacetGrid(graph_df, col="grade", col_wrap = 2)
g = g.map(sns.pointplot, "earliest_cr_line_year", "mean")
g.set(xlabel = 'Year', ylabel = '')
axes = plt.gca()
axes.set_ylim([0, graph_df['mean'].max()])
_ = plt.tight_layout()


# ### 9. Treatment of missing values
# This section is a toughie because there really is no correct answer. A pure data science/mining approach would test each of the approaches here using a CV split and include the most accurate treatment in their modelling pipeline.
# Here I have included the code for the following treatments:
# 
# - Mean imputation
# - Median imputation
# - Algorithmic imputation
# 
# I spent a large amount of time looking at 3. because I couldn't find anyone else who has implemented it, so I built it myself. In R it's very easy to use supervised learning techniques to impute missing values for a given variable (as shown here: https://www.kaggle.com/mrisdal/shelter-animal-outcomes/quick-dirty-randomforest) but sadly I couldn't find it done in Python.

# In[ ]:


#data['emp_length'].fillna(data['emp_length'].mean())
#data['emp_length'].fillna(data['emp_length'].median())
#data['emp_length'].fillna(data['earliest_cr_line_year'].median())

from sklearn.ensemble import RandomForestClassifier
rf =  RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1)

data['emp_length'].replace(to_replace=0, value=np.nan, inplace=True, regex=True)

cat_variables = ['term', 'purpose', 'grade']
columns = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'grade', 'purpose', 'term']

def impute_missing_algo(df, target, cat_vars, cols, algo):

    y = pd.DataFrame(df[target])
    X = df[cols].copy()
    X.drop(cat_vars, axis=1, inplace=True)

    cat_vars = pd.get_dummies(df[cat_vars])

    X = pd.concat([X, cat_vars], axis = 1)

    y['null'] = y[target].isnull()
    y['null'] = y.loc[:, target].isnull()
    X['null'] = y[target].isnull()

    y_missing = y[y['null'] == True]
    y_notmissing = y[y['null'] == False]
    X_missing = X[X['null'] == True]
    X_notmissing = X[X['null'] == False]

    y_missing.loc[:, target] = ''

    dfs = [y_missing, y_notmissing, X_missing, X_notmissing]
    
    for df in dfs:
        df.drop('null', inplace = True, axis = 1)

    y_missing = y_missing.values.ravel(order='C')
    y_notmissing = y_notmissing.values.ravel(order='C')
    X_missing = X_missing.as_matrix()
    X_notmissing = X_notmissing.as_matrix()
    
    algo.fit(X_notmissing, y_notmissing)
    y_missing = algo.predict(X_missing)

    y.loc[(y['null'] == True), target] = y_missing
    y.loc[(y['null'] == False), target] = y_notmissing
    
    return(y[target])

data['emp_length'] = impute_missing_algo(data, 'emp_length', cat_variables, columns, rf)
data['earliest_cr_line_year'] = impute_missing_algo(data, 'earliest_cr_line_year', cat_variables, columns, rf)


# ### 10. Running a simple classification model
# Here I take my cleaned variables (missing values have been imputed using random forests) and run a simple sklearn algo to classify the term of the loan.
# This step in the analytics pipeline does take longer in Python than in R (as R handles factor variables out of the box while sklearn only accepts numeric features) but it isn't that hard.
# This is just indicative though! A number of the variables are likely to introduce leakage to the prediction problem as they'll influence the term of the loan either directly or indirectly.

# In[ ]:


y = data.term

cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'grade', 'emp_length', 'purpose', 'earliest_cr_line_year']
X = pd.get_dummies(data[cols])

from sklearn import preprocessing

y = y.apply(lambda x: x.lstrip())

le = preprocessing.LabelEncoder()
le.fit(y)

y = le.transform(y)
X = X.as_matrix()

from sklearn import linear_model

logistic = linear_model.LogisticRegression()

logistic.fit(X, y)


# ### 11. Pipelining in sklearn
# 
# In this section I'll go through how you can combine multiple techniques (supervised an unsupervised) in a pipeline.
# These can be useful for a number of reasons:
# 
# - You can score the output of the whole pipeline
# - You can gridsearch for the whole pipeline making finding optimal parameters easier
# 
# So next we'll combine some a PCA (unsupervised) and Random Forests (supervised) to create a pipeline for modelling the data. 
# 
# In addition to this I'll show you an easy way to grid search for the optimal hyper parameters.

# In[ ]:


from sklearn import linear_model, decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

rf = RandomForestClassifier(max_depth=5, max_features=1)

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])

n_comp = [3, 5]
n_est = [10, 20]

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_comp,
                              rf__n_estimators=n_est))

estimator.fit(X, y)

