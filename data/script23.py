
# coding: utf-8

# # Introduction 
# 
# ![](https://i.ytimg.com/vi/7CJaDIkyvsU/maxresdefault.jpg)
# 
# This dataset is the result of an industry-wide survey conducted by Kaggle to establish a comprehensive view of the state of data science and machine learning as of the current year 2017. We have kindly been made available to the survey data so let us see what kind of analysis we might be able to craft out of the data and get an insight into the minds of a generation of data scientists (potential data scientists) that may come to shape the strategy and plans of this field in the years to come.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)


# Let us load in three (out of the five) different spreadsheets - namely that of the 1.) Multiple choice responses 2.) Free form responses and 3.) Conversion Rates

# In[2]:


mcr = pd.read_csv('../input/multipleChoiceResponses.csv', 
                  encoding="ISO-8859-1", 
                  low_memory=False)
ffr = pd.read_csv('../input/freeformResponses.csv', 
                  low_memory=False)
conversion = pd.read_csv('../input/conversionRates.csv')


# In[3]:


mcr.head(3)


# ## Countries surveyed
# 
# Let us take a quick look at the countries that participated in the survey and a count of the number of respondees via a Plotly globe plot (unhide the code if you want to see how it is generated).

# In[4]:


# Plotting 2010 and 2014 visuals
metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], [0.15, 'rgb(171,221,164)'], 
              [0.2, 'rgb(230,245,152)'], [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale1,
        showscale = True,
        locations = mcr.Country.value_counts().index,
        z = mcr.Country.value_counts().values,
        locationmode = 'country names',
        text = mcr.Country.value_counts().index,
        marker = dict(
            line = dict(color = 'rgb(250,250,225)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Number of Respondees')
            )
       ]

layout = dict(
    title = 'Number of People Surveyed\n (Countries in white represent no survey participation)',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,107,160)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap')


# **Takeaways from the Plot**
# 
# From the globe plot with the patches of red (greater number of survey respondees), we can see that the United States and India were the two countries that responded and contributed to the survey questions the most relative to the rest of the participating countries.

# ## What is the Gender balance of the survey population
# 
# Let us take a quick look at the gender ratio of the survey population to get a brief idea

# In[22]:


data = [go.Bar(
            x = mcr.GenderSelect.value_counts().index,
            y = mcr.GenderSelect.value_counts().values,
            marker= dict(colorscale='Jet',
                         color =  mcr.GenderSelect.value_counts().values
                        ),
            text='Number of respondees'
    )]

layout = go.Layout(
    title='Gender distribution of Survey participants'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')


# **Takeaway from the plots**
# 
# Most interesting indeed. Parking the two categories of "A different identity" and "Non-binary, genderqueer, or gender non-conforming" aside, we can see that there is still a huge disproprotion of Males participating in this survey (over Females).

# In[6]:


plt.style.use('dark_background')
x_sort = mcr.groupby(['GenderSelect','MLMethodNextYearSelect']).size().sort_values(ascending= False)
x_sort.unstack().plot(kind='bar',stacked=True, colormap= 'tab20', grid=False,  figsize=(13,11), )
plt.title('Stacked Barplot of ML methods across the different Genders')
plt.ylabel('Number of Respondees')


# ## Gender frequency for using Python?

# In[7]:


plt.style.use('dark_background')
x_sort = mcr.groupby(['GenderSelect','WorkToolsFrequencyPython']).size().sort_values(ascending= False)
x_sort.unstack().plot(kind='bar',stacked=True, colormap= 'Reds', grid=False,  figsize=(13,11), )
plt.title('Stacked Barplot of frequency of use of Python across the different Genders')
plt.ylabel('Number of Respondees')


# ## Gender frequency for using R?

# In[8]:


plt.style.use('dark_background')
x_sort = mcr.groupby(['GenderSelect','WorkToolsFrequencyR']).size().sort_values(ascending= False)
x_sort.unstack().plot(kind='bar',stacked=True, colormap= 'Blues', grid=False,  figsize=(13,11), )
plt.title('Stacked Barplot of frequency of use of R')
plt.ylabel('Number of Respondees')


# In[9]:


# plt.style.use('white_background')
x_sort = mcr.groupby(['GenderSelect','EmploymentStatus']).size().sort_values(ascending= False)
x_sort.unstack().plot(kind='bar',stacked=True, colormap= 'viridis_r', grid=False,  figsize=(13,11), )
plt.title('Stacked Barplot of Gender with Employment Status')
plt.ylabel('Number of Respondees')


# **Takeaway from the Plots**
# 
# The employment statuses of within the male-female gender buckets seem to be well distributed from a first glance. As we can see, the majority of the respondents are full-time employees (dark blue) 

# In[10]:


mcr.WorkToolsFrequencyPython.value_counts()


# In[11]:


mcr.WorkToolsFrequencyR.value_counts()


# # Choice of Recommended Language
# 
# Let us look at how the selection of recommended programming language ties in with the recorded job satisfaction.

# In[12]:


plt.style.use('dark_background')
x_sort = mcr.groupby(['LanguageRecommendationSelect','MLMethodNextYearSelect']).size().sort_values(ascending= False)
x_sort.unstack().plot(kind='bar',stacked=True, colormap= 'tab20', grid=False,  figsize=(13,11), )
plt.title('Stacked Barplot of Language with Machine Learning method')
plt.ylabel('Counts')


# **Takeaway from Plots**
# 
# Quite a well distributed set

# ## Employment Status

# In[13]:


mcr.EmploymentStatus.value_counts()


# In[14]:


mcr.head(3)


# # Categorical Correlations

# In[15]:


import numpy as np
mcr_cat = mcr.select_dtypes(exclude=['float64']).copy()


# In[16]:


col_names = []
for colname, colvalue in mcr_cat.iteritems():
    if len(colvalue.unique()) <= 4:
        col_names.append(colname)


# In[17]:


mcr_cat_encoded = pd.get_dummies(mcr_cat[col_names].fillna('missing'))


# In[18]:


# for colname, colvalue in mcr_cat.iteritems():
#     if len(colvalue.unique()) <= 4:
#         print(colname,colvalue.isnull().sum()/colvalue.shape[0])


# In[19]:


cols = [i for i in mcr_cat_encoded.columns if "missing" not in i]


# In[20]:


colormap = plt.cm.inferno
plt.figure(figsize=(26,21))
#fig, ax = plt.subplots(1, 1, figsize = (14,12), dpi=300)
#plt.title('Pearson Correlation of Features', y=1.05, size=20)
g = sns.heatmap(mcr_cat_encoded[cols].iloc[:,:60].corr(),linewidths=0,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=False)
#g.set_xticklabels(g.get_xticklabels(),rotation=30)
g.set_yticklabels(g.get_yticklabels(),rotation=30);
#g.set_yticklabels('')
#ax.set_ylabel('') 


# Seems the upper left-hand corner of the correlation plots show some interesting behaviour. Perhaps we could inspect that on a more granular level by zooming down into it.

# In[21]:


colormap = plt.cm.cubehelix
plt.figure(figsize=(26,21))
#fig, ax = plt.subplots(1, 1, figsize = (14,12), dpi=300)
#plt.title('Pearson Correlation of Features', y=1.05, size=20)
g = sns.heatmap(mcr_cat_encoded[cols].iloc[:,:10].corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#g.set_xticklabels(g.get_xticklabels(),rotation=30)
g.set_yticklabels(g.get_yticklabels(),rotation=30);

