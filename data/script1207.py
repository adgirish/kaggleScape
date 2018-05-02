
# coding: utf-8

# ![banner][1]
# 
# 
#   [1]: https://s21.postimg.org/piyv8hmh3/Banner1.png

# The following is an analysis of global terrorist attacks from 1970 until 2015. It features detailed textual analysis, visual exploration, as well as modeling using various machine learning algorithms.
# 
# For more information, visit the [GTD website](https://www.start.umd.edu/gtd/), the [kaggle webpage](https://www.kaggle.com/START-UMD/gtd), or feel free to reach out to me!<br>
# <br>
# 
# This project is still under progress

# # Table of Contents:
# 
# I. [Introduction](#I---Introduction)
# 1. [Context](#Context)
# 2. [Aims and Objectives](#Aims-and-Objectives)
# 3. [Data Dictionary](#Data-Dictionary)
# 
# II. [Pre-Processing](#II---Pre-Processing)
# 1. [Describing the Data](#Describing-the-Data)
# 2. [Cleaning the Data](#Cleaning-the-Data)
# 3. [Categorizing Perpetrators](#Categorizing-Perpetrators)
# 4. [Categorizing Targets](#Categorizing-Targets)
# 5. [Filling Missing Data](#Filling-Missing-Data)
# 
# III. [Exploratory Data Analysis](#III---Exploratory-Data-Analysis)
# 
# 1. [Visual Exploration](#Visual-Exploration)
#     - [Casualties by Year](#Casualties-by-Year)
#     - [Casualties by Region](#Casualties-by-Region)
#     - [Casualties and Attacks by Country](#Casualties-and-Attacks-by-Country)
#     - [Top Perpetrators](#Top-Perpatrors)
#     - [Terror Attacks by Weapon Type](#Terror-Attacks-by-Weapon-Type)
#     - [Terror Tactics throught the Years](#Terror-Tactics-throughout-the-Years)
#     <br><br>
#         
# 2. [Feature Analysis](#Feature-Analysis)
#     - [Feature Selection](#Feature-Selection)
#     - [Target Selection](#Target-Selection)
#     - [Feature Importance](#Feature-Importance)
#         - [PCA](#Principal-Component-Analysis)
#         - [Chi2](#Chi2)
#         - [Scoring Metrics](#Scoring-Metrics)
#         - [Model Selection](#Model-Selection)
#         
# 
# IV. [Predictive Analysis](#IV---Predictive-Analysis)
# 1. [Model Calibration](#Model-Calibration)
#  - [Logistic Regression](#Logistic-Regression)
#  - [SGD Classifier](#SGD-Classifier)
#  - [Linear SVC](#Linear-SVC)
#  - [Random Forest Classifier](#Random-Forest-Classifier)
#  - [Multi-Layer Perceptron](#Multi-Layer-Perceptron)
#     <br><br>
# 
# 2. [Performance Summary](#Performance-Summary)
#     - [Score Table](#Score-Table)
#     - [Feature Coefficients](#Feature-Coefficients)
#     - [Confusion Matrices](#Confusion-Matrices)
#     - [ROC Curves](#ROC-Curves)
# 
# 
# V. [Appendix](#VII---Appendix)
# - [Plot Confusion Matrix](#Plot-Confusion-Matrix)
# - [Plot Cummulative Variance](#Plot-Confusion-Matrix)
# - [Plot ROC Curve](#Plot-ROC)
#     

# # I - Introduction

# [Table of Contents](#Table-of-Contents:)

# ## Context
# [Table of Contents](#Table-of-Contents:)

# *"It is two and a half minute to midnight" reads the Doomsday clock.*

# The past few decades have been painted with growing geo-political instability across the world and terrorism has been one of the main ways through which this global decay manifested itself.
# 
# In recent years, increased access to technology has allowed the average individual to get a deeper insight in what terror acts occurring around the world. As such, terrorism attacks have been increasingly mediatized. I thought it would be interesting to explore what is all over the news from another approach, a more methodic one so to say.

# 
# The Global Terrorism  Database (GTD) records terror attacks around the world from 1970 through 2015, it includes systematic data on domestic and international terrorist incidents durint this time period. 
# 
# - Contains information on over 150,000 terrorist attacks
# 
# - Currently the most comprehensive unclassified data base on terrorist events in the world
# 
# - Includes information on more than 75,000 bombings, 17,000 assassinations, and 9,000 kidnappings since 1970
# 
# - Includes information on at least 45 variables for each case, with more recent incidents including information on more than 120 variables
# 
# - Over 4,000,000 news articles and 25,000 news sources were reviewed to collect incident data from 1998 to 2015 alone
# 
# ---
# 
# For the purpose of this study,  the data analyzed will be strictly restricted to attacks that satisfy the three criteria per the Codebook guidelines.

# > **CRITERIA 1 : POLITICAL, ECONOMIC, RELIGIOUS, OR SOCIAL GOAL**
# 
# >The violent act must be aimed at attaining a political, economic, religious, or social goal. This criterion is not satisfied in those cases where the perpetrator(s) acted out of a pure profit motive or from an idiosyncratic personal motive unconnected with broader societal change
# 
# > **CRITERIA 2 : INTENTION TO COERCE, INTIMIDATE OR PUBLICIZE TO LARGER AUDIENCE(S)**
# 
# > To satisfy this criterion there must be evidence of an intention to coerce, intimidate, or convey some other message to a larger audience (or audiences) than the immediate victims. Such evidence can include (but is not limited to) the following: pre‐ or post‐attack statements by the perpetrator(s), past behavior by the perpetrators, or the particular nature of the target/victim, weapon, or attack type.
# 
# > **CRITERIA 3 : OUTSIDE INTERNATIONAL HUMANITARIAN LAW**
# 
# > The action is outside the context of legitimate warfare activities, insofar as it targets non‐combatants (i.e. the act must be outside the parameters permitted by international humanitarian law as reflected in the Additional Protocol to the Geneva Conventions of 12 August 1949 and elsewhere).

# ## Aims and Objectives
# [Table of Contents](#Table-of-Contents:)

# This notebook aims to explore the global terrorism database in two ways:
# - Visually exploring the extremely rich data on the 150,000 terrorist attacks, trying to answer questions such as:
#     
#     - Which countries/region are the most targeted?
#     
#     - Where are there the most casualties?
#     
#     - How have casualties evolved throughout the years?
#     
#     - What are the casualties by weapon type?
#     
#     - Are certain nationalities more targeted?
#     
#     - Are some countries better at defending themselves against terrorist attacks?
#     
#     
# - Analyze the data using scikit learn's classification library:
#     
#     - Can we predict if a terrorist attacks will result in casualties or yield no death and wounded.
#     
#     - If so, which model will be the most appropriate, should we use linear models, ensemble, support vector machines or neural networks?
#     
#     - How accurate will they be? How do we determine our scoring metrics?
#     
#     - Can we delve deeper by predicting thresholds of casualties? Will an attack be devastating or benign?
#     
#     - What about predicting the future? Can we get descent results running an ARIMA model?
# 
# 

# Now let's look at this database!

# ## Data Dictionary
# [Table of Contents](#Table-of-Contents:)
# 
# Here is a quick look at the data that will be used in this notebook, the complete descriptions can be found in the [codebook.](https://www.start.umd.edu/gtd/downloads/Codebook.pdf)
# 
# ```
# ## Spatio-Temporal Variables:
# 
# 'iyear'            : year of the incident
# 'imonth'           : month of the incident
# 'iday'             : day of the incident
# 
# 'latitude'         : latitude of the incident
# 'longitude'        : longitude of the incident
# 
# ## Continous variables: 
# 
# 'nkill'            : number of dead
# 'nwound'           : number of wounded
# 
# ## Binary Variables:
# 
# 'crit1'            : was the attack aimed at attaining a political, economic, religious, or social goal?
# 'crit2'            : was there intent to coerce or intimidate a larger audience than the victims?
# 'crit3'            : was the incident outside legitimate warfare activities (i.e. target non-combattants)?
# 'doubtter'         : aws there doubt as to whether or not the incident is a terrorist attack
# 
# 'extended'         : has the incident lasted for more than 24 hours?
# 'multiple'         : is the incident connected to other attacks?
# 'success'          : did the terrorist attack achieve its goal (i.e. assassination, etc.)?
# 'suicide'          : did the incident involve a suicide attack?
# 'guncertain1'      : was the terrorist group confirmed?
# 'claimed'          : was the incident claimed by a particular group?
# 'property'         : was property damaged during the attack?
# 'ishostkid'        : were victims taken hostages or kidnapped?
# 
# ## Categorical Variables:
# 
# 'country_txt'      : country in which the incident occured
# 'region_txt'       : region in which the incident occured
# 'alternative_txt'  : type of attack if it was not terrorist for certain
# 'attacktype1_txt'  : general method of attack used (i.e. assassination, hijacking, bombing/explosion, etc.)
# 'targtype1_txt'    : general type of target/victim (i.e. business, government, police, military, etc.)
# 'natlty1_txt'      : nationality of the target/victim
# 'weaptype1_txt'    : general type of weapon used in the incident (i.e. biological, chemical, firearms, etc.)
# 
# ## Descriptive Variables: 
# 
# 'target1'          : specific person, building, installation, etc. that was targeted
# 'gname'            : terrorist group responsible for the attack
# 'summary'          : summary of the incident, when avaialble
# 
# ```
# 
# 

# ---
# # II - Pre-Processing
# [Table of Contents](#Table-of-Contents:)

# We start by importing our packages! 
# 
# In the following we will be using matplotlib, seaborn, and plotly for plotting purposes, and scikit learn's classification library for modeling purposes

# In[ ]:


## Importing Packages

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import scipy.stats as stats
import os, sys, operator, warnings


# Scikit-learn Auxiliary Modules
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix
from sklearn.metrics import explained_variance_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve, precision_score, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, learning_curve, StratifiedKFold, train_test_split, validation_curve 
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline 


# Scikit-learn Classification Models
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# Natural Language Processing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from textblob import TextBlob, Word, WordList 



# Plotly 
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

py.offline.init_notebook_mode(connected=True)

# Other imports
import itertools
# import pprint
import patsy

# Setting some styles and options
sns.set_style('whitegrid') 
pd.options.display.max_columns = 40 

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
 
get_ipython().run_line_magic('matplotlib', 'inline')

print('Packages Imported Successfully!')


# In[ ]:


data = pd.read_csv('../input/globalterrorismdb_0616dist.csv', low_memory = False, encoding='ISO-8859-1')

print('Data Loaded Successfuly!')


# In[ ]:


print 'The dataset documents', data.shape[0], 'terror attacks with', data.shape[1], 'different features'


# ## Describing the Data
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


data_columns = [
    
    ## Spatio-Temporal Variables:
                'iyear', 'imonth', 'iday', 'latitude', 'longitude',
    
    ## Binary Variables: 
                'extended', 'vicinity', 'crit1', 'crit2', 'crit3', 'doubtterr',
                'multiple', 'success', 'suicide', 'guncertain1', ## check back guncertain
                'claimed', 'property', 'ishostkid',
    
    ## Continuous Variables:
                'nkill', 'nwound',               
    
    ## Categorical variables (textual): 
                'country_txt', 'region_txt', 'alternative_txt', 'attacktype1_txt', 'targtype1_txt',
                'natlty1_txt', 'weaptype1_txt', 
    
    ## Descriptive Variables: 
                'target1', 'gname', 'summary',    
    
                                            ]

gtd = data.loc[:, data_columns]

# To avoid confusion, we restrict the dataset to only attacks that were of terrorist nature.

gtd = gtd[(gtd.crit1 == 1) & (gtd.crit2 == 1) & (gtd.crit3 == 1) & (gtd.doubtterr == 0)]


# `First, we call pandas' describe method to summarize the data`

# In[ ]:


gtd.describe()


# `Here is an example of what the data looks like, these rows correspond to the 9/11 attacks`

# In[ ]:


print '9/11 attacks:'
gtd[(gtd.iyear == 2001) & (gtd.imonth == 9) & (gtd.iday == 11) & (gtd.country_txt == 'United States')]


# ## Cleaning the Data
# [Table of Contents](#Table-of-Contents:)

# We start by shortening the name of one of our weapon type categories which is way too long!

# In[ ]:


gtd.weaptype1_txt.replace(
    'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
    'Vehicle', inplace = True)


# Next, we replace the unknown values in columns vicinity, claimed, property, and ishostkid with zeros

# In[ ]:


gtd.iloc[:,[6, 15, 16, 17]] = gtd.iloc[:,[6, 15, 16, 17]].replace(-9,0)


# Fixing a couple of strange values in the claimed category

# In[ ]:


gtd.claimed.replace(2,1, inplace = True) # (3)


# We change the textual variables to lowercase for future ease of analysis

# In[ ]:


gtd.target1 = gtd.target1.str.lower()
gtd.gname = gtd.gname.str.lower()
gtd.summary = gtd.summary.str.lower()    
gtd.target1 = gtd.target1.fillna('unknown').replace('unk','unknown')


# We replace missing values for the number of dead and wounded using the median, rounding them in the process

# In[ ]:


gtd.nkill = np.round(gtd.nkill.fillna(gtd.nkill.median())).astype(int) 
gtd.nwound = np.round(gtd.nwound.fillna(gtd.nwound.median())).astype(int) 


# We then create two new features: total number of casualties and its binary interpretation

# In[ ]:


gtd['casualties'] = gtd.nkill + gtd.nwound
gtd['nclass'] = gtd.casualties.apply(lambda x: 0 if x == 0 else 1) 


# ## Categorizing Perpetrators
# [Table of Contents](#Table-of-Contents:)

# `Here, we define a function to categorize perpetrators which assigns terrorist groups with less than 10 attacks to a 'small_time_perpetrator' category.`

# In[ ]:


def categorize_perpetrators(column):
    '''
    This function reorganizes perpetrator groups based on their value_counts, perpetrator groups with
    less than 10 occurences are re-assigned to a new category called 'small_time_perpetrator'
    Parameter is of the type <pandas.core.series.Series>
    '''
    perpetrators_count = column.value_counts()
    small_time_perpetrator = perpetrators_count[perpetrators_count < 10].index.tolist()
    column = column.apply(lambda x: 'small time perpetrator' if x in small_time_perpetrator else x).astype(str)
    return column


# `We then apply the function to the target feature, in this case gtd.gname. We will now consider this column categorical.`

# In[ ]:



gtd.gname = categorize_perpetrators(gtd.gname)
print 'Perpetrators categorized!'


# ## Categorizing Targets
# [Table of Contents](#Table-of-Contents:)

# `Here, we define a function to categorize the textual target variable by performing a few re-assignments`

# In[ ]:


def categorize_target1(column):
    '''
    This function performs three operations:
    - It uses TextBlop in order to lemmatize (e.g. transform a word into its cannonical form) the textual data,
    for example, converting 'civilians' to 'civilian'. This enables us to increase the value count for recurrent
    words.
    - The second part of the function defines a list of top_targets, which include targets mentioned more than
    50 times. It then loops through every target string and re-assigns sentences that contain top_targets words.
    - Finally, it assigns every target not in top_targets to a new 'isolated target' category.
    Parameter is of the type <pandas.core.series.Series>
    '''
    
    temp_target = []
    for target in column:
        blob = TextBlob(target)
#         blob.ngrams = 2
        blop = blob.words
        lemma = [words.lemmatize() for words in blop]
        temp_target.append(" ".join(lemma))
    column = pd.Series(temp_target, index = column.index)
    target_count = column.value_counts()
    top_targets = target_count[target_count > 50].index.tolist()
    for item in top_targets: 
        column = column.apply(lambda x: item if item in x else x)
    column = column.apply(lambda x: 'isolated target' if x not in top_targets else x)
    return column


# `We then apply the function to the target variable`

# In[ ]:


gtd.target1 = categorize_target1(gtd.target1)
print('Targets categorized!')


# ## Filling Missing Data
# [Table of Contents](#Table-of-Contents:)

# Our data is almost ready for modeling, we only have a few changes to make. 
# 
# Let's look at the missing values

# In[ ]:


print 'missing data : \n'
print gtd.drop(['latitude','longitude','summary'], axis = 1).isnull().sum().sort_values(ascending = False).head(4)


# As we can see, for guncertain1 and ishostkid, the missing data is negligible and we can afford to fill NaNs with zeros.
# 
# However, given the size of missing data for the claimed variable, we need to find a way to input the data.
# 
# Given the binary nature of the column, let's use logistic regression

# In[ ]:


df = gtd.drop(['longitude','latitude', 'summary'], axis =1)


# In[ ]:


df.shape


# In[ ]:


df.guncertain1.fillna(0, inplace = True)
df.ishostkid.fillna(0, inplace = True)


# We start by assigning our target variable to claimed

# In[ ]:


y_temp = df.claimed
y_temp.shape


# Here we select our features for modeling.
# 
# We split them into categorical and numerical (binary) groups

# In[ ]:


categorical = ['country_txt', 'alternative_txt', 'attacktype1_txt',
               'targtype1_txt', 'weaptype1_txt', 'gname', 'target1']

numerical = ['extended', 'vicinity', 'multiple', 'success',
             'suicide', 'guncertain1', 'casualties', 'property', 'ishostkid',]


# Here we use Patsy, which allows us to create dummy columns for our categorical data per the above.

# In[ ]:


formula =  ' + '.join(numerical)+ ' + ' + ' + '.join(['C('+i+')' for i in categorical]) + ' -1' 
formula


# In[ ]:


X_temp = patsy.dmatrix(formula, data = df, return_type= 'dataframe')
print(X_temp.shape, y_temp.shape)


# Now that we have our target and features, we define our training and testing sets.
# 
# For fitting purposes we use every incident where claimed is known for our training set.
# 
# We then use the predictions from that model to fill missing values in our test set (the one with the NaN values)

# In[ ]:


X_train = X_temp[~y_temp.isnull()]
X_test = X_temp[y_temp.isnull()]


# In[ ]:


y_train = y_temp[~y_temp.isnull()]
y_test = y_temp[y_temp.isnull()]


# Checking the shape

# In[ ]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# Fitting our model

# In[ ]:


lr = LogisticRegression(random_state = 42).fit(X_train, y_train) 


# In[ ]:


predictions = pd.Series(lr.predict(X_test), index = X_test.index)


# Filling missing values, at last!

# In[ ]:


df.claimed.fillna(predictions, inplace = True)


# In[ ]:


# imputed_values = [ pred + sampling_from_norma_distribution for pred in predicted]


# ---
# # III - Exploratory Data Analysis
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


df = pd.read_csv('./Assets/modeling.csv') 
print 'Data Loaded Successfuly!'


# ## Visual Exploration
# [Table of Contents](#Table-of-Contents:)

# ### Map of Casualties from 1970 to 2015

# In[ ]:


trace = dict(
    type = 'choropleth',
    locationmode = 'country names',
    locations = cpc['country_txt'],

    z = cpc['casualties'],
    name = 'Casualties',
    text = cpc['country_txt'].astype(str) + '<br>' + cpc['casualties'].astype(str),
    hoverinfo = 'text+name',
    autocolorscale = False,
    colorscale = 'Viridis',
#     reversescale = True,
    marker = dict( line = dict ( color = 'rgb(255,255,255)', width = 0.5))
        
    )
        

layout = dict(
    title = 'Cummulative Casualties World Map from 1970 until 2015 ',
    geo = dict( showframe = False, showcoastlines = True,
               projection = dict(type = 'Mercator'), showlakes = True,
               lakecolor = 'rgb(255, 255, 255)'       
              )
    )
    

py.iplot(dict( data=[trace], layout=layout ))


# ### *Casualties by Year*

# In[ ]:


cpy = df.groupby('iyear', as_index=False)['casualties'].sum()

trace = go.Scatter(x = cpy.iyear, y = cpy.casualties,
                   name = 'Casualties', line = dict(color = 'salmon', width = 4, dash ='dot'),
                   hoverinfo = 'x+y+name')

layout = go.Layout(title = 'Casualties per Year')

py.iplot(dict(data = [trace], layout = layout))     


# ### *Casualties by Region*

# In[ ]:



cpr = df.groupby('region_txt', as_index= False)['casualties'].sum()
apr = df.groupby('region_txt')['region_txt'].count()

trace_1 = go.Bar(x = cpr.region_txt, y = cpr.casualties,
                 marker = dict(color = 'rgb(100, 229, 184)'),
                 name = 'Casualties')

trace_2 = go.Bar(x = apr.index, y = apr,
                 marker = dict(color = 'rgb(255, 188, 214)'),
                 name = 'Terror Attacks')

layout = go.Layout(title = "Total Casualties and Terror Attacks by Region", barmode='group' )


py.iplot(dict(data = [trace_1,trace_2], layout = layout))


# ### *Casualties and Attacks by Country*

# In[ ]:


### Top 10 countries by attack/fatalities
apc = df.groupby('country_txt')['country_txt'].count().sort_values(ascending= False)
cpc = df.groupby('country_txt', as_index= False)['casualties'].sum().sort_values(by = 'casualties', ascending= False)
cc = pd.merge(pd.DataFrame(apc), cpc, on = 'country_txt')


trace = go.Bar(x = apc.index[:20],y = apc,
                 marker = dict(color = 'rgb(255, 188, 214)'),
                 name = 'Terror Attacks')

layout = go.Layout(title = 'top 20 most targeted countries', barmode='relative' )

py.iplot(dict(data = [trace], layout = layout)) 


# In[ ]:


#### Notes to self

## Are nationalities more likely to get killed?
## Are certain countries better at capturing perpetrators?
## Are there countries particularily focused on kidnappings/hostages?
## How well do certain countries defend against terrorist attacks?
# 3d filled lines for number of attacks per year per top country
# Fatalities by target


# ## Feature Analysis
# [Table of Contents](#Table-of-Contents:)

# ### Target Selection

# Since we are trying to predict whether or not there will be casualties during a terrorist attack. We define our target as the binary counterpart to our casualties feature
# 
# - Casualties : 1
# 
# - No Casualties : 0
# 

# Our current casualties variable is continuous. As such, we hereby apply a lambda function in order to transform our target from its continuous form to a binary state.

# In[ ]:


y = df.casualties.apply(lambda x: 0 if x == 0 else 1).values


# ### Feature Selection

# In[ ]:



numerical = ['extended', 'vicinity', 'multiple', 'success', 'claimed',
             'suicide', 'guncertain1', 'property', 'ishostkid','natlty1_txt']

categorical = ['country_txt', 'alternative_txt', 'attacktype1_txt',
              'targtype1_txt', 'weaptype1_txt', 'gname', 'target1']


# Once again, we create a Patsy formula in order to frame our feature variables. This time, claimed is included in the features.

# In[ ]:


formula =  ' + '.join(numerical)+ ' + ' + ' + '.join(['C('+i+')' for i in categorical]) + ' -1' 
formula


# In[ ]:


X = patsy.dmatrix(formula, data = df, return_type= 'dataframe')


# In[ ]:


print X.shape, y.shape


# Let's take a look at our features

# In[ ]:


X.head(2) 


# In order to test the accuracy of our model, we split our dataset into a training set and a testing set.

# ### Feature Importance
# [Table of Contents](#Table-of-Contents:)

# #### Principal Component Analysis

# In[ ]:


pca_model = PCA(n_components=len(X.columns)) 
pca = pca_model.fit(X)


# In[ ]:


var_ratio = pca.explained_variance_ratio_
var_ratio = np.cumsum(var_ratio)
plot_cumsum_variance(var_ratio)


# The above tells us that about 200 principal components justify 90% of the explained variance

# #### Chi2

# Here we look at individual importance by delivering a chi2 test

# In[ ]:


X_columns = list(X.columns) #Here we transfrom our variables into a list

#We then apply a chi2 statistical measure
skb_chi2 = SelectKBest(chi2, k=20)
skb_chi2.fit(X, y)

# examine results
top_15_chi2 = pd.DataFrame([X_columns, list(skb_chi2.scores_)], 
                     index=['feature','chi2 score']).T.sort_values('chi2 score', ascending=False)[:15]
top_15_chi2


# In[ ]:


plt.figure(figsize=(13,6))

sns.barplot(x = top_15_chi2['chi2 score'], y = top_15_chi2.feature, palette= 'viridis')
plt.show()


# ### Scoring Metrics
# [Table of Contents](#Table-of-Contents:)

# In order to evaluate our model we simultaneously use two approaches:
# - **Train-Test Splitting:** 
#     
#     Splitting our data into training and testing set allows us to train our model on one set and test its accuracy on the other.
# 
# 
# - **Recall Scoring: **
#     
#     Given the nature of the data, we are better off predicting casualties and sending an ambulance when there is no casualties than not to do anything when there are. As such, we are trying to maximize our true positives and reduce our false negatives, hence trying to maximize our recall score!

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 102)

print X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ### Model Selection
# [Table of Contents](#Table-of-Contents:)

# Let's get started! 
# 
# 
# We start by creating a dictionary of models to test on our data.

# In[ ]:


vanilla_models = { 
    
    # Linear Models
    'Logistic Regression' : LogisticRegression(n_jobs = -1, random_state = 56, penalty = 'l1'),
    'Perceptron' : Perceptron(n_iter = 20, n_jobs = -1, random_state= 56),
    'SGD Classifier' : SGDClassifier(penalty = 'l1', n_jobs = -1, random_state= 56),
    
    # Support Vector Machine
    'Linear SVC' : LinearSVC(penalty = 'l1', random_state = 56, dual = False),
    
    # Naive Bayes:
    'Gaussian Naive-Bayes' : GaussianNB(),
    
    # Decision Tree & Ensemble
    'Decision Tree Classifier' : DecisionTreeClassifier(random_state= 56),
    'Random Forest Classifier': RandomForestClassifier(n_jobs = -1, random_state= 56),
    'Gradient Boosting Classifier' : GradientBoostingClassifier(random_state= 56),
    'AdA Boost Classifier': AdaBoostClassifier(random_state = 56),
    'Bagging Classifier' : BaggingClassifier(random_state= 56, n_jobs = -1),
    
    # K-Nearest Neighbor:
    
    # Multi-Layer Perceptron (Neural Network):
    'MLP Classifier' : MLPClassifier(activation = 'logistic', random_state = 56, max_iter=400),  
    
}


# We now instantiate a pipeline to fit our models with our training data

# In[ ]:


score_table = pd.DataFrame(columns = ['model', 'cv_10'])


for model, n in zip(vanilla_models, np.arange(len(vanilla_models))):
                    
    clf = Pipeline([
          ('classification', vanilla_models[model]),
        ])
    
    clf.fit(X_train, y_train)
    
    cv_10 = cross_val_score(clf, X_test, y_test, cv = 10, scoring = 'recall').mean()
    
    score_table.loc[n,'model'] = model
    score_table.loc[n,'cv_10'] = cv_10


# In[ ]:


score_table.sort_values(by = 'cv_10', ascending = False)


# In[ ]:



plt.figure(figsize = (11,4))
plt.xticks(rotation = 45, ha = 'center')
sns.barplot(score_table.model, score_table.cv_10, palette = 'viridis');


# From the above, we can see that our top performers are:
# 
# - Logistic Regression
# 
# - SGD Classifier
# 
# - SVC Classification
# 
# - Random Forest Classifier
# 
# - Multi-Layer Perceptron

# ---
# # IV - Predictive Analysis
# [Table of Contents](#Table-of-Contents:)

# ## Model Calibration
# 

# ### Logistic Regression
# [Table of Contents](#Table-of-Contents:)

# The first model we will be trying to improve is the traditional logistic regression. It is one of sklearn's linear models.
# 
# We use a GridSearchCV process in order to iterate over different hyper-parameters, which can sometimes bring quite significant improvements over our model performance.
# 
# In our case, we will iterate over different regularization strengths.

# We start by defining our model and the hyper-parameters we want to search through.

# In[ ]:


lr = LogisticRegression(random_state = 56, n_jobs = -1, penalty = 'l1')

lr_params = {
    'C': np.linspace(0.001, 1, 20),
}


# We define our GridSearchCV with recall-scoring and five cross-validations

# In[ ]:


lr_grid = GridSearchCV(lr, lr_params, scoring = 'recall', cv = 5, n_jobs = -1, error_score = 0)


# Finally we fit the model, this is where everything happens. The model uses the training set in order to define the best optimization

# In[ ]:


lr_grid.fit(X_train, y_train)


# We print the results of the search

# In[ ]:


lr_best_estimator = lr_grid.best_estimator_

print 'best estimator: \n', lr_grid.best_estimator_

print '\naccuracy_score: \n', lr_grid.score(X_test, y_test)

print '\nbest_params: \n', lr_grid.best_params_


# Let's look at the results table from the grid search to see how different values of C influence our test score

# In[ ]:


lr_results = pd.DataFrame(lr_grid.cv_results_).sort_values(by = 'param_C')


# In[ ]:


lr_results.head(3)


# Plotting the relationship betweeen C strength and mean_test_score

# In[ ]:


lr_results.plot(x ='param_C', y = 'mean_test_score');


# In[ ]:


lr_score = cross_val_score(lr_grid.best_estimator_, X_test, y_test, cv = 10, scoring = 'recall').mean()
lr_score


# In[ ]:


# lr_coef = pd.DataFrame(lr_best_estimator.coef_, columns = X.columns).T.sort_values(by = 0, ascending = False).rename(columns = {0: 'coef'})


# In[ ]:


# sns.barplot(x=lr_coef[:10].coef, y=lr_coef[:10].index)


# ### SGD Classifier
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


sgd = SGDClassifier(random_state = 56, n_jobs = -1, n_iter = 200, penalty = 'elasticnet', l1_ratio = 0.01)

sgd_params = {

    'alpha' : np.logspace(-5,0, 10)
    
}


# In[ ]:


sgd_grid = GridSearchCV(sgd, sgd_params, cv = 5, scoring = 'recall', n_jobs = -1, error_score = 0, random_state=212)


# In[ ]:


sgd_grid.fit(X_train, y_train)


# In[ ]:


sgd_best_estimator = sgd_grid.best_estimator_ 

print sgd_grid.best_estimator_
print
print sgd_grid.score(X_test, y_test)
print 
print sgd_grid.best_params_


# In[ ]:


sgd_results = pd.DataFrame(sgd_grid.cv_results_).sort_values(by = 'param_alpha')


# In[ ]:


sgd_results.head(3)


# In[ ]:


sgd_results.plot(x= 'param_alpha', y = 'mean_test_score', logx=True);


# ***We are clearly over-fitting here!***

# In[ ]:


sgd_score = cross_val_score(sgd_grid.best_estimator_, X_test, y_test, cv = 10, scoring = 'recall', n_jobs = -1).mean()
sgd_score


# ### Linear SVC
# [Table of Contents](#Table-of-Contents:)

# Support Vector Machines can be very useful as they allow classification of non-linearly seperable data.
# 
# Here is an example of linearly and non-linearly seperable data

# ![SVM Example](./Assets/SVM_example.png)

# Given the size of our dataset, we are constrained to the LinearSVC, which is a liblinear (large linear classification) implementation of the traditional SVC
# 

# In[ ]:


svm = LinearSVC(random_state = 56, penalty = 'l1', dual = False)
svm_params = {
    
    'C': np.linspace(0.001, 10, 15),
    
}


# In[ ]:


svm_grid = GridSearchCV(svm, svm_params, cv = 5, scoring = 'recall', n_jobs = -1, error_score = 0)


# In[ ]:


warnings.filterwarnings('ignore')

svm_grid.fit(X_train, y_train)


# In[ ]:


warnings.filterwarnings('default')


# In[ ]:


svm_best_estimator = svm_grid.best_estimator_
get_ipython().run_line_magic('store', 'svm_best_estimator')
print svm_grid.best_estimator_
print
print svm_grid.score(X_test, y_test)

print svm_grid.best_params_


# In[ ]:


svm_results = pd.DataFrame(svm_grid.cv_results_).sort_values(by = 'param_C')


# In[ ]:


svm_results.head(3)


# In[ ]:


svm_results.plot(x = 'param_C', y = 'mean_test_score');


# In[ ]:


svm_score = cross_val_score(svm_grid.best_estimator_, X_test, y_test, scoring = 'recall', cv = 10, n_jobs = -1).mean()
svm_score


# In[ ]:


get_ipython().run_line_magic('store', 'svm_score')
get_ipython().run_line_magic('store', 'svm_results')


# ### Random Forest Classifier
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


rf = RandomForestClassifier(random_state = 56, n_jobs = -1, n_estimators= 300)

rf_params = {
    
    'criterion': ['gini','entropy'],
    'max_features' : ['auto', 'sqrt'],
    
    
}


# In[ ]:


rf_grid = GridSearchCV(rf, rf_params, scoring = 'recall', cv = 5, n_jobs = -1, error_score= 0)


# In[ ]:


rf_grid.fit(X_train, y_train)


# In[ ]:


rf_best_estimator =rf_grid.best_estimator_
print rf_grid.best_estimator_
print
print rf_grid.score(X_test, y_test)
print
print rf_grid.best_params_


# In[ ]:


rf_results = pd.DataFrame(rf_grid.cv_results_).sort_values(by = 'rank_test_score')


# In[ ]:


rf_results.head(3)


# In[ ]:


get_ipython().run_line_magic('store', 'rf_results')
get_ipython().run_line_magic('store', 'rf_best_estimator')


# In[ ]:


rf_score = cross_val_score(rf_grid.best_estimator_, X_test, y_test, cv = 10, scoring = 'recall', n_jobs = -1).mean()
rf_score


# In[ ]:


get_ipython().run_line_magic('store', '-r rf_best_estimator')


# ### Multi-Layer Perceptron
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


mlp = MLPClassifier(

    hidden_layer_sizes= (40,), 
    activation = 'logistic',
    learning_rate = 'adaptive',
    learning_rate_init = 0.2,
    random_state = 56,
    max_iter = 500,    
    
)

mlp_params = {
#     'hidden_layer_sizes' : [10, 20, 30, 40, 50],
    'alpha' : np.logspace(-5,1,10),
    'solver' : ['adam', 'sgd'],
#     'solver' : ['adam','sgd'],
#     'learning_rate_init' : [0.2, 0.0001],
       
    }






# In[ ]:


mlp_grid = GridSearchCV(mlp, mlp_params, scoring = 'recall', cv = 5, n_jobs = -1, error_score= 0)


# In[ ]:


# mlp_grid.fit(X_train, y_train)


# In[ ]:


print mlp_grid.best_estimator_
print mlp_grid.score(X_test, y_test)
print mlp_grid.best_params_


# In[ ]:


mlp_best_estimator = mlp_grid.best_estimator_
mlp_best_estimator


# In[ ]:


get_ipython().run_line_magic('store', 'mlp_best_estimator')
get_ipython().run_line_magic('store', 'mlp_results')


# In[ ]:


mlp_results.sort_values(by = 'param_alpha', inplace = True, ascending = False)


# In[ ]:


mlp_results[mlp_results.param_solver == 'adam'].plot(x = 'param_alpha', y = 'mean_test_score', logx = True);


# In[ ]:


mlp_results[mlp_results.param_solver == 'sgd'].plot(x = 'param_alpha', y = 'mean_test_score', logx = True);


# In[ ]:


mlp_score = cross_val_score(mlp_best_estimator, X_test, y_test, cv = 10, scoring = 'recall', n_jobs = -1).mean()
mlp_score


# ## Performance Summary
# [Table of Contents](#Table-of-Contents:)

# ### Score Table
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


score_table = pd.DataFrame(columns = ['model', 'cv_10_score'])
models = ['Logistic Regression', 'SGD Classifier', 'SVC Classifier', 'Random Forest Classifier', 'Multi-Layer Perceptron']
score_list = [lr_score, sgd_score, svm_score, rf_score,  mlp_score]

for model, n, score in zip(models, np.arange(len(models)), score_list):
    score_table.loc[n,'model'] = model
    score_table.loc[n,'cv_10_score'] = score           


# In[ ]:


score_table


# In[ ]:



plt.figure(figsize = (11,4))
plt.xticks(rotation = 45, ha = 'center')
sns.barplot(score_table.model, score_table.cv_10_score, palette = 'viridis');


# ### Feature Coefficients
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


#Rank by mean coefficient value across models
# coef_table = pd.DataFrame(zip(lasso.coef_[0],lr.coef_[0],X_train.columns), columns = ['Lasso_coef','Ridge_coef','Features'])
# coef_table.head(10)


# ### Confusion Matrices
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


plot_confusion_matrix(confusion_matrix(y_test, lr_grid.best_estimator_.predict(X_test)), title = 'Logistic Regression', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, sgd_grid.best_estimator_.predict(X_test)), title = 'SGD Classifier', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, svm_grid.best_estimator_.predict(X_test)), title = 'SVC Classifier', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, rf_grid.best_estimator_.predict(X_test)), title = 'Random Forest Classifier', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, mlp_grid.best_estimator_.predict(X_test)), title = 'Multi-Layer Perceptron', classes = np.array([0,1]))


# In[ ]:


plt.figure(1)
plot_confusion_matrix(confusion_matrix(y_test, lr_grid.best_estimator_.predict(X_test)), title = 'Logistic Regression', classes = np.array([0,1]))
plt.figure(2)
plot_confusion_matrix(confusion_matrix(y_test, sgd_grid.best_estimator_.predict(X_test)), title = 'SGD Classifier', classes = np.array([0,1]))


# ### ROC Curves
# [Table of Contents](#Table-of-Contents:)

# In[ ]:


plot_roc(lr_grid.best_estimator_, 'Logistic Regression')
# plot_roc(sgd_grid.best_estimator_, 'SGD Classifier')
# plot_roc(svm_grid.best_estimator_, 'SVC Classifier')
plot_roc(rf_grid.best_estimator_, 'Random Forest Classifier')
plot_roc(mlp_best_estimator, 'Multi-Layer Perceptron')


# # VII - Appendix
# [Table of Contents](#Table-of-Contents:)

# ### Plot Confusion Matrix
# [Table of Contents](#Table-of-Contents:)

# In[1]:


def plot_confusion_matrix(cm, classes, title, cmap='viridis'):
    '''
    This function simply gives a nice loooking layout to the confusion matrix.
    '''
    plt.figure(figsize = (3,3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# ### Plot Cummulative Sum Variance
# [Table of Contents](#Table-of-Contents:)

# In[2]:


def plot_cumsum_variance(var_ratio):
    '''
    This function plots cummulative explained variance, ranking features by PCA importance.
    '''
    fig = plt.figure(figsize=(15,5))#init figure 
    ax = fig.gca()
    
    x_vals = range(1,len(var_ratio)+1)#set x&y values
    y_vals = var_ratio
    
    ax.set_title('Explained Variance over Principal Components')#set title and labels 
    ax.set_ylabel('Cumulative Sum of Variance Explained')
    ax.set_xlabel('Number of Principal Components')
    
    ax.plot(x_vals, y_vals)


# ### Plot ROC Curve
# [Table of Contents](#Table-of-Contents:)

# In[3]:


def plot_roc(model, varname):
    y_pp = model.predict_proba(X_test)[:, 1]
    fpr_, tpr_, _ = roc_curve(y_test, y_pp)
    auc_ = auc(fpr_, tpr_)
    acc_ = np.abs(0.5 - np.mean(y)) + 0.5
    
    fig, axr = plt.subplots(figsize=(5,4))

    axr.plot(fpr_, tpr_, label='ROC (area = %0.2f)' % auc_,
             color='darkred', linewidth=4,
             alpha=0.7)
    axr.plot([0, 1], [0, 1], color='grey', ls='dashed',
             alpha=0.9, linewidth=4, label='baseline accuracy = %0.2f' % acc_)

    axr.set_xlim([-0.05, 1.05])
    axr.set_ylim([0.0, 1.05])
    axr.set_xlabel('false positive rate', fontsize=16)
    axr.set_ylabel('true positive rate', fontsize=16)
    axr.set_title(varname+' ROC', fontsize=20)

    axr.legend(loc="lower right", fontsize=12)

    plt.show()


# In[ ]:


# The end!


# In[ ]:


# sns.distplot(data.year_salary,bins=60, color ="dimgrey",kde_kws={"color": "darkred", "lw": 1, "label": "KDE"})

