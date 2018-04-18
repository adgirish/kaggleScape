
# coding: utf-8

# # Implementing Advanced Regression  Techniques for Prediction:
# 
# There are several factors that impact the overall price of the house, some of those factors are more **tangible** as the quality of the house or the overall size (area) of the house and other factors are more **intrinsic** such as the performance of the economy. Coming with an accurate model that predicts with such precision the actual value is an arduous task since there are both internal and external factors that will affect the price of a single house. Nevertheless, what we can do is **detect** those features that carry a heavier weight on the overall output (Price of the house). <br><br>
# 
# Before the housing crisis that occurred during the years of (2007-2008), most people believed that the prices of houses tended to go up throughout the years and that people that invested into properties were certain that they will get a return. This was not the case since banks were basically approving loans to people that were not able to afford to pay a house, there were even financial institutions who were approving loans to ordinary individuals at a variable interest rate (meaning rate will change depending on the current market rate) and when the crisis occurred lots of those ordinary individuals were not able to afford to pay back their mortgages. Of course, there were other reasons that caused the financial crisis in the first place such as the introduction of complex financial instruments (*derivatives are still not widely understood*), hedging financial instruments (credit default swaps), and the deregulation of the financial industry as a whole. While we can argue about the factors that caused the financial crisis, the main objective of this post is to determine what possible features could have a real impact on the overall value of a house. We will try to answer questions such as to what extent did the recession impacted the value house prices? What materials were most commonly used with houses that had a high price range? (Rooftop, walls etc.) Which neighborhoods were the most exclusive? <br><br>
# 
# I believe that in order to perform an extensive analysis of this data we should explore our data, by this I mean getting a sense of what is the **story behind the data**. Most of the time I tend to reject the idea of just building a model that have a good accuracy score for predicting values instead, I analyze my data carefully (determining distributions, missing values, visualizations) in order to have a better understanding of what is going on. ONly after my extensive analysis I proceed to developing the predictive model, in this case we will use **regression models.** The downside of this to many of you who will see this post, is that it will be somewhat long, so if you think you should **skip** all the sections and start from the regression model step, please feel free to do so! I will create an outline so it will help you find the section you wish to start with. <br><br>
# 
# **I'd rather have a full house at a medium price than a half-full at a high price. - George Shinn**
# ***

# ## Goal of this Project:
# ***
# ### Achieving our goal is split into two phases: <br>
# 1) **Exploratory Data Analysis (EVA)**: In this phase our main aim is to have a better understanding of the features involved in our data. It might be possible that some are left behind but I will be focusing on the features that have the highest correlation towards SalePrice. <br><br>
# 
# 2) **Regression and Classification Models**: We will implement a multiclassification model first (**Price Ranges: High, Medium, Low**) and a Regression model to predict a possible SalePrice (label) of the house.

# ## Outline: 
# ***
# I. **Understanding our Data**<br>
# a) [Splitting into Different Categories](#splitting)<br>
# b) [Gathering Basic Insight](#insight) <br><br>
# 
# II. **Economic Activity**<br><br>
# III. [Outside Surroundings](#outside)<br>
# a) [Type of Zoning](#zoning)<br>
# b) [Neighborhoods](#neighborhoods) <br><br>
# 
# IV. **Areas of the House** <br>
# a) [The Impact of Space towards Price](#space)<br><br>
# 
# V. **Building Characteristics**<br>
# a) [Correlations with SalePrice](#correlation)<br>
# b) [What garages tell about House Prices?](#garage)<br><br>
# 
# VI. **Miscellaneous and Utilities**<br>
# a) [What determines the quality of the house?](#quality)<br>
# b) [Intersting insights](#interesting)<br>
# c) [Which Material Combination increased the Price of Houses?](#material)<br><br>
# 
# VII. [Quality of Neighborhoods](#quality_neighborhoods)<br><br>
# 
# VIII. **The Purpose of using Log Transformations** <br>
# a)[Log Transformations](#log_transformations)<br>
# b) [Skewedness and Kurtosis](#skew_kurt)<br>
# c) [Outliers Analysis](#analysis_outliers)<br>
# d) [Bivariate Analysis](#bivariate) <br><br>
# 
# IX. **Feature Engineering** <br>
# a) [Dealing with Missing Values](#missing_values)<br>
# b) [Transforming Values](#transforming_values)<br>
# c) [Combining Attributes](#combining_atributes) <br>
# d) [Dealing with numerical and categorical values](#num_cat_val) <br><br>
# 
# X. **Scaling** <br>
# a) [Categorical Encoding Class](#categorical_class)<br>
# b) [Combine Attribute Class](#combining)<br>
# c) [Pipelines](#combining)<br><br>
# 
# XI. **Predictive Models** <br>
# a) [Residual Plot](#residual_plot) <br>
# b) [RandomForests Regressor](#random_forest) <br>
# c) [GradientBoosting Regressor](#gradient_boosting)<br>
# d) [Stacking Regressor](#stacking_regressor)

# ### References: 
# 1) <a href="https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard">Stacked Regressions : Top 4% on LeaderBoard</a> by Serigne.
# - Good if you are looking for stacking models and to gather an in-depth analysis for feature engineering. <br><br>
# 
# 2) <a href="https://www.kaggle.com/vhrique/simple-house-price-prediction-stacking"> Simple House Price Prediction Stacking </a> by Victor Henrique Alves Ribeiro.  
# - Gave me an idea of which algorithms to implement in my ensemble methods. <br>
# - Also Victor is really open to answer any doubts with regards to this project. <br><br>
# 
# 3) <a href="https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python"> Comprehensive data exploration with Python </a> by Pedro Marcelino.
# - Help me understand more in depth the different linear regularization methods and its parameters. <br><br>
# 
# 4) <b> Hands on Machine Learning with Scikit-Learn & TensorFlow by Aurélien Géron (O'Reilly). CopyRight 2017 Aurélien Géron   </b><br>
# - Good reference for understanding how Pipelines work. <br>
# - Good for understanding ensemble methods such as RandomForests and GradientBoosting. <br>
# - This book is a must have for people starting in the area of machine learning.<br><br>
# 
# 
# 5) <a href="https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/"> A comprehensive beginners guide for Linear, Ridge and Lasso Regression </a> by Shubham Jain at Analytics Vidhya.
# - Helped me with the residual plot. <br>
# - Better understanding of Ridge, Lasso and ElasticNet (Good for Beginners).

# In[83]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from scipy import stats

# Common imports
import numpy as np

# Plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Maintain the Ids for submission
train_id = train['Id']
test_id = test['Id']


# In[84]:


train['SalePrice'].describe()


# In[85]:


# It seems we have nulls so we will use the imputer strategy later on.
Missing = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['train', 'test'])
Missing[Missing.sum(axis=1) > 0]


# In[86]:


# We have several columns that contains null values we should replace them with the median or mean those null values.
train.info()


# In[87]:


train.describe()


# In[88]:


corr = train.corr()
plt.figure(figsize=(14,8))
plt.title('Overall Correlation of House Prices', fontsize=18)
sns.heatmap(corr,annot=False,cmap='BrBG',linewidths=0.2,annot_kws={'size':20})
plt.show()


# # Splitting the Variables into Different Categories:
# <a id="splitting"></a>
# ## Data Analysis:
# For data analysis purposes I am going to separate the different features into different categories in order to segment our analysis. These are the steps we are going to take in our analysis: Nevertheless, I will split the categories so you can analyse thoroughly the different categories.<br>
# 1) Separate into different categories in order to make our analysis easier. <br>
# 2) All of our categories will contain sales price in order to see if there is a significant pattern.<br>
# 3) After that we will create our linear regression model in order to make accurate predictions as to what will the price of the houses will be.<br><br>
# 4) For all the categories we have id, salesprice, MoSold, YearSold, SaleType and SaleCondition.
# 
# **Note:** At least for me, it is extremely important to make a data analysis of our data, in order to have a grasp of what the data is telling us, what might move salesprice higher or lower. Instead of just running a model and just predict prices, we must make a thorough analysis of our data. Also, using these different categories is completely optional in case you want to make a more in-depth analysis of the different features.

# In[89]:


# Create the categories
outsidesurr_df = train[['Id', 'MSZoning', 'LotFrontage', 'LotArea', 'Neighborhood', 'Condition1', 'Condition2', 'PavedDrive', 
                    'Street', 'Alley', 'LandContour', 'LandSlope', 'LotConfig', 'MoSold', 'YrSold', 'SaleType', 'LotShape', 
                     'SaleCondition', 'SalePrice']]

building_df = train[['Id', 'MSSubClass', 'BldgType', 'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 
                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'Functional', 
                    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'MoSold', 'YrSold', 'SaleType',
                    'SaleCondition', 'SalePrice']]

utilities_df = train[['Id', 'Utilities', 'Heating', 'CentralAir', 'Electrical', 'Fireplaces', 'PoolArea', 'MiscVal', 'MoSold',
                     'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']]

ratings_df = train[['Id', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                   'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature',
                   'GarageCond', 'GarageQual', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']]

rooms_df = train[['Id', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','TotRmsAbvGrd', 
                 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'YrSold', 'SaleType',
                 'SaleCondition', 'SalePrice']]




# Set Id as index of the dataframe.
outsidesurr_df = outsidesurr_df.set_index('Id')
building_df = building_df.set_index('Id')
utilities_df = utilities_df.set_index('Id')
ratings_df = ratings_df.set_index('Id')
rooms_df = rooms_df.set_index('Id')

# Move SalePrice to the first column (Our Label)
sp0 = outsidesurr_df['SalePrice']
outsidesurr_df.drop(labels=['SalePrice'], axis=1, inplace=True)
outsidesurr_df.insert(0, 'SalePrice', sp0)

sp1 = building_df['SalePrice']
building_df.drop(labels=['SalePrice'], axis=1, inplace=True)
building_df.insert(0, 'SalePrice', sp1)

sp2 = utilities_df['SalePrice']
utilities_df.drop(labels=['SalePrice'], axis=1, inplace=True)
utilities_df.insert(0, 'SalePrice', sp2)

sp3 = ratings_df['SalePrice']
ratings_df.drop(labels=['SalePrice'], axis=1, inplace=True)
ratings_df.insert(0, 'SalePrice', sp3)

sp4 = rooms_df['SalePrice']
rooms_df.drop(labels=['SalePrice'], axis=1, inplace=True)
rooms_df.insert(0, 'SalePrice', sp4)


# # Gathering a Basic Insight of our Data:
# <a id="insight"></a>
# <br><br>
# <img src="http://blog.algoscale.com/wp-content/uploads/2017/06/algoscale_data_analytics4.jpg">
# <br><br>
# 
# ## Summary:
# <ul>
# <li> The distribution of <b> house prices </b> is right skewed.</li>
# <li> There is a <b>drop</b> in the number of houses sold during the year of 2010. </li>
# </ul>

# In[90]:


import seaborn as sns
sns.set_style('white')

f, axes = plt.subplots(ncols=4, figsize=(16,4))

# Lot Area: In Square Feet
sns.distplot(train['LotArea'], kde=False, color="#DF3A01", ax=axes[0]).set_title("Distribution of LotArea")
axes[0].set_ylabel("Square Ft")
axes[0].set_xlabel("Amount of Houses")

# MoSold: Year of the Month sold
sns.distplot(train['MoSold'], kde=False, color="#045FB4", ax=axes[1]).set_title("Monthly Sales Distribution")
axes[1].set_ylabel("Amount of Houses Sold")
axes[1].set_xlabel("Month of the Year")

# House Value
sns.distplot(train['SalePrice'], kde=False, color="#088A4B", ax=axes[2]).set_title("Monthly Sales Distribution")
axes[2].set_ylabel("Number of Houses ")
axes[2].set_xlabel("Price of the House")

# YrSold: Year the house was sold.
sns.distplot(train['YrSold'], kde=False, color="#FE2E64", ax=axes[3]).set_title("Year Sold")
axes[3].set_ylabel("Number of Houses ")
axes[3].set_xlabel("Year Sold")

plt.show()


# ## Right-Skewed Distribution Summary:
# In a right skew or positive skew the mean is most of the times to the right of the median. There is a higher frequency of occurence to the left of the distribution plot leading to more exceptions (outliers to the right). Nevertheless, there is a way to transform this histogram into a normal distributions by using <b>log transformations</b> which will be discussed further below.

# In[91]:


# Maybe we can try this with plotly.
plt.figure(figsize=(12,8))
sns.distplot(train['SalePrice'], color='r')
plt.title('Distribution of Sales Price', fontsize=18)

plt.show()


# <h1 align="center"> Economic Activity: </h1>
# <a id="economy"></a>
# <img src="http://vietsea.net/upload/news/2016/12/1/11220161528342876747224.jpg">
# We will visualize how the housing market in **Ames, IOWA** performed during the years 2006 - 2010 and how bad it was hit by the economic recession during the years of 2007-2008.  
# 
# ## Level of Supply and Demand (Summary):
# <ul>
# <li><b>June</b> and <b>July</b> were the montnths in which most houses were sold. </li>
# <li> The <b> median house price </b> was at its peak in 2007 (167k) and it was at its lowest point during the year of 2010 (155k) a difference of 12k. This might be a consequence of the economic recession. </li>
# <li> Less houses were <b>sold</b> and <b>built</b> during the year of 2010 compared to the other years. </li>
# </ul>
# 
# 

# In[92]:


# People tend to move during the summer
sns.set(style="whitegrid")
plt.figure(figsize=(12,8))
sns.countplot(y="MoSold", hue="YrSold", data=train)
plt.show()


# In[93]:


plt.figure(figsize=(12,8))
sns.boxplot(x='YrSold', y='SalePrice', data=train)
plt.xlabel('Year Sold', fontsize=14)
plt.ylabel('Price sold', fontsize=14)
plt.title('Houses Sold per Year', fontsize=16)


# In[94]:


plt.figure(figsize=(14,8))
plt.style.use('seaborn-white')
sns.stripplot(x='YrSold', y='YearBuilt', data=train, jitter=True, palette="Set2", linewidth=1)
plt.title('Economic Activity Analysis', fontsize=18)
plt.xlabel('Year the house was sold', fontsize=14)
plt.ylabel('Year the house was built', rotation=90, fontsize=14)
plt.show()


# <h1 align="center"> Outside Surroundings of the House: </h1>
# <a id="outside"></a>
# <img src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Lot_map.PNG">
# ## Features from Outside: 
# In this section we will create an in-depth analysis of how the outside surroundings affect the price. Which variables have the highest weight on price. You can use the **train** dataframe or the **outsidesurr_df** to simplify the amount of features and have a closer look as to how they behave towards **"SalePrice"**. For the correlation matrix I will be using outsidesurr_df so you can have a better look as to which variables from the **outside surroundings category** impact the most on the price of a house. <br><br>
# 
# ## Summary:
# <ul>
# <li> The <b>mean price</b> of the house of is 180,921, this will explain why the data is right skewed. </li>
# <li> <b>Standard deviation</b> is pretty high at 79442.50 meaning the data deviates a lot from the mean (many outliers) </li>
# <li> <b>LotArea</b> and <b>LotFrontage</b> had the highest correlation with the price of a house from the <b> outside surroundings category </b>. </li>
# <li> Most of the houses that were sold were from a <b> Residential Low Density Zone </b>.</li>
# <li> The most exclusive Neighborhoods are <b>Crawfor</b>, <b>Sawyer</b> and <b>SawyerW</b></li>
# </ul>

# In[95]:


outsidesurr_df.describe()


# In[96]:


outsidesurr_df.columns


# In[97]:


# Lot Area and Lot Frontage influenced hugely on the price. 
# However, YrSold does not have that much of a negative correlation with SalePrice as we previously thought.
# Meaning the state of IOWA was not affected as other states.
plt.style.use('seaborn-white')
corr = outsidesurr_df.corr()

sns.heatmap(corr,annot=True,cmap='YlOrRd',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(14,10)
plt.title("Outside Surroundings Correlation", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# ## Type of Zoning:
# <a id="zoning"></a>

# In[98]:


# We already know which neighborhoods were the most sold but which neighborhoods gave the most revenue. 
# This might indicate higher demand toward certain neighborhoods.
plt.style.use('seaborn-white')
zoning_value = train.groupby(by=['MSZoning'], as_index=False)['SalePrice'].sum()
zoning = zoning_value['MSZoning'].values.tolist()


# Let's create a pie chart.
labels = ['C: Commercial', 'FV: Floating Village Res.', 'RH: Res. High Density', 'RL: Res. Low Density', 
          'RM: Res. Medium Density']
total_sales = zoning_value['SalePrice'].values.tolist()
explode = (0, 0, 0, 0.1, 0)

fig, ax1 = plt.subplots(figsize=(12,8))
texts = ax1.pie(total_sales, explode=explode, autopct='%.1f%%', shadow=True, startangle=90, pctdistance=0.8,
       radius=0.5)


ax1.axis('equal')
plt.title('Sales Groupby Zones', fontsize=16)
plt.tight_layout()
plt.legend(labels, loc='best')
plt.show()


# In[99]:


plt.style.use('seaborn-white')
SalesbyZone = train.groupby(['YrSold','MSZoning']).SalePrice.count()
SalesbyZone.unstack().plot(kind='bar',stacked=True, colormap= 'gnuplot',  
                           grid=False,  figsize=(12,8))
plt.title('Building Sales (2006 - 2010) by Zoning', fontsize=18)
plt.ylabel('Sale Price', fontsize=14)
plt.xlabel('Sales per Year', fontsize=14)
plt.show()


# ## Neighborhoods: 
# <a id="neighborhoods">
# 

# In[100]:


fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(x="Neighborhood", data=train, palette="Set2")
ax.set_title("Types of Neighborhoods", fontsize=20)
ax.set_xlabel("Neighborhoods", fontsize=16)
ax.set_ylabel("Number of Houses Sold", fontsize=16)
ax.set_xticklabels(labels=train['Neighborhood'] ,rotation=90)
plt.show()


# In[101]:


# Sawyer and SawyerW tend to be the most expensive neighberhoods. Nevertheless, what makes them the most expensive
# Is it the LotArea or LotFrontage? Let's find out!
fig, ax = plt.subplots(figsize=(12,8))
ax = sns.boxplot(x="Neighborhood", y="SalePrice", data=train)
ax.set_title("Range Value of the Neighborhoods", fontsize=18)
ax.set_ylabel('Price Sold', fontsize=16)
ax.set_xlabel('Neighborhood', fontsize=16)
ax.set_xticklabels(labels=train['Neighborhood'] , rotation=90)
plt.show()


# <h1 align="center">The Impact of Space towards Price:</h1>
# <a id="space"></a>
# <img src="http://www.archiii.com/wp-content/uploads/2013/06/Office-Orchard-House-Interior-Design-by-Arch11-Architecture-Interior.jpg" width=700 height=300>
# <br><br>
# 
# ## The Influence of Space:
# How much influence does space have towards the price of the house. Intuitively, we might think the bigger the house the higher the price but let's take a look in order to see ifit actually has a positive correlation towards **SalePrice**.
# 
# ## Summary:
# <ul>
# <li><b>GrlivingArea:</b> The living area square feet is positively correlated with the price of the house.</li>
# <li> <b> GarageArea:</b> Apparently, the space of the garage is an important factor that contributes to the price of the house. </li>
# <li>  <b>TotalBsmft:</b> The square feet of the basement contributes positively to the value of the house. </li>
# <li> <b>LotArea and LotFrontage:</b> I would say from all the area features these are the two that influencess the less on the price of the house.  </li>
# </ul>

# In[102]:


sns.jointplot(x='GrLivArea',y='SalePrice',data=train,
              kind='hex', cmap= 'CMRmap', size=8, color='#F84403')

plt.show()


# In[103]:


sns.jointplot(x='GarageArea',y='SalePrice',data=train,
              kind='hex', cmap= 'CMRmap', size=8, color='#F84403')

plt.show()


# In[104]:


sns.jointplot(x='TotalBsmtSF',y='SalePrice',data=train,
              kind='hex', cmap= 'CMRmap', size=8, color='#F84403')

plt.show()


# In[105]:


plt.figure(figsize=(16,6))
plt.subplot(121)
ax = sns.regplot(x="LotFrontage", y="SalePrice", data=train)
ax.set_title("Lot Frontage vs Sale Price", fontsize=16)

plt.subplot(122)
ax1 = sns.regplot(x="LotArea", y="SalePrice", data=train, color='#FE642E')
ax1.set_title("Lot Area vs Sale Price", fontsize=16)

plt.show()


# <h1 align="center"> Building Characteristics: </h1>
# <a id="building_characteristics"></a>
# 

# In[106]:


building_df.head()


# # High Correlated Variables with SalePrice:
# <a id="correlation"></a>
# 1) YearBuilt - The Date the building was built. <br>
# 2) YearRemodAdd - The last time there wasa building remodeling. <br>
# 3) MasVnArea - Masonry veneer area in square feet. <br>
# 4) GarageYrBlt - Year garage was built. <br>
# 5) GarageCars - Size of garage in car capacity. <br>
# 6) GarageArea - Size of garage in square feet. <br>

# In[107]:


corr = building_df.corr()

g = sns.heatmap(corr,annot=True,cmap='coolwarm',linewidths=0.2,annot_kws={'size':20})
g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 8)
fig=plt.gcf()
fig.set_size_inches(14,10)
plt.title("Building Characteristics Correlation", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[108]:


# To understand better our data I will create a category column for SalePrice.
train['Price_Range'] = np.nan
lst = [train]

# Create a categorical variable for SalePrice
# I am doing this for further visualizations.
for column in lst:
    column.loc[column['SalePrice'] < 150000, 'Price_Range'] = 'Low'
    column.loc[(column['SalePrice'] >= 150000) & (column['SalePrice'] <= 300000), 'Price_Range'] = 'Medium'
    column.loc[column['SalePrice'] > 300000, 'Price_Range'] = 'High'
    
train.head()


# ## What Garages tells us about each Price Category:
# <a id="garage"></a>
# <img src="https://www.incimages.com/uploaded_files/image/970x450/garage-office-970_24019.jpg">

# In[109]:


import matplotlib.pyplot as plt
palette = ["#9b59b6", "#BDBDBD", "#FF8000"]
sns.lmplot('GarageYrBlt', 'GarageArea', data=train, hue='Price_Range', fit_reg=False, size=7, palette=palette,
          markers=["o", "s", "^"])
plt.title('Garage by Price Range', fontsize=18)
plt.annotate('High Price \nCategory Garages \n are not that old', xy=(1997, 1100), xytext=(1950, 1200), 
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()


# # Miscellaneous and Utilities:
# <a id="utilities"></a>

# In[110]:


plt.style.use('seaborn-white')
types_foundations = train.groupby(['Price_Range', 'PavedDrive']).size()
types_foundations.unstack().plot(kind='bar', stacked=True, colormap='Set1', figsize=(13,11), grid=False)
plt.ylabel('Number of Streets', fontsize=16)
plt.xlabel('Price Category', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.title('Condition of the Street by Price Category', fontsize=18)
plt.show()


# In[111]:


# We can see that CentralAir impacts until some extent the price of the house.

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
plt.suptitle('Relationship between Saleprice \n and Categorical Utilities', fontsize=18)
sns.pointplot(x='CentralAir', y='SalePrice', hue='Price_Range', data=train, ax=ax1)
sns.pointplot(x='Heating', y='SalePrice', hue='Price_Range', data=train, ax=ax2)
sns.pointplot(x='Fireplaces', y='SalePrice', hue='Price_Range', data=train, ax=ax3)
sns.pointplot(x='Electrical', y='SalePrice', hue='Price_Range', data=train, ax=ax4)

plt.legend(loc='best')
plt.show()


# In[112]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

fig, ax = plt.subplots(figsize=(14,8))
palette = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#FF8000", "#AEB404", "#FE2EF7", "#64FE2E"]

sns.swarmplot(x="OverallQual", y="SalePrice", data=train, ax=ax, palette=palette, linewidth=1)
plt.title('Correlation between OverallQual and SalePrice', fontsize=18)
plt.ylabel('Sale Price', fontsize=14)
plt.show()


# <h1 align="center"> What determines the quality of the House? </h1>
# <a id="quality"></a>
# 
# Remember quality is the most important factor that contributes to the SalePrice of the house. <br>
# **Correlations with OverallQual:**<br>
# 1) YearBuilt <br>
# 2) TotalBsmtSF <br>
# 3) GrLivArea <br>
# 4) FullBath <br>
# 5) GarageYrBuilt <br>
# 6) GarageCars <br>
# 7) GarageArea <br><br>

# <img src="http://tibmadesignbuild.com/images/female-hands-framing-custom-kitchen-design.jpg">
# 
# ## Interesting insights:
# <a id="interesting"></a>
# 1) **Overall Condition**: of the house or building, meaning that further remodelations are likely to happen in the future, either for reselling or to accumulate value in their real-estate.. <br>
# 2) **Overall Quality**: The quality of the house is one of the factors that mostly impacts SalePrice. It seems that the overall material that is used for construction and the finish of the house has a great impact on SalePrice. <br>
# 3) **Year Remodelation**: Houses in the **high** price range remodelled their houses sooner. The sooner the remodelation the higher the value of the house. <br>
# 

# In[113]:


with sns.plotting_context("notebook",font_scale=2.8):
    g = sns.pairplot(train, vars=["OverallCond", "OverallQual", "YearRemodAdd", "SalePrice"],
                hue="Price_Range", palette="Dark2", size=6)


g.set(xticklabels=[]);

plt.show()


# ## Which Material Combination increased the Price of Houses?
# <a id="material"></a>
# <ul>
# <li> <b>Roof Material</b>: <b>Hip</b> and <b>Gable</b> was the most expensive since people who bought <b>high value</b> houses tended to buy this material bor he rooftop.</li>
# <li> <b>House Material</b>: Houses made up of <b>stone</b> tend to influence positively the price of the house. (Except in 2007 for <b>High Price House Values. </b>)  </li>
# </ul>
# 

# In[114]:


# What type of material is considered to have a positive effect on the quality of the house?
# Let's start with the roof material

with sns.plotting_context("notebook",font_scale=1):
    g = sns.factorplot(x="SalePrice", y="RoofStyle", hue="Price_Range",
                   col="YrSold", data=train, kind="box", size=5, aspect=.75, sharex=False, col_wrap=3, orient="h",
                      palette='Set1');
    for ax in g.axes.flatten(): 
        for tick in ax.get_xticklabels(): 
            tick.set(rotation=20)

plt.show()


# **Note:** Interestingly, the Masonry Veneer type of stone became popular after 2007 for the houses that belong to the **high** Price Range category. I wonder why? <br>
# **For some reason during the year of 2007, the Saleprice of houses within the high range made of stone dropped drastically! 
# 
# 

# In[115]:


with sns.plotting_context("notebook",font_scale=1):
    g = sns.factorplot(x="MasVnrType", y="SalePrice", hue="Price_Range",
                   col="YrSold", data=train, kind="bar", size=5, aspect=.75, sharex=False, col_wrap=3,
                      palette="YlOrRd");
    
plt.show()


# <h1 align="center"> Quality of Neighborhoods </h1>
# <a id="quality_neighborhoods"></a>
# <img src="http://www.unitedwaydenver.org/sites/default/files/UN_neighborhood.jpg">
# 
# ## Which Neighborhoods had the best Quality houses?
# <a id="which_neighborhoods"></a>

# In[116]:


plt.style.use('seaborn-white')
types_foundations = train.groupby(['Neighborhood', 'OverallQual']).size()
types_foundations.unstack().plot(kind='bar', stacked=True, colormap='RdYlBu', figsize=(13,11), grid=False)
plt.ylabel('Overall Price of the House', fontsize=16)
plt.xlabel('Neighborhood', fontsize=16)
plt.xticks(rotation=90, fontsize=12)
plt.title('Overall Quality of the Neighborhoods', fontsize=18)
plt.show()


# In[117]:


# Which houses neighborhoods remodeled the most.
# price_categories = ['Low', 'Medium', 'High']
# remod = train['YearRemodAdd'].groupby(train['Price_Range']).mean()

fig, ax = plt.subplots(ncols=2, figsize=(16,4))
plt.subplot(121)
sns.pointplot(x="Price_Range",  y="YearRemodAdd", data=train, order=["Low", "Medium", "High"], color="#0099ff")
plt.title("Average Remodeling by Price Category", fontsize=16)
plt.xlabel('Price Category', fontsize=14)
plt.ylabel('Average Remodeling Year', fontsize=14)
plt.xticks(rotation=90, fontsize=12)

plt.subplot(122)
sns.pointplot(x="Neighborhood",  y="YearRemodAdd", data=train, color="#ff9933")
plt.title("Average Remodeling by Neighborhood", fontsize=16)
plt.xlabel('Neighborhood', fontsize=14)
plt.ylabel('')
plt.xticks(rotation=90, fontsize=12)
plt.show()


# ## The Purpose of Log Transformations:
# <a id="log_transformations"></a>
# The main reason why we use log transformation is to reduce **skewness** in our data. However, there are other reasons why we log transform  our data: <br>
# <ul>
# <li> Easier to interpret patterns of our data. </li>
# <li> For possible statistical analysis that require the data to be normalized.</li>
# </ul>

# In[118]:


numeric_features = train.dtypes[train.dtypes != "object"].index

# Top 5 most skewed features
skewed_features = train[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_features})
skewness.head(5)


# In[119]:


from scipy.stats import norm

# norm = a normal continous variable.

log_style = np.log(train['SalePrice'])  # log of salesprice

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14,10))
plt.suptitle('Probability Plots', fontsize=18)
ax1 = sns.distplot(train['SalePrice'], color="#FA5858", ax=ax1, fit=norm)
ax1.set_title("Distribution of Sales Price with Positive Skewness", fontsize=14)
ax2 = sns.distplot(log_style, color="#58FA82",ax=ax2, fit=norm)
ax2.set_title("Normal Distibution with Log Transformations", fontsize=14)
ax3 = stats.probplot(train['SalePrice'], plot=ax3)
ax4 = stats.probplot(log_style, plot=ax4)

plt.show()


# ## Skewedness and Kurtosis:
# <a id="skew_kurt"></a>
# **Skewedness**: <br>
# <ul>
# <li> A skewness of <b>zero</b> or near zero indicates a <b>symmetric distribution</b>.</li>
# <li> A <b>negative value</b> for the skewness indicate a <b>left skewness</b> (tail to the left) </li>
# <li> A <b>positive value</b> for te skewness indicate a <b> right skewness </b> (tail to the right) </li>
# <ul>

# **Kurtosis**:
# <ul>
# <li><b>Kourtosis</b> is a measure of how extreme observations are in a dataset.</li>
# <li> The <b> greater the kurtosis coefficient </b>, the more peaked the distribution around the mean is. </li>
# <li><b>Greater coefficient</b> also means fatter tails, which means there is an increase in tail risk (extreme results) </li>
# </ul>
# 
# **Reference**:
# Investopedia: https://www.investopedia.com/terms/m/mesokurtic.asp
# 

# In[120]:


print('Skewness for Normal D.: %f'% train['SalePrice'].skew())
print('Skewness for Log D.: %f'% log_style.skew())
print('Kurtosis for Normal D.: %f' % train['SalePrice'].kurt())
print('Kurtosis for Log D.: %f' % log_style.kurt())


# # Outliers Analysis:
# <a id="analysis_outliers"></a>
# **Analysis**:
# <ul>
# <li> The year of <b>2007</b> had the highest outliers (peak of the housing market before collapse). </li>
# <li>  The highest outliers are located in the <b> High category </b> of the Price_Range column.</li>
# </ul>

# In[121]:


# Most outliers are in the high price category nevertheless, in the year of 2007 saleprice of two houses look extremely high!

fig = plt.figure(figsize=(12,8))
ax = sns.boxplot(x="YrSold", y="SalePrice", hue='Price_Range', data=train)
plt.title('Detecting outliers', fontsize=16)
plt.xlabel('Year the House was Sold', fontsize=14)
plt.ylabel('Price of the house', fontsize=14)
plt.show()


# In[122]:


corr = train.corr()
corr['SalePrice'].sort_values(ascending=False)[:11]


# ## Bivariate Analysis (Detecting outliers through visualizations):
# <a id="bivariate"></a>
# **There are some outliers in some of this columns but there might be a reason behind this, it is possible that these outliers in which the area is high but the price of the house is not that high, might be due to the reason that these houses are located in agricultural zones.**

# In[123]:


fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2, figsize=(14,8))
var1 = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var1]], axis=1)
sns.regplot(x=var1, y='SalePrice', data=data, fit_reg=True, ax=ax1)


var2 = 'GarageArea'
data = pd.concat([train['SalePrice'], train[var2]], axis=1)
sns.regplot(x=var2, y='SalePrice', data=data, fit_reg=True, ax=ax2, marker='s')

var3 = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var3]], axis=1)
sns.regplot(x=var3, y='SalePrice', data=data, fit_reg=True, ax=ax3, marker='^')

var4 = '1stFlrSF'
data = pd.concat([train['SalePrice'], train[var4]], axis=1)
sns.regplot(x=var4, y='SalePrice', data=data, fit_reg=True, ax=ax4, marker='+')

plt.show()


# <h1 align="center"> Feature Engineering </h1>
# <a id="feature_engineering"></a>
# ## Dealing with Missing Values:
# <a id="missing_values"></a>

# In[124]:


y_train = train['SalePrice'].values
# We will concatenate but we will split further on.
rtrain = train.shape[0]
ntest = test.shape[0]
train.drop(['SalePrice', 'Price_Range', 'Id'], axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# In[125]:


complete_data = pd.concat([train, test])
complete_data.shape


# In[126]:


total_nas = complete_data.isnull().sum().sort_values(ascending=False)
percent_missing = (complete_data.isnull().sum()/complete_data.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total_nas, percent_missing], axis=1, keys=['Total_M', 'Percentage'])


# missing.head(9) # We have 19 columns with NAs


# ## Transforming Missing Values:
# <a id="transforming_values"></a>
# 

# In[127]:


complete_data["PoolQC"] = complete_data["PoolQC"].fillna("None")
complete_data["MiscFeature"] = complete_data["MiscFeature"].fillna("None")
complete_data["Alley"] = complete_data["Alley"].fillna("None")
complete_data["Fence"] = complete_data["Fence"].fillna("None")
complete_data["FireplaceQu"] = complete_data["FireplaceQu"].fillna("None")
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    complete_data[col] = complete_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    complete_data[col] = complete_data[col].fillna('None')
complete_data['MSZoning'] = complete_data['MSZoning'].fillna(complete_data['MSZoning'].mode()[0])
complete_data["MasVnrType"] = complete_data["MasVnrType"].fillna("None")
complete_data["Functional"] = complete_data["Functional"].fillna("Typ")
complete_data['Electrical'] = complete_data['Electrical'].fillna(complete_data['Electrical'].mode()[0])
complete_data['KitchenQual'] = complete_data['KitchenQual'].fillna(complete_data['KitchenQual'].mode()[0])
complete_data['Exterior1st'] = complete_data['Exterior1st'].fillna(complete_data['Exterior1st'].mode()[0])
complete_data['Exterior2nd'] = complete_data['Exterior2nd'].fillna(complete_data['Exterior2nd'].mode()[0])
complete_data['SaleType'] = complete_data['SaleType'].fillna(complete_data['SaleType'].mode()[0])
complete_data['MSSubClass'] = complete_data['MSSubClass'].fillna("None")


# In[128]:


# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
complete_data["LotFrontage"] = complete_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    complete_data[col] = complete_data[col].fillna(0)
    
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    complete_data[col] = complete_data[col].fillna(0)
    
complete_data["MasVnrArea"] = complete_data["MasVnrArea"].fillna(0)


# In[129]:


# Drop
complete_data = complete_data.drop(['Utilities'], axis=1)


# ## Combining Attributes
# <a id="combining_atributes"></a>

# In[130]:


# Adding total sqfootage feature 
complete_data['TotalSF'] = complete_data['TotalBsmtSF'] + complete_data['1stFlrSF'] + complete_data['2ndFlrSF']


# ## Dealing with Numerical and Categorical Values:
# <a id="num_cat_val"></a>

# ## Transforming our Data:
# <ul>
# <li> Separate the <b> features </b> and <b> labels </b> from the training dataset. </li>
# <li> Separate <b> numeric </b> and <b> categorical </b> variables for the purpose of running them in separate pipelines and scaling them with their respective scalers. </li>
# 
# </ul>

# In[131]:


complete_data.head()


# In[132]:


# splitting categorical variables with numerical variables for encoding.
categorical = complete_data.select_dtypes(['object'])
numerical = complete_data.select_dtypes(exclude=['object'])

print(categorical.shape)
print(numerical.shape)


# ## Categorical Encoding Class:
# <a id="categorical_class"></a>
# This is a way to encode our features in a way that it avoids the assumption that two nearby values are more similar than two distant values. This is the reason we should avoid using LabelEncoder to scale features (inputs) in our dataset and in addition the word **LabelEncoder** is used for scaling labels (outputs). This could be used more often in **binary classification problems** were no *association* exists between the outputs.

# In[133]:



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# ## Combine Attribute Class:
# <a id="combining"></a>
# This class will help us to include the total area variable into our pipeline for further scaling.

# In[134]:


from sklearn.base import BaseEstimator, TransformerMixin

# class combination attribute.
# First we need to know the index possition of the other cloumns that make the attribute.
numerical.columns.get_loc("TotalBsmtSF") # Index Number 37
numerical.columns.get_loc("1stFlrSF") # Index NUmber 42
numerical.columns.get_loc("2ndFlrSF") # Index NUmber 43

ix_total, ix_first, ix_second = 9, 10, 11
# complete_data['TotalSF'] = complete_data['TotalBsmtSF'] + complete_data['1stFlrSF'] + complete_data['2ndFlrSF']

class CombineAttributes(BaseEstimator, TransformerMixin):
    
    def __init__(self, total_area=True): # No args or kargs
        self.total_area = total_area
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        total_sf = X[:,ix_total] + X[:,ix_first] + X[:,ix_second]
        if self.total_area:
            return np.c_[X, total_sf]
        else: 
            return np.c_[X]

attr_adder = CombineAttributes(total_area=True)
extra_attribs = attr_adder.transform(complete_data.values)


# In[135]:


# Scikit-Learn does not handle dataframes in pipeline so we will create our own class.
# Reference: Hands-On Machine Learning
from sklearn.base import BaseEstimator, TransformerMixin
# Create a class to select numerical or cateogrical columns.
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit (self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# ## Pipelines:
# <a id="pipelines"></a> 
# 
# Create our numerical and cateogircal pipelines to scale our features.

# In[136]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lst_numerical = list(numerical)

numeric_pipeline = Pipeline([
    ('selector', DataFrameSelector(lst_numerical)),
    ('extra attributes', CombineAttributes()),
    ('std_scaler', StandardScaler()),
])

categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 
                                    'Neighborhood', 'Condition1', 'Condition2','BldgType', 'HouseStyle', 'RoofStyle',
                                    'RoofMatl', 'Exterior1st',  'Exterior2nd','ExterQual','ExterCond', 'Foundation',
                                    'Heating','HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                                    'PavedDrive', 'SaleType', 'SaleCondition'])),
    ('encoder', CategoricalEncoder(encoding="onehot-dense")),
])


# In[137]:


# Combine our pipelines!
from sklearn.pipeline import FeatureUnion

main_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', numeric_pipeline),
    ('cat_pipeline', categorical_pipeline)
])

data_prepared = main_pipeline.fit_transform(complete_data)
data_prepared


# In[138]:


features = data_prepared
labels = np.log1p(y_train) # Scaling the Saleprice column.

train_scaled = features[:rtrain] 
test_scaled = features[rtrain:]


# <h1 align="center"> Implementing Predictive Models </h1>
# 
# <img src="http://precisionanalytica.com/blog/wp-content/uploads/2014/09/Predictive-Modeling.jpg">
# 
# ## Residual Plot:
# <a id="residual_plot"></a>
# <ul>
# <li><b>Residual plots</b> will give us more or less the actual prediction errors our models are making. In this example, I will use <b>yellowbrick library</b> (statistical visualizations for machine learning) and a simple linear regression model.  In our <b>legends</b> of the residual plot it says training and test data but in this scenario instead of the test set it is the <b>validation set</b> we are using. [If there is a possibility to change the name of the legend to validation I will make the update whenever possible.</li>
# <li> Create a validation set within the training set to actually predict values. (Remember the test set does not have the training price, and also when testing data it should be done during the last instance of the project.) </li>
# 
# </ul>

# In[139]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.linear_model import Ridge
from yellowbrick.regressor import PredictionError, ResidualsPlot


# In[140]:


# This is data that comes from the training test.
X_train, X_val, y_train, y_val = train_test_split(train_scaled, labels, test_size=0.25, random_state=42)


# In[141]:


# Our validation set tends to perform better. Less Residuals.
ridge = Ridge()
visualizer = ResidualsPlot(ridge, train_color='#045FB4', test_color='r', line_color='#424242')
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_val, y_val)
g = visualizer.poof(outpath="residual_plot")


# In[142]:


#Validation function
n_folds = 5

def rmsle_cv(model, features, labels):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(features) # Shuffle the data.
    rmse= np.sqrt(-cross_val_score(model, features, labels, scoring="neg_mean_squared_error", cv = kf))
    return(rmse.mean())


# In[143]:


rid_reg = Ridge()
rid_reg.fit(X_train, y_train)
y_pred = rid_reg.predict(X_val)
rmsle_cv(rid_reg, X_val, y_val)


# ### RandomForestRegressor:
# <a id="random_forest"></a> 
# <img src="https://techblog.expedia.com/wp-content/uploads/2017/06/BoostedTreeExample.jpg">
# **RandomForestRegressor** gives us more randomness, insead of searching through impurity the best feature, RandomForest picks features in a randomly manner to reduce variance at the expense of a higher bias. Nevertheless, this helps us find what the trend is. After all the trees have predicted the outcome for a specific instance, the average from all the DecisionTree models is taken and that will be the prediction for a specific instance.

# In[144]:


from sklearn.model_selection import GridSearchCV

params = {'n_estimators': list(range(50, 200, 25)), 'max_features': ['auto', 'sqrt', 'log2'], 
         'min_samples_leaf': list(range(50, 200, 50))}

grid_search_cv = GridSearchCV(RandomForestRegressor(random_state=42), params, n_jobs=-1)
grid_search_cv.fit(X_train, y_train)


# In[145]:


grid_search_cv.best_estimator_


# In[146]:


# Show best parameters.
grid_search_cv.best_params_


# In[147]:


# You can check the results with this functionof grid search.
# RandomSearchCV takes just a sample not all possible combinations like GridSearchCV.
# Mean test score is equivalent to 0.2677
grid_search_cv.cv_results_
df_results = pd.DataFrame(grid_search_cv.cv_results_)
df_results.sort_values(by='mean_test_score', ascending=True).head(2)


# In[148]:


rand_model = grid_search_cv.best_estimator_

rand_model.fit(X_train, y_train)


# In[149]:


# Final root mean squared error.
y_pred = rand_model.predict(X_val)
rand_mse = mean_squared_error(y_val, y_pred)
rand_rmse = np.sqrt(rand_mse)
rand_rmse


# In[150]:


# It was overfitting a bit.
score = rmsle_cv(rand_model, X_val, y_val)
print("Random Forest score: {:.4f}\n".format(score))


# In[151]:


# Display scores next to attribute names.
# Reference Hands-On Machine Learning with Scikit Learn and Tensorflow
attributes = X_train
rand_results = rand_model.feature_importances_
cat_encoder = categorical_pipeline.named_steps["encoder"]
cat_features = list(cat_encoder.categories_[0])
total_features = lst_numerical + cat_features
feature_importance = sorted(zip(rand_results, total_features), reverse=True)
feature_arr = np.array(feature_importance)
# Top 10 features.
feature_scores = feature_arr[:,0][:10].astype(float)
feature_names = feature_arr[:,1][:10].astype(str)


d = {'feature_names': feature_names, 'feature_scores': feature_scores}
result_df = pd.DataFrame(data=d)

fig, ax = plt.subplots(figsize=(12,8))
ax = sns.barplot(x='feature_names', y='feature_scores', data=result_df, palette="coolwarm")
plt.title('RandomForestRegressor Feature Importances', fontsize=16)
plt.xlabel('Names of the Features', fontsize=14)
plt.ylabel('Feature Scores', fontsize=14)


# ## GradientBoostingRegressor:
# <img src="https://image.slidesharecdn.com/slides-140224130205-phpapp02/95/gradient-boosted-regression-trees-in-scikitlearn-21-638.jpg?cb=1393247097">
# <a id="gradient_boosting"></a>
# The Gradient Boosting Regressor class trains the models over the residuals (prediction errors) leading to smaller variances and higher accuracy.  

# In[152]:


params = {'learning_rate': [0.05], 'loss': ['huber'], 'max_depth': [2], 'max_features': ['log2'], 'min_samples_leaf': [14], 
          'min_samples_split': [10], 'n_estimators': [3000]}


grad_boost = GradientBoostingRegressor(learning_rate=0.05, loss='huber', max_depth=2, 
                                       max_features='log2', min_samples_leaf=14, min_samples_split=10, n_estimators=3000,
                                       random_state=42)


grad_boost.fit(X_train, y_train)


# In[153]:


y_pred = grad_boost.predict(X_val)
gboost_mse = mean_squared_error(y_val, y_pred)
gboost_rmse = np.sqrt(gboost_mse)
gboost_rmse


# In[154]:


# Gradient Boosting was considerable better than RandomForest Regressor.
# scale salesprice.
# y_val = np.log(y_val)
score = rmsle_cv(grad_boost, X_val, y_val)
print("Gradient Boosting score: {:.4f}\n".format(score))


# ## StackingRegressor:
# <img src="https://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor_files/stackingregression_overview.png">
# <a id="stacking_regressor"></a>
# In stacking regressor we combine different models and use the predicted values in the training set to mae further predictions. In case you want to go deeper into parameter <b>"tuning"</b> I left you the code above the different models so you can perform your own GridSearchCV and find even more efficient parameters! <br>
# <ul>
# <li> ElasticNet </li>
# <li> DecisionTreeRegressor </li>
# <li> MLPRegressor (Later I will include it after learning more about neural networks) </li>
# <li> SVR </li>
# </ul>

# In[155]:


# Define the models
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, Ridge

# Parameters for Ridge
# params = {"alpha": [0.5, 1, 10, 30, 50, 75, 125, 150, 225, 250, 500]}
# grid_ridge = GridSearchCV(Ridge(random_state=42), params)
# grid_ridge.fit(X_train, y_train)

# Parameters for DecisionTreeRegressor
# params = {"criterion": ["mse", "friedman_mse"], "max_depth": [None, 2, 3], "min_samples_split": [2,3,4]}

# grid_tree_reg = GridSearchCV(DecisionTreeRegressor(), params)
# grid_tree_reg.fit(X_train, y_train)



# Parameters for SVR
# params = {"kernel": ["rbf", "linear", "poly"], "C": [0.3, 0.5, 0.7, 0.7, 1], "degree": [2,3]}
# grid_svr = GridSearchCV(SVR(), params)
# grid_svr.fit(X_train, y_train)



# Tune Parameters for elasticnet
# params = {"alpha": [0.5, 1, 5, 10, 15, 30], "l1_ratio": [0.3, 0.5, 0.7, 0.9, 1], "max_iter": [3000, 5000]}
# grid_elanet = GridSearchCV(ElasticNet(random_state=42), params)

# Predictive Models
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=3000)
svr = SVR(C=1, kernel='linear')
tree_reg = DecisionTreeRegressor(criterion='friedman_mse', max_depth=None, min_samples_split=3)
ridge_reg = Ridge(alpha=10)

# grid_elanet.fit(X_train, y_train)


# In[156]:


from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
# Try tomorrow with svr_rbf = SVR(kernel='rbf')
# Check this website!
# Consider adding two more models if the score does not improve.
lin_reg = LinearRegression()

ensemble_model = StackingRegressor(regressors=[elastic_net, svr, rand_model, grad_boost], meta_regressor=SVR(kernel="rbf"))

ensemble_model.fit(X_train, y_train)


score = rmsle_cv(ensemble_model, X_val, y_val)
print("Stacking Regressor score: {:.4f}\n".format(score))


# In[157]:


# We go for the stacking regressor model
# although sometimes gradientboosting might show to have a better performance.
final_pred = ensemble_model.predict(test_scaled)


# In[158]:


# # Dataframe
final = pd.DataFrame()

# Id and Predictions
final['Id'] = test_id
final['SalePrice'] = np.expm1(final_pred)

# CSV file
final.to_csv('submission.csv', index=False) # Create Submission File


# ## TensorFlow:
# Although the accuracy of our neural network  is still not as accurate as our ensemble model, I wanted to share two main aspects of tensorflow.
# <ul>
# <li> Implementing a Neural Network with a real life <b>regression scenario</b>. </li>
# <li>Show the structure of Neural Networks through <b>tensorboard</b> (we will do this with ipython display.) </li>
# </ul>
# 
# **Note: There are things still to be done to improve the accuracy of this neural network nevertheles, this notebook will be subjected to future changes to improve the effectiveness of our neural network. (We still are missing layers and activation functions! which are critical for improving accuracy of our NN.)** <br><br>
# 
# (Reference: Hands On Machine Learning and TensorFlow by Aurélien Géron)

# In[159]:


# TensorFlow

import tensorflow as tf
from datetime import datetime

# Functions (Hands On Machine Learning by Aurelien Geron)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(42)
    

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(len(X_train), size=batch_size)
    X_batch = X_train[indices]
    y_batch = y_train[indices]
    return X_batch, y_batch

reset_graph()

address = datetime.utcnow().strftime("%Y%m%D%H%M%S")
root_log = 'log_dir'
logdir = '{}/run-{}/'.format(root_log, address)

n_inputs = X_train.shape[1]
# Write the hidden layers.
n_hiddenlayer1 = 25

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None), name="y")

theta = tf.Variable(tf.random_uniform([n_inputs, 1],-1.0, 1.0, seed=42), name="theta") # These are the weights
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse)
writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) # Writer stays out of the session to avoid overwrite!


# In[160]:


n_epochs = 10
batch_size = 50
n_batches = int(round(np.ceil(X_train.shape[0]/ batch_size), 2))


# In[161]:


with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 50 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                writer.add_summary(summary_str, step)
                # print("batch_index: ", step, "MSE:{}".format(sess.run(mse, feed_dict={X: X_batch, y: y_batch})))
                acc_val = mse.eval(feed_dict={X: X_val, y: y_val})
                print('Batch accuracy: {:.2f}'.format(sess.run(mse, feed_dict={X: X_batch, y: y_batch})), 
                'Validation Set Accuracy:{:.2f} '.format(acc_val))
                test_predictions = y_pred.eval(feed_dict={X: test_scaled})
                
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
            
    best_theta = theta.eval()   
    
# print(logdir)
writer.close()


# In[162]:


from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


# In[163]:


show_graph(tf.get_default_graph())


# ## Conclusion:
# I got a 0.13 score approximately, in the future I aim to fix some issues with regards to the tuning of hyperparameters and implement other concepts of feature engineering that will help algorithms make a more concise prediction. Nevertheless, this project helped me understand more complex models that could be implemented in practical situations. Hope you enjoyed our in-depth analysis of this project and the predictive models used to come with close to accurate predictions. Open to constructive criticisms!
