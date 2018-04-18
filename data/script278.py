
# coding: utf-8

# # **PPP, Salaries and XGBoost ! **
# ***
# 
# **Mhamed Jabri — 11/22/17**
# 
# When I started working on this dataset, I checked other kernels that had a salary-approach and noticed that it was really complicated to give unbiased insights when you compare incomes from all over the world. So in my previous [kernel](http://www.kaggle.com/mhajabri/what-do-kagglers-say-about-data-science), I restricted my income analysis to US Data Scientists and , as I explained there, the reason why I did so is that comparing wages across countries can be misleading and actually false.
# The main problem that arises is the fact that **the cost of living** is not the same for every country. As a consequence to that, the wages we get after converting to the same currency using market exchange rates are different from one another.
# 
# In this kernel, I'll try to show you how you can analyze income data when the respondents come from differents countries and we'll see who are the most valued data scientists in different parts of the world. After that, we'll use the adjusted incomes from part 1 and the insights from part 2 to build a ML model that predicts the income of a Kaggler Data Scientist.

# # Table of contents
# 
# * [Purchasing Power Parity (PPP) for dummies](#ppp)
# * [PPP-Adjusted Kagglers' incomes ](#adjusted)
# * [How valued are Data Scientists across the world](#valuation)
# * [What makes a Data Scientist valuable for each country](#properties)
#    * [Most valuated ML methods](#methods)
#    * [Most valuated DS tools](#tools)
#    * [Most valuated job titles](#titles)
#    * [Most valuated job functions](#functions)
# * [Income prediction across the world using xgboost](#xgboost)
#     * [Feature engineering](#features)
#     * [Prediction and scores](#prediction)
#     * [Analysis of the results](#analysis)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import xgboost as xgb
from sklearn.metrics import mean_squared_log_error as MSLE

# Any results you write to the current directory are saved as output.


# # What's Purchasing Power Parity (PPP) ?
# <a id="ppp"></a>
# ***

# When you're traveling across the country you live in, you notice that prices for food, rent and common goods vary from one city to another. This happens in a much larger scale when changing countries !      
# Let's take an example : Consider 1 USD, Google tells you that it equals 64.92 Indian Rupee. An average dinner in the US may cost you 10-12 USD which corresponds to 640-760 INR. But in fact, it appears that you could get yourself a good meal with 150-200INR in India. In other words : **You can do with 1USD in India more than what you would do with 1USD in the US.**
# 
# As a basic example, let's say we find that 1USD in India = 3USD in the US, that would mean that if you get paid 3000 USD in India you would lead the same someone would be leading in the US if he's getting paid 9000 USD.
# 
# Here comes the **Purchasing Power Parity (PPP from now on)** : Faced with this problematic of different costs of livings, economists came up with a ratio to convert currencies without using market exchange rates. To do so, they select a **basket of goods**, say orange/Milk/tomato ..., and check its value in the USD and in the country they're interested in.     
# Let's say they find that : 
# * Value of the basket in the US = 10 USD
# * Value of the basket in India = 200 INR        
# 
# Then they would say that **PPP(India) = 200/10 = 20**, in other words, you need 20 local currency units to get what you would get with 1 USD in the US.   
# 
# **All in all,* The purchasing power of a currency refers to the quantity of the currency needed to purchase a given unit of a good, or common basket of goods and services.***

# # Kagglers incomes adjusted according to PPPs
# <a id="adjusted"></a>
# ***

# In the following, we'll convert the incomes using the PPP rates and see what happens for different countries.      
# To get significant results, we'll only keep countries from where at least 80 respondents gave their income informations.

# In[ ]:


data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")

#We convert the salaries to numerical values and keep salaries between 1000 and 1.000.000 Local currency
data['CompensationAmount'] = data['CompensationAmount'].fillna(0)
data['CompensationAmount'] = data.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0))
                                                       else float(x.replace(',','')))
df = data[(data['CompensationAmount']>1000) & (data['CompensationAmount']<2000000)]


#We only keep the countries with more than 80 respondents to get significant results later on
s_temp = df['Country'].value_counts()
s_temp = s_temp[s_temp>80]
countries=list(s_temp.index)
countries.remove('Other')
df=df[df.Country.isin(countries)]


# We have some missing values in the Currency column.    
# We won't drop the rows with missing values, instead, we'll fill them with the currency of the country the respondent lives in, here's the code to do so : 

# In[ ]:


df['CompensationCurrency'] =df.groupby('Country')['CompensationCurrency'].apply(lambda x: x.fillna(x.value_counts().idxmax()))


# Now, we'll provide PPP rates for the countries we kept earlier.
# > I tried to use the latest PPP rates available, you can found the PPP for each country [here](http://www.imf.org/external/datamapper/PPPEX@WEO/OEMDC/ADVEC/WEOWORLD/IND).

# In[ ]:


#The PPP rates
rates_ppp={'Countries':['United States','India','United Kingdom','Germany','France','Brazil','Canada','Spain','Australia','Russia','Italy',"People 's Republic of China",'Netherlands'],
           'Currency':['USD','INR','GBP','EUR','EUR','BRL','CAD','EUR','AUD','RUB','EUR','CNY','EUR'],
           'PPP':[1.00,17.7,0.7,0.78,0.81,2.05,1.21,0.66,1.46,25.13,0.74,3.51,0.8]}

rates_ppp = pd.DataFrame(data=rates_ppp)
rates_ppp


# We notice that **the currency used for each country respondents is most of the time the local currency but not always** so we can't directly use the PPP rates.    
# What we'll do instead is the following : 
# 1. Convert all incomes to USD using Market Exchange Rates that were given by kaggle
# 2. Calculature the ratio of PPP rates to MER rates
# 3. Calculate the adjusted salaries using the ratio adjusting factor

# In[ ]:


#we load the exchange rates that were given by Kaggle a
rates_mer=pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")
rates_mer.drop('Unnamed: 0',inplace=True,axis=1)

rates=rates_ppp.merge(rates_mer,left_on='Currency',right_on='originCountry',how='left')
rates['PPP/MER']=rates['PPP']*rates['exchangeRate']

#keep the PPP/MER rates plus the 'Countries' column that will be used for the merge
rates=rates[['Countries','PPP','PPP/MER']]
rates


# In[ ]:


df=df.merge(rates_mer,left_on='CompensationCurrency',right_on='originCountry',how='left')
df=df.merge(rates,left_on='Country',right_on='Countries',how='left')

df['AdjustedSalary']=df['CompensationAmount']*df['exchangeRate']/df['PPP/MER']

d_salary = {}
for country in df['Country'].value_counts().index :
    d_salary[country]=df[df['Country']==country]['AdjustedSalary'].median()
    
median_wages = pd.DataFrame.from_dict(data=d_salary, orient='index').round(2)
median_wages.sort_values(by=list(median_wages),axis=0, ascending=True, inplace=True)
ax = median_wages.plot(kind='barh',figsize=(15,8),width=0.7,align='center')
ax.legend_.remove()
ax.set_title("Adjusted incomes over the world",fontsize=16)
ax.set_xlabel("Amount", fontsize=14)
ax.set_ylabel("Country", fontsize=14)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(10)
plt.tight_layout()

plt.show();


# Let's analyze this plot.    
# Other kernels, such as this [one](http://www.kaggle.com/drgilermo/salary-analysis), have already plotted the medians of the yearly incomes  using Market Exchanges Rates. We notice there for example that the median for India was about 12k USD while it was 36k USD for China.      
# Using the PPP rates as above, the results are more balanced, with 33k USD for India and 50K for USD for China. That being said, the problem is not entirely solved as the gap is still quite big comparing to the US. Why is that ? 
# 
# Two main reasons : 
# * **Very populated and/or developing countries often set the value of their currency to be lower than the U.S. dollar** because they want the cost of living to be lower, so that they can pay their workers less. This also mean that they are more attractive markers to big companies and investors who are interested in implementing themselves in countries where it would cost them less (Hello Apple) ! 
# * **There are some limitations to PPP rates**. One of them is that, as we've said before, we use baskets of products and compare their price in different countries. By doing that, we are regaling with a similar group of commodities in both countries which is not entirely reasonable since production for some products depends on geographical parameters.

# # Which country values its Data Scientists the most ?
# <a id="valuation"></a>
# ***

# **We know the median salary for each country so we can compare between that and the median income in that country to measure how well is a data scientist paid by the standards of the country he lives in.**
# 
# Unfortunately, the information we find easily on the internet is the *average income* which is not trustworthy at all because as we know the mean is very sensitive to abnormally high values and, in any country in the world, the wealthiest people have ridiculous incomes.
# 
# That being said [Gallup](http://news.gallup.com/poll/166211/worldwide-median-household-income-000.aspx) conducted a survey between 2005 and 2012 to publish the medians over the world for 2013. This information is valuable, but how to use it ? 
# 
# You certainly heard your grandfather saying *When I was your age, I traveled across the country with 100\$ in my pocket* and you tought to yourself *that's absurd, I can't believe it.* Well, that's **inflation** ! Inflation is the fact that, in the same country with the same currency, you can't do in the present with some amount X what you could have done in the past with that same amount. So yeah obviously, **100\$ in the 1930s is much more *valuable* than 100\$ in 2017**.    
# What we'll do is the following : 
# 1. We take the median incomes that were measured in 2013 
# 2. We calculate the inflation rate between 2013 and now ([source](http://data.oecd.org/price/inflation-cpi.htm))
# 3. We use the inflation rate to calculate new medians that would correspond to the present ([tutorial](http://www.cpwr.com/sites/default/files/annex_how_to_calculate_the_real_wages.pdf) on how to do that)

# In[ ]:


inflations={'Countries':['United States','India','United Kingdom','Germany','France','Brazil','Canada','Spain','Australia','Russia','Italy',"People 's Republic of China",'Netherlands'],
           'CPI_2013':[106.83,131.98,110.15,105.68,105.01,119.37,105.45,107.21,107.70,121.64,107.20,111.16,107.48],
           'CPI_2017':[113.10,162.01,116.51,109.6,107.1,156.73,112.39,109.13,113.48,168.50,108.61,119.75,111.55],
           'medians_2013':[15480,615,12399,14098,12445,2247,15181,7284,15026,4129,6874,1786,14450]}

rates_inflations = pd.DataFrame(inflations)
rates_inflations['adjusted_medians']=(rates_inflations['medians_2013']*rates_inflations['CPI_2017']/rates_inflations['CPI_2013']).round(2)
rates_inflations


# The three steps are now completed.    
# We move on to calculate the ratio of data scientists incomes to median incomes for each country.

# In[ ]:


tmp=median_wages.reset_index()
tmp = tmp.rename(columns={'index': 'Country', 0: 'median_income'})

rates_inflations=rates_inflations.merge(tmp,left_on='Countries',right_on='Country',how='left')
rates_inflations['ratio_incomes']=(rates_inflations['median_income']/rates_inflations['adjusted_medians']).round(2)

tmp2=rates_inflations[['Country','ratio_incomes']]
tmp2.sort_values(by='ratio_incomes',axis=0, ascending=True, inplace=True)


# In[ ]:


tmp2.plot.barh(x='Country',figsize=(12,8))
plt.show();


# **Indian Data Scientists are paid 44 times more than average workers, WOW !    **
# In second and third position we find China (25 times more) and Brazil (15 times more). Is that to be expected ? In fact, Yes !   
# Actually, India, China and Brazil are part of what's called [BRICS](http://en.wikipedia.org/wiki/BRICS) which is the acronym for an association of five major emerging national economies: *Brazil, Russia, India, China and South Africa*. These countries have fast-growing economies but are still suffering from high poverty rates (Russia not to be included). The consequence of that is a social imbalance where workers with high status (such as data scientists) are getting paid much more higher (at least 15 times more) than average workers, hence the results above.   
# 
# Looking at developed countries, we find that data scientists are pretty well payed comparing to average, with minimums in Canada and France and maximums in Spain and the US.

# So much for the economics course ! Personnaly, I enjoyed digging to find data about inflation rates and PPP rates. I hope you guys learned a thing or two from what's above and hopefully you can use that in the future when you deal with incomes from various years (inflations) or countries (PPP).
# 
# We move on to more usual stuff to understand what impacts most the income of data scientists across the world.

# # Who are the most valued Data Scientists ? 
# <a id="properties"></a>
# ***

# In this section, we'll try to see who are the data scientists who earn by analyzing what they do at work : their job function, their roles, the ML/DS methods they use the most ...
# 
# In order to progress in our analysis and obtain significant results, we'll construct baskets of countries for which salaries are similar.
# 1. First group will only contain the US,
# 2. Second group will contain *Australia, Netherlands, Germany, United Kingdom and Canada*,
# 3. Third group will contain *Spain, France, Italy, China, Brazil*,
# 4. At last, the last group will contain *India and Russia*

# In[ ]:


datasets = {'USA' : df[df['Country']=='United States'] , 
            'Eur+Ca' :df[df.Country.isin(['Australia','Germany','Canada','United Kingdom','Netherlands'])],
            'Eur2+Bra+Chi' : df[df.Country.isin(['Spain','France','Brazil',"People 's Republic of China",'Italy'])],
            'India/Russia' : df[df.Country.isin(['India','Russia'])]}


# ## Methods used at work
# <a id="methods"></a>

# In[ ]:


methods=['WorkMethodsFrequencyBayesian','WorkMethodsFrequencyNaiveBayes','WorkMethodsFrequencyLogisticRegression',
       'WorkMethodsFrequencyDecisionTrees','WorkMethodsFrequencyRandomForests',
       'WorkMethodsFrequencyEnsembleMethods','WorkMethodsFrequencyDataVisualization','WorkMethodsFrequencyPCA',
       'WorkMethodsFrequencyNLP','WorkMethodsFrequencyNeuralNetworks',
       'WorkMethodsFrequencyTextAnalysis',
       'WorkMethodsFrequencyRecommenderSystems','WorkMethodsFrequencyKNN','WorkMethodsFrequencySVMs',
       'WorkMethodsFrequencyTimeSeriesAnalysis']


d_method_countries={} 
for key, value in datasets.items():
    d_method_countries[key]={}
    for col in methods : 
        method = col.split('WorkMethodsFrequency')[1]
        d_method_countries[key][method]=value[value[col].isin(['Most of the time','Often'])]['AdjustedSalary'].median()
        
positions=[(0,0),(1,0),(0,1),(1,1)]
f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))
for ((key, value), pos) in zip(d_method_countries.items() , positions):
    methods = pd.DataFrame.from_dict(data=value, orient='index').round(2)
    methods.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)
    methods.plot(kind='barh',figsize=(12,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])
    ax[pos[0],pos[1]].set_title(key,fontsize=14)
    ax[pos[0],pos[1]].legend_.remove()
    

plt.tight_layout()
plt.show();
    


# People who often use Recommender Systems in their work are generally well paid no matter the country (2nd in the US, 3rd in India/Russia...).      
# 
# It's interesting to notice a clear contrast in the 4 plots : In the US, Canada and the first group of european countries (where the median income was high), the most valued method seems to be Naive Bayes. On the other hand, in the BRICS countries (see above) + other european countries, not only Native Bayes users aren't the best paid like their counterparties in the US, they're actually the lowest paid !      
# The same thing happens with Data Visualization but the other way around this time : not too valuable in the US + first basket of Europe, quite important in the rest of the world.
# 
# **For me, this really shows how data science has a different meaning and value depending on the country you live in. Some countries would pay more if you're good at visualizing data while others will pay more if you know how to use bayesian methods so the needs and the purpose of DS/ML vary across the world.**
# 
# 

# ## Tools and programming languages used at work
# <a id="tools"></a>

# In[ ]:


tools=['WorkToolsFrequencyC','WorkToolsFrequencyJava','WorkToolsFrequencyMATLAB',
       'WorkToolsFrequencyPython','WorkToolsFrequencyR','WorkToolsFrequencyTensorFlow',
       'WorkToolsFrequencyHadoop','WorkToolsFrequencySpark','WorkToolsFrequencySQL',
       'WorkToolsFrequencyNoSQL','WorkToolsFrequencyExcel','WorkToolsFrequencyTableau',
       'WorkToolsFrequencyJupyter','WorkToolsFrequencyAWS',
       'WorkToolsFrequencySASBase','WorkToolsFrequencyUnix']

d_tools_countries={} 
for key, value in datasets.items():
    d_tools_countries[key]={}
    for col in tools : 
        tool = col.split('WorkToolsFrequency')[1]
        d_tools_countries[key][tool]=value[value[col].isin(['Most of the time','Often'])]['AdjustedSalary'].median()
        
positions=[(0,0),(1,0),(0,1),(1,1)]
f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))
for ((key, value), pos) in zip(d_tools_countries.items() , positions):
    tools = pd.DataFrame.from_dict(data=value, orient='index').round(2)
    tools.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)
    tools.plot(kind='barh',figsize=(12,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])
    ax[pos[0],pos[1]].set_title(key,fontsize=14)
    ax[pos[0],pos[1]].legend_.remove()
    

plt.tight_layout()
plt.show();
        


# I found the results on this plot quite surprising and not that expected.    
# 
# Technologies that are often used with big data architecture such as Hadoop, Spark and AWS grant good incomes abslutely everywhere, which is stunning.        
# 
# Java users in the US seem to be well payed too, I didn't expect that actually but I thought about it and my guess is that data scientsts that still work with Java occupy senior positions and have been using it for a long time while Python and R coders can be junior data scientists.        
# 
# MATLAB coders are the lowest paid in all countries.

# ## Job titles
# <a id="titles"></a>

# In[ ]:


titles=list(df['CurrentJobTitleSelect'].value_counts().index)
d_titles_countries={} 
for key, value in datasets.items():
    d_titles_countries[key]={}
    for title in titles : 
        d_titles_countries[key][title]=value[value['CurrentJobTitleSelect']==title]['AdjustedSalary'].median()
        
positions=[(0,0),(1,0),(0,1),(1,1)]
f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))
for ((key, value), pos) in zip(d_titles_countries.items() , positions):
    tools = pd.DataFrame.from_dict(data=value, orient='index').round(2)
    tools.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)
    tools.plot(kind='barh',figsize=(12,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])
    ax[pos[0],pos[1]].set_title(key,fontsize=14)
    ax[pos[0],pos[1]].legend_.remove()
    

plt.tight_layout()
plt.show();


# In the US, business analysts and data analysts are far from getting paid as much as data scientists or ML engineers. On the other hand, in Russia and India, business analysts are better paid than data scientists actually !      
# 
# This plot gives us yet another view of what we said above : **countries and companies have different needs across the world**, it seems that US employers are hungry for ML practitioners and they pay them good money while in other emerging countries, the most important role is Predictive Modeler and analysts are still quite needed.

# ## Job Functions
# <a id="functions"></a>

# In[ ]:


func = list(df['JobFunctionSelect'].value_counts().index)
tmp = df
tmp=tmp.replace(to_replace=func, value=['Analyze data','Build a ML service','Build prototypes',
                                        'Build the Data Infrastructure','Other','Research'])

datasets_tmp = {'USA' : tmp[tmp['Country']=='United States'] , 
            'Eur+Ca' :tmp[tmp.Country.isin(['Australia','Germany','Canada','United Kingdom','Netherlands'])],
            'Eur2+Bra+Chi' : tmp[tmp.Country.isin(['Spain','France','Brazil',"People 's Republic of China",'Italy'])],
            'India/Russia' : tmp[tmp.Country.isin(['India','Russia'])]}

functions=list(tmp['JobFunctionSelect'].value_counts().index)
d_functions_countries={} 
for key, value in datasets_tmp.items():
    d_functions_countries[key]={}
    for function in functions : 
        d_functions_countries[key][function]=value[value['JobFunctionSelect']==function]['AdjustedSalary'].median()
        
positions=[(0,0),(1,0),(0,1),(1,1)]
f,ax=plt.subplots(nrows=2, ncols=2,figsize=(15,8))
for ((key, value), pos) in zip(d_functions_countries.items() , positions):
    tools = pd.DataFrame.from_dict(data=value, orient='index').round(2)
    tools.sort_values(by=list(methods),axis=0, ascending=True, inplace=True)
    tools.plot(kind='barh',figsize=(15,8),width=0.7,align='center',ax=ax[pos[0],pos[1]])
    ax[pos[0],pos[1]].set_title(key,fontsize=14)
    ax[pos[0],pos[1]].legend_.remove()
    
plt.tight_layout()
plt.show();


# Building prototypes seems to be the function that's the most rewarded everywhere. It's not the most paid one in the US but it's a close second.** Being a researcher in a lab to advance the SOTA of ML isn't rewarding, no matter the country.** That's a pity :( I always felt like researchers aren't valued enough in France and that's honestly a real issue for many researchers here who switch to industry based roles for salary purposes. This plot shows that it's in fact the case everywhere.   
# Let's hope that this will change in the future ! 

# # How hard is it to predict the incomes for different respondents ?
# <a id="xgboost"></a>
# ***

# In this part, I try to predict the income for kagglers.   
# 
# First, I would like to show how to benefit from the data analysis part to build a model. When you complete your data exploration and find insights, you have to perform some feature engineering according to your findings. For example, we found that MATLAB users are always under-paid while Hadoop/Spark users are always amongst the most paid. We'll add specific binarized features that indicate whether the instance uses that tool or not.    
# Then we do the exact same thing with ML methods using again the results we found above.
# 
# At last, we keep the feature we think will be the most important to our prediction.

# **Feature engineering and preprocessing**
# <a id="features"></a>

# In[ ]:


df['MATLABUsers']=[1 if df['WorkToolsFrequencyMATLAB'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]
df['AWSUsers']=[1 if df['WorkToolsFrequencyAWS'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]
df['HadoopUsers']=[1 if df['WorkToolsFrequencyHadoop'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]
df['SparkUsers']=[1 if df['WorkToolsFrequencySpark'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]

df['NaiveBayesUsers']=[1 if df['WorkMethodsFrequencyNaiveBayes'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]
df['RecommenderSystemsUsers']=[1 if df['WorkMethodsFrequencyRecommenderSystems'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]
df['DataVisualizationUsers']=[1 if df['WorkMethodsFrequencyDataVisualization'].iloc[i] in ['Most of the time','Often'] else 0 for i in range(df.shape[0])]

features= ['GenderSelect','Country','Age','FormalEducation','MajorSelect','ParentsEducation',
           'EmploymentStatus','StudentStatus','DataScienceIdentitySelect','CodeWriter',
           'CurrentEmployerType','SalaryChange','RemoteWork','WorkMLTeamSeatSelect',
           'Tenure','EmployerIndustry','EmployerSize','CurrentJobTitleSelect','JobFunctionSelect',
           'MATLABUsers','AWSUsers','HadoopUsers','SparkUsers',
           'NaiveBayesUsers','RecommenderSystemsUsers','DataVisualizationUsers',
           'TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights',
           'AdjustedSalary']




# Now that we don't really have to worry about having the same range of salary in the same dataset, we'll group countries using a more meaningful way.
# * USA will be kept alone as it already has enough respondents
# * We talked about BRICS earliers, we'll make a dataset with those countries
# * We merge all european countries and add to them Canada and Australia

# In[ ]:


df_us = df[df['Country']=='United States'][features]
df_eur = df[df.Country.isin(['Spain','France','Germany','Canada','United Kingdom','Netherlands','Italy','Australia','Canada'])][features]
df_bric = df[df.Country.isin(['India','Russia','Brazil',"People 's Republic of China"])][features]


# The scoring metric we'll use is the [Mean Squared Logarithmic Error](http://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError). This metric is appropriate when the target variable in the regression problem takes high values so that large differences aren't too penalized. For example, using RMSE, if you predict an annual income of 80k when the real income is 82k, your prediction is quite good but the error score you commit would be $2000^2$ ! Hence our choice.     
# 
# In my previous Kernel, I used logistic regression as it's the most simple model around for classification. Here, instead of using linear regression, I'll go with the most used algorithm on Kaggle, the beloved **XGBoost** !   
# Here we go ! 

# **Predictions**
# <a id="prediction"></a>

# In[ ]:


for (dataset,zone) in zip([df_us,df_bric,df_eur],['USA','BRIC','Europe + Canada and Australia']) : 
    
    dataset=pd.get_dummies(dataset,columns=['GenderSelect','Country','FormalEducation','MajorSelect','ParentsEducation',
           'EmploymentStatus','StudentStatus','DataScienceIdentitySelect','CodeWriter',
           'CurrentEmployerType','SalaryChange','RemoteWork','WorkMLTeamSeatSelect',
           'Tenure','EmployerIndustry','EmployerSize','CurrentJobTitleSelect','JobFunctionSelect'])
    for col in ['Age','TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights']:
        dataset[col] = dataset[col].fillna(value=dataset[col].median())
    dataset.dropna(axis=0,inplace=True)

    np.random.seed(42)
    perm = np.random.permutation(dataset.shape[0])
    train = dataset.iloc[perm[0:round(0.85*len(perm))]]
    test = dataset.iloc[perm[round(0.85*len(perm))::]]
    y_train , y_test = train['AdjustedSalary'] , test['AdjustedSalary']
    X_train , X_test = train.drop('AdjustedSalary',axis=1) , test.drop('AdjustedSalary',axis=1)

    clf=xgb.XGBRegressor(learning_rate=0.05, n_estimators=500, objective='reg:linear',reg_lambda=0.5, 
                         random_state=42)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    print('Prediction for : %s'%zone)
    print('The RMSLE score is : {:0.4f}'.format(np.sqrt(MSLE(y_test,y_pred)) /np.sqrt(len(y_pred)) ))
    print('-------------------------------------------------')


# **Results analysis**
# <a id="analysis"></a>

# Those scores are actually not bad at all ! I consider them to be really great.
# Just to get an idea, the public leaderboard for beginners competition "Bike Sharing Demand" on kaggle displays a best RMSLE of $0.33$. I didn't expect the regression prediction problem to yield results that are as good. If you want to to get an idea about what a RMSLE score means and how it can be interpreted, I invite you to check @j_scheibel response to this [Kaggle topic](https://www.kaggle.com/general/9933).
# 
# We notice that it's way harder to predict the salary for BRIC countries as the RMSLE is 4 times as high as the RMSLE for US Prediction or Europe prediciton.          
# Let's check the salary distribution (plot histograms) for each one of those datasets.

# In[ ]:


f,ax=plt.subplots(nrows=1, ncols=3,figsize=(15,8))
df_bric['AdjustedSalary'].plot.hist(bins=50,ax=ax[0],figsize=(15,8),title='Salary distribution in BRIC countries')
df_eur['AdjustedSalary'].plot.hist(bins=50,ax=ax[1],figsize=(15,8),title='Salary distribution in Europe + Ca+ Aus')
df_us['AdjustedSalary'].plot.hist(bins=50,ax=ax[2],figsize=(15,8),title='Salary distribution in the the US')
plt.show();


# We notice that the salary distributions in the US + Euro zone are quite similar : a bit skewed to the right because of some very high values. For the BRIC countries, it's really different : it's highly skewed to the right and most values of the income are actually near 0 ! As a consequence to that, 'outlier bins' appear earlier for BRIC countries than for the rest.   
# Let's confirm that with boxplots.

# In[ ]:


f,ax=plt.subplots(nrows=1, ncols=3,figsize=(15,8))
sns.boxplot(y=df_bric['AdjustedSalary'],data=df_bric,ax=ax[0]).set_title('Quartiles and outliers in BRIC')
sns.boxplot(y=df_eur['AdjustedSalary'],data=df_eur,ax=ax[1]).set_title('Quartiles and outliers in EUR')
sns.boxplot(y=df_us['AdjustedSalary'],data=df_us,ax=ax[2]).set_title('Quartiles and outliers in USA')
plt.show();


# As we observed before, too many points near the 0 mark for BRIC zone and also a higher density of outliers (isolated points). Those points are the hardest for algorithms because they don't have enough data that looks like them, I think that this is the main reason why the error we got for BRIC was much higher than the errors we got for EUR/USA.

# # Conclusion and perspectives
# ***
# What I truly appreciate about Data Analytics is that you can always dig deeper, try to find additional informations elsewhere to complete the dataset that you have and then gain new insights. Moreover, your knowledge from different fields can be put into practice, you just have to think about what's the best way to use it !    
# 
# I hope you enjoyed learning some basic economics (PPP and inflation rates) and its application to this particular dataset that helped us analyzing the incomes for several countries all at once. Throughout the second part of the kernel, I tried to emphasize on the fact that data science importance is really country-related : the most needed and valuable tools / methods / functions in a country will be very different from those of another one, espacially if their economies are very different from one another (USA vs India for example). At last, I wanted to see how related the income was to some features by building a predictive model and I was surprised by how good the model was.
# 
# **To do list **       
# I would suggest you guys think of other features that can show the difference of perception and application of Data Science and Machine Learning across the globe; maybe the method to learn next year isn't the same for everyone for example ?      
# Also, since the regresison model for BRIC countries gave a score not as good as the one for the US and Europe, I would suugest building a specific ML model for BRIC that would be more suited to its specifities.
# 
# Of course, if you guys have any suggestions about informations I could add to this kernel or improvements to what's already here,  don't hesitate to put that in the comments section, I'd appreciate that !       
# Also, an upvote is always welcome if you feel like the content here is worth it :)
# 
# Thanks for your time and see on another one fellow Kagglers ! 
# 
