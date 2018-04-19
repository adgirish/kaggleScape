
# coding: utf-8

# <a id='1'>1.About Kiva</a><br>
# <a id='2'>2.Load Libraries</a><br>
# <a id='3'>3.Getting the Data</a><br>
# <a id='4'>4.Overview Of The Data</a><br>
# <a id='5'>5.Looking For Missing Values</a><br>
# <a id='6'>6.Lets Analyse The Data</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.1'>6.1 Top Countries to get Frequent Loans</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.2'>6.2.Top Sector to get Frequent Loans</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.3'>6.3 Top Uses to get Frequent Loans</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.4'>6.4 Top Activity to get Frequent Loans</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.5'>6.5 Distribution of loan across different region around the globe </a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.6'>6.6 Top Lender per Loan</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.7'>6.7 Popular Repayment Mode Of Loan</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='6.8'>6.8 Popular Loan Term(In Months)</a><br>
# <a id='7'>7.Diving Deeper Now</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='7.1'>7.1 Distribution Of Loan Amount</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='7.2'>7.2 Distribution of Funded Amount</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='7.3'>Top Countries to get the highest Laont</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='7.4'>Top Sector to get the highest Laont</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='7.5'>Top Activity to get the highest Laont</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='7.6'>Top Use to get the highest Laont</a><br>
# <a id='8'>8.Kiva Over The Years</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='8.1'>8.1 Year-wise Breakdown of Loan/Funding</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='8.2'>8.2 Month-Wise Breakdown of Loan </a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='8.3'>8.3.Average loan Amount Over the Years</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='8.4'>8.4 Average Fund amount over the years</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a id='8.5'>8.5 Different Region Of Funding Over The Years</a><br>

# # [1. ](# 1)About KIVA
# 
# Kiva Microfunds (commonly known by its domain name, Kiva.org) is a 501(c)(3) non-profit organization[2] that allows people to lend money via the Internet to low-income entrepreneurs and students in over 80 countries. 
#                    Kiva's mission is “to connect people through lending to alleviate poverty
# Kiva operates two models—Kiva.org and KivaZip.org. Kiva.org relies on a network of field partners to administer the loans on the ground. These field partners can be microfinance institutions, social businesses, schools or non-profit organizations. KivaZip.org facilitates loans at 0% directly to entrepreneurs via mobile payments and PayPal. In both Kiva.org and KivaZip.org, Kiva includes personal stories of each person who needs a loan because they want their lenders to connect with their entrepreneurs on a human level.                   
#      [Another Cell](# another_cell)              

# # [2. ](# 2)Lets Load the libraries
# 

# In[2]:



import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
import mpl_toolkits
from numpy import array
from matplotlib import cm
import folium
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# # [3. ](# 3) Lets get the dataset

# ### Kiva CrowdFunding dataset.....

# In[ ]:



kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv',parse_dates=['date'])
kiva_mpi_locations= pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
loan_theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')
loan_theme_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')


# ### Additional Kiva snapshot....

# In[ ]:


country_stats=pd.read_csv('../input/additional-kiva-snapshot/country_stats.csv')
lenders=pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')
loan_coords=pd.read_csv('../input/additional-kiva-snapshot/loan_coords.csv')
loans=pd.read_csv('../input/additional-kiva-snapshot/loans.csv')
locations=pd.read_csv('../input/additional-kiva-snapshot/locations.csv')


# ### Multidimensional Poverty Index dataset...

# In[ ]:


mpi_national=pd.read_csv('../input/mpi/MPI_national.csv')


# ## [4. ](# 4)Lets have a quicklook of the datasets

# #### kiva_loans data

# In[ ]:


kiva_loans.head(5)


# #### Kiva Mpi Region data

# In[ ]:


kiva_mpi_locations.head(5)


# #### Loan_Theme_ids data

# In[ ]:


loan_theme_ids.head(5)


# #### Loan_themes_by_region data

# In[ ]:


loan_theme_region.head(5)


# #### country_stats data

# In[ ]:


country_stats.head(5)


# #### Lenders data

# In[ ]:


lenders.head(5)


# #### Loans data

# In[ ]:


loans.head(5)


# #### Mpi national data..

# In[ ]:


mpi_national.head()


# # [5. ](# 5)Missing Values

# ### Kiva_loans Missing

# In[ ]:


kiva_miss=kiva_loans.isnull().sum().sort_values(ascending=False).reset_index()
kiva_miss.columns=['Column','Count%']
kiva_miss=kiva_miss[kiva_miss['Count%']>0]
kiva_miss['Count%']=(kiva_miss['Count%']*100)/kiva_loans.shape[0]
kiva_miss


# ## Country_stats Missing Data

# In[ ]:


country_miss=country_stats.isnull().sum().sort_values(ascending=False).reset_index()
country_miss.columns=['Column','Count%']
country_miss=kiva_miss[country_miss['Count%']>0]
country_miss['Count%']=(country_miss['Count%']*100)/country_stats.shape[0]
country_miss


# # [6. ](# 6)Analysis Of The Data

# ## [6.1 ](# 6.1)Top Countries to get Frequent loan
# 1.  As we know kiva operates in many countries,but here we will only consider top 10 countries.
# 2. As kiva's policy is to lend money to low-income entrepreneurs so chances are high that countries with low GDP tends to get more loan.

# In[ ]:


plt.figure(figsize=(12,10))

country_count=kiva_loans['country'].value_counts()
top_country=country_count.head(10)
sns.barplot(top_country.values,top_country.index)

plt.xlabel('Loan Counts',fontsize=12)
plt.ylabel('Country Name',fontsize=12)
plt.title('Top countries to take loan from Kiva',fontsize=18)
plt.show()


# ### Top 5 countries to get highest number of loans are:<br>
# **Philippines<br>
# Kenya<br>
# El Salvador<br>
# Combodia<br>
# Pakistan**<br>

# In[ ]:


country = kiva_loans['country'].value_counts().reset_index()
country.columns = ['country', 'Total_loans']



data = [ dict(
        type = 'choropleth',
        locations = country['country'],
        locationmode = 'country names',
        z = country['Total_loans'],
        text = country['country'],
        colorscale='Rainbow',
        marker = dict(line = dict (width = 0.5) ),
        colorbar = dict(autotick = False,tickprefix = '',title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Total Loans for Different Countries',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='loans-world-map')


# ## [6.2 ](# 6.2)Top Sectors to get frequent Number of Loans

# In[ ]:


plt.figure(figsize=(12,10))

sector_count=kiva_loans['sector'].value_counts()

top_sector=sector_count.head(10)
sns.barplot(top_sector.values,top_sector.index)

plt.xlabel('Loan Counts',fontsize=12)
plt.ylabel('Sector Name',fontsize=12)
plt.title('Important sector for Kiva',fontsize=18)
plt.show()


# ### Top sector is Agriculture which is also an indication that most of the loans are given to developing countries as agro-food sector is the backbone for their gwoth.

# ## [6.3 ](# 6.3)Top Uses to get frequent Number of Loans

# In[ ]:


plt.figure(figsize=(12,10))

use_count=kiva_loans['use'].value_counts()
top_use=use_count.head(10)
sns.barplot(top_use.values,top_use.index)

plt.xlabel('Loan Counts',fontsize=12)
plt.ylabel('Use of the Loan Name',fontsize=12)
plt.title('Important Use of Loan',fontsize=18)
plt.show()


# ## [6.4 ](# 6.4)Top Activities to get frequent Number of Loans

# In[ ]:


plt.figure(figsize=(12,10))

country_count=kiva_loans['activity'].value_counts()
top_country=country_count.head(10)
sns.barplot(top_country.values,top_country.index)

plt.xlabel('Loan Counts',fontsize=12)
plt.ylabel('Activity Name',fontsize=12)
plt.title('Top activities for Kiva laon',fontsize=18)
plt.show()


# #### Agriculture,Food,Farming are important activies which is predictable 

# In[ ]:


index=['Higher education costs','Home Applicances','Clothing Sales','Retail','Pigs','Agriculture','Food Production/Sales','Personal Housing Expenses','General Store','Farming']
activity_list = []
top=kiva_loans[kiva_loans['activity'].isin(index)]
for sector in top["activity"].values:
    activity_list.extend( [lst.strip() for lst in sector.split(",")] )
temp = pd.Series(activity_list).value_counts()

tag = (np.array(temp.index))
sizes = (np.array((temp / temp.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Activity Distribution')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Activity Distribution")
plt.show()


# ## [6.5 ](# 6.5)Distribution Of loan across differnt regions around the globe

# In[ ]:


plt.figure(figsize=(12,10))

world_region_count=kiva_mpi_locations['world_region'].value_counts()
top_sector=world_region_count
sns.barplot(top_sector.values,top_sector.index)

plt.xlabel('Loan Counts',fontsize=12)
plt.ylabel('World Regions',fontsize=12)
plt.title('World Region for Kiva loans',fontsize=18)
plt.show()


# ## [6.6](# 6.6)Top Lender Count Per Loan

# In[ ]:


plt.figure(figsize=(12,10))

lender_count=kiva_loans['lender_count'].value_counts()
top_lender=lender_count.head(10)
sns.barplot(top_lender.index,top_lender.values)

plt.xlabel('Leander Count per Lo',fontsize=12)
plt.ylabel('Total cases',fontsize=12)
plt.title('Number Of Lenders per Loan',fontsize=18)
plt.show()


# ## [6.7 ](# 6.7)Popular Repayment Mode of Loan

# In[ ]:


repay_list = []
for repay in kiva_loans["repayment_interval"].values:
    repay_list.extend( [lst.strip() for lst in repay.split(",")] )
temp = pd.Series(repay_list).value_counts()

tag = (np.array(temp.index))
sizes = (np.array((temp / temp.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=tag, values=sizes)
layout = go.Layout(title='Repay Distribution')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Repay Distribution")
plt.show()


# ## [6.8 ](# 6.8)Popular Loan Term(in months)

# In[ ]:


plt.figure(figsize=(12,10))

country_count=kiva_loans['term_in_months'].value_counts().sort_values(ascending=False)
top_country=country_count.head(30)
sns.barplot(top_country.index,top_country.values)

plt.xlabel('Number Of Weeks',fontsize=12)
plt.ylabel('Total Case',fontsize=12)
plt.title('Loan Term for Kiva loans',fontsize=18)
plt.show()


# # [7. ](# 7)Lets Dive Deeper Now

# ## [7.1 ](# 7.1)Distribution of Loan Amount<br>
# Let's take a look at how requested loan amounts are distributed

# In[ ]:


fig=plt.figure(figsize=(10,8))
sns.distplot(kiva_loans['loan_amount'])

plt.show()


# Let's look at 95th percentile for plotting this data.

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(kiva_loans[kiva_loans['loan_amount'] < kiva_loans['loan_amount'].quantile(.95) ]['loan_amount'])
plt.show()


# ## [7.2 ](# 7.2)Distribution Of Funded Amount<br>
# Lets look at the spread of distributed amount

# In[ ]:


fig=plt.figure(figsize=(10,8))
sns.distplot(kiva_loans['funded_amount'])

plt.show()


# Lets look at the 95th percentile of the data

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(kiva_loans[kiva_loans['funded_amount'] < kiva_loans['funded_amount'].quantile(.95) ]['funded_amount'])
plt.show()


# ## [7.3 ](# 7.3)Top Country which Get Highest Loan(In terms of mean loan value)

# In[ ]:


plt.figure(figsize=(15,8))
country = (kiva_loans.groupby(['country'])['loan_amount'].mean().sort_values(ascending=False).head(10))
sns.barplot(country.values, country.index, )

plt.xlabel('Mean loan amount ', fontsize=20)
plt.ylabel('Countries', fontsize=20)
plt.title('Countries Getting Highest Loan Amount', fontsize=24)
plt.show()


# ## [7.4 ](# 7.4)Top Sector which got highest loan(average loan value)

# In[ ]:


plt.figure(figsize=(15,8))
country = (kiva_loans.groupby(['sector'])['loan_amount'].mean().sort_values(ascending=False).head(10))
sns.barplot(country.values, country.index, )

plt.xlabel('Mean loan amount ', fontsize=20)
plt.ylabel('Sector', fontsize=20)
plt.title('Sector Getting Highest Loan Amount', fontsize=24)
plt.show()


# ## [7.5 ](# 7.5)Top Activity which got the highest loan

# In[ ]:


plt.figure(figsize=(15,8))
activity = (kiva_loans.groupby(['activity'])['loan_amount'].mean().sort_values(ascending=False).head(10))
sns.barplot(activity.values, activity.index, )

plt.xlabel('Mean loan amount ', fontsize=20)
plt.ylabel('Top Activity', fontsize=20)
plt.title('Activity Getting Highest Loan Amount', fontsize=24)
plt.show()


# ## [7.6 ](# 7.6)Top use of the Loan(In terms of mean amount Spend on that activity)

# In[ ]:


plt.figure(figsize=(15,12))
use = (kiva_loans.groupby(['use'])['loan_amount'].mean().sort_values(ascending=False).head(10))
sns.barplot(use.values, use.index, )

plt.xlabel('Mean loan amount ', fontsize=20)
plt.ylabel('Top Use', fontsize=20)
plt.title('Top use of the Loan Money', fontsize=24)
plt.show()


# # [8. ](# 8)Kiva Over The Years(Kiva's Spending) 

# ## [8.1 ](# 8.1)Year-Wise Breakdown of Loan/Funding

# In[ ]:


plt.figure(figsize=(12,10))
kiva_loans['Year']=kiva_loans.date.dt.year
year_count=kiva_loans['Year'].value_counts().sort_values(ascending=False)
top_country=year_count.head(30)
sns.barplot(top_country.index,top_country.values)

plt.xlabel('Year',fontsize=12)
plt.ylabel('Total',fontsize=12)
plt.title('Year-wise Breakdown of Loan/Funding',fontsize=18)
plt.show()


# In[ ]:


df=kiva_loans.loc[:,['Year','funded_amount']]
year_count=df['Year'].value_counts()

figure=plt.figure(figsize=(12,10))
sns.pointplot(year_count.index,year_count.values)

plt.xlabel('Year',fontsize=12)
plt.ylabel('Total NUmber Of Loans',fontsize=12)
plt.title('Year-Wise Breakdown',fontsize=18)
plt.show()


# ## [8.2 ](# 8.2)Monthwise Breakdown of Kiva Loans for differnt Years

# In[ ]:


kiva_loans['Months']=kiva_loans.date.dt.month
fig=plt.figure(figsize=(15,8))
df=kiva_loans.groupby(['Year', 'Months']).count()
df=df.reset_index()
sns.pointplot(df.Months,df.loan_amount,hue=df.Year)

plt.xlabel('Month(Jan-Dec)',fontsize=12)
plt.ylabel('Total number Of Loans',fontsize=12)
plt.title('Month-Wise Breakdown',fontsize=18)
plt.show()


# ## [8.3 ](# 8.3)Mean Loan Amount monthwise over-the Years

# In[ ]:


fig=plt.figure(figsize=(15,8))
df=kiva_loans.groupby(['Year', 'Months']).mean()
df=df.reset_index()
sns.pointplot(df.Months,df.loan_amount,hue=df.Year)

plt.xlabel('Month(Jan-Dec)',fontsize=12)
plt.ylabel('Mean Loan Amount',fontsize=12)
plt.title('Month-Wise Breakdown',fontsize=18)
plt.show()


# ## [8.4 ](# 8.4)Mean Funded Amount monthwise over-the Years

# In[ ]:


fig=plt.figure(figsize=(15,8))
df=kiva_loans.groupby(['Year', 'Months']).mean()
df=df.reset_index()
sns.pointplot(df.Months,df.funded_amount,hue=df.Year)

plt.xlabel('Month(Jan-Dec)',fontsize=12)
plt.ylabel('Average Funded Amount',fontsize=12)
plt.title('Month-Wise Breakdown',fontsize=18)
plt.show()


# ## [8.5 ](# 8.5)Different Area Of Funding Over The Years

# In[ ]:


for x in range(2014,2018):
    kiva_loans_14=kiva_loans[kiva_loans['Year']==x]
    country = kiva_loans_14['country'].value_counts().reset_index()
    country.columns = ['country', 'Total_loans']



    data = [ dict(
        type = 'choropleth',
        locations = country['country'],
        locationmode = 'country names',
        z = country['Total_loans'],
        text = country['country'],
        colorscale='Reds',
        marker = dict(line = dict (width = 0.5) ),
        colorbar = dict(autotick = False,tickprefix = '',title = 'Number of Loans'),
      ) ]

    layout = dict(
      title = ('Total Loans for Different Countries ('+str(x))+')',
      geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
           )
        )
     )

    fig = dict( data=data, layout=layout )
    py.iplot( fig, validate=False, filename='loans-world-map')

