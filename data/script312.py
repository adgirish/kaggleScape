
# coding: utf-8

# # ** Kiva - Data analysis and Poverty estimation **
# ***
# 
# **Mhamed Jabri — 02/27/2018**
# 
# Machine Learning is **disruptive**. That's no news, everyone knows that by now. Every industry is being affected by AI, AI/ML startups are booming ... No doubt, now sounds like the perfect time to be a data scientist !  
# That being said, two industries stand out for me when it comes to applying machine learning : **Healthcare and Economic Development**. Not that other applications aren't useful or interesting but those two are, in my opinion, showing how we can really use technology to make the world a better place. Some impressive projects on those fields are being conducted right now by several teams; *Stanford* has an ongoing project about predicting poverty rates with satellite imagery, how impressive is that ?!
# 
# Here in Kaggle, we already have experience with the first one (healthcare), for example, every year there's the *Data Science Bowl* challenge where competitors do their very best to achieve something unprecedented, in 2017, the competition's goal was to **improve cancer screening care and prevention**.  
# I was very excited and pleased when I got the email informing me about the Kiva Crowdfunding challenge and it's nice to know that this is only the beggining, with many more other competitions to come in the Data Science for Good program.
# 
# Being myself very interested in those issues and taking courses such as *Microeconomics* and *Data Analysis for Social Scientists* (If interested, you can find both courses [here](https://micromasters.mit.edu/dedp/), excellent content proposed by MIT and Abdul Latif Jameel Poverty Action Lab), I decided to publish a notebook in this challenge and take the opportunity to use everything I've learned so far.     
# **Through this notebook**, I hope that not only will you learn some Data Analysis / Machine Learning stuff, but also (and maybe mostly) learn a lot about economics (I'll do my best), learn about poverty challenges in the countries where Kiva is heavily involved, learn about how you can collect data that's useful in those problems and hopefully inspire you to apply your data science skills to build a better living place in the future !
# 
# **P.S : This will be a work in progress for at least a month. I will constantly try to improve the content, add new stuff and make use of any interesting new dataset that gets published for this competition.**
# 
# Enjoy !

# # Table of contents
# ***
# 
# * [About Kiva and the challenge](#introduction)
# 
# * [1. Exploratory Data Analysis](#EDA)
#    * [1.1. Data description](#description)
#    * [1.2. Use of Kiva around the world](#users)
#    * [1.3. Loans, how much and what for ?](#projects)
#    * [1.4. How much time until you get funded ?](#dates)
#    * [1.5. Amount of loan VS Repayment time ?](#ratio)
#    * [1.6. Lenders : who are they and what drives them ?](#lenders)
# 
# * [2. Poverty estimation by region](#predict)
#    * [2.1. What's poverty ?](#definition)
#    * [2.2. Multidimensional Poverty Index](#mpi)
#    * [2.3 Proxy Means Test](#pmt)     
# 
# * [Conclusion](#conclusion)

# #  About Kiva and the challenge
# ***
# 
# Kiva is a non-profit organization that allows anyone to lend money to people in need in over 80 countries. When you go to kiva.org, you can choose a theme (Refugees, Shelter, Health ...) or a country and you'll get a list of all the loans you can fund with a description of the borrower, his needs and the time he'll need for repayment. So far, Kiva has funded more than 1 billion dollars to 2 million borrowers and is considered a major actor in the fight against poverty, especially in many African countries.
# 
# In this challenge, the ultimate goal is to obtain as precise informations as possible about the poverty level of each borrower / region because that would help setting investment priorities. Kagglers are invited to use Kiva's data as well as any external public datasets to build their poverty estimation model.  
# As for Kiva's data, here's what we've got : 
# * **kiva_loans** : That's the dataset that contains most of the informations about the loans (id of borrower, amount of loan, time of repayment, reason for borrowing ...)
# * **kiva_mpi_region_locations** : This dataset contains the MPI of many regions (subnational) in the world.
# * **loan_theme_ids** : This dataset has the same unique_id as the kiva_loans (id of loan) and contains information about the theme of the loan.
# * **loan_themes_by_region** : This dataset contains specific informations about geolocation of the loans.
# 
# This notebook will be divided into two parts : 
# 1. First I will conduct an EDA using mainly the 4 datasets provided by Kiva. 
# 2. After that, I'll try to use the informations I got from the EDA and external public datasets to build a model for poverty level estimation.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
from datetime import datetime, timedelta


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# # 1. Exploratory Data Analysis
# <a id="EDA"></a>
# *** 
# In this part, the goal is to understand the data that was given to us through plots and statistics, draw multiple conclusions and see how we can use those results to build the features that will be needed for our machine learning model. 
# 
# Let's first see what this data is about.

# ## 1.1 Data description
# <a id="description"></a>
# *** 
# Let's load the 4 csv files we have and start by analyzing the biggest one : kiva loans.

# In[2]:


df_kiva_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_loc = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
df_themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
df_mpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")

df_kiva_loans.head(5)


# Before going any further, let's take a look at the missing values so that we don't encounter any bad surprises along the way.

# In[3]:


msno.matrix(df_kiva_loans);


# Seems that this dataset is pretty clean ! the *tags* column got a lot of missing values but that's not a big deal. The *funded_time* has little less than 10% of missing values, that's quite a few but since we have more than 600 000 rows, we can drop the missing rows if we need to and we'll still get some telling results !  
# Let's get some global information about each of our columns.

# In[4]:


df_kiva_loans.describe(include = 'all')


# Plenty of useful informations in this summary :
# * There are exactly 87 countries where people borrowed money according to this snapshot.
# * There are 11298 genders in this dataset ! That's obviously impossible so we'll see later on why we have this value. 
# * The funding mean over the world is 786 dollars while the funding median is 450 dollars.
# * More importantly : there's only 1298 different dates on which loans were posted. If we calculate the ratio, **it means that there's more than 500 loans posted per day on Kiva** and that's just a snapshot (a sample of their entire data). This gives you a clear idea about how important this crowdsourcing platform is and what impact it has.

# ## 1.2. Kiva users 
# <a id="users"></a>
# *** 
# In this part we will focus on the basic demographic properties of people who use Kiva to ask for loans : Where do they live ? what's their gender ? Their age would be a nice property but we don't have direct access to that for now, we'll get to that later.
# 
# Let's first start with their countries : as seen above, the data contains 671205 rows. In order to have the most (statistically) significant results going further, I'll only keep the countries that represent at least 0.5% of Kiva's community. 

# In[5]:


countries = df_kiva_loans['country'].value_counts()[df_kiva_loans['country'].value_counts()>3400]
list_countries = list(countries.index) #this is the list of countries that will be most used.


# In[6]:


plt.figure(figsize=(13,8))
sns.barplot(y=countries.index, x=countries.values, alpha=0.6)
plt.title("Number of borrowers per country", fontsize=16)
plt.xlabel("Nb of borrowers", fontsize=16)
plt.ylabel("Countries", fontsize=16)
plt.show();


# Philippines is the country with most borrowers with approximately 25% of all users being philippinians. Elliott Collins, from the Kiva team, explained that this is due to the fact that a couple of Philippine field partners tend to make smaller short-term loans (popular low-risk loans + fast turnover rate). 
# 
# 
# We also notice that several african countries are in the list such as *Kenya, Mali, Nigeria, Ghana ...* and no european union country at all !     
# For me, the most surprising was actually the presence of the US in this list, as it doesn't have the same poverty rate as the other countries but it turns out it's indeed a specific case, **I'll explain that in 1.4**.
# 
# Let's now move on to the genders.

# In[7]:


df_kiva_loans['borrower_genders']=[elem if elem in ['female','male'] else 'group' for elem in df_kiva_loans['borrower_genders'] ]
#to replace values such as "woman, woman, woman, man"

borrowers = df_kiva_loans['borrower_genders'].value_counts()
labels = (np.array(borrowers.index))
values = (np.array((borrowers / borrowers.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=True)

layout = go.Layout(
    title="Borrowers' genders"
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Borrowers_genders")


# In many loans (16.4% as you can see), the borrower is not actually a single person but a group of people that have a project, here's an [example](https://www.kiva.org/lend/1440912). In the dataset, they're listed as 'female, female, female' or 'male, female' ... I decided to use the label *mixed group* to those borrowers on the pie chart above.
# 
# You can see that most borrowers are female, I didn't expect that and it was actually a great surprise. This means that **women are using Kiva to get funded and work on their projects in countries (most of them are third world countries) where breaking in as a woman is still extremely difficult.**

# ## 1.3 Activities, sectors and funding amounts
# ***
# 
# Now let's take a peek at what people are needing loans for and what's the amounts they're asking for. Let's start with the sectors. There were 15 unique sectors in the summary we've seen above, let's see how each of them fare.

# In[8]:


plt.figure(figsize=(13,8))
sectors = df_kiva_loans['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.xlabel('Number of loans', fontsize=16)
plt.ylabel("Sectors", fontsize=16)
plt.title("Number of loans per sector")
plt.show();


# **The most dominant sector is Agriculture**, that's not surprising given the list of countries that heavily use Kavi. A fast research for Kenya for example shows that all the top page is about agriculture loans, here's a sample of what you would find:  *buy quality seeds and fertilizers to use in farm*, *buy seeds to start a horticulture farming business so as a single mom*, *Purchase hybrid maize seed and fertilizer* ... Food sector occupies an important part too because many people are looking to buy fish, vegetables and stocks for their businesses to keep running.  
# It's important to note that *Personal Use* occupy a significant part too, this means there are people who don't use Kavi to get a hand with their work but because they are highly in need.
# 
# Let's see the more detailed version and do a countplot for **activities**,

# In[9]:


plt.figure(figsize=(15,10))
activities = df_kiva_loans['activity'].value_counts().head(50)
sns.barplot(y=activities.index, x=activities.values, alpha=0.6)
plt.ylabel("Activity", fontsize=16)
plt.xlabel('Number of loans', fontsize=16)
plt.title("Number of loans per activy", fontsize=16)
plt.show();


# This plot is only a confirmation of the previous one, activities related to agriculture come in the top : *Farming, Food production, pigs ...*. All in all, we notice that none of the activities belong to the world of 'sophisticated'. Everything is about basic daily needs or small businesses like buying and reselling clothes ...
# 
# How about the money those people need to pursue their goals ?

# In[10]:


plt.figure(figsize=(12,8))
sns.distplot(df_kiva_loans['loan_amount'])
plt.ylabel("density estimate", fontsize=16)
plt.xlabel('loan amount', fontsize=16)
plt.title("KDE of loan amount", fontsize=16)
plt.show();


# Some outliers are clearly skewing the distribution and the plot doesn't give much information in this form : We need to **truncate the data**, how do we do that ? 
# 
# We'll use a basic yet really powerful rule : the **68–95–99.7 rule**. This rule states that for a normal distribution :
# * 68.27% of the values $ \in [\mu - \sigma , \mu + \sigma]$
# * 95.45% of the values $ \in [\mu - 2\sigma , \mu + 2\sigma]$
# * 99.7% of the values $ \in [\mu - 3\sigma , \mu + 3\sigma]$     
# where $\mu$ and $\sigma$ are the mean and standard deviation of the normal distribution.
# 
# Here it's true that the distribution isn't necessarily normal but for a shape like the one we've got, we'll see that applying the third filter will **improve our results radically**.
# 

# In[11]:


temp = df_kiva_loans['loan_amount']

plt.figure(figsize=(12,8))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())]);
plt.ylabel("density estimate", fontsize=16)
plt.xlabel('loan amount', fontsize=16)
plt.title("KDE of loan amount (outliers removed)", fontsize=16)
plt.show();


# Well, that's clearly a lot better !    
# * Most of the loans are between 100\$ and 600\$ with a first peak at 300\$.
# * The amount is naturally decreasing but we notice that we have a clear second peak at 1000\$. This suggets that there may be a specific class of projects that are more 'sophisticated' and get funded from time to time, interesting.
# 
# How about some specification ? We have information how the loan amount in general, let's see now sector-wise : 

# In[12]:


plt.figure(figsize=(15,8))
sns.boxplot(x='loan_amount', y="sector", data=df_kiva_loans);
plt.xlabel("Value of loan", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.title("Sectors loans' amounts boxplots", fontsize=16)
plt.show();


# As you can see, for any sector, we have outlier loans. For example it seems someone asked for a 100k loan for an agriculture project. There are also many 20k, 50k ... loans. But as seen earlier, the mean amount in general is around 500 dollars so we have to get rid of those outliers to obtain better boxplots.  
# First, let's see the median loan amount for each sector, this would give an idea about the value to use as a treshold.

# In[13]:


round(df_kiva_loans.groupby(['sector'])['loan_amount'].median(),2)


# The highest median corresponds to 950 dollars for the sector *Wholesale*. Basically, using a treshold that doubles this value (so 2000 dollars) is more than safe and we wouldn't be using much information.

# In[14]:


temp = df_kiva_loans[df_kiva_loans['loan_amount']<2000]
plt.figure(figsize=(15,8))
sns.boxplot(x='loan_amount', y="sector", data=temp)
plt.xlabel("Value of loan", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.title("Sectors loans' amounts boxplots", fontsize=16)
plt.show();


# This is obviously better visually ! We can also draw some important conclusions. The median and the inter-quartile for 'Personal Use' are much lower than for any other sector. Education and health are a bit higher compared to other 'standard sectors'.    
# Why such information can be considered important ? Well keep in mind throughout this notebook that our final goal is to estimate poverty levels.   
# Since the amount of loans aren't the same for all sectors (distribution-wise), it may mean that for example borrowers with Personnal Use are poorer and need a small amout for critical needs. This is not necessarily true but it's indeed an hypothesis we can take advantage of going further.

# ## 1.4. Waiting time for funds
# <a id="dates"></a>
# *** 
# 
# So far we got to see where Kiva is most popular, the nature of activities borrowers need the money for and how much money they usually ask for, great !    
# 
# An interesting question now would also be : **how long do they actually have to wait for funding ?** As we've seen before, some people on the plateform are asking for loans for critical needs and can't afford to wait for months to buy groceries or have a shelter. Fortunately, we've got two columns that will help us in our investigation : 
# * funded_time : corresponds to the date + exact hour **when then funding was completed.**
# * posed_time : corresponds to the date + exact hour **when the post appeared on the website.**
# 
# We've also seen before that we have some missing values for 'funded_time' so we'll drop those rows, get the columns in the correct date format and then calculate the difference between them.

# In[15]:


loans_dates = df_kiva_loans.dropna(subset=['disbursed_time', 'funded_time'], how='any', inplace=False)

dates = ['posted_time','disbursed_time','funded_time']
loans_dates[dates] = loans_dates[dates].applymap(lambda x : x.split('+')[0])

loans_dates[dates]=loans_dates[dates].apply(pd.to_datetime)
loans_dates['time_funding']=loans_dates['funded_time']-loans_dates['posted_time']
loans_dates['time_funding'] = loans_dates['time_funding'] / timedelta(days=1) 
#this last line gives us the value for waiting time in days and float format,
# for example: 3 days 12 hours = 3.5


# Now first thing first, we'll plot the this difference that we called *time_funding*. To avoid any outliers, we'll apply the same rule for normal distribution as before.

# In[16]:


temp = loans_dates['time_funding']

plt.figure(figsize=(12,8))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())]);


# I was really surprised when I got this plot (and happy too), you'll rarely find a histogram where the distribution fits in this smoothly !   
# On top of that, getting two peaks was the icing on the cake, it makes perfect sense ! **We've seen above that there are two peaks for loans amounts, at 300\$ and 1000\$, we're basically saying that for the first kind of loan you would be waiting 7 days and for the second kind a little more than 30 days !   **
# This gives us a great intuition about how those loans work going forward.
# 
# Let's be more specific and check for both loan amounts and waiting time country-wise :   
# We'll build two new DataFrames using the groupby function and we'll aggregate using the median : what we'll get is the median loan amount (respectively waiting time) for each country.

# In[17]:


df_ctime = round(loans_dates.groupby(['country'])['time_funding'].median(),2)
df_camount = round(df_kiva_loans.groupby(['country'])['loan_amount'].median(),2)


# In[18]:


df_camount = df_camount[df_camount.index.isin(list_countries)].sort_values()
df_ctime = df_ctime[df_ctime.index.isin(list_countries)].sort_values()

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.barplot(y=df_camount.index, x=df_camount.values, alpha=0.6, ax=ax[0])
ax[0].set_title("Medians of funding amounts per loan country wise ")
ax[0].set_xlabel('Amount in dollars')
ax[0].set_ylabel("Country")

sns.barplot(y=df_ctime.index, x=df_ctime.values, alpha=0.6,ax=ax[1])
ax[1].set_title("Medians of waiting days per loan to be funded country wise  ")
ax[1].set_xlabel('Number of days')
ax[1].set_ylabel("")

plt.tight_layout()
plt.show();


# **Left plot**    
# We notice that in most countries, funded loans don't usually exceed 1000\$. For Philippines, Kenya and El Salvador (the three most present countries as seen above), the medians of fund per loan are respectively : 275.00\$, 325.00\$ and 550.00\$ .
# 
# The funded amount for US-based loans seem to be a lot higher than for other countries. I dug deeper and looked in Kiva's website. **It appears that there's a special section called 'Kiva U.S.' which goal is to actually fund small businesses for *financially excluded and socially impactful borrowers*.  ** 
# Example of such businesses : Expanding donut shop in Detroit (10k\$),  Purchasing equipment and paying for services used to properly professionally train basketball kids ... You can see more of that in [here](https://www.kiva.org/lend/kiva-u-s).    
# This explains what we've been seeing earliers : the fact that the US is among the countries, the big amount of loan, the two-peaks plots ...
# 
# **Right plot**   
# The results in this one aren't that intuitive. 
# * Paraguay is the second fastest country when it comes to how much to wait for a loan to be funded while it was also the country with the second highest amount per loan in the plot above !  
# * The US loans take the most time to get funded and that's only natural since their amount of loans are much higher than the other countries.
# * Most of African countries are in the first half of the plot.

# ## 1.5. Amount of loan vs Repayment time
# <a id="ratio"></a>
# *** 
# 
# We have information about months needed for borrowers to repay their loans. Simply ploting the average / median repayment time per country can give some insights but what's even more important is **the ratio of the amount of loan to repayment time**. Indeed, let's say in country A, loans are repayed after 12 months in average and in country B after 15 months in average; if you stop here, you can just say *people in country B need more time to repay their loans compared to people in country A*. Now let's say the average amount of loans in country A is 500\$ while it's 800\$ in country B, then it means that *people in country A repay 41.66\$ per month while people in country B repay 51.33\$ per month* !   
# This ratio gives you an idea about **how much people in a given country can afford to repay per month**.

# In[19]:


df_repay = round(df_kiva_loans.groupby(['country'])['term_in_months'].median(),2)
df_repay = df_repay[df_repay.index.isin(list_countries)].sort_values()

df_kiva_loans['ratio_amount_duration']= df_kiva_loans['funded_amount']/df_kiva_loans['term_in_months'] 
temp = round(df_kiva_loans.groupby('country')['ratio_amount_duration'].median(),2)
temp = temp[temp.index.isin(list_countries)].sort_values()

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.barplot(y=temp.index, x=temp.values, alpha=0.6, ax=ax[0])
ax[0].set_title("Ratio of amount of loan to repayment period per country", fontsize=16)
ax[0].set_xlabel("Ratio value", fontsize=16)
ax[0].set_ylabel("Country", fontsize=16)

sns.barplot(y=df_repay.index, x=df_repay.values, alpha=0.6,ax=ax[1])
ax[1].set_title("Medians of number of months per repayment, per country",fontsize=16)
ax[1].set_xlabel('Number of months', fontsize=16)
ax[1].set_ylabel("")

plt.tight_layout()
plt.show();


# From these 2 plots, we notice that ** Nigeria** is the country with the smallest ratio (around 8 dollars per month) while Paraguay has a surprinsinly high one. Also, it seems that on average, loans are being repayed after 1 year except for India where they take much longer.    
# In the second part of this kernel, we'll see weither the fact that a country/region repays a loan rapidly (in other words if the ratio of a country is high) is correlated with poverty or not.

# ## 1.6. Lenders community
# <a id="lenders"></a>
# *** 
# We said that we would talk about Kiva users, that include lenders too ! It's true that our main focus here remains the borrowers and their critical need but it's still nice to know more about who uses Kiva in the most broad way and also get an idea about **what drives people to actually fund projects ?   **
# Thanks to additional datasets, we got freefrom text data about the lenders and their reasons for funding, let's find about that.

# In[20]:


lenders = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')
lenders.head()


# Seems like this dataset is filled with missing values :). We'll still be able to retrieve some informations, let's start by checking which country has most lenders.

# In[21]:


lender_countries = lenders.groupby(['country_code']).count()[['permanent_name']].reset_index()
lender_countries.columns = ['country_code', 'Number of Lenders']
lender_countries.sort_values(by='Number of Lenders', ascending=False,inplace=True)
lender_countries.head(7)


# Two things here :    
# * The US is, by far, the country with most lenders. It has approximately 9 times more lenders than any other country. If we want to plot a map or a barplot with this information, we have two choices : either we leave out the US or we use a logarithmic scale, which means we'll apply $ ln(1+x) $ for each $x$ in the column *Number of Lenders*. The logarithmic scale allows us to respond to skewness towards large values when one or more points are much larger than the bulk of the data (here, the US).
# * We don't have a column with country names so we'll need to use another dataset to get those and plot a map.
# 
# Here's another additional dataset that contains poverty informations about each country. For the time being, we'll only use the column *country_name* to merge it with our previous dataset.

# In[22]:


countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
countries_data.head()


# In[23]:


countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
lender_countries = pd.merge(lender_countries, countries_data[['country_name','country_code']],
                            how='inner', on='country_code')

data = [dict(
        type='choropleth',
        locations=lender_countries['country_name'],
        locationmode='country names',
        z=np.log10(lender_countries['Number of Lenders']+1),
        colorscale='Viridis',
        reversescale=False,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Lenders'),
    )]
layout = dict(
    title = 'Lenders per country in a logarithmic scale ',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='lenders-map')


# The US have the largest community of lenders and it is followed by Canada and Australia. On the other hand, the African continent seems to have the lowest number of funders which is to be expected, since it's also the region with highest poverty rates and funding needs.
# 
# So now that we know more about lenders location, let's analyze the textual freeform column *loan_because* and construct a wordcloud to get an insight about their motives for funding proejcts on Kiva.

# In[24]:


import matplotlib as mpl 
from wordcloud import WordCloud, STOPWORDS
import imageio

heart_mask = imageio.imread('../input/poverty-indicators/heart_msk.jpg') #because displaying this wordcloud as a heart seems just about right :)

mpl.rcParams['figure.figsize']=(12.0,8.0)    #(6.0,4.0)
mpl.rcParams['font.size']=10                #10 

more_stopwords = {'org', 'default', 'aspx', 'stratfordrec','nhttp','Hi','also','now','much','username'}
STOPWORDS = STOPWORDS.union(more_stopwords)

lenders_reason = lenders[~pd.isnull(lenders['loan_because'])][['loan_because']]
lenders_reason_string = " ".join(lenders_reason.loan_because.values)

wordcloud = WordCloud(
                      stopwords=STOPWORDS,
                      background_color='white',
                      width=3200, 
                      height=2000,
                      mask=heart_mask
            ).generate(lenders_reason_string)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('./reason_wordcloud.png', dpi=900)
plt.show()


# Lenders' answers are heartwarming :) Most reasons contain *help people / others* or *want to help*. We also find that it's the *right thing* (to do), it helps *less fortunate* and makes the world a *better place*.  
# Kiva provides a platform for people who need help to fund their projects but it also provides a platform for people who want to make a difference by helping others and maybe changing their lives !

# # 2. Welfare estimation
# <a id="prediction"></a>
# *** 
# In this part we'll delvo into what's this competition is really about : **welfare and poverty estimation.  ** 
# As a lender, you basically have two criterias when you're looking for a loan to fund : the loan description and how much the borrower does really need that loan. For the second, Kiva's trying to have as granular poverty estimates as possible through this competition.
# 
# In this part, I'll be talking about what poverty really means and how it is measures by economists. I'll also start with a country-level model as an example to what will be said.
# 
# Let's start.

#     No society can surely be flourishing and happy, of which by far the greater part of the numbers are poor and miserable. - Adam Smith, 1776       
# 

# ## 2.1 What's poverty ?
# <a id="definition"></a>
# *** 
# The World Bank defines poverty in terms of **income**. The bank defines extreme poverty as living on less than US\$1.90 per day (PPP), and moderate poverty as less than \$3.10 a day.  
# P.S: In this part, we'll say (PPP) a lot. It refers to Purchasing Power Parity. I have a notebook that is entirely dedicated to PPP and if interested and want to know more about how it works, you can check it [here](https://www.kaggle.com/mhajabri/salary-and-purchasing-power-parity).  
# Over the past half century, significant improvements have been made and still, extreme poverty remains widespread in the developing countries. Indeed, an estimated **1.374 billion people live on less than  1.25 \$ per day** (at 2005 U.S. PPP) and around **2.6 billion (which is basically 40% of the worlds's population !!) live on less than \$ 2 per day**. Those impoverished people suffer from : undernutrition / poor health, live in environmentally degraded areas, have little literacy ...
# 
# As you can see, poverty seems to be defined exactly by the way it's actually measured, but what's wrong with that definition ? **In developing countries, many of the poor work in the informal sector and lack verifiable income records => income data isn't reliable**. Suppose you're the government and you have a specific program that benefits the poorest or you're Kiva and you want to know who's in the most critical condition, then relying on income based poverty measures in developing countries will be misleading and using unreliable information to identify eligible households can result in funds being diverted to richer households and leave fewer resources for the program’s intended beneficiaries. We need another way of measuring poverty. 

# ## 2.2 Multidimensional Poverty Index
# <a id="mpi"></a>
# *** 
# 
# Well one day the UNDP (United Nations Development Programme) came and said well *salary* is only one **dimension** that can describe poverty levels but it's far from the only indicator. Indeed, if you visit someone's house and take a look at how it is and what it has, it gives an intuition. Based on that, the UNDP came up with the **Multidimensional Poverty Index **, an index that has **3 dimensions and a total of 10 factors **assessing poverty : 
# * **Health **: Child Mortality - Nutrition
# * **Education** : Years of schooling - School attendance
# * **Living Standards** : Cooking fuel - Toilet - Water - Electricity - Floor - Assets
# 
# How is the MPI calculated ? Health's and Education's indicators (there are 4 in total) are weighted equally at 1/6. Living standards' indicators are weighted equally at 1/18. The sum of the weights $2*1/6 + 2*1/6 + 6*1/18 = 1$. Going from here, **a person is considered poor if they are deprived in at least a third of the weighted indicators.** 
# Example : Given a household with no electricity, bad sanitation, no member with more than 6 years of schooling and no access to safe drinking water, the MPI score would be : 
# $$ 1/18 + 1/18 + 1/6 + 1/18 = 1/3$$
# So this household is deprived in at least a third of the weighted indicators (MPI > 0.33) and is considered MPI-poor.
# 
# Kiva actually included MPI data so let's get a look at it :

# In[25]:


df_mpi.head(7)


# This dataset gives the MPI of different regions for each country, to have a broad view, let's use a groupby *country* and take the average MPI and plot that in a map.

# In[26]:


mpi_country = df_mpi.groupby('country')['MPI'].mean().reset_index()

data = [dict(
        type='choropleth',
        locations=mpi_country['country'],
        locationmode='country names',
        z=mpi_country['MPI'],
        colorscale='Greens',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
    )]

layout = dict(
    title = 'Average MPI per country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='mpi-map')


# As you can notice, the data provides the MPI for the African continent essentially. That shouldn't surpise you, as we said before, for developed countries, income data is actually reliable and good enough to measure poverty so we don't need to send researchers on the field to run surveys and get the necessary data for the MPI. That's why you'll find more data about MPI measurements in developing / poor countries.   
# 
# Now that we know what's MPI is about, we can use an Oxford Poverty & Human Development Initiative dataset that's been uploaded here to analyze the **Poverty headcount ratio (% of population listed as poor) in the countries that are using kiva**.

# In[27]:


df_mpi_oxford = pd.read_csv('../input/mpi/MPI_subnational.csv')
temp = df_mpi_oxford.groupby('Country')['Headcount Ratio Regional'].mean().reset_index()

temp = temp[temp.Country.isin(list_countries)].sort_values(by="Headcount Ratio Regional", ascending = False)

plt.figure(figsize=(15,10))
sns.barplot(y=temp.Country, x=temp['Headcount Ratio Regional'], alpha=0.6)
plt.ylabel("Country", fontsize=16)
plt.xlabel('Headcount Ratio National', fontsize=16)
plt.title("Headcount Ratio National per Country", fontsize=16)
plt.show();


# First of all, the dataset provides us with regional headcount ratios; as you can see in the code above, I consider that the national headcount ratio is the mean of all the regional ratios. That's not perfectly true. A more precise formula would weight the ratios by the % of total population living in that region. For example, if a region in a given country has 40% of total habitants, then it would count 10 times more than a region that has 4% of total habitants.   
# So those results are far from perfect, many countries may have a higher ratio, other would have a smaller ratio but this gives us a first glance at least !    
# 
# As you can notice, African countries come on top, with the top 3 having more than 70% of their total population listed as poor ! Kenya, which is the most present country in Kiva, has more than 40% of its population listed as poor while Philippines' is 11%.

# ## 2.3 Proxy Means Test
# <a id="pmt"></a>
# *** 
# 
# ### Definition 
# 
# Eventually, the World Bank had to come up with it's own way of estimating consumption when income data are either unreliable or unavailable. The results are used for **mean-testing** as the name suggests, where a **mean-test is a determination of whether an individual or family is eligible for government assistance, based upon whether the individual or family possesses the means to do without that help.** Sounds familiar ? Yeah, we want to determine if a person is "eligible" for funding here ! So let's know more about this technique.
# 
# For the MPI, we have exactly 10 indicators and those same indicators are used whenever / wherever you're conducting your research. For PMT, there's two key differences : 
# 1. There is no longer an exact number of  **proxies**. You choose them based on household surveys and you can actually come up with what you think will be the most effective when it comes to estimating poverty.
# 2. You don't have equal weights in PMT. You use statistical methods (mainly regressions) with the dependant variable being either consumption-related or income related. Then regression will give you the $\beta$s (weights). Example : Ownership of a house could have a weight of 200, number of persons per living room a weight of (-50) ....
# 
# One advantage of PMT is that** it is not required that a single model (same proxies or same weights) is used for the entire country ** and so this gives us a better model overall.

# ### Performing a PMT    
# 
# For a start, I decided actually run a very general Proxy Means Test where I consider the most frequent countries in Kiva !   
# #### **Step 1**     
# First we have to decide what proxies to use. Mine fall under 4 categories : 
# 1. **Location** : 
#     * % of population living in rural Area. 
# 2. **Housing** :
#     * % of population with access to improved water sources
#     * % of population with access to electricity
#     * % of population with access to improved sanitation facilities 
# 3. **Family** : 
#     * Average size of the household
#     * Average number of children per family
#     * Sex of the head of household : % of families with a male head of household    
#     * Level of education attained
#     * Employment rate : Employment to population (age >15) ratio.
#     * Agriculture employment : % of total workers who have an agriculture-related job
# 4. **Ownership** : 
#     * Ownership of a telephone : Mobile cellular per capita
#  
# #### **Step 2**     
# Assembling the data ! Let's get back to coding :)

# In[28]:


#load all needed data
df_household =pd.read_csv('../input/poverty-indicators/household_size.csv',sep=';')
df_indicators = pd.read_csv('../input/poverty-indicators/indicators.csv',
                            sep=';', encoding='latin1', decimal=',').rename(columns={'country_name': 'country'})
df_education= pd.read_csv('../input/additional-kiva-snapshot/country_stats.csv')[['country_name','mean_years_of_schooling']].rename(columns={'country_name': 'country'})
df_mobile = pd.read_csv('../input/poverty-indicators/mobile_ownership.csv',sep=';',encoding='latin1',decimal=',')
df_mobile['mobile_per_100capita']=df_mobile['mobile_per_100capita'].astype('float')

#merge data for most frequent countries
temp = pd.merge(df_indicators, df_household, how='right', on='country')
temp = pd.merge(temp, df_mobile, how='left', on='country')
indicators = pd.merge(temp, df_education, how='left', on='country').round(2)

indicators


# Okay, so we notice that there is missing data about Palestine and Kyrgyzstan. I searched for that to fill those two rows instead of deleting them.

# In[29]:


palestine_data = ['Palestine','PLS', 4550000, 24.52, 91, 89, 60, 73, 7.4, 0.90, 5.9, 3.1, 97.8, 8]
kyrgyzstan_data = ['Kyrgyzstan','KGZ', 6082700, 64.15, 90, 93.3, 99.8, 58.3, 29.20, 0.73, 4.2, 2.1, 123.7, 10.80]

indicators.loc[35]=palestine_data
indicators.loc[36]=kyrgyzstan_data


# Now before going further with PMT, let's play a bit with this data.  
# 
# Just above, I've calculated average MPIs per country to plot the map. Let's merge the MPI information with those indicators, plot a correlation matrix and see what we'll have.

# In[30]:


indicators_mpi = pd.merge(indicators,mpi_country, how='inner', on='country').drop(['country_code','population'],axis=1)

corre = indicators_mpi.corr()

mask1 = np.zeros_like(corre, dtype=np.bool)
mask1[np.triu_indices_from(mask1)] = True

f, axs = plt.subplots(figsize=(10, 8))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corre, mask=mask1, cmap=cmap, vmax=.3, center=0, square=False, linewidths=.5, cbar_kws={"shrink": .5});


# The results here are **comforting**. According to the matrix, the MPI is significantly related to every indicator except the sex of household headship. Fair enough.    More importantly, notice that the most correlated indicators to the MPI, which also happen to have really high scores (0.8 approximately) are the following : **access to water / sanitation / electricity and years of schooling**. Now if you go back to how the MPI is defined, you'll find among the 10 indicators : Years of schooling - Toilet - Water - Electricity !!
# 
# #### **Step 3**     
# Back to our Proxy Means Test now, let's do some regression now. The dependant variable here will be **Household final consumption expenditure per capita (PPP)**.

# In[31]:


consumption = pd.read_csv('../input/poverty-indicators/consumption.csv',sep=";", decimal=',',encoding='latin1')

df = pd.merge(indicators, consumption[['country','consumption_capita']] , how='left', on='country').round(2).dropna()
df.rename(columns={'rural_population_%': 'rural_ratio','access_water_%':'ratio_wateraccess', 'access_electricity_%':'ratio_electricityaccess',
                  'employment_%':'ratio_employment','agriculture_employment_%':'ratio_agriculture',
                  'access_sanitation_%':'ratio_sanitation'},inplace=True)

from statsmodels.formula.api  import ols
model = ols(formula = 'consumption_capita ~ rural_ratio+ratio_wateraccess+ratio_electricityaccess+ratio_sanitation+ratio_employment+ratio_agriculture+            male_headship+average_household_size+avg_children_nb+mean_years_of_schooling+mobile_per_100capita',
          data = df).fit()
print(model.summary())


# Back to reality where first models don't actually give you the best results right away :( The p-values for the coefficients are too high (which translates to not statistically significant) and it seems that we also have multicolinearity issues.   
# Let's see what's happening here :   
# * The data is very **heterogeneous**. Indeed, we're dealing with country-level data here and poverty in different countries might as well have different causes. As explained before, PMT is really intended to be done on the data of a specific country / region.
# * We're using **11 features to perform a linear regression on less than 40 points**. That can't be good ! We have way too much features for such a small sample.
# 
# Let's try a simplified model with less features.

# In[32]:


model2 = ols(formula = 'consumption_capita ~ rural_ratio+ratio_sanitation+mobile_per_100capita+average_household_size ',
          data = df).fit()
print(model2.summary())


# This is much better already. You can see that the **p-values are much lower**, the overall **F-statistic is lower too**, the BIC score is lower (which means better) and the multicolinearity is no longer an issue.   
# You would think that this also translates in a less accurate model (because we have less information) but that's not true actually ! The $R^2$ and adjusted $R^2$ improved as well and are higher for the second model. Well as the saying goes, *sometimes less is more* !
# 
# So far we've only used external data to assess welfare, but what about Kiva's data actually ? Can't we use some properties of the loans or borrowers and put them into our model ? Well actually we can ! We've already talked up above about a couple things : 
# 1. Time for a project to get entirely funded (date_funded - date_posted)
# 2. The median amount of loan per country, median duration to repay the loan and the ratio of the 1st to the 2nd.

# In[33]:


df_ratio = round(df_kiva_loans.groupby('country')['ratio_amount_duration'].median(),2)
df_ctime = round(loans_dates.groupby(['country'])['time_funding'].median(),2)
df_camount = round(df_kiva_loans.groupby(['country'])['loan_amount'].median(),2)
df_repay = round(df_kiva_loans.groupby(['country'])['term_in_months'].median(),2)

kiva_indic = pd.concat([df_ratio, df_ctime, df_camount, df_repay], axis=1, join='inner').reset_index()
indicators_mpi_kiva = pd.merge(indicators_mpi,kiva_indic,how='left',on='country')

indicators_mpi_kiva = indicators_mpi_kiva[['MPI','ratio_amount_duration','time_funding','loan_amount','term_in_months']]
indicators_mpi_kiva.corr()


# The MPI doesn't seem related at all to the ratio, which is quite surprising to me. Nonetheless, the correlation between the MPI and funding time and repayment time isn't negligible which means those two can prove to be good resources going further.

# # **Work in progress, stay tuned**
