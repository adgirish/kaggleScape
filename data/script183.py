
# coding: utf-8

# ![](http://sabtrends.com/wp-content/uploads/2017/05/funding-word-money-100-dollar-bill-currency-ball.jpg)

# # More To Come. Stay Tuned. !! 
#   If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# **This notebook will always be a work in progress**. Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!. **If you like it or it helps you , you can upvote and/or leave a comment :).**

# ## This notebook explores the analysis of indian startup funding and basically gives answer of following questions :-
# 1.  How does the funding ecosystem change with time ?(Number of funding per month)
# 2. How much funds does startups generally get in India ?(maximum funding, minimum funding , average funding and number of fundings)
# 3. Which industries are favored by investors for funding ? (OR) Which type of companies got more easily funding ?
# 4. Do cities play a major role in funding ? (OR) Which city has maximum startups ?
# 5. Who is the important investors in the Indian Ecosystem?
# 6. What are different types of funding for startups ?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
import seaborn as sns
color = sns.color_palette()
import squarify

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None
#pd.options.display.max_columns = 999


# ## Obtaining the data

# In[ ]:


funding_data = pd.read_csv("../input/startup_funding.csv")
funding_data.head()


# ### Column names of the table

# In[ ]:


funding_data.columns


# In[ ]:


print("Size of data(Rows, Columns)",funding_data.shape)


# **Lets see How much data is missing**

# In[ ]:


# missing data 
total = funding_data.isnull().sum().sort_values(ascending = False)
percent = ((funding_data.isnull().sum()/funding_data.isnull().count())*100).sort_values(ascending = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent %'])
missing_data.head()


# Deleting "**Remarks**" from table and displaying remaining data

# In[ ]:


del funding_data["Remarks"]
funding_data.head()


# Now convert the string **"AmountInUSD" **into numeric

# In[ ]:


funding_data["AmountInUSD"] = funding_data["AmountInUSD"].apply(lambda x: float(str(x).replace(",","")))
funding_data["AmountInUSD"] = pd.to_numeric(funding_data["AmountInUSD"])
funding_data.head()


# ## Ques 1:  How does the funding ecosystem change with time ?(Number of funding per month)

# In[ ]:


### Some more fixes in the data format. Will try to fix in the input file in next version #
funding_data['Date'][funding_data['Date']=='12/05.2015'] = '12/05/2015'
funding_data['Date'][funding_data['Date']=='13/04.2015'] = '13/04/2015'
funding_data['Date'][funding_data['Date']=='15/01.2015'] = '15/01/2015'
funding_data['Date'][funding_data['Date']=='22/01//2015'] = '22/01/2015'
funding_data["yearmonth"] = (pd.to_datetime(funding_data['Date'],format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(funding_data['Date'],format='%d/%m/%Y').dt.month)
temp = funding_data['yearmonth'].value_counts().sort_values(ascending = False).head(10)
print("Number of funding per month in decreasing order(Top 10)\n",temp)
year_month = funding_data['yearmonth'].value_counts()
plt.figure(figsize=(15,8))
sns.barplot(year_month.index, year_month.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Year-Month of transaction', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Year-Month Distribution", fontsize=16)
plt.show()


# As we can see that startups got more funding in **January 2016**(Total funding in January 2016 are 104). Above visualization shows how funding
# varies from one month to another.

# ## Ques 2 : How much funds does startups generally get in India ?(maximum funding, minimum funding , average funding and number of fundings)
# 

# In[ ]:


print("Maximum funding to a Startups is : ",funding_data["AmountInUSD"].dropna().sort_values().max())


# In[ ]:


funding_data[funding_data.AmountInUSD == 1400000000.0]


# In[ ]:


funding_data[funding_data.StartupName == 'Paytm']


# As we can see** Paytm** and **Flipkart** got maximum funding of  1400000000 USD. Now lats see least funding.

# In[ ]:


print("Minimum funding to a Startups is : ",funding_data["AmountInUSD"].dropna().sort_values().min())


# In[ ]:


funding_data[funding_data.AmountInUSD == 16000.0]


# Now as we can see **Hostel Dunia, Play your sport, Yo Grad, Enabli and CBS** are least funded Startups i.e, 16000 USD

# In[ ]:


print("On Average indian startups got funding of : ",funding_data["AmountInUSD"].dropna().sort_values().mean())


# On an Average indian startups got funding of :  12031073.099016393

# In[ ]:


print("Total startups funded : ", len(funding_data["StartupName"].unique()))
print(funding_data["StartupName"].value_counts().head(10))
startupname = funding_data['StartupName'].value_counts().head(20)
plt.figure(figsize=(15,8))
sns.barplot(startupname.index, startupname.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Startup Name', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Number of funding a startup got", fontsize=16)
plt.show()


# As we can see that **Swiggy** got maximum number of fundings(Total funding = 7) and total there are 2001 indian startups funded from January 2015 to August 2017. The above visulization is only for Top 20 startups.

# ## Ques 3 :  Which industries are favored by investors for funding ? (OR) Which type of companies got more easily funding ?

# In[ ]:


industry = funding_data['IndustryVertical'].value_counts().head(10)
print(industry)
plt.figure(figsize=(15,8))
sns.barplot(industry.index, industry.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Industry vertical of startups', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Industry vertical of startups with number of funding", fontsize=16)
plt.show()


# If we see Above **"Consumer Internet" **got maximum number of funding = 772 followed by technology and E-Commerce.

# In[ ]:


industry = funding_data['SubVertical'].value_counts().head(10)
print(industry)
plt.figure(figsize=(15,8))
sns.barplot(industry.index, industry.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Subvertical of startups', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Subvertical of startups with number of funding", fontsize=16)
plt.show()


# In Subcategores, **"Online Phamacy"** got maximim number of fundings.

# ## Ques 4 : Do cities play a major role in funding ? (OR) Which city has maximum startups ?
# 

# In[ ]:


city = funding_data['CityLocation'].value_counts().head(10)
print(city)
plt.figure(figsize=(15,8))
sns.barplot(city.index, city.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('city location of startups', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("city location of startups with number of funding", fontsize=16)
plt.show()


# **Distribution of startups across Top different cities**

# In[ ]:


plt.figure(figsize=(15,8))
count = funding_data['CityLocation'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of Startups across Top cities')


# We can see **Bangalore** attracts more number of investotrs followed by **Mumbai** and **New** **Delhi**

# ## Ques 5 : Who is the important investors in the Indian Ecosystem?
# 

# In[ ]:


from wordcloud import WordCloud

names = funding_data["InvestorsName"][~pd.isnull(funding_data["InvestorsName"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Investor Names", fontsize=35)
plt.axis("off")
plt.show()


# In[ ]:


funding_data['InvestorsName'][funding_data['InvestorsName'] == 'Undisclosed investors'] = 'Undisclosed Investors'
funding_data['InvestorsName'][funding_data['InvestorsName'] == 'undisclosed Investors'] = 'Undisclosed Investors'
funding_data['InvestorsName'][funding_data['InvestorsName'] == 'undisclosed investors'] = 'Undisclosed Investors'
funding_data['InvestorsName'][funding_data['InvestorsName'] == 'Undisclosed investor'] = 'Undisclosed Investors'
funding_data['InvestorsName'][funding_data['InvestorsName'] == 'Undisclosed Investor'] = 'Undisclosed Investors'
funding_data['InvestorsName'][funding_data['InvestorsName'] == 'Undisclosed'] = 'Undisclosed Investors'


# In[ ]:


investors = funding_data['InvestorsName'].value_counts().head(10)
print(investors)
plt.figure(figsize=(15,8))
sns.barplot(investors.index, investors.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Investors Names', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Investors Names with number of funding", fontsize=16)
plt.show()


# **Indian Angel network **and** Ratan tata** funded maximum number of startups followed by **Kalaari Caitals**.

# ## Ques 6 : What are different types of funding for startups ?

# In[ ]:


investment = funding_data['InvestmentType'].value_counts()
print(investment)


# In[ ]:


funding_data['InvestmentType'][funding_data['InvestmentType'] == 'SeedFunding'] = 'Seed Funding'
funding_data['InvestmentType'][funding_data['InvestmentType'] == 'Crowd funding'] = 'Crowd Funding'
funding_data['InvestmentType'][funding_data['InvestmentType'] == 'PrivateEquity'] = 'Private Equity'


# In[ ]:


investment = funding_data['InvestmentType'].value_counts()
print(investment)
plt.figure(figsize=(15,8))
sns.barplot(investment.index, investment.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Investment Type', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Investment Type with number of funding", fontsize=16)
plt.show()


# In[ ]:


temp = funding_data["InvestmentType"].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Types of investment funding with %')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")


# We can see **Seed Funding** is in **Top** followed by Private Equity.

#  # More is coming and if you find useful please upvote the Kernel.
