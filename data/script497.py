
# coding: utf-8

# ![](https://i.imgur.com/dsg47jV.jpg)
# 
# # Introduction
# > About this notebook: In this notebook, I will be analyzing Kiva Crowdfunding dataset. I will try to find helpful information on this dataset. If you find something wrong or have any suggestion feel free to reach me at the comment. And don't forget to upvote. I will continue adding new information, so please visit frequentrly.
# 
# ## Table of contents
# 1. [Popular loan sector](#popular_loan_sector)
# 2. [Loan due](#loan_due)
# 3. [Gender combination](#gender_cobination)
# 4. [Usecase of loan](#usecase_of_loan)
# 5. [Who needs help ](#who_needs_help)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS # this module is for making wordcloud in python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import plotly
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
df_kiva_region_location = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
df_kiva_loans.head()


# # TL;DR

# In[ ]:


df_kiva_loans['due_loan'] = df_kiva_loans['loan_amount'] - df_kiva_loans['funded_amount']
df_kiva_loans['country_iso_3'] = df_kiva_loans['country'].map(pd.DataFrame(df_kiva_region_location.groupby(['country','ISO']).size()).reset_index().drop(0,axis=1).set_index('country')['ISO'])


# In[ ]:


plot_df_country_popular_loan = pd.DataFrame(df_kiva_loans.groupby(['country','country_iso_3'])['loan_amount', 'funded_amount'].mean()).reset_index()


# In[ ]:


plot_df_country_popular_due = pd.DataFrame(df_kiva_loans.groupby(['country','country_iso_3'])['due_loan'].max()).reset_index()
plot_df_country_popular_due = plot_df_country_popular_due[plot_df_country_popular_due['due_loan'] > 0]  
plot_df_country_popular_loan['due_loan'] = plot_df_country_popular_loan['country_iso_3'].map(plot_df_country_popular_due.set_index('country_iso_3')['due_loan'])
plot_df_country_popular_loan = plot_df_country_popular_loan.fillna(0)


# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],[0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = plot_df_country_popular_loan['country_iso_3'],
        z = plot_df_country_popular_loan['loan_amount'].astype(float),
        text =  'Country name: ' + plot_df_country_popular_loan['country'] +'</br>' + 'Average loan amount: ' + plot_df_country_popular_loan['loan_amount'].astype(str) \
    + '</br>' + 'Average funded_amount: ' + plot_df_country_popular_loan['funded_amount'].astype(str) + '</br>' \
     + 'Loan due: ' + plot_df_country_popular_loan['due_loan'].astype(str),
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Average loan amount in dollar")
        ) ]

layout = dict(
        title = 'Average loan amount taken by different country<br>(Hover for breakdown)',
        geo = dict(
            projection=dict( type='orthographic' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )




    
fig = dict( data=data, layout=layout )

py.iplot( fig, filename='d3-cloropleth-map' )


# <a id="popular_loan_sector"></a>
# ## 1. Popular loan sector
# > In this section, I will be analyzing which loan sector and activity take more loan. 

# In[ ]:


plot_df_sector_popular_loan = pd.DataFrame(df_kiva_loans.groupby(['sector'])['loan_amount'].mean()).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='sector',y='loan_amount',data=plot_df_sector_popular_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan sector in terms of loan amount', fontsize=24)
plt.savefig('popular_loan_amount_sector.png')
plt.show()


# It looks like Entertainment sector is popular for taking large amount of loan!

# In[ ]:


plot_df_sector_popular_loan = pd.DataFrame(df_kiva_loans.groupby(['activity'])['loan_amount'].mean().sort_values(ascending=False)[:20]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='activity',y='loan_amount',data=plot_df_sector_popular_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan activity in terms of loan amount', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()


# The plot shows us that Technology, Gardening, communication is most popular activity that takes large amount of loan. 

# <a id="loan_due"></a>
# 
# # 2. Loan due
# > In this section, I will to find which country, gender, sector have most due loan. 

# In[ ]:


plot_df_due_loan = pd.DataFrame(df_kiva_loans.groupby(['country'])['due_loan'].max().sort_values(ascending=False)[:20]).reset_index()
plot_df_due_loan = plot_df_due_loan[plot_df_due_loan['due_loan'] > 0]  
plot_df_due_loan['country'] = plot_df_due_loan['country'].replace('The Democratic Republic of the Congo', 'Congo')
plt.subplots(figsize=(15,7))
sns.barplot(x='country',y='due_loan',data=plot_df_due_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Due loan($)', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Country', fontsize=20)
plt.title('Country with highest due loan', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()


# Haiti, Mexico and Peru has most due loans. 

# In[ ]:


plot_df_due_loan = pd.DataFrame(df_kiva_loans.groupby(['sector'])['due_loan'].max().sort_values(ascending=False)[:20]).reset_index()
plot_df_due_loan = plot_df_due_loan[plot_df_due_loan['due_loan'] > 0]  
plt.subplots(figsize=(15,7))
sns.barplot(x='sector',y='due_loan',data=plot_df_due_loan,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Due loan($)', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan sector with highest due loan', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()


# It looks like Wholesale, Transportation and Food sector has highest due loan. 

# <a id="gender_cobination"></a>
# 
# 
# # 3. Gender combination
# > Some time loan only taken by one woman/man or sometime woman and man together take lone. Let's see what's going on in data set

# In[ ]:


from collections import Counter
def count_word(x):
    y = Counter(x.split(', '))
    return y
df_kiva_loans_without_null_gender = df_kiva_loans.dropna(subset = ['borrower_genders'])
plot_df_gender = pd.DataFrame.from_dict(df_kiva_loans_without_null_gender['borrower_genders'].apply(lambda x: count_word(x)))
plot_df_gender['borrower_genders'] = plot_df_gender['borrower_genders'].astype(str).replace({'Counter':''}, regex=True)


# In[ ]:


plot_df_gender = pd.DataFrame(plot_df_gender['borrower_genders'].value_counts()[:10]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='index',y='borrower_genders',data=plot_df_gender,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Gender count', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Gender combinations', fontsize=20)
plt.title('Popular gender combinations', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()


# It's look like most of the time female alone take loan!

# <a id="usecase_of_loan"></a>
# # 4. Usecase of loan
# > In this section, I will create a wordcloud of usecase described by the borrower when taking loan. 

# In[ ]:


wc = WordCloud(width=1600, height=800, random_state=1,max_words=200000000)
# generate word cloud using df_yelp_tip_top['text_clear']
wc.generate(str(df_kiva_loans['use']))
# declare our figure 
plt.figure(figsize=(20,10), facecolor='k')
# add title to the graph
plt.title("Usecase of loan", fontsize=40,color='white')
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)
# after lot of congiguration finally plot the graph
# plt.savefig('word.png', dpi=900)
plt.show()


# It's look like most of the people take loan for purchase something!

# <a id="who_needs_help"></a>
# 
# # 5. Let's see who needs help more! 
# > In this section, I will analyze Kiva's dataset with other dataset and try to find some informative information. Like where Kiva needs to provide more loans etc. 

# In[ ]:


df_youth = pd.read_csv('../input/youth-unemployment-gdp-and-literacy-percentage/youth.csv', sep='\s*,\s*',engine='python')
df_youth['country'] = df_youth['country'].replace('Congo (Democratic Republic)', 'The Democratic Republic of the Congo')
df_youth['country'] = df_youth['country'].replace('United States of America', 'United States')
df_youth['country'] = df_youth['country'].replace('Palestinian Territories', 'Palestine')
df_youth['country'] = df_youth['country'].replace('East Timor', 'Timor-Leste')
df_youth['country'] = df_youth['country'].replace('East Timor', 'Timor-Leste')
df_youth['country'] = df_youth['country'].replace('Laos', "Lao People's Democratic Republic")
df_youth['country'] = df_youth['country'].replace('Congo (Republic)', "Congo")
df_youth['country'] = df_youth['country'].replace('Virgin Islands of the U.S.', "Virgin Islands")


# In[ ]:


df_youth = df_youth.set_index('country')
df_youth.index.names = [None]
df_kiva_loans['youth'] = df_kiva_loans['country'].map(df_youth['youth_percentage'])


# In[ ]:


plot_df_top_youth_country = pd.DataFrame(df_kiva_loans.groupby('country')['youth'].mean()).reset_index().nlargest(20, 'youth')
plot_df_pop_country = pd.DataFrame(df_kiva_loans.country.value_counts().reset_index()).nlargest(20, 'country')
fig, axs = plt.subplots(figsize=(15,10), ncols=2)
# plt.subplots_adjust(right=0.9)
ax1 = sns.barplot(y='country',x='youth',data=plot_df_top_youth_country,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7), ax=axs[0])
ax2 = sns.barplot(y='index', x = 'country', data=plot_df_pop_country, ax=axs[1])
ax1.invert_xaxis()
ax1.set_xlabel('Youth percentage', fontsize=20)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax1.set_ylabel('Country', fontsize=20)
ax1.set_title('Country with highest youth percentage', fontsize=24)


ax2.set_xlabel('Number of loans', fontsize=20)
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax2.set_ylabel('')
ax2.set_title('Country with highest number of loan', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()


# **Takeaways from the plot:**
# > The left plot shows us the countries with the highest percentage. Youth are the main backbone of a country. If you can empower the youth the country can prosper. Unemployment is the disaster for a country. Because Kiva wants to improve people lives, it needs to invest in those countries with the highest percentage of youth. On the right side plot, it shows us the countries with the highest number of loans. It clearly shows the gap between the popular country with youth percentage and popular country with the number of loans. So Kiva should increase the number of loans in those countries with the highest percentage of youth. 

# In[ ]:


df_unemployment = pd.read_csv('../input/youth-unemployment-gdp-and-literacy-percentage/unemployment.csv', sep='\s*,\s*',engine='python')
df_unemployment['country'] = df_unemployment['country'].replace('Congo (Democratic Republic)', 'The Democratic Republic of the Congo')
df_unemployment['country'] = df_unemployment['country'].replace('United States of America', 'United States')
df_unemployment['country'] = df_unemployment['country'].replace('Palestinian Territories', 'Palestine')
df_unemployment['country'] = df_unemployment['country'].replace('East Timor', 'Timor-Leste')
df_unemployment['country'] = df_unemployment['country'].replace('East Timor', 'Timor-Leste')
df_unemployment['country'] = df_unemployment['country'].replace('Laos', "Lao People's Democratic Republic")
df_unemployment['country'] = df_unemployment['country'].replace('Congo (Republic)', "Congo")
df_unemployment['country'] = df_unemployment['country'].replace('Virgin Islands of the U.S.', "Virgin Islands")


# In[ ]:


df_unemployment = df_unemployment.set_index('country')
df_unemployment.index.names = [None]
df_kiva_loans['unemployment'] = df_kiva_loans['country'].map(df_unemployment['unemployment_percentage'])


# In[ ]:


plot_df_top_unemployment_country = pd.DataFrame(df_kiva_loans.groupby('country')['unemployment'].mean()).reset_index().nlargest(20, 'unemployment')
plot_df_pop_country = pd.DataFrame(df_kiva_loans.country.value_counts().reset_index()).nlargest(20, 'country')
fig, axs = plt.subplots(figsize=(15,10), ncols=2)
# plt.subplots_adjust(right=0.9)
ax1 = sns.barplot(y='country',x='unemployment',data=plot_df_top_unemployment_country,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7), ax=axs[0])
ax2 = sns.barplot(y='index', x = 'country', data=plot_df_pop_country, ax=axs[1])
ax1.invert_xaxis()
ax1.set_xlabel('Unemployment percentage', fontsize=20)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax1.set_ylabel('Country', fontsize=20)
ax1.set_title('Country with highest unemployment rate', fontsize=24)


ax2.set_xlabel('Number of loans', fontsize=20)
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax2.set_ylabel('')
ax2.set_title('Country with highest number of loan', fontsize=24)
# plt.savefig('unemployment.png')
plt.show()


# **Takeaways from the plot: **
# > If Kiva wants to help countries for fighting unemployment Kiva should focus on those countries with highest unemployment countries. The graph shows us that those countries with highest unemployment rate have really less number of loans compare to other countries. So Kiva should focus on this matter. 

# In[ ]:


df_literacy = pd.read_csv('../input/youth-unemployment-gdp-and-literacy-percentage/literacy_rate.csv', sep='\s*,\s*',engine='python')
df_literacy['country'] = df_literacy['country'].replace('Congo (Democratic Republic)', 'The Democratic Republic of the Congo')
df_literacy['country'] = df_literacy['country'].replace('United States of America', 'United States')
df_literacy['country'] = df_literacy['country'].replace('Palestinian Territories', 'Palestine')
df_literacy['country'] = df_literacy['country'].replace('East Timor', 'Timor-Leste')
df_literacy['country'] = df_literacy['country'].replace('East Timor', 'Timor-Leste')
df_literacy['country'] = df_literacy['country'].replace('Laos', "Lao People's Democratic Republic")
df_literacy['country'] = df_literacy['country'].replace('Congo (Republic)', "Congo")
df_literacy['country'] = df_literacy['country'].replace('Virgin Islands of the U.S.', "Virgin Islands")

df_literacy = df_literacy.set_index('country')
df_literacy.index.names = [None]
df_kiva_loans['literacy'] = df_kiva_loans['country'].map(df_literacy['literacy_rate_percent_all'])


# In[ ]:


df_kiva_loans.literacy = df_kiva_loans.literacy.astype(float)
plot_df_top_literacy_country = pd.DataFrame(df_kiva_loans.groupby('country')['literacy'].mean()).reset_index().nsmallest(20, 'literacy')
plot_df_top_literacy_country['literacy'] = plot_df_top_literacy_country.literacy.astype(float)
plot_df_pop_country = pd.DataFrame(df_kiva_loans.country.value_counts().reset_index()).nlargest(20, 'country')
fig, axs = plt.subplots(figsize=(15,10), ncols=2)
# plt.subplots_adjust(right=0.9)
ax1 = sns.barplot(y='country',x='literacy',data=plot_df_top_literacy_country,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7), ax=axs[0])
ax2 = sns.barplot(y='index', x = 'country', data=plot_df_pop_country, ax=axs[1])
ax1.invert_xaxis()
ax1.set_xlabel('Literacy percentage', fontsize=20)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax1.set_ylabel('Country', fontsize=20)
ax1.set_title('Country with lowest literacy rate', fontsize=24)


ax2.set_xlabel('Number of loans', fontsize=20)
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)
    tick.set_fontsize(20)
# ax1.set_r(rotation=90,fontsize=20)
ax2.set_ylabel('')
ax2.set_title('Country with highest number of loan', fontsize=24)
plt.show()


# **Takeaways from the plot: **
# > On the left side it shows the countries with lowest literacy rate and right side with the highest number of loans. Those countries with the lowest number of literacy need more help compare to other countries. So Kiva should consider this.  

# <a id="loan_themes"></a>
# # 6. Loan themes
# > In this section, I will analyse what is popular loan theme like education, agriculture etc. 

# In[ ]:


df_loan_theme = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')


# In[ ]:


plot_df_loan_theme = df_loan_theme['Loan Theme Type'].value_counts()[:10].reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='index',y='Loan Theme Type',data=plot_df_loan_theme,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Loan theme count', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan theme names', fontsize=20)
plt.title('Popular loan theme', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()


# In[ ]:


plot_df_sector_popular_loan_by_amount = pd.DataFrame(df_loan_theme.groupby(['Loan Theme Type'])['amount'].mean().sort_values(ascending=False)[:10]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(y='Loan Theme Type',x='amount',data=plot_df_sector_popular_loan_by_amount,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average loan amount in Dollar', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular loan sector in terms of loan amount', fontsize=24)
# plt.savefig('popular_loan_amount_sector.png')
plt.show()


# Hm! It looks like most popular loan theme in terms of frequency is General, Agriculture and Higher education. But most popular loan theme in terms of amount is totally different. 

# # To be continued.
