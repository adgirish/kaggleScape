
# coding: utf-8

# <h1>Welcome to my Kernel.</h1>
# 
# I will do some analysis trying to understand the Kiva data.
# 
# Kiva is a excellent crowdfunding plataform that helps the poor and financially excluded people around the world. 

# <h2> Note that this analysis is not finised. To follow the all updates please <i>votesup</i> and also give me your feedback.</h2>
# 
# 
# <i>**English is not my native language, so sorry about any error</i>

# <h2> OBJECTIVES OF THIS EXPLORATION </h2>
# - Understand the distribuitions of loan values
# - Understand the principal sectors that was helped
# - Understand the countrys that receive the loan's
# - Understand the Date's through this loans
# - Understand what type of loan have more lenders
# - And much more... Everything that we can get of information about this dataset

# <h2> About the Dataset</h2>
# 
# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical. However, this requires inference based on a limited set of information for each borrower.
# 
# In Kaggle Datasets' inaugural Data Science for Good challenge, Kiva is inviting the Kaggle community to help them build more localized models to estimate the poverty levels of residents in the regions where Kiva has active loans. Unlike traditional machine learning competitions with rigid evaluation criteria, participants will develop their own creative approaches to addressing the objective. Instead of making a prediction file as in a supervised machine learning problem, submissions in this challenge will take the form of Python and/or R data analyses using Kernels, Kaggle's hosted Jupyter Notebooks-based workbench.
# 
# Kiva has provided a dataset of loans issued over the last two years, and participants are invited to use this data as well as source external public datasets to help Kiva build models for assessing borrower welfare levels. Participants will write kernels on this dataset to submit as solutions to this objective and five winners will be selected by Kiva judges at the close of the event. In addition, awards will be made to encourage public code and data sharing. With a stronger understanding of their borrowers and their poverty levels, Kiva will be able to better assess and maximize the impact of their work.
# 
# The sections that follow describe in more detail how to participate, win, and use available resources to make a contribution towards helping Kiva better understand and help entrepreneurs around the world.

# <h2>Importing the librarys</h2>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# <h2>Importing the data</h2>

# In[ ]:


df_kiva = pd.read_csv("../input/kiva_loans.csv")
df_kiva_loc = pd.read_csv("../input/kiva_mpi_region_locations.csv")


# <h2>First Look to our data</h2>

# In[ ]:


print(df_kiva.shape)
print(df_kiva.nunique())


# In[ ]:


print(df_kiva.describe())


# <h2>Looking how the data is</h2>

# In[ ]:


df_kiva.head()


# <h1>Let's start exploring the Funded and and Loan Amount</h1>

# In[ ]:


print("Description of distribuition")
print(df_kiva[['funded_amount','loan_amount']].describe())

plt.figure(figsize=(12,10))

plt.subplot(221)
g = sns.distplot(np.log(df_kiva['funded_amount'] + 1))
g.set_title("Funded Amount Distribuition", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Frequency", fontsize=12)

plt.subplot(222)
g1 = plt.scatter(range(df_kiva.shape[0]), np.sort(df_kiva.funded_amount.values))
g1= plt.title("Funded Amount Residual Distribuition", fontsize=15)
g1 = plt.xlabel("")
g1 = plt.ylabel("Amount(US)", fontsize=12)

plt.subplot(223)
g2 = sns.distplot(np.log(df_kiva['loan_amount'] + 1))
g2.set_title("Loan Amount Distribuition", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Frequency", fontsize=12)

plt.subplot(224)
g3 = plt.scatter(range(df_kiva.shape[0]), np.sort(df_kiva.loan_amount.values))
g3= plt.title("Loan Amount Residual Distribuition", fontsize=15)
g3 = plt.xlabel("")
g3 = plt.ylabel("Amount(US)", fontsize=12)

plt.subplots_adjust(wspace = 0.3, hspace = 0.3,
                    top = 0.9)

plt.show()


# Cool. We have a normal distribuition to the both values.

# <h2>Another interesting numerical values is the lender number and the term in months.<br><br>
# 
# I will start exploring further the Lenders_count column</h2>

# In[ ]:


#First plot
trace0 = go.Histogram(x=np.log(df_kiva['lender_count'] + 1), 
                      name='Log',
                      nbinsx=30,
                      histnorm='probability')

#Second plot
trace1 = go.Histogram(x=df_kiva['lender_count'], 
                      name='Normal', 
                      nbinsx=600)

#Third plot
trace2 = go.Bar(
    x = df_kiva["lender_count"].value_counts()[:200].index.values,
    y = df_kiva["lender_count"].value_counts()[:200].values, 
    marker=dict(
        color=df_kiva["lender_count"].value_counts()[:200].values,
        colorscale = 'Viridis'
    )
)

#Creating the grid
fig = tls.make_subplots(rows=2, cols=2, specs=[[{'colspan': 2}, None], [{}, {}]],
                          subplot_titles=('Lender Count Filter first 200',
                                          'Lender Count Log Dist',
                                          'Lender Count Normal Dist'))

#setting the figs
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace0, 2, 1)
fig.append_trace(trace1, 2, 2)

fig['layout'].update(showlegend=True, title='Lender Count Distribuition', 
                     bargap=0.05, 
                     height=700, width=800)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')


# We have a interesting distribuition...Or you have 1 lender or you will have a great chance to chave through 4 ~ 12 lenders in the project

# <h2>Term in Months</h2>

# In[ ]:


months = df_kiva.term_in_months.value_counts()

plt.figure(figsize=(25,8))

plt.subplot(122)
g = sns.distplot(np.log(df_kiva['term_in_months'] + 1))

g.set_title("Term in Months Log", fontsize=25)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=18)

plt.subplot(121)
g1 = sns.distplot(df_kiva['term_in_months'])
g1.set_title("Term in Months without filter", fontsize=25)
g1.set_xlabel("")
g1.set_ylabel("Frequency", fontsize=18)

trace0 = go.Bar(
            x = df_kiva["term_in_months"].value_counts()[:60].index.values,
            y = df_kiva["term_in_months"].value_counts()[:60].values,
            marker=dict(
                color=df_kiva["term_in_months"].value_counts()[:60].values,
                colorscale = 'Viridis',
                reversescale = True
        )
    )

data = [trace0]

layout = go.Layout(
    
)

layout = go.Layout(
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Lender Counts Distribuition'
    ),
    title='The Month terms distribuition'
)

fig = go.Figure(data=data, layout=layout)


plt.subplots_adjust(wspace = 0.2, hspace = 0.3,top = 0.9)

plt.show()
py.iplot(fig, filename='term-months')


# Curious... In a "normal" loan the term almost ever is 6 ~ 12 ~ 24 and so on. <br>
# It's the first time that I see 8 ~ 14 ~ 20; Very Interesting
# 

# <h1>Let's look through the Sectors to known them</h1>

# In[ ]:


sectors = df_kiva.sector

sectors_amount = pd.DataFrame(df_kiva.groupby(['sector'])['loan_amount'].mean().sort_values(ascending=False)[:30]).reset_index()

sectors_count = go.Bar(
    x = sectors.value_counts().index.values,
    y = sectors.value_counts().values,
            marker=dict(
                color=sectors.value_counts().values,
                colorscale = 'Viridis',
                reversescale = True
        )
)


sectors_mean = go.Bar(
    x = sectors_amount.sector,
    y = sectors_amount.loan_amount,
    marker=dict(
        color=sectors.value_counts().values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

fig = tls.make_subplots(rows=2, cols=1,
                          subplot_titles=('Sectors Count',
                                          'Sectors Mean Loan Amount'))

#setting the figs
fig.append_trace(sectors_count, 1, 1)
fig.append_trace(sectors_mean, 2, 1)

fig['layout'].update(height=900, width=800, title='Sectors')

py.iplot(fig, filename='Sectors Count')


# Very cool graph. It show us that the highest mean values ins't to the most frequent Sectors.... Transportation, Arts and Servies have a low frequency but a high mean. 

# <h2>Now I will look some values through the sectors.</h2>

# In[ ]:


df_kiva['loan_amount_log'] = np.log(df_kiva['loan_amount'])
df_kiva['funded_amount_log'] = np.log(df_kiva['funded_amount'] + 1)
df_kiva['diff_fund'] = df_kiva['loan_amount'] / df_kiva['funded_amount'] 

plt.figure(figsize=(12,14))

plt.subplot(312)
g1 = sns.boxplot(x='sector', y='loan_amount_log',data=df_kiva)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Loan Distribuition by Sectors", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount(log)", fontsize=12)

plt.subplot(311)
g2 = sns.boxplot(x='sector', y='funded_amount_log',data=df_kiva)
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Funded Amount(log) by Sectors", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Funded Amount", fontsize=12)

plt.subplot(313)
g3 = sns.boxplot(x='sector', y='term_in_months',data=df_kiva)
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("Term Frequency by Sectors", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Term Months", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6,top = 0.9)
plt.show()


# The values have an equal distribution through all data, being little small to Personal Use, that make sense.
# 
# The highest Term months is to Agriculture, Education and Health.
# 
# 

# <h1> Taking advantage of sectors, let's look the Acitivities'</h1>

# In[ ]:


acitivies = df_kiva.activity.value_counts()[:30]

activities = go.Bar(
    x = acitivies.index.values,
    y = acitivies.values,
    name='Activities Count',
    marker=dict(
        color=acitivies.values,
        colorscale = 'Viridis',
        reversescale = False
    )
)

data = [activities]

layout = go.Layout(
    title='Activities Count Distribuition'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='term-months-count')



# In[ ]:


activies_amount = pd.DataFrame(df_kiva.groupby(['activity'])['loan_amount'].mean().sort_values(ascending=False)[:30]).reset_index()

activities_amount = go.Bar(
    x = activies_amount.activity,
    y = activies_amount.loan_amount,
    name='Activities Amount',
    marker=dict(
        color=activies_amount.loan_amount.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

data = [activities_amount]

layout = go.Layout(
    title='Loan Amount mean by Activity type'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='term-months-amount')


# We can see that the activities with highest mean loan amount aren't the same as more frequent...
# This is an interesting distribution of Acitivies but it isn't so meaningful... Let's further to understand this.

# <h1>Now I will explore the activies by the top 3 sectors</h1>

# In[ ]:


plt.figure(figsize=(12,14))

plt.subplot(311)
g1 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Agriculture'])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Activities by Agriculture Sector", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g2 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Food'])
g2.set_xticklabels(g2.get_xticklabels(),rotation=80)
g2.set_title("Activities by Food Sector", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplot(313)
g3 = sns.countplot(x='activity', data=df_kiva[df_kiva['sector'] == 'Retail'])
g3.set_xticklabels(g3.get_xticklabels(),rotation=90)
g3.set_title("Activiies by Retail Sector", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Count", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)
plt.show()


# Looking the activities by Sector is more insightful than just look through the sectors or activities

# <h2>Another important value on this Dataset is the Repayment Interval.... <br>
# Might it is very meaningful about the loans to sectors  </h2>

# In[ ]:


repayment = df_kiva.repayment_interval.value_counts()

trace = go.Bar(
    x=repayment.index,
    y=repayment.values,
    marker=dict(
        color=repayment.values,
        colorscale = 'Viridis',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Repayment Interval Distribuition'
)

data = [trace]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="RepaymentIntervals")


# Humm.. An high number of loans have a Irregular Repayment... It's very interesting.... <br>
# Let's explore further the Sector's

# <h2> I will plot the distribuition of loan by Repayment Interval to see if they have the same distribuition </h2>

# In[ ]:


(sns
  .FacetGrid(df_kiva, 
             hue='repayment_interval', 
             size=5, aspect=2)
  .map(sns.kdeplot, 'loan_amount_log', shade=True)
 .add_legend()
)
plt.show()


# <h2>And with the Lender Count? How is the distribuition of lenders over the Repayment Interval</h2>

# In[ ]:


df_kiva['lender_count_log'] = np.log(df_kiva['lender_count'] + 1)

(sns
  .FacetGrid(df_kiva, 
             hue='repayment_interval', 
             size=5, aspect=2)
  .map(sns.kdeplot, 'lender_count_log', shade=True)
 .add_legend()
)
plt.show()


# Intesresting behavior of Irregular Payments, this have a little differenc... The first peak in lenders distribuition is about the zero values that I add 1.

# <h2>Let's take a better look on Sectors and Repayment Intervals in this heatmap of correlation</h2>

# In[ ]:


sector_repay = ['sector', 'repayment_interval']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[sector_repay[0]], df_kiva[sector_repay[1]]).style.background_gradient(cmap = cm)


# We have 3 sector's that have a high number of Irregular Repayments. Why this difference, just because is the most frequent ? 

# In[ ]:


df_kiva.loc[df_kiva.country == 'The Democratic Republic of the Congo', 'country'] = 'Republic of Congo'
df_kiva.loc[df_kiva.country == 'Saint Vincent and the Grenadines', 'country'] = 'S Vinc e Grenadi'


# <h1>And what's the most frequent countrys? </h1>

# In[ ]:


country_counts = df_kiva['country'].value_counts().head(40)

countrys_loan = go.Bar(
    x=country_counts.values[::-1],
    y=country_counts.index[::-1],
    orientation = 'h',
    marker=dict(
        color=country_counts.values[::-1],
        colorscale = 'Viridis'
    ),
)

layout = go.Layout(
    title="Country Loan Distribuition",
    width=700,
    height=900,
    )

data = [countrys_loan]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Country-counts")


# In[ ]:


philip_uses = df_kiva['use'][df_kiva['country'] == 'Brazil']

percent = round(philip_uses.value_counts() / len(philip_uses) * 100, 2)[:10]

pie_philip = go.Pie(labels=percent.keys(), values=percent.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))

data = [pie_philip]

layout = go.Layout(width=800, height=800, title='The top 10 loan in Philippines',titlefont= dict(size=20), 
                   legend=dict(x=0.1,y=-5))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


countrys_amount = pd.DataFrame(df_kiva.groupby(['country'])['loan_amount'].mean().sort_values(ascending=False)[:30]).reset_index()

countrys_amount = go.Bar(
    x = countrys_amount.country,
    y = countrys_amount.loan_amount,
    name='Activities Amount',
    marker=dict(
        color=countrys_amount.loan_amount.values,
        colorscale = 'Viridis',
        reversescale = True
        )
)

data = [countrys_amount]

layout = go.Layout(
    title='Loan Amount mean by Countrys'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='country-mean-amount')


# The highest mean value without the filter is Cote D'Ivoire is 50000 because have just 1 loan to this country, that was lended by 1706 lenders, how we can look below. 

# In[ ]:


df_kiva[df_kiva['country'] == "Cote D'Ivoire"]


# The most frequent Regions with more projects is really of poor regions... 
# - One interesting information is that almost all borrowers values means are under $ 10k

# <h2>Taking a look at localization loans dataset</h2> 
# - it's a 2772 rows and 9 columns  dataset
# 

# In[ ]:


df_kiva_loc.head()


# <h2>Let's start looking the counting of countrys on this dataset<h2>

# In[ ]:


data = go.Bar(
    x = df_kiva_loc.country.value_counts().values[50::-1],
    y = df_kiva_loc.country.value_counts().index[50::-1],
    orientation = 'h',
    marker=dict(
        color=df_kiva_loc.country.value_counts().values[50::-1],
        colorscale = 'Viridis'
    ),
)

layout = go.Layout(
    title='TOP 50 Countries Around the world with Loans',
    width=800,
    height=1000,
    )
figure = go.Figure(data=[data], layout=layout)
py.iplot(figure, filename="Loans-country-count")


# We can see 

# In[ ]:


regions = df_kiva_loc['world_region'].value_counts()

countrys_loan = go.Bar(
    x=regions.index,
    y=regions.values,
    orientation = 'v',
    marker=dict(
        color=regions.values,
        colorscale = 'Viridis',
        reversescale = True
    ),
)

layout = go.Layout(
    title="World Regions Distribuition",
    width=900,
    height=500,
    )

data = [countrys_loan]

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Regions-counts")


# In[ ]:


maps = pd.DataFrame(df_kiva_loc['country'].value_counts()).reset_index()
maps.columns=['country', 'counts']
maps = maps.reset_index().drop('index', axis=1)


# In[ ]:


data = [ dict(
        type = 'choropleth',
        locations = maps['country'],
        locationmode = 'country names',
        z = maps['counts'],
        text = maps['country'],
        autocolorscale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Couting of Loans Per Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

figure = dict( data=data, layout=layout )
py.iplot(figure, validate=False, filename='map-countrys-count')


# <h2>If you want to see the heatmap correlations below, click on Show Output</h2>

# In[ ]:


country_repayment = ['country', 'repayment_interval']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[country_repayment[0]], df_kiva[country_repayment[1]]).style.background_gradient(cmap = cm)


# On this heatmap correlation above we can see that just Kenya have Weekly payments and Kenya have the highest number of Irregular payments

# In[ ]:


#To see the result output click on 'Output' 
country_sector = ['country','sector']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_kiva[country_sector[0]], df_kiva[country_sector[1]]).style.background_gradient(cmap = cm)


# We can look a lot of interesting values on this heatmap.
# - Philipines have high number of loan in almost all sectors
# - The USA is the country with the highest number of Entertainment loans
# - Cambodia have the highest number of Loans to Personal Use
# - Paraguay is the country with highest Education Loan request
# - Pakistan have highest loan requests to Art and Whosale
# - Tajikistan have highest requests in Education and Health
# - Kenya and Philipines have high loan requests to Construction
# - Kenya also have high numbers to Services Loans

# <h2>Let's verify the most frequent currency's </h2>

# In[ ]:


currency = df_kiva['currency'].value_counts()

plt.figure(figsize=(10,5))
g = sns.barplot(x=currency.index[:35], y=currency.values[:35])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("The 35 most Frequency Currencies at Platform", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.show()


# PHP THAT IS THE CURRENCY PHILIPINE<br>
# USD EVERYONE KNOWS<br>
# KES THAT IS THE CURRENCY OF KENYA

# <h2>Let's take a look at the Genders</h2>
# - We will start cleaning the column borrower_genders and create a new column with this data clean 

# In[ ]:


df_kiva.borrower_genders = df_kiva.borrower_genders.astype(str)

df_sex = pd.DataFrame(df_kiva.borrower_genders.str.split(',').tolist())

df_kiva['sex_borrowers'] = df_sex[0]

df_kiva.loc[df_kiva.sex_borrowers == 'nan', 'sex_borrowers'] = np.nan


# In[ ]:


sex_mean = pd.DataFrame(df_kiva.groupby(['sex_borrowers'])['loan_amount'].mean().sort_values(ascending=False)).reset_index()


# <h2> First I will look through the Repayment Intervals column </h2>

# In[ ]:


plt.figure(figsize=(10,6))

g = sns.countplot(x='sex_borrowers', data=df_kiva, 
                  hue='repayment_interval')
g.set_title("Exploring the Genders by Repayment Interval", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count Distribuition", fontsize=12)

plt.show()


# Let's look further this analysis

# In[ ]:



print("Gender Counting and Loan Amount Mean")
print(round(df_kiva.groupby('sex_borrowers')['loan_amount'].agg(['mean','count']),2))

plt.figure(figsize=(12,14))

plt.subplot(321)
g = sns.countplot(x='sex_borrowers', data=df_kiva, 
              order=['male','female'])
g.set_title("Gender Distribuition", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(322)
g1 = sns.violinplot(x='sex_borrowers', y='loan_amount_log', data=df_kiva)
g1.set_title("Mean Loan Amount by Gender ", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Average Loan Amount(US)", fontsize=12)

plt.subplot(313)
g2 = sns.boxplot(x='sector', y='loan_amount_log', data=df_kiva, 
              hue='sex_borrowers', hue_order=['male','female'])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Exploring the Genders by Sectors", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g3 = sns.countplot(x='term_in_months',data=df_kiva[df_kiva['term_in_months'] < 45], 
              hue='sex_borrowers', hue_order=['male','female'])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("Exploring the Genders by Term in Months", fontsize=15)
g3.set_xlabel("")
g3.set_ylabel("Term in months", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()

print("Gender Percentual Average")
print(round(df_kiva['sex_borrowers'].value_counts() / len(df_kiva['sex_borrowers'] )* 100),2)


# Why my plot are getting this color? 

# We have a greeeat difference between the genders, and very meaninful informations with this filter by Gender.
# Although we have 77% of women in our dataset, the men have the highest mean of Loan Amount
# 
# On the sectors, we have also a interesting distribution of genders

# <h2>Now I will do some transformation in Dates to verify the informations contained on this. </h2>

# In[ ]:


df_kiva['date'] = pd.to_datetime(df_kiva['date'])
df_kiva['funded_time'] = pd.to_datetime(df_kiva['funded_time'])
df_kiva['posted_time'] = pd.to_datetime(df_kiva['posted_time'])

df_kiva['date_month_year'] = df_kiva['date'].dt.to_period("M")
df_kiva['funded_year'] = df_kiva['funded_time'].dt.to_period("M")
df_kiva['posted_month_year'] = df_kiva['posted_time'].dt.to_period("M")

df_kiva['date_year'] = df_kiva['date'].dt.to_period("A")
df_kiva['funded_year'] = df_kiva['funded_time'].dt.to_period("A")
df_kiva['posted_year'] = df_kiva['posted_time'].dt.to_period("A")


# In[ ]:


df_kiva.columns


# In[ ]:


plt.figure(figsize=(10,14))

plt.subplot(311)
g = sns.countplot(x='date_month_year', data=df_kiva)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Month-Year Loan Counting", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Count", fontsize=12)

plt.subplot(312)
g1 = sns.pointplot(x='date_month_year', y='loan_amount', 
                   data=df_kiva, hue='repayment_interval')
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_title("Mean Loan by Month Year", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount", fontsize=12)

plt.subplot(313)
g2 = sns.pointplot(x='date_month_year', y='term_in_months', 
                   data=df_kiva, hue='repayment_interval')
g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
g2.set_title("Term in Months by Month-Year", fontsize=15)
g2.set_xlabel("")
g2.set_ylabel("Term in Months", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.50,top = 0.9)

plt.show()


# It looks nice and very meaninful
# - we have mean of 15k of projects by month
# - The weekly payments might was ended in 2015
# - The peak of projects was in 2017-03

# <h1>Quick look through the use description</h1>

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import nltk.tokenize as word_tokenize
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction import stop_words


# In[ ]:


stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=800,
                          random_state=42,
                         ).generate(" ".join(df_kiva['use'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - USE", fontsize=25)
plt.axis('off')
plt.show()


# In[ ]:






# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=800,
                          random_state=42,
                         ).generate(" ".join(df_kiva['region'].astype(str)))

print(wordcloud)
fig = plt.figure(figsize = (12,14))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - REGION DESCRIPTION",fontsize=25)
plt.axis('off')
plt.show()


# <h2>Thank you very much and <i>don't forget </i> to VOTEUP my kernel.</h2>

# Also feel free to fork this
