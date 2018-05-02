
# coding: utf-8

# Welcome to my notebook!<br>
# This is my first notebook that is not based in a class or project from a course so please let me know how I can make it better. I bet many of you will have suggestions and criticism. In the end, I'm here to learn, be better and decided to test my knowledge with Kiva's project.<br>
# If you think this notebook is heading somewhere and can be meaningful to others then upvote it! I'll like to have as many people looking at my notebook and share their point of view. You don't get better by just being told you are doing a good job, but rather by suggestions and criticism.<br>
# Thank you!

# # Glossary
# 
# - <a href='#introduction'>1 <b>Intruction:</b></a>
# - <a href='#importing'>2 <b>Importing and installing dependencies:</b></a>
# - <a href='#manipulating'>3 <b>Manipulating the dataset for better usage:</b></a>
#     - <a href='#dataset'>3.1 Saving and displaying parts of the dataset</a>
#     - <a href='#converting'>3.2 Converting columns to lower case and displaying their meanings</a>
#     - <a href='#description'>3.3 Total amount of rows and types in kiva_loans</a>
#     - <a href='#overview'>3.4 Overview of the dataset</a>
#     - <a href='#dates'>3.5 Splitting the dates to year, month, day for future use</a>
#     - <a href='#missingvalues'>3.6 Checking if any of the columns have any missing values<a/>
# - <a href='#kiva_loans'>4 <b>Analyzing kiva_loans:</b></a>
#     - <a href='#kiva_loans_regions'>4.1 Continents that got the most loans</a>
#         - <a href='#kiva_loans_countries'>4.1.1 Countries that got the most money invested</a>
#         - <a href='#kiva_loans_country_mpi'>4.1.2 Countries by mpi</a>
#     - <a href='#kiva_loans_currency'>4.2 Most popular currency</a>
#     - <a href='#kiva_loans_sectors'>4.3 Sectors that got the most money invested</a>
#         - <a href='#kiva_loans_frequent_investment'>4.3.1 Sector investment average</a>
#         - <a href='#kiva_loans_counting'>4.3.2 Counting the total amount of loans in Entertainment, Wholesale and Agriculture</a>
#         - <a href='#kiva_loans_sectors_years'>4.3.3 How sectors get funded through the years</a>
#     - <a href='#kiva_loans_use'>4.4 Most common usage of loans</a>
#         - <a href='#kiva_loans_activity_funded'>4.4.1 Kiva's ability to fund these loans</a>
#         - <a href='#kiva_loans_use_years'>4.4.2 Usage throughout the years</a>
#     - <a href='#kiva_loans_lenders'>4.5 Lenders from kiva</a>
#         - <a href='#kiva_loans_sex_amount'>4.5.1 Sex that lends more money</a>
#         - <a href='#kiva_loans_lender_amount'>4.5.2 How many lenders usually invest in 1 loan</a>
#         - <a href='#kiva_loans_sector_sex'>4.5.3 How each sex invest in each sector</a>
#         - <a href='#kiva_loans_activity_sex'>4.5.4 How each sex invest in each activity</a>
#     - <a href='#kiva_loans_repayment'>4.6 Most popular repayment type</a>
#         - <a href='#kiva_loans_repayment_year'>4.6.1 Repayments by year</a>
#         - <a href='#kiva_loans_terms'>4.6.2 Average how long it takes to pay the loans in months</a>
# - <a href='#loan_themes'>5 <b>Analyzing loan_theme_by_region file:</b></a>
#     - <a href='#loan_themes_partners'>5.1 Partners that invest frequently</a>
#         - <a href='#loan_themes_amount'>5.1.1 Partners that invest the most money</a>
#         - <a href='#loan_themes_regions'>5.1.2 Regions that partners invest the most</a>
#         - <a href='#loan_themes_country'>5.1.3 Countries that partners invest the most</a>
#     - <a href='#loan_theme_correlation'> 5.2 Most common themes</a>
#         - <a href='#loan_theme_general_theme'>5.2.1 General theme in more detail</a>
#     - <a href='#kiva_specific'>5.3 Loans created specifically for Kiva</a>
# - <a href='#philippines'>6 <b>Analyzing the Philippines:</b></a>
#     - <a href='#philippines_intro'>6.1 Introduction of the Philippines</a>
#         - <a href='#philippines_dataset'>6.1.1 Gathering only the Philippines from the dataset</a>
#         - <a href='#philippines_currency'>6.1.2 Philippines currency</a>
#         - <a href='#philippines_mpi'>6.1.3 Philippines MPI</a>
#     - <a href='#philippines_sector'>6.2 Sectors</a>
#         - <a href='#philippines_sector_average'>6.2.1 Philippines' sectors investment average</a>
#         - <a href='#philippines_sector_comparison'>6.2.2 Comparing investments in Agriculture and Wholesale</a>
#         - <a href='#philippines_sector_gender'>6.2.3 Gender in sectors</a>
#         - <a href='#philippines_sector_partners'>6.2.4 Partners in sectors</a>
#         - <a href='#philippines_sector_years'>6.2.5 Sectors throughout the years</a>
#     - <a href='#philippines_activity'>6.3 Activities</a>
#         - <a href='#philippines_activity_generalstore'>6.3.1 General store from activities</a>
#         - <a href='#philippines_activity_gender'>6.3.2 Activities invested by gender</a>
#         - <a href='#philippines_activity_years'>6.3.3 Funding of activities throughout the years</a>
#     - <a href='#philippines_use'>6.4 Use</a>
#     - <a href='#philippines_years'>6.5 Total amount loaned throughout the years</a>
#          - <a href='#philippines_years_correlation'>6.5.1 Kiva's ability to fund the Philippines loans</a>
#          - <a href='#philippines_years_repayment'>6.5.2 Repayment of loans</a>
#          - <a href='#phillipines_years_months'>6.5.3 Repayment of loans in months</a>
#     - <a href='#philippines_genders'>6.6 Gender</a>
#     - <a href='#philippines_partners'>6.7 Partners</a>
# - <a href='#kenya'>7 <b>Analyzing Kenya:</b></a>
#     - <a href='#kenya_intro'>7.1 Introduction of Kenya</a>
#         - <a href='#kenya_dataset'>7.1.1 Gathering only Kenya from the dataset</a>
#         - <a href='#kenya_mpi'>7.1.2 Kenya's MPI</a>
#         - <a href='#kenya_currency'>7.1.3 Kenya's currency value against USD</a>
#     - <a href='#kenya_sector'>7.2 Sectors</a>
#         - <a href='#kenya_sector_average'>7.2.1 Sector investment average</a>
#         - <a href='#kenya_sector_comparison'>7.2.2 Comparing the average with the most invested popular</a>
#         - <a href='#kenya_sector_gender'>7.2.3 Genders in sector</a>
#         - <a href='#kenya_sector_partner'>7.2.4 Partners in sector</a>
#         - <a href='#kenya_sector_years'>7.2.5 Sectors funded thoughout the years</a>
#     - <a href='#kenya_activity'>7.3 Kenya's activity</a>
#         - <a href='#kenya_activity_farming'>7.3.1 Farming from activities</a>
#         - <a href='#kenya_activity_gender'>7.3.2 Activities invested by gender</a>
#         - <a href='#kenya_activity_years'>7.3.3 Funding of activities throughout the years</a>
#     - <a href='#kenya_use'>7.4 Use</a>
#     - <a href='#kenya_loans'>7.5 Total amount loaned throughout the years</a>
#         - <a href='#kenya_loans_funding'>7.5.1 Kiva's ability to fund these loans</a>
#         - <a href='#kenya_loans_repayment'>7.5.2 Repayment of loans</a>
#         - <a href='#kenya_loans_months'>7.5.3 Repayment of loans in months</a>
#     - <a href='#kenya_genders'>7.6 Gender</a>
#     - <a href='#kenya_partners'>7.7 Partners</a>
# - <a href='#salvador'>8 <b>Analyzing El Salvador:</b></a>
#     - <a href='#salvador_intro'>8.1 Introduction of El Salvador</a>
#         - <a href='#salvador_dataset'>8.1.1 Gathering only El Salvador from the dataset</a>
#         - <a href='#salvador_mpi'>8.1.2 Kenya's MPI</a>
#     - <a href='#salvador_sector'>8.2 Sectors</a>
#         - <a href='#salvador_sector_average'>8.2.1 Sector investment average</a>
#         - <a href='#salvador_sector_comparison'>8.2.2 Comparing the average with the most invested popular</a>
#         - <a href='#salvador_sector_gender'>8.2.3 Genders in sector</a>
#         - <a href='#salvador_sector_partner'>8.2.4 Partners in sector</a>
#         - <a href='#salvador_sector_theme'>8.2.5 Themes in general inclusion</a>
#         - <a href='#salvador_sector_years'>8.2.6 Sectors funded thoughout the years</a>
#     - <a href='#salvado_ractivity'>8.3 Analyzing each activity in agriculture</a>
#         - <a href='#salvador_activity_uses'>8.3.1 Uses in Personal Housing Expenses</a>
#         - <a href='#salvador_activity_gender'>8.3.2 Gender in activity</a>
#         - <a href='#salvador_activity_years'>8.3.3 Activities throughout the years</a>
#     - <a href='#salvador_use'>8.4 Analyzing the different uses of the money in agriculture</a>
#     - <a href='#salvador_loans'>8.5 Total amount loaned throughout the years</a>
#         - <a href='#salvador_loans_funding'>8.5.1 Kiva's ability to fund these loans</a>
#         - <a href='#salvador_loans_repayment'>8.5.2 Repayment of loans</a>
#         - <a href='#salvador_loans_months'>8.5.3 Repayment of loans in months</a>
#     - <a href='#salvador_genders'>8.6 Gender</a>
#     - <a href='#salvador_partners'>8.7 Partners</a>
# - <a href='#mpi'>9 <b>Deprivation levels:</b></a>
# - <a href='#conclusions'>10 <b>Conclusions:</b></a>

# # <a id='introduction'>1 Introduction</a>

# # Data Science for Good: Kiva Crowdfunding

# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical. However, this requires inference based on a limited set of information for each borrower.
# 
# In Kaggle Datasets' inaugural Data Science for Good challenge, Kiva is inviting the Kaggle community to help them build more localized models to estimate the poverty levels of residents in the regions where Kiva has active loans. Unlike traditional machine learning competitions with rigid evaluation criteria, participants will develop their own creative approaches to addressing the objective. Instead of making a prediction file as in a supervised machine learning problem, submissions in this challenge will take the form of Python and/or R data analyses using Kernels, Kaggle's hosted Jupyter Notebooks-based workbench.
# 
# Kiva has provided a dataset of loans issued over the last two years, and participants are invited to use this data as well as source external public datasets to help Kiva build models for assessing borrower welfare levels. Participants will write kernels on this dataset to submit as solutions to this objective and five winners will be selected by Kiva judges at the close of the event. In addition, awards will be made to encourage public code and data sharing. With a stronger understanding of their borrowers and their poverty levels, Kiva will be able to better assess and maximize the impact of their work.
# 
# The sections that follow describe in more detail how to participate, win, and use available resources to make a contribution towards helping Kiva better understand and help entrepreneurs around the world.

# # <a id="importing">2 Importing and installing dependencies:</a>

# In[105]:


# Importing dependencies:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print('Installed all dependencies!')


# # <a id='manipulating'>3 Manipulating the dataset for better usage</a>

# ## <a id='dataset'>3.1 Saving and displaying parts of the dataset:</a>

# In[106]:


# Exploring the data:
# I'm creating this notebook locally so the path of the files differ from Kaggle.

# Kiva dataset - Kaggle format:
kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva_mpi_region_locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding//kiva_mpi_region_locations.csv')
loan_theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')
loan_themes_by_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')

# Multidimensional Poverty Index - Kaggle format:
MPI_national = pd.read_csv('../input/mpi/MPI_national.csv')
MPI_subnational = pd.read_csv('../input/mpi/MPI_subnational.csv')

# # Kiva dataset:
# kiva_loans = pd.read_csv('../kiva_loans.csv')
# kiva_mpi_region_locations = pd.read_csv('../kiva_mpi_region_locations.csv')
# loan_theme_ids = pd.read_csv('../loan_theme_ids.csv')
# loan_themes_by_region = pd.read_csv('../loan_themes_by_region.csv')

# # Multidimensional Poverty Index:
# MPI_national = pd.read_csv('../MPI_national.csv')
# MPI_subnational = pd.read_csv('../MPI_subnational.csv')

MPI_subnational.head()


# ## <a id='converting'>3.2 Converting columns to lower case and displaying their meanings:</a>

# In[107]:


# Converting dataset to lowercase for easier future use:

kiva_loans.columns = [x.lower() for x in kiva_loans.columns]
kiva_mpi_region_locations.columns = [x.lower() for x in kiva_mpi_region_locations.columns]
loan_theme_ids.columns = [x.lower() for x in loan_theme_ids.columns]
loan_themes_by_region.columns = [x.lower() for x in loan_themes_by_region.columns]
MPI_national.columns = [x.lower() for x in MPI_national.columns]
MPI_subnational.columns = [x.lower() for x in MPI_subnational.columns]


# <b>File: kiva_loans.csv</b>
# - id: Unique ID for loan
# - funded_amount: Dollar value of loan funded on Kiva.org
# - loan_amount: Total dollar amount of loan
# - activity: Loan activity type
# - sector: Sector of loan activity as shown to lenders
# - use: text description of how loan will be used
# - country_code: 2-letter Country ISO Code
# - country: country name
# - region: name of location within country
# - currency: currency in which loan is disbursed
# - partner_id: Unique ID for field partners
# - posted_time: date and time when loan was posted on kiva.org
# - disbursed_time: date and time that the borrower received the loan
# - funded_time: date and time at which loan was fully funded on kiva.org
# - term_in_months: number of months over which loan was scheduled to be paid back
# - lender_count: number of lenders contributing to loan
# - tags: tags visible to lenders describing loan type
# - borrower_genders: gender of borrower(s)
# - repayment_interval: frequency at which lenders are scheduled to receive installments
# - date: date on which loan was posted<br>
# 
# <b>File: kiva_mpi_region_locations.csv</b>
# - LocationName: "{region}, {country}" - Unique ID for region
# - ISO: Unique ID for country
# - country: country name
# - region: name of location within country
# - world_region: General global region
# - MPI: Multi-dimensional poverty index for this region
# - geo: Lat-Lon pair
# - lat: latitude
# - lon: longitude <br>
# 
# <b>File: loan_theme_ids.csv</b>
# - id: Unique ID for loan (Loan ID)
# - Loan Theme ID: Unique ID for loan theme
# - Loan Theme Type: General description of the loan theme category
# - Parner ID: Unique ID for field partners (Partner ID) <br>
# 
# <b>File: loan_themes_by_region.csv</b>
# - Partner ID: Unique ID for field partners
# - Field Partner Name: Name of Field Partner
# - sector: Sector in which a loan is placed on Kiva's main page
# - Loan Theme ID: Unique ID for loan theme
# - Loan Theme Type: General description of the loan theme category
# - country: country name
# - forkiva: Was this loan theme created specifically for Kiva?
# - region: Region within country
# - geocode_old: Kiva's old geocoding system Lots of missing values
# - ISO: Unique ID for country
# - number: Number of loans funded in this LocationName and this loan theme
# - amount: Dollar value of loans funded in this LocationName and this loan theme
# - LocationName: "{region}, {country}" - Unique ID for region
# - geocode: Lat-Lon pair
# - names: All placenames that the Gmaps API associates with LocationName
# - geo: Lat-Lon pair
# - lat: latitude
# - lon: longitude
# - mpi_region: MPI Region where we think this loan theme is located
# - mpi_geo: Lat-Lon pair where we think this MPI region is located
# - rural_pct: The percentage of this field partners' borrowers that are in rural areas- 
# 
# <em>Source: https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding/discussion/50585</em>

# ## <a id='description'>3.3 Total amount of rows and types in kiva_loans:</a>

# In[108]:


# Analizing the dataset:

print('kiva_loans Shape: ', kiva_loans.shape)
print('-' * 40)
print(pd.DataFrame(kiva_loans.info()))


# ## <a id='overview'>3.4 Overview of kiva_loans:</a>

# In[109]:


# Overview of the dataset:

kiva_loans.describe(include=['O'])


# ## <a id='dates'>3.5 Splitting the dates to year, month, day for future use:</a>

# In[110]:


# Changing the date column to to_datetime:

kiva_loans['date'] = pd.to_datetime(kiva_loans['date'])
kiva_loans['year'] = pd.DataFrame(kiva_loans['date'].dt.year)
kiva_loans['month'] = pd.DataFrame(kiva_loans['date'].dt.month)
kiva_loans['day'] = pd.DataFrame(kiva_loans['date'].dt.day)
kiva_loans.head()


# ## <a id='missingvalues'>3.6 Checking if any of the columns have any missing values:</a>

# In[111]:


null_values = kiva_loans.isnull().sum()
null_values.columns = ['total_null']
total_cells = np.product(kiva_loans.shape)
missing_values = null_values.sum()

print('Only ', (missing_values/total_cells) * 100, 'of the dataset is missing.')


# # <a id='kiva_loans'>4 Analyzing kiva_loans</a>

# ## <a id='kiva_loans_regions'>4.1 Continents that got the most loans:</a>
# The Sub-Saharan region has the most loans invested. Let's see if the individual countries confirm this.

# In[112]:


kiva_loan_regions = pd.DataFrame(kiva_mpi_region_locations['world_region'].value_counts())
kiva_loan_regions.reset_index(inplace=True)
kiva_loan_regions.columns = ['world_region', 'total_amount']

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=kiva_loan_regions['total_amount'], y=kiva_loan_regions['world_region'])
barplot.set(xlabel='', ylabel='')
plt.title('Regions that got most of the loans:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kiva_loans_countries'>4.1.1 Countries that got the most loans:</a>
# According to the dataset (only showing 25 countries), Kiva is investing more than twice the amount of money into the Philippines than the 2nd country (Kenya). After that, the investments have more of a steady decline.

# In[113]:


kiva_loans_countries = pd.DataFrame(kiva_loans['country'].value_counts(sort=['loan_amount']))
kiva_loans_countries.reset_index(inplace=True)
kiva_loans_countries.columns = ['country', 'total_loaned']

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=kiva_loans_countries['total_loaned'][:20], y=kiva_loans_countries['country'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Top 20 countries that got the most loans:', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# ### <a id='kiva_loans_country_mpi'>4.1.2 Countries by mpi:</a>
# Even though the Sub-Saharan world region includes more countries with low MPI, the lowest MPI found are in Latin America and Caribbean.

# In[114]:


plt.figure(figsize=(20, 20))

pointplot = sns.pointplot(x=kiva_mpi_region_locations['mpi'], y=kiva_mpi_region_locations['country'], hue=kiva_mpi_region_locations['world_region'])
pointplot.set(xlabel='', ylabel='')
plt.yticks(fontsize=17)
plt.yticks(fontsize=12)
plt.show()


# ## <a id='kiva_loans_currency'>4.2 Most popular currency:</a>
# The Philippine currency is the most popular, followed by the USD.

# In[115]:


kiva_currency = pd.DataFrame(kiva_loans['currency'].value_counts(sort='country'))

plt.figure(figsize=(20, 7))
sns.set_style("whitegrid")

barplot = sns.barplot(x=kiva_currency.index[:15], y=kiva_currency['currency'][:15])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the 15 most popular currency used:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ## <a id='kiva_loans_sectors'>4.3 Sectors that got the most money invested:</a>
# Agriculture seems to be the most popular investment for Kiva, followed by food and retail slighlty lagging behind. This will make sense since agriculture tends to be the most common investment for poor countries, with a low level of training and easily accessible to people.

# In[116]:


kiva_loans['loan_amount_log'] = np.log(kiva_loans['loan_amount'])

plt.figure(figsize=(20, 7))

sns.set_style("whitegrid")
boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=kiva_loans)
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying all sectors that got loans:', fontsize=20)
plt.xticks(rotation=60, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kiva_loans_frequent_investment'>4.3.1 Sectors investment average:</a>
# Even though most of the money gets invested in agriculture and food, entertainment seems to have a high mean, followed by wholesale. This discrepancy can be due to low amount of loans and a high investment value.

# In[117]:


kiva_loans_sectors = pd.DataFrame(kiva_loans.groupby(['sector'])['loan_amount'].mean())
kiva_loans_sectors.reset_index(inplace=True)
kiva_loans_sectors.columns = ['sector', 'average_frequent_sectors']

plt.figure(figsize=(20, 7))

sns.set_style("whitegrid")
boxplot = sns.barplot(x='sector', y='average_frequent_sectors', data=kiva_loans_sectors)
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying the most frequent sectors that get loans:', fontsize=20)
plt.xticks(rotation=60, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kiva_loans_counting'>4.3.2 Counting the total amount of loans in Entertainment, Wholesale and Agriculture :</a>
# As compared in the graph, entertainment has less amounts of investments than agriculture and some of the investments have a high value. For this reasin the average is higer than normal.

# In[118]:


kiva_loans_counting_entertainment = pd.DataFrame(kiva_loans[kiva_loans['sector'] == 'Entertainment']['loan_amount'].value_counts())
kiva_loans_counting_entertainment.reset_index(inplace=True)
kiva_loans_counting_entertainment.columns = ['total_amount', 'times_invested']
kiva_loans_counting_wholesale = pd.DataFrame(kiva_loans[kiva_loans['sector'] == 'Wholesale']['loan_amount'].value_counts())
kiva_loans_counting_wholesale.reset_index(inplace=True)
kiva_loans_counting_wholesale.columns = ['total_amount', 'times_invested']
kiva_loans_counting_agriculture = pd.DataFrame(kiva_loans[kiva_loans['sector'] == 'Agriculture']['loan_amount'].value_counts())
kiva_loans_counting_agriculture.reset_index(inplace=True)
kiva_loans_counting_agriculture.columns = ['total_amount', 'times_invested']

fig = plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(311)
ax1 = sns.pointplot(x=kiva_loans_counting_entertainment['times_invested'], y=kiva_loans_counting_entertainment['total_amount'], color='green')
ax1.set(xlabel='Times Invested', ylabel='Amount')
ax1.set_title('Displaying the frequency and values of loans in entertainment:', fontsize=20)

plt.subplot(312)
ax1 = sns.pointplot(x=kiva_loans_counting_wholesale['times_invested'], y=kiva_loans_counting_wholesale['total_amount'], color='purple')
ax1.set(xlabel='Times Invested', ylabel='Amount')
ax1.set_title('Displaying the frequency and values of loans in wholesale:', fontsize=20)

plt.subplot(313)
ax2 = sns.pointplot(x=kiva_loans_counting_agriculture['times_invested'], y=kiva_loans_counting_agriculture['total_amount'], color='pink')
ax2.set(xlabel='Times Invested', ylabel='Amount')
ax2.set_title('Displaying the frequency and values of loans in agriculture:', fontsize=20)

plt.tight_layout()
plt.show()


# ### <a id='kiva_loans_sectors_years'>4.3.3 Sectors funding throughout the years:</a>
# Agriculture seems to be the most invested sector through the years presented in the dataset.

# In[119]:


light_palette = sns.light_palette("green", as_cmap=True)
pd.crosstab(kiva_loans['year'], kiva_loans['sector']).style.background_gradient(cmap=light_palette)


# ## <a id='kiva_loans_use'>4.4 Most common usage of loans: </a>
# The top usages of investments are to provide for basic human needs. As shown in the graph above, the top 3 are for clean water, then the 45h & 5th are for hygene; and the rest are for food supplies or farming food.

# In[120]:


kiva_use = pd.DataFrame(kiva_loans['use'].value_counts(sort='loan_amount'))
kiva_use.reset_index(inplace=True)
kiva_use.columns = ['use', 'total_amount']

plt.figure(figsize=(15, 10))

barplot = sns.barplot(x=kiva_use['total_amount'][:20], y=kiva_use['use'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Top 20 usages of loans:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=15)
plt.show()


# ### <a id='kiva_loans_activity_funded'>4.4.1 Kiva's ability to fund these loans:</a>
# Kiva seems to be doing very well funding what they actually say they will fund.

# In[121]:


plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(221)
ax1 = plt.scatter(range(kiva_loans['sector'].shape[0]), np.sort(kiva_loans['loan_amount'].values))
# ax1.title('Displaying funding of usage of investments:', fontsize=20)

plt.subplot(222)
ax2 = plt.scatter(range(kiva_loans['sector'].shape[0]), np.sort(kiva_loans['funded_amount'].values))
# ax2.title('Displaying funding of usage of investments:', fontsize=20)

plt.tight_layout()
plt.show()


# ### <a id='kiva_loans_use_years'>4.4.2 Uses through the years:</a>

# In[122]:


light_palette = sns.light_palette("green", as_cmap=True)
pd.crosstab(kiva_loans['year'], kiva_loans['activity']).style.background_gradient(cmap=light_palette)


# ## <a id='kiva_loans_lenders'>4.5 Lenders from Kiva:</a>
# Females tend to invest more frequently than males. Let's investigate whether or not they invest more money than males.

# In[123]:


kiva_loans['borrower_genders'] = kiva_loans['borrower_genders'].astype(str)
gender_list = pd.DataFrame(kiva_loans['borrower_genders'].str.split(',').tolist())
kiva_loans['clean_borrower_genders'] = gender_list[0]
kiva_loans.loc[kiva_loans['clean_borrower_genders'] == 'nan', 'clean_borrower_genders'] = np.nan

kiva_gender = kiva_loans['clean_borrower_genders'].value_counts()
labels = kiva_gender.index

plt.figure(figsize=(15, 5))

patches = plt.pie(kiva_gender, autopct='%1.1f%%')
plt.legend(labels, fontsize=20)
plt.axis('equal')
plt.tight_layout()
plt.show()


# ### <a id='kiva_loans_sex_amount'>4.5.1 Which sex lends more money:</a>
# Interestingly, males tend to invest less frequently but the majority of their investments present a consistent number; even though females compose more than 75% of the total investments.

# In[124]:


sex_mean = kiva_loans.groupby('clean_borrower_genders').count()

fig = plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(211)
ax1 = sns.violinplot(kiva_loans['loan_amount'], kiva_loans['clean_borrower_genders'])
ax1.set(xlabel='', ylabel='')
ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)

plt.subplot(212)
ax2 = sns.violinplot(kiva_loans['loan_amount'], kiva_loans['clean_borrower_genders'])
ax2.set(xlabel='', ylabel='')
ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)
ax2.set_xlim(0, 2500)

plt.tight_layout()
plt.show()


# ### <a id='kiva_loans_lender_amount'>4.5.2 How many lenders usually invest in 1 loan:</a>
# It seems that a few investors are capable of investing high amounts of money. 

# In[125]:


kiva_loan = pd.DataFrame(kiva_loans['lender_count'].value_counts())
kiva_loan.reset_index(inplace=True)
kiva_loan.columns = ['lenders', 'total_amount']
kiva_loan

plt.figure(figsize=(20, 7))

pointplot = sns.pointplot(x=kiva_loan['lenders'], y=kiva_loan['total_amount'], color='g')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying the 25 most common amounts of lenders that invest in one loan:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim(0, 25)
plt.show()


# ### <a id='kiva_loans_sector_sex'>4.5.3 How each sex invest in each sector.</a>
# This confirms that Agriculture is the most invested sector. Also, interestingly males tend to invest a higher amount than females.

# In[126]:


plt.figure(figsize=(20, 7))

boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=kiva_loans, hue='clean_borrower_genders')
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying how each gender invest in each sector:', fontsize=17)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kiva_loans_activity_sex'>4.5.3 How each sex invest in each activity:</a>
# There seems to be an pretty even correlation between gender when investing in the different activities.

# In[127]:


plt.figure(figsize=(25, 7))

boxplot = sns.pointplot(x='activity', y='loan_amount_log', data=kiva_loans, hue='clean_borrower_genders')
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying how each gender invest in each activity:', fontsize=17)
plt.xticks(rotation=80, fontsize=8)
plt.yticks(fontsize=17)
plt.show()


# ## <a id='kiva_loans_repayment'>4.6 Most popular repayment type:</a>
# It seems that most of the loans get repaid in a monthly basis, followed by irregular payments.

# In[128]:


facetgrid = sns.FacetGrid(kiva_loans, hue='repayment_interval', size=5, aspect=3)
facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))


# ### <a id='kiva_loans_repayment_year'>4.6.1 Repayments by year:</a>
# It seems that all the years that are found in the dataset have an overall similar repayment.

# In[129]:


facetgrid = sns.FacetGrid(kiva_loans, hue='year', size=5, aspect=3)
facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))


# ### <a id='kiva_loans_terms'>4.6.2 How long on average it takes to pay the loans in months:</a>
# On average it takes for loans to be repaid between 7 to 15 months.

# In[130]:


kiva_terms = pd.DataFrame(kiva_loans['term_in_months'].value_counts(sort='country'))
kiva_terms.reset_index(inplace=True)
kiva_terms.columns = ['term_in_months', 'total_amount']

plt.figure(figsize=(20, 7))

pointplot = sns.pointplot(x=kiva_terms['term_in_months'], y=kiva_terms['total_amount'], color='g')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying how long in average the monthly terms are:', fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(0, 30)
plt.show()


# # <a id='loan_themes'>5 Analyzing loan_themes_by_region file:</a>

# In[131]:


# Analyzing loan themes by region:
loan_themes_by_region.head()


# ## <a id='loan_themes_partners'>5.1 Partners that invest frequently:</a>

# In[132]:


loan_partner = pd.DataFrame(loan_themes_by_region['field partner name'].value_counts(sort=['amount']))
loan_partner.reset_index(inplace=True)
loan_partner.columns = ['partner_name', 'total_amount']
loan_partner.head()

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=loan_partner['total_amount'][:20], y=loan_partner['partner_name'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the partners that invest the most times:', fontsize=20)
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='loan_themes_amount'>5.1.1 Partners that invest the most money:</a>

# In[133]:


loan_amount = loan_themes_by_region.groupby('field partner name').sum().sort_values(by='amount', ascending=False)
loan_amount.reset_index(inplace=True)
loan_amount.head()

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=loan_amount['amount'][:20], y=loan_amount['field partner name'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the lenders that invested the most money:', fontsize=20)
plt.xticks(rotation=80, fontsize=15)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='loan_themes_regions'>5.1.2 Regions that partners invest the most:</a>

# In[134]:


loan_region = pd.DataFrame(loan_themes_by_region['region'].value_counts())
loan_region.reset_index(inplace=True)
loan_region.columns = ['region', 'total_amount']

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=loan_region['total_amount'][:20], y=loan_region['region'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the most common regions that partners invest in:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='loan_themes_country'>5.1.3 Countries that partners invest the most:</a>
# Philippines as shown before is the most common country that gets invested in. However, there are some differences on the following countries.

# In[135]:


loan_country = pd.DataFrame(loan_themes_by_region['country'].value_counts())
loan_country.reset_index(inplace=True)
loan_country.columns = ['region', 'total_amount']

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=loan_country['total_amount'][:20], y=loan_country['region'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the most common countries that partners invest in:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ## <a id='loan_theme_correlation'> 5.2 Most common themes:</a>

# In[136]:


loan_theme = pd.DataFrame(loan_themes_by_region['loan theme type'].value_counts()).reset_index()
loan_theme.columns = ['theme', 'total_amount']

plt.figure(figsize=(20, 7))

barplot = sns.pointplot(x=loan_theme['theme'][:15], y=loan_theme['total_amount'][:15], color='g')
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the most common themes that partners invest in:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='loan_theme_general_theme'>5.2.1 General theme in more detail:</a>
# It seems that the theme correlates with kiva_loans file. Even though it does not specifically say where the loans where invested, a general terminolgy was used to group these loans.

# In[137]:


loan_general = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['loan theme type'] == 'General'])
loan_general = pd.DataFrame(loan_general['sector'].value_counts().reset_index())
loan_general.columns = ['sector', 'total_amount']

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=loan_general['sector'][:10], y=loan_general['total_amount'][:10])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the most common sector with general theme that partners invest in:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ## <a id='kiva_specific'>5.3 Loans created specifically for Kiva:</a>

# In[138]:


# loans_kiva = pd.DataFrame(loan_themes_by_region['forkiva'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

pointplot = sns.pointplot(x='sector', y='amount', hue='forkiva', data=loan_themes_by_region)
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying loans that were for Kiva based in sector:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# # <a id='philippines'>6 Analyzing the Philippines</a>

# ## <a id='philippines_intro'>6.1 Introduction of the Philippines:</a>
# The Philippine Islands became a Spanish colony during the 16th century; they were ceded to the US in 1898 following the Spanish-American War. In 1935 the Philippines became a self-governing commonwealth. Manuel QUEZON was elected president and was tasked with preparing the country for independence after a 10-year transition. In 1942 the islands fell under Japanese occupation during World War II, and US forces and Filipinos fought together during 1944-45 to regain control. On 4 July 1946 the Republic of the Philippines attained its independence. A 20-year rule by Ferdinand MARCOS ended in 1986, when a "people power" movement in Manila ("EDSA 1") forced him into exile and installed Corazon AQUINO as president. Her presidency was hampered by several coup attempts that prevented a return to full political stability and economic development. Fidel RAMOS was elected president in 1992. His administration was marked by increased stability and by progress on economic reforms. In 1992, the US closed its last military bases on the islands. Joseph ESTRADA was elected president in 1998. He was succeeded by his vice-president, Gloria MACAPAGAL-ARROYO, in January 2001 after ESTRADA's stormy impeachment trial on corruption charges broke down and another "people power" movement ("EDSA 2") demanded his resignation. MACAPAGAL-ARROYO was elected to a six-year term as president in May 2004. Her presidency was marred by several corruption allegations but the Philippine economy was one of the few to avoid contraction following the 2008 global financial crisis, expanding each year of her administration. Benigno AQUINO III was elected to a six-year term as president in May 2010 and was succeeded by Rodrigo DUTERTE in May 2016.<br>
# The Philippine Government faces threats from several groups, some of which are on the US Government's Foreign Terrorist Organization list. Manila has waged a decades-long struggle against ethnic Moro insurgencies in the southern Philippines, which has led to a peace accord with the Moro National Liberation Front and ongoing peace talks with the Moro Islamic Liberation Front. The decades-long Maoist-inspired New People's Army insurgency also operates through much of the country. In 2016, Philippine armed forces battled an ISIS-Philippines siege in Marawi City, driving DUTERTE to declare martial law in the region. The Philippines faces increased tension with China over disputed territorial and maritime claims in the South China Sea.<br>
# <em>source: https://www.cia.gov/library/publications/the-world-factbook/geos/rp.html</em>
# 

# ### <a id='philippines_dataset'>6.1.1 Gathering only the Philippines from the dataset:</a>

# In[139]:


philippines = pd.DataFrame(kiva_loans[kiva_loans['country'] == 'Philippines'])
philippines_partners = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['country'] == 'Philippines'])
philippines.head()


# ### <a id='philippines_currency'>6.1.2 Philippines currency:</a>

# In[140]:


kiva_php = 52.0640
kiva_usd = 1.00
# source: http://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To=PHP

philippines_php = philippines['loan_amount'].sum()
philippines_transform = philippines_php / kiva_php
print('Total amount invested in PHP: ₱', philippines_php)
print('Amount invested in USD: #',  philippines_transform)


# ### <a id='philippines_mpi'>6.1.3 Philippines MPI:</a>

# In[141]:


kiva_mpi_region_locations_philippines = pd.DataFrame(kiva_mpi_region_locations[kiva_mpi_region_locations['country'] == 'Philippines'])
kiva_mpi_region_locations_philippines_mpi = kiva_mpi_region_locations_philippines['mpi'].mean()
print('Philippines has a MPI of: ', kiva_mpi_region_locations_philippines_mpi)


# ## <a id='philippines_sector'>6.2 Sectors:</a>
# Agriculture is the most invested sector.

# In[142]:


plt.figure(figsize=(20, 7))

sns.set_style('whitegrid')
boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=philippines)
boxplot.set(xlabel='', ylabel='')
plt.title("Total loaned in Philippines' sectors :", fontsize=20)
plt.xticks(rotation=60, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_sector_average'>6.2.1 Sector investment average: </a>
# Almost each sector has a similar mean in comparison to the others, except for Wholesale.

# In[143]:


philippines_sector_average = pd.DataFrame(philippines.groupby(['sector'])['loan_amount'].mean().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=philippines_sector_average['sector'], y=philippines_sector_average['loan_amount'])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the mean of loans in each sector:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_sector_comparison'>6.2.2 Comparing investments in Agriculture and Wholesale: </a>
# Clearly the mean in Wholesale is a little skewed given the low amount of investment and a couple of investment being really high.

# In[144]:


philippines_counting_agriculture = pd.DataFrame(philippines[philippines['sector'] == 'Agriculture']['loan_amount'].value_counts().reset_index())
philippines_counting_wholesale = pd.DataFrame(philippines[philippines['sector'] == 'Wholesale']['loan_amount'].value_counts().reset_index())

plt.figure(figsize=(20, 7))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(211)
ax1 = sns.pointplot(x=philippines_counting_agriculture['loan_amount'], y=philippines_counting_agriculture['index'], color='purple')
ax1.set(xlabel='Times Invested', ylabel='Amount')
ax1.set_title('Displaying the frequency and values of loans in agriculture:', fontsize=20)

plt.subplot(212)
ax2 = sns.pointplot(x=philippines_counting_wholesale['loan_amount'], y=philippines_counting_wholesale['index'], color='pink')
ax2.set(xlabel='Times Invested', ylabel='Amount')
ax2.set_title('Displaying the frequency and values of loans in wholesale:', fontsize=20)

plt.tight_layout()
plt.show()


# ### <a id='philippines_sector_gender'>6.2.3 Genders in sectors: </a>

# In[145]:


plt.figure(figsize=(20, 7))

boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=philippines, hue='clean_borrower_genders')
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying how each sector got funded based on gender:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_sector_partners'>6.2.4 Partners in sectors:</a>

# In[146]:


philippines_sector_partner = pd.DataFrame(philippines_partners['sector'].value_counts().reset_index())
philippines_sector_partner.head()

plt.figure(figsize=[20, 7])

barplot = sns.barplot(x='sector', y='index', data=philippines_sector_partner)
barplot.set(xlabel='', ylabel='')
plt.title('Displaying how each sector got funded by partners:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_sector_years'>6.2.5 Sectors throughout the years:</a>

# In[147]:


light_palette = sns.light_palette("green", as_cmap=True)
pd.crosstab(philippines['year'], philippines['sector']).style.background_gradient(cmap=light_palette)


# ## <a id='philippines_activity'>6.3 Activities</a>

# In[148]:


philippines_activity = pd.DataFrame(philippines['activity'].value_counts().reset_index())
philippines_activity.columns = ['activity', 'total_amount']

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=philippines_activity['total_amount'][:20], y=philippines_activity['activity'][:20])
barplot.set(xlabel='', ylabel='')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_activity_generalstore'>6.3.1 General store from activities:</a>
# Purchasing groceries for resell is a very popular usa of loans in the Philippines.

# In[149]:


philippines_generalstore = pd.DataFrame(philippines[philippines['activity'] == 'General Store'])
philippines_generalstore = pd.DataFrame(philippines_generalstore['use'].value_counts())
philippines_generalstore.reset_index(inplace=True)
philippines_generalstore.columns = ['use', 'total_amount']

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=philippines_generalstore['total_amount'][:20], y=philippines_generalstore['use'][:20])
barplot.set(xlabel='', ylabel='')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_activity_gender'>6.3.2 Gender in activty:</a>
# It seems there is not much of a correlation between gender regarding the activities that get invested on.

# In[150]:


plt.figure(figsize=(25, 7))

pointplot = sns.pointplot(x='activity', y='loan_amount_log', data=philippines, hue='clean_borrower_genders')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying how each gender invest in each activity:', fontsize=17)
plt.xticks(rotation=80, fontsize=8)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_activity_years'>6.3.3 Activities throughout the years:</a>

# In[151]:


light_palette = sns.light_palette("green", as_cmap=True)
pd.crosstab(philippines['year'], philippines['activity']).style.background_gradient(cmap=light_palette)


# ## <a id='philippines_use'>6.4 Uses of loans</a>

# In[152]:


philippines_use = pd.DataFrame(philippines['use'].value_counts())
philippines_use.reset_index(inplace=True)
philippines_use.columns = ['use', 'total']

# Displaying the results in a pie chart:

plt.figure(figsize=(20, 8))

barplot = sns.barplot(x=philippines_use['total'][:20], y=philippines_use['use'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the top 20 most common usage of loans:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ## <a id='philippines_years'>6.5 Total amount loaned throughout the years:</a>
# The yearly loans are pretty even throughout the years, 2016 being the highest and 2017 being the lowest.

# In[153]:


philippines_amount_loan = philippines.groupby('year').count().sort_values(by='loan_amount', ascending=False)
philippines_amount_loan.reset_index(inplace=True)

plt.figure(figsize=(20, 7))

barplot = sns.pointplot(x=philippines_amount_loan['year'], y=philippines_amount_loan['loan_amount'], color='g')
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the yearly loan amounts:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='philippines_years_correlation'>6.5.1 Kiva's ability to fund the Philippines loans:</a>

# In[154]:


plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(221)
ax1 = plt.scatter(range(philippines['sector'].shape[0]), np.sort(philippines['loan_amount'].values))
# ax1.title('Displaying funding of usage of investments:', fontsize=20)

plt.subplot(222)
ax2 = plt.scatter(range(philippines['sector'].shape[0]), np.sort(philippines['funded_amount'].values))
# ax2.title('Displaying funding of usage of investments:', fontsize=20)

plt.tight_layout()
plt.show()


# ### <a id='philippines_years_repayment'>6.5.2 Repayment of loans:</a>
# The philippines has an irregular payment system rather than a monthly one. It seems that the borrowers pay their loans whenever they can instead of having a specified time frame.

# In[155]:


facetgrid = sns.FacetGrid(philippines, hue='repayment_interval', size=5, aspect=3)
facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))


# ### <a id='phillipines_years_months'>6.5.3 Repayment of loans in months:</a>
# Philippines tends to repay their loans faster than average.

# In[156]:


philippines_terms = pd.DataFrame(philippines['term_in_months'].value_counts(sort='country'))
philippines_terms.reset_index(inplace=True)
philippines_terms.columns = ['term_in_months', 'total_amount']

plt.figure(figsize=(20, 7))

pointplot = sns.pointplot(x=philippines_terms['term_in_months'], y=philippines_terms['total_amount'], color='g')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying how long in average the monthly terms are:', fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(0, 30)
plt.show()


# ## <a id='philippines_genders'>6.6 Gender:</a>
# Females tend to invest more than their male counterpart.

# In[157]:


sex_mean = philippines.groupby('clean_borrower_genders').count()

fig = plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(211)
ax1 = sns.violinplot(philippines['loan_amount'], philippines['clean_borrower_genders'])
ax1.set(xlabel='', ylabel='')
ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)

plt.subplot(212)
ax2 = sns.violinplot(philippines['loan_amount'], philippines['clean_borrower_genders'])
ax2.set(xlabel='', ylabel='')
ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)
ax2.set_xlim(0, 2500)

plt.tight_layout()
plt.show()


# ## <a id='philippines_partners'>6.7 Partners:</a>

# In[158]:


philippines_partners_count = pd.DataFrame(philippines_partners['field partner name'].value_counts().reset_index())
plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=philippines_partners_count['field partner name'], y=philippines_partners_count['index'])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying partners that invest in the Philippines:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# # <a id='kenya'>7 Analyzing Kenya</a>

# ## <a id='kenya_intro'>7.1 Introduction of Kenya:</a>

# Founding president and liberation struggle icon Jomo KENYATTA led Kenya from independence in 1963 until his death in 1978, when Vice President Daniel MOI took power in a constitutional succession. The country was a de facto one-party state from 1969 until 1982, after which time the ruling Kenya African National Union (KANU) changed the constitution to make itself the sole legal party in Kenya. MOI acceded to internal and external pressure for political liberalization in late 1991. The ethnically fractured opposition failed to dislodge KANU from power in elections in 1992 and 1997, which were marred by violence and fraud, but were viewed as having generally reflected the will of the Kenyan people. President MOI stepped down in December 2002 following fair and peaceful elections. Mwai KIBAKI, running as the candidate of the multiethnic, united opposition group, the National Rainbow Coalition (NARC), defeated KANU candidate Uhuru KENYATTA, the son of founding president Jomo KENYATTA, and assumed the presidency following a campaign centered on an anticorruption platform.<br>
# KIBAKI's reelection in December 2007 brought charges of vote rigging from Orange Democratic Movement (ODM) candidate Raila ODINGA and unleashed two months of violence in which approximately 1,100 people died. African Union-sponsored mediation led by former UN Secretary General Kofi ANNAN in late February 2008 resulted in a power-sharing accord bringing ODINGA into the government in the restored position of prime minister. The power sharing accord included a broad reform agenda, the centerpiece of which was constitutional reform. In August 2010, Kenyans overwhelmingly adopted a new constitution in a national referendum. The new constitution introduced additional checks and balances to executive power and significant devolution of power and resources to 47 newly created counties. It also eliminated the position of prime minister following the first presidential election under the new constitution, which occurred in March 2013. Uhuru KENYATTA won the election and was sworn into office in April 2013; he began a second term in November 2017.<br>
# <em>Source: https://www.cia.gov/library/publications/the-world-factbook/geos/ke.html</em>

# ### <a id='kenya_dataset'>7.1.1 Gathering only Kenya from the dataset:</a>

# In[159]:


kenya = pd.DataFrame(kiva_loans[kiva_loans['country'] == 'Kenya'])
kenya_partners = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['country'] == 'Kenya'])
kenya.head()


# ### <a id='kenya_mpi'>7.1.2 Kenya's MPI:</a>

# In[160]:


kenya_mpi = pd.DataFrame(MPI_subnational[MPI_subnational['country'] == 'Kenya'])
kenya_mpi = kenya_mpi['mpi national'].mean()
print("Kenya's MPI is:", kenya_mpi)


# ### <a id='kenya_currency'>7.1.3 Kenya's currency value against USD:</a>

# In[161]:


kiva_kes = 100.801

kenya_loan_amount = kenya['loan_amount'].sum()
# sounrce: http://www.xe.com/currencyconverter/convert/?Amount=1&From=USD&To=KES

kiva_transform = kenya_loan_amount / kiva_kes
print('Total amount invested in Kenya: KSh', kenya_loan_amount)
print('Amount invested in USD: $', kiva_transform)


# ## <a id='kenya_sector'>7.2 Sectors:</a>
# Agriculture has the highest investment amount followed by retail.

# In[162]:


# Creating and saving the different sectors:
plt.figure(figsize=(20, 7))

sns.set_style('whitegrid')
boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=kenya)
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying all the sectors and their repective loan amounts:', fontsize=20)
plt.xticks(rotation=60, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kenya_sector_average'>7.2.1 Sector investment average:</a>
# Even though Agriculture is the most popular investment, Health has a higher average.

# In[163]:


kenya_sector_average = kenya.groupby(['sector'])['loan_amount'].mean().reset_index()
kenya_sector_average.head()

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=kenya_sector_average['sector'], y=kenya_sector_average['loan_amount'])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the average loan amount in each sector:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kenya_sector_comparison'>7.2.2 Comparing the average with the most invested popular:</a>

# In[164]:


kenya_average_health = pd.DataFrame(kenya[kenya['sector'] == 'Health']['loan_amount'].value_counts().reset_index())
kenya_average_agriculture = pd.DataFrame(kenya[kenya['sector'] == 'Agriculture']['loan_amount'].value_counts().reset_index())

plt.figure(figsize=(20, 7))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(211)
ax1 = sns.pointplot(x=kenya_average_health['loan_amount'], y=kenya_average_health['index'], color='green')
ax1.set(xlabel='Times Invested', ylabel='Amount')
ax1.set_title('Displaying the frequency and values of loans in Health:', fontsize=20)

plt.subplot(212)
ax2 = sns.pointplot(x=kenya_average_agriculture['loan_amount'], y=kenya_average_agriculture['index'], color='pink')
ax2.set(xlabel='Times Invested', ylabel='Amount')
ax2.set_title('Displaying the frequency and values of loans in Agriculture:', fontsize=20)

plt.tight_layout()
plt.show()


# ### <a id='kenya_sector_gender'>7.2.3 Genders in sector:</a>
# It seems that males tend to invest more than demales in every secotr but not by that much.

# In[165]:


plt.figure(figsize=(20, 7))

boxplot = sns.boxplot(x='sector', y='loan_amount_log', hue='clean_borrower_genders', data=kenya)
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying genders investment in each sector:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kenya_sector_partner'>7.2.4 Partners in sector</a>

# In[166]:


kenya_sector_partners = pd.DataFrame(kenya_partners['sector'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=kenya_sector_partners['sector'], y=kenya_sector_partners['index'])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying how each partner invest in each sector:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kenya_sector_years'>7.2.5 Sectors funded thoughout the years:</a>
# Consistently, Agriculture is the most popular investment. Also, 2017 is the year with the lowest o investment.

# In[167]:


light_palette = sns.light_palette('green', as_cmap=True)
pd.crosstab(kenya['year'], kenya['sector']).style.background_gradient(cmap=light_palette)


# ## <a id='kenya_activity'>7.3 Activity:</a>
# Kenya's investment is more in par to the overall investments since agriculture is considered to be the more succesful type of investment in poor countries, with the highest yield of return and tends to have an accessible entry point for most people.

# In[168]:


kenya_activity = pd.DataFrame(kenya['activity'].value_counts(sort=['loan_amount']))
kenya_activity.reset_index(inplace=True)
kenya_activity.columns = ['activity', 'total_amount']

# Displaying each activity in a pie chart:

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=kenya_activity['total_amount'][:20], y=kenya_activity['activity'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the activities in the agriculture sector:', fontsize=20)
plt.show()


# ### <a id='kenya_activity_farming'>7.3.1 Farming from activities:</a>
# Most popular uses of loans in farming are to modernize the industry with efficient technologies or to upgrade them.

# In[169]:


kenya_farming = pd.DataFrame(kenya[kenya['activity'] == 'Farming']['use'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=kenya_farming['use'][:20], y=kenya_farming['index'][:20])
plt.show()


# ### <a id='kenya_activity_gender'>7.3.2 Gender in activity:</a>

# In[170]:


plt.figure(figsize=(25, 7))

pointplot = sns.pointplot(x='activity', y='loan_amount_log', data=kenya, hue='clean_borrower_genders')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying how each gender invest in each activity:', fontsize=17)
plt.xticks(rotation=80, fontsize=8)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kenya_activity_years'>7.3.3 Activities throughout the years:<a/>

# In[171]:


light_palette = sns.light_palette("green", as_cmap=True)
pd.crosstab(kenya['year'], kenya['activity']).style.background_gradient(cmap=light_palette)


# # <a id='kenya_uses'>7.4 Uses:</a>

# In[172]:


kenya_use = pd.DataFrame(kenya['use'].value_counts().reset_index())

plt.figure(figsize=(20, 8))

barplot = sns.barplot(x=kenya_use['use'][:20], y=kenya_use['index'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the top 20 most common usage of loans:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ## <a id='kenya_years'>7.5 Total amount loaned throughout the years:</a>
# Kenya also showed a steady increase in loan investment; however, it declined steeply on 2017

# In[173]:


kenya_amount_year = kenya.groupby('year').count().sort_values(by='loan_amount', ascending=False)
kenya_amount_year.reset_index(inplace=True)

plt.figure(figsize=(20, 7))

pointplot = sns.pointplot(x=kenya_amount_year['year'], y=kenya_amount_year['loan_amount'], color='g')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying the yearly loan amounts:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='kenya_loans_funding'>7.5.1 Kiva's ability to fund these loans:</a>
# Kiva is also good at funding Kenya's loans.

# In[174]:


plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(221)
ax1 = plt.scatter(range(kenya['sector'].shape[0]), np.sort(kenya['loan_amount'].values))
# ax1.title('Displaying funding of usage of investments:', fontsize=20)

plt.subplot(222)
ax2 = plt.scatter(range(kenya['sector'].shape[0]), np.sort(kenya['funded_amount'].values))
# ax2.title('Displaying funding of usage of investments:', fontsize=20)

plt.tight_layout()
plt.show()


# ### <a id='kenya_loans_repayment'>7.5.2 Displaying how loans get repaid:</a>
# Monthly payments seem to be the most common form of payments but weekly payments have a big repayment amounts.

# In[175]:


facetgrid = sns.FacetGrid(kenya, hue='repayment_interval', size=5, aspect=3)
facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))


# ### <a id='kenya_loans_months'>7.5.3 How long it takes to repay the loans in months:</a>
# Kenya takes 14 months on average to repay it's loans which is within the average.

# In[176]:


kenya_terms = pd.DataFrame(kenya['term_in_months'].value_counts(sort='country'))
kenya_terms.reset_index(inplace=True)
kenya_terms.columns = ['term_in_months', 'total_amount']

plt.figure(figsize=(20, 7))

pointplot = sns.pointplot(x=kenya_terms['term_in_months'], y=kenya_terms['total_amount'], color='g')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying how long in average the monthly terms are:', fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(0, 30)
plt.show()


# # <a id=kenya_gender''>7.6 Gender</a>

# In[177]:


fig = plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(211)
ax1 = sns.violinplot(kenya['loan_amount'], kenya['clean_borrower_genders'])
ax1.set(xlabel='', ylabel='')
ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)

plt.subplot(212)
ax2 = sns.violinplot(kenya['loan_amount'], kenya['clean_borrower_genders'])
ax2.set(xlabel='', ylabel='')
ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)
ax2.set_xlim(0, 2000)

plt.tight_layout()
plt.show()


# ## <a id='kenya_partners'>7.7 Partners:</a>

# In[178]:


kenya_partners_count = pd.DataFrame(loan_themes_by_region['field partner name'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=kenya_partners_count['field partner name'][:20], y=kenya_partners_count['index'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying partners that invest in the Kenya:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# # <a id='salvador'>8 Analyzing El Salvador</a>

# ## <a id='salvador_intro'>8.1 Introduction to El Salvador:</a>

# El Salvador achieved independence from Spain in 1821 and from the Central American Federation in 1839. A 12-year civil war, which cost about 75,000 lives, was brought to a close in 1992 when the government and leftist rebels signed a treaty that provided for military and political reforms. El Salvador is beset by one of the world’s highest homicide rates and pervasive criminal gangs.<br>
# <em>Source: https://www.cia.gov/library/publications/the-world-factbook/geos/es.html</em>

# ## <a id='salvador_dataset'>8.1.1 Gathering only El Salvador from the dataset:</a>

# In[179]:


salvador = kiva_loans[kiva_loans['country'] == 'El Salvador']
salvador_partners = pd.DataFrame(loan_themes_by_region[loan_themes_by_region['country'] == 'El Salvador'])
salvador.head()


# ### <a id='salvador_mpi'> 8.1.2 Salvador's MPI</a>

# In[180]:


salvador_mpi = pd.DataFrame(MPI_subnational[MPI_subnational['country'] == 'El Salvador'])
salvador_mpi = salvador_mpi['mpi national'].mean()
print("El Salvador's MPI is:", salvador_mpi)


# ## <a id='salvador_sector'>8.2 Sectors</a>

# In[181]:


plt.figure(figsize=(20, 7))

sns.set_style('whitegrid')
boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=salvador)
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying each sector with their respective loans invested:', fontsize=20)
plt.xticks(rotation=60, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_sector_average'>8.2.1 Sector investment average:</a>

# In[182]:


salvador_sector_average = salvador.groupby('sector')['loan_amount'].mean().reset_index()

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=salvador_sector_average['sector'], y=salvador_sector_average['loan_amount'])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the average investment in sectors:', fontsize=(20))
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_sector_comparison'>8.2.2 Comparing the averages:</a>
# Wholesale had loans that were invested by a single investors and most of the loans were bigger skewing the average up.

# In[183]:


salvador_sector_wholesale = pd.DataFrame(salvador[salvador['sector'] == 'Wholesale']['loan_amount'].value_counts().reset_index())
salvador_sector_transportation = pd.DataFrame(salvador[salvador['sector'] == 'Transportation']['loan_amount'].value_counts().reset_index())
salvador_sector_agriculture = pd.DataFrame(salvador[salvador['sector'] == 'Agriculture']['loan_amount'].value_counts().reset_index())

plt.figure(figsize=(20, 7))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(311)
ax1 = sns.pointplot(x='loan_amount', y='index', data=salvador_sector_wholesale, color='red')
ax1.set(xlabel='Times Invested', ylabel='Amount')
ax1.set_title('Displaying the frequency and values of loans in Wholesale:', fontsize=20)

plt.subplot(312)
ax2 = sns.pointplot(x='loan_amount', y='index', data=salvador_sector_transportation, color='purple')
ax2.set(xlabel='Times Invested', ylabel='Amount')
ax2.set_title('Displaying the frequency and values of loans in Transportation:', fontsize=20)

plt.subplot(313)
ax3 = sns.pointplot(x='loan_amount', y='index', data=salvador_sector_agriculture, color='pink')
ax3.set(xlabel='Times Invested', ylabel='Amount')
ax3.set_title('Displaying the frequency and values of loans in Agriculture:', fontsize=20)

plt.tight_layout()
plt.show()


# ### <a id='salvador_sector_gender'>8.2.3 Genders in sector:</a>
# There is a pretty even correlation between males and females. However, males to invest more frequently in sectors where there is a difference.

# In[184]:


plt.figure(figsize=(20, 7))

boxplot = sns.boxplot(x='sector', y='loan_amount_log', data=salvador, hue='clean_borrower_genders')
boxplot.set(xlabel='', ylabel='')
plt.title('Displaying genders investment in each sector:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.xticks(fontsize=17)
plt.show()


# ### <a id='salvador_sector_partner'>8.2.4 Partners in sector:</a>
# Partners only invest in general financial inclusion.

# In[185]:


salvador_sector_partners = pd.DataFrame(salvador_partners['sector'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x='sector', y='index', data=salvador_sector_partners)
barplot.set(xlabel='', ylabel='')
plt.title('Displaying partners investments:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_sector_theme'>8.2.5 Themes in general inclusion:</a>

# In[186]:


salvador_partner_theme_type = pd.DataFrame(salvador_partners['loan theme type'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x='loan theme type', y='index', data=salvador_partner_theme_type)
barplot.set(xlabel='', ylabel='')
plt.title('Displaying themes from partners:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_sector_years'>8.2.6 Sectors throughout the years:</a>

# In[187]:


light_palette = sns.light_palette("green", as_cmap=True)
pd.crosstab(salvador['year'], salvador['sector']).style.background_gradient(cmap=light_palette)


# ## <a id='salvador_activity'>8.3 Activity</a>
# Personal housing expenses is the most popular activity in El Salvador.

# In[188]:


salvador_activity = pd.DataFrame(salvador['activity'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=salvador_activity['activity'][:20], y=salvador_activity['index'][:20])
barplot.set(xlabel='', ylabel='')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_activity_uses'>8.3.1 Uses in Personal Housing Expenses:</a>

# In[189]:


salvador_housing = pd.DataFrame(salvador[salvador['activity'] == 'Personal Housing Expenses']['use'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=salvador_housing['use'][:20], y=salvador_housing['index'][:20])
barplot.set(xlabel='', ylabel='')
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_activity_gender'>8.3.2 Gender in activity:</a>

# In[190]:


plt.figure(figsize=(25, 7))

pointplot = sns.pointplot(x='activity', y='loan_amount_log', data=salvador, hue='clean_borrower_genders')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying how each gender invest in each activity:', fontsize=17)
plt.xticks(rotation=80, fontsize=8)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_activity_years'>8.3.3 Activities throughout the years:</a>

# In[191]:


light_palette = sns.light_palette("green", as_cmap=True)
pd.crosstab(salvador['year'], salvador['activity']).style.background_gradient(cmap=light_palette)


# ## <a id='salvador_use'>8.4 Uses:</a>

# In[192]:


salvador_use = pd.DataFrame(salvador['use'].value_counts().reset_index())

plt.figure(figsize=(20, 8))

barplot = sns.barplot(x=salvador_use['use'][:20], y=salvador_use['index'][:20])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the top 20 most common usage of loans:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ## <a id='salvador_years'>8.5 Total amount loaned thoughout the years:</a>

# In[193]:


salvador_amount_loan = salvador.groupby('year').count().sort_values(by='loan_amount', ascending=False).reset_index()

plt.figure(figsize=(20, 7))

barplot = sns.pointplot(x=salvador_amount_loan['year'], y=salvador_amount_loan['loan_amount'], color='g')
barplot.set(xlabel='', ylabel='')
plt.title('Displaying the yearly loan amounts:', fontsize=20)
plt.xticks(rotation=80, fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# ### <a id='salvador_years_funding'>8.5.1 Kiva's ability to fund the Philippines loans:</a>

# In[194]:


plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(221)
ax1 = plt.scatter(range(salvador['sector'].shape[0]), np.sort(salvador['loan_amount'].values))

plt.subplot(222)
ax2 = plt.scatter(range(salvador['sector'].shape[0]), np.sort(salvador['funded_amount'].values))

plt.tight_layout()
plt.show()


# ### <a id='salvador_years_repayment'>8.5.2 Repayment of loans:</a>

# In[195]:


facetgrid = sns.FacetGrid(salvador, hue='repayment_interval', size=5, aspect=3)
facetgrid = (facetgrid.map(sns.kdeplot, 'loan_amount_log', shade=True).set_axis_labels('Months', 'Total Amount (log)').add_legend(fontsize=17))


# ### <a id='salvador_years_months'>8.5.3 Repayment of loans in months:</a>

# In[196]:


salvador_terms = pd.DataFrame(salvador['term_in_months'].value_counts(sort='country').reset_index())

plt.figure(figsize=(20, 7))

pointplot = sns.pointplot(x=salvador_terms['index'], y=salvador_terms['term_in_months'], color='g')
pointplot.set(xlabel='', ylabel='')
plt.title('Displaying how long in average the monthly terms are:', fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlim(0, 30)
plt.show()


# ## <a id='salvador_gender'>8.6 Gender</a>

# In[197]:


fig = plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17

plt.subplot(211)
ax1 = sns.violinplot(salvador['loan_amount'], salvador['clean_borrower_genders'])
ax1.set(xlabel='', ylabel='')
ax1.set_title('Displaying the total amount of money loaned by gender:', fontsize=20)

plt.subplot(212)
ax2 = sns.violinplot(salvador['loan_amount'], salvador['clean_borrower_genders'])
ax2.set(xlabel='', ylabel='')
ax2.set_title('Displaying a closer look of the initial part of the violinplot for better visualization of distribution:', fontsize=20)
ax2.set_xlim(0, 1750)

plt.tight_layout()
plt.show()


# ## <a id='salvador_partners'>8.7 Partners:</a>

# In[198]:


salvador_partners_count = pd.DataFrame(salvador_partners['field partner name'].value_counts().reset_index())

plt.figure(figsize=(20, 7))

barplot = sns.barplot(x=salvador_partners_count['field partner name'], y=salvador_partners_count['index'])
barplot.set(xlabel='', ylabel='')
plt.title('Displaying partners that invest in the Philippines:', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# # <a id='mpi'>9 Deprivation levels:</a>

# In[199]:


MPI_national.head()


# In[200]:


MPI_subnational.head()


# In[222]:


ordered_MPI_national = MPI_national.sort_values(by='intensity of deprivation rural', ascending=False)
ordered_MPI_subnational = MPI_subnational.sort_values(by='intensity of deprivation regional', ascending=False)

fig = plt.figure(figsize=(20, 10))
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 13

plt.subplot(121)
ax1 = sns.barplot(x=ordered_MPI_national['intensity of deprivation rural'][:20], y=ordered_MPI_national['country'][:20])
ax1.set(xlabel='', ylabel='')
ax1.set_title('Rural deprivation intensity:', fontsize=20)

plt.subplot(122)
ax2 = sns.barplot(x=ordered_MPI_subnational['intensity of deprivation regional'][:10], y=ordered_MPI_subnational['sub-national region'][:10])
ax2.set(xlabel='', ylabel='')
ax2.set_title('Sub-regional deprivation intensity:', fontsize=20)

plt.show()


# # <a id='conclusions'>10 Conclusions:</a>

# - Loans:
#     - The Sub-Saharan Africa is the region most heavely invested.
#     - Philippines is the country most heavely invested.
#     - Latin American and Caribean have the lowest MPI on average.
#     - The PHP (Philippine currency) is the most common (can be due by how low it compares against the USD).
#     - Kiva has a good reputation funding their investments.
# - Sectors:
#     - Agriculture is the most heavely invested followed by Food.
#     - Entertainment holds a higher average meandue to lower frequency of investments and larger amounts loaned in one loan.
#     - 
# - Uses:
#     - Buying filters for clean water is the most common investment.
# - Gender:
#     - Females invest more frequently in Kiva than their male counter part.
#     - Males invest more money overall.
#     - On average it requires between 1 to 12 lender to fully fund an investment.
#     
