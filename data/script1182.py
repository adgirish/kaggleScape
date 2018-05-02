
# coding: utf-8

# ![](http://www.homeandbuild.ie/wp-content/uploads/2018/01/Best-loan-advisor-in-Rajkot3_adtubeindia.jpg)

# # More To Come. Stay Tuned. !!
# If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# **This notebook will always be a work in progress.** Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!.** If you like it or it helps you , you can upvote and/or leave a comment :).**

# - <a href='#intro'>1. Introduction</a>  
# - <a href='#rtd'>2. Retrieving the Data</a>
#      - <a href='#ll'>2.1 Load libraries</a>
#      - <a href='#rrtd'>2.2 Read the Data</a>
# - <a href='#god'>3. Glimpse of Data</a>
#      - <a href='#oot'>3.1 Overview of tables</a>
#      - <a href='#sootd'>3.2 Statistical overview of the Data</a>
# - <a href='#dp'>4. Data preparation</a>
#      - <a href='#cfmd'> 4.1 Check for missing data</a>
# - <a href='#de'>5. Data Exploration</a>
#      - <a href='#tsiwm'>5.1 Top sectors in which more loans were given</a>
#      - <a href='#tori'>5.2 Types of repayment intervals</a>
#      - <a href='#mfcwg'>5.3 Most frequent countries who got loans</a>
#      - <a href='#d'>5.4 Distribution </a>
#          - <a href='#dofa'> 5.4.a Distribution of funded anount</a>
#          - <a href='#dola'> 5.4.b Distribution of loan amount</a>
#          - <a href='#dowr'> 5.4.c Distribution of world regions</a>
#          - <a href='#dolc'> 5.4.d Distribution of lender count</a>
#          - <a href='#dolat'> 5.4.e Distribution of loan Activity type</a>
#          - <a href='#dotim'> 5.4.f Distribution of terms_in_month</a>
#          - <a href='#dos'> 5.4.g Distribution of Sectors</a>
#          - <a href='#doa'> 5.4.h Distribution of Activity</a>
#          - <a href='#dori'> 5.4.i Distribution of repayment intervals</a>
#          - <a href='#dompu'>5.4.j Distribution of Most popular uses of loans</a>
#      - <a href='#fm'> 5.5 Borrower Gender: Female V.S. Male</a>  
#      - <a href='#sbri'> 5.6 Sex_borrower V.S. Repayment_intervals</a>
#      - <a href='#kf'>5.7 Kiva Field partner name V.S. Funding count</a>
#      - <a href='#tcwfa'> 5.8 Top Countries with funded_amount</a>
#      - <a href='#tmrwa'> 5.9 Top mpi_regions with amount</a>
#      - <a href='#vpito'> 5.10 Various popularities in terms of loan amount</a>
#          - <a href='#plsit'> 5.10.a Popular loan sectors in terms of loan amount</a>
#          - <a href='#plait'> 5.10.b Popular loan Activities in terms of loan amount</a>
#          - <a href='#pcito'> 5.10.c Popular Countries in terms of loan amount</a>
#          - <a href='#prito'> 5.10.d Popular regions in terms of loan amount</a>
#      - <a href='#wcfcn'> 5.11 Word Cloud for countries names</a>
#      - <a href='#mlbmy'> 5.12 Mean loan by month-year with repayment intervals</a>
#      - <a href='#ydoco'> 5.13 Yearwise distribution of count of loan availed by each country</a>
#      - <a href='#cmah'> 5.14 Correlation Matrix and Heatmap</a>
#          - <a href='#saric'> 5.14.a Sectors and Repayment Intervals correlation</a>
#          - <a href='#caric'> 5.14.b Country and Repayment Intervals correlation</a>
#          - <a href='#cmaho'> 5.14.c Correlation Matrix and Heatmap of kiva_loans_data</a>
#      - <a href='#timri'> 5.15 Term_In_Months V.S. Repayment_Interval</a>
#      - <a href='#ltcsf'> 5.16 Loan theme created specifically for Kiva or not ?</a>
#      - <a href='#twfa'>5.17 Time Series Analysis</a>
#         - <a href='#toaafa'>5.17.1 Trend of loan amount V.S. funded amount</a> 
#         - <a href='#toaaf'>5.17.2 Trend of unfunded amount V.S. funded amount</a>
#         - <a href='#tdbpf'>5.17.3 Trend of Disbursed to borrower V.S. Posted on kiva.org V.S. Funded on kiva.org</a>
#      - <a href='#i'> 5.18 India</a>
#         - <a href='#tluii'> 5.18.a Top 13 loan uses in India</a>
#         - <a href='#tfrii'> 5.18.b Top 7 funded regions in India(On tha Map)</a>
#         - <a href='#mdfpn'>5.18.c Most dominant field partner names in India</a>
#         - <a href='#tolii'>5.18.d Trend of loans in India</a>
#         - <a href='#tdbpfi'>5.18.e Trend of Disbursed to borrower V.S. Posted on kiva.org V.S. Funded on kiva.org in India</a>
#      - <a href='#p'> 5.19 Philippines</a>
#         - <a href='#tluip'>5.19.a Top 13 loan uses in Philippines</a>
#         - <a href='#tfrip'> 5.19.b Top 7 funded regions in Philippines</a>
#         - <a href='#mdfpnp'>5.19.c Most dominant field partner names in Philippines</a>
#         - <a href='#tolip'>5.19.d Trend of loans in Philippines</a>
#         - <a href='#tdbpfp'>5.19.e Trend of Disbursed to borrower V.S. Posted on kiva.org V.S. Funded on kiva.org in Philippines</a>
#      - <a href='#sols'>5.20 Status of loans</a>
#      - <a href='#dmols'>5.21 Distribution models of loans</a>
#      - <a href='#trtgl'>5.22 Top Reasons to give loan</a>
#      - <a href='#uplob'>5.23 Understanding Poverty levels of Borrowers</a>
#        - <a href="#mdpia">5.23.1 Multi-dimensional Poverty Index(MPI) for different regions</a>
#        - <a href='#tcwhh'>5.23.2 Top 10 countries with higher Human Development Index(HDI) </a>
#        - <a href='#pbpl'>5.23.3. Country level Analysis</a> 
#           - <a href='#pbplc'>5.23.3.a Population below poverty line for different countries in %</a>
#           - <a href='#pphdi'>2.23.3.b Population below poverty line V.S. HDI</a>
#           - <a href='#ledc'>2.23.3.c  Life_expectancy for different countries</a> 
#           - <a href='#tcrmup'>2.23.3.d  Top countries with higher rural MPI's along with urban MPI's</a>
#           - <a href='#tcrhruhr'>2.23.3.e  Top countries with higher rural Headcount Ratio along with urban Headcount Ratio</a>
#           - <a href='#tcriduid'>2.23.3.f  Top countries with higher rural Intensity of Deprivation along with urban Intensity of Deprivation</a>
#           
# - <a href='#s'>6. Summary/Conclusion</a> 

# ## <a id='intro'>1. Intoduction</a>
# ---------------------------------------
# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical. However, this requires inference based on a limited set of information for each borrower.
# 
# For the locations in which Kiva has active loans, our objective is to pair Kiva's data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics.
# 

# # <a id='rtd'>2. Retrieving the Data</a>

# ## <a id='ll'>2.1 Load libraries</a>

# In[ ]:


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
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ## <a id='rrtd'>2.2 Read tha Data</a>

# In[ ]:


kiva_loans_data = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
kiva_mpi_locations_data = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
loan_theme_ids_data = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_by_region_data = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
# Additional kiva snapshot data set
loans_data = pd.read_csv("../input/additional-kiva-snapshot/loans.csv")
lenders_data = pd.read_csv("../input/additional-kiva-snapshot/lenders.csv")
loans_lenders_data = pd.read_csv("../input/additional-kiva-snapshot/loans_lenders.csv")
country_stats_data = pd.read_csv("../input/additional-kiva-snapshot/country_stats.csv")
#geconv4_data = pd.read_csv("../input/additional-kiva-snapshot/GEconV4.csv")
# Multidimensional Poverty Measures Data set
mpi_national_data = pd.read_csv("../input/mpi/MPI_national.csv")
mpi_subnational_data = pd.read_csv("../input/mpi/MPI_subnational.csv")


# In[ ]:


print("Size of kiva_loans_data",kiva_loans_data.shape)
print("Size of kiva_mpi_locations_data",kiva_mpi_locations_data.shape)
print("Size of loan_theme_ids_data",loan_theme_ids_data.shape)
print("Size of loan_themes_by_region_data",loan_themes_by_region_data.shape)
print("***** Additional kiva snapshot******")
print("Size of loans_data",loans_data.shape)
print("Size of lenders_data",lenders_data.shape)
print("Size of loans_lenders_data",loans_lenders_data.shape)
print("Size of country_stats_data",country_stats_data.shape)
#print("Size of geconv4_data",geconv4_data.shape)
print("*****Multidimensional Poverty Measures Data set******")
print("Size of mpi_national_data",mpi_national_data.shape)
print("Size of mpi_subnational_data",mpi_subnational_data.shape)


# # <a id='god'>3. Glimpse of Data</a>

# ## <a id='oot'>3.1 Overview of tables</a>

# **kiva_loans_data**

# In[ ]:


kiva_loans_data.head()


# **kiva_mpi_locations_data**

# In[ ]:


kiva_mpi_locations_data.head()


# **loan_theme_ids_data**

# In[ ]:


loan_theme_ids_data.head()


# **loan_themes_by_region_data**

# In[ ]:


loan_themes_by_region_data.head()


# **loans_data **

# In[ ]:


loans_data .head()


# **lenders_data** 

# In[ ]:


lenders_data.head()


# **loans_lenders_data** 

# In[ ]:


loans_lenders_data.head()


# **country_stats_data** 

# In[ ]:


country_stats_data.head() 


# **mpi_national_data**

# In[ ]:


mpi_national_data.head()


# **mpi_subnational_data**

# In[ ]:


mpi_subnational_data.head()


# ## <a id='sootd'>3.2 Statistical Overview of the Data</a>

# **kiva_loans_data some little info**

# In[ ]:


kiva_loans_data.info()


# **Little description of kiva_loans_data for numerical features**

# In[ ]:


kiva_loans_data.describe()


# **Little description of kiva_loans_data for categorical features**

# In[ ]:


kiva_loans_data.describe(include=["O"])


# # <a id='dp'>4. Data preparation</a>

# ## <a id='cfmd'>4.1 Checking for missing data</a>

# **Missing data in kiva_loans data**

# In[ ]:


# checking missing data in kiva_loans data 
total = kiva_loans_data.isnull().sum().sort_values(ascending = False)
percent = (kiva_loans_data.isnull().sum()/kiva_loans_data.isnull().count()).sort_values(ascending = False)
missing_kiva_loans_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_kiva_loans_data


# **Missing data in kiva_mpi_locations data**

# In[ ]:


# missing data in kiva_mpi_locations data 
total = kiva_mpi_locations_data.isnull().sum().sort_values(ascending = False)
percent = (kiva_mpi_locations_data.isnull().sum()/kiva_mpi_locations_data.isnull().count()).sort_values(ascending = False)
missing_kiva_mpi_locations_data= pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_kiva_mpi_locations_data


# **Missing data in loan_theme_ids data **

# In[ ]:


# missing data in loan_theme_ids data 
total = loan_theme_ids_data.isnull().sum().sort_values(ascending = False)
percent = (loan_theme_ids_data.isnull().sum()/loan_theme_ids_data.isnull().count()).sort_values(ascending = False)
missing_loan_theme_ids_data= pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_loan_theme_ids_data


# **Missing data in loan_themes_by_region data**

# In[ ]:


# missing data in loan_themes_by_region data 
total = loan_themes_by_region_data.isnull().sum().sort_values(ascending = False)
percent = (loan_themes_by_region_data.isnull().sum()/loan_themes_by_region_data.isnull().count()).sort_values(ascending = False)
missing_loan_themes_by_region_data= pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_loan_themes_by_region_data


# # <a id='de'>5. Data Exploration</a>

# ## <a id='tsiwm'>5.1 Top sectors in which more loans were given</a>

# In[ ]:


# print("Top sectors in which more loans were given : ", len(kiva_loans_data["sector"].unique()))
# print(kiva_loans_data["sector"].value_counts().head(10))
plt.figure(figsize=(15,8))
sector_name = kiva_loans_data['sector'].value_counts()
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xticks(rotation='vertical')
plt.xlabel('Number of loans were given')
plt.ylabel('Sector Name')
plt.title("Top sectors in which more loans were given")
plt.show()


# Agriculture sector is very frequent followed by Food in terms of number of loans.

#  ## <a id='tori'>5.2 Types of repayment intervals</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = kiva_loans_data['repayment_interval'].value_counts().head(10)
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Types of repayment interval', fontsize=12)
plt.title("Types of repayment intervals with their count", fontsize=16)


# In[ ]:


kiva_loans_data['repayment_interval'].value_counts().plot(kind="pie",figsize=(12,12))


# Types of repayment interval
# * Monthly (More frequent)
# * irregular
# * bullet
# * weekly (less frequent)

# ## <a id='mfcwg'>5.3 Most frequent countries who got loans</a>

# **Most frequent countries**

# In[ ]:


# Plot the most frequent countries
plt.figure(figsize=(15,8))
count = kiva_loans_data['country'].value_counts().head(10)
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('country name', fontsize=12)
plt.title("Most frequent countries for kiva loan", fontsize=16)


# * **Philippines** is most frequent **country** who got more loans** followed by Kenya**

# In[ ]:


kiva_loans_data.columns


# ## <a id='d'>5.4 Distribution </a>

# ### <a id='dofa'>5.4.a Distribution of funded anount</a>

# In[ ]:


# Distribution of funded anount
plt.figure(figsize = (12, 8))

sns.distplot(kiva_loans_data['funded_amount'])
plt.show() 
plt.figure(figsize = (12, 8))
plt.scatter(range(kiva_loans_data.shape[0]), np.sort(kiva_loans_data.funded_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()


# ### <a id='dola'>5.4.b Distribution of loan amount</a>

# In[ ]:


# Distribution of loan amount
plt.figure(figsize = (12, 8))

sns.distplot(kiva_loans_data['loan_amount'])
plt.show()
plt.figure(figsize = (12, 8))

plt.scatter(range(kiva_loans_data.shape[0]), np.sort(kiva_loans_data.loan_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()


# In[ ]:


kiva_mpi_locations_data.columns


# ### <a id='dowr'>5.4.c Distribution of world regions</a>

# In[ ]:


# Distribution of world regions
plt.figure(figsize=(15,8))
count = kiva_mpi_locations_data['world_region'].value_counts()
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('world region name', fontsize=12)
plt.title("Distribution of world regions", fontsize=16)


# * A we can see **sub-Saharan Africa** got more number of loans.
# * **Europe** and **central Asia** is least frequent world region.

# ### <a id='dolc'>5.4.d Distribution of Lender counts</a>

# In[ ]:


#Distribution of lender count(Number of lenders contributing to loan)
print("Number of lenders contributing to loan : ", len(kiva_loans_data["lender_count"].unique()))
print(kiva_loans_data["lender_count"].value_counts().head(10))
lender = kiva_loans_data['lender_count'].value_counts().head(40)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('lender count(Number of lenders contributing to loan)', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Distribution of lender count", fontsize=16)
plt.show()



# * Distribution is highly Skewed.
# * Number of lenders contributing to loan(lender_count) is 8 whose count is high followed by 7 and 9.

# ### <a id='dolat'>5.4.e Distribution of Loan Activity type</a>

# In[ ]:


#Distribution of Loan Activity type

plt.figure(figsize=(15,8))
count = kiva_loans_data['activity'].value_counts().head(30)
sns.barplot(count.values, count.index)
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Activity name?', fontsize=12)
plt.title("Top Loan Activity type", fontsize=16)


# Top 2 loan activity which got more number of funded are **Farming** and **general Store**

# ### <a id='dotim'>5.4.f Distribution of terms_in_month(Number of months over which loan was scheduled to be paid back)</a>

# In[ ]:


#Distribution of Number of months over which loan was scheduled to be paid back
print("Number of months over which loan was scheduled to be paid back : ", len(kiva_loans_data["term_in_months"].unique()))
print(kiva_loans_data["term_in_months"].value_counts().head(10))
lender = kiva_loans_data['term_in_months'].value_counts().head(70)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical')
plt.xlabel('Number of months over which loan was scheduled to be paid back', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Distribution of Number of months over which loan was scheduled to be paid back", fontsize=16)
plt.show()


# * 14 months over which loan was scheduled to be paid back have taken higher times followed by 8 and 11.

# ### <a id='dos'>5.4.g Distribution of sectors</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = kiva_loans_data['sector'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of sectors')


# ### <a id='doa'>5.4.h Distribution of Activities</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = kiva_loans_data['activity'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of Activities')


# ### <a id='dori'>5.4.i Distribution of repayment_interval</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = kiva_loans_data['repayment_interval'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of repayment_interval')


# ### <a id='dompu'>5.4.j Distribution of Most popular uses of loans</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = kiva_loans_data['use'].value_counts().head(10)
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('uses of loans', fontsize=12)
plt.title("Most popular uses of loans", fontsize=16)


# * **Most popupar use of loan** is **to buy a water filter to provide safe drinking water for their family**.

# ## <a id='fm'>5.5 Borrower Gender: Female V.S. Male</a>

# In[ ]:


gender_list = []
for gender in kiva_loans_data["borrower_genders"].values:
    if str(gender) != "nan":
        gender_list.extend( [lst.strip() for lst in gender.split(",")] )
temp_data = pd.Series(gender_list).value_counts()

labels = (np.array(temp_data.index))
sizes = (np.array((temp_data / temp_data.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Borrower Gender')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")


# As we can see Approx. **80 % borrower** are **Female** and approx. **20 % borrowers **are **Male**.

# In[ ]:


kiva_loans_data.borrower_genders = kiva_loans_data.borrower_genders.astype(str)
gender_data = pd.DataFrame(kiva_loans_data.borrower_genders.str.split(',').tolist())
kiva_loans_data['sex_borrowers'] = gender_data[0]
kiva_loans_data.loc[kiva_loans_data.sex_borrowers == 'nan', 'sex_borrowers'] = np.nan
sex_mean = pd.DataFrame(kiva_loans_data.groupby(['sex_borrowers'])['funded_amount'].mean().sort_values(ascending=False)).reset_index()
print(sex_mean)
g1 = sns.barplot(x='sex_borrowers', y='funded_amount', data=sex_mean)
g1.set_title("Mean funded Amount by Gender ", fontsize=15)
g1.set_xlabel("Gender")
g1.set_ylabel("Average funded Amount(US)", fontsize=12)


# The average amount is **funded** **more** by **Male** than Female.

# ## <a id='sbri'>5.6 Sex_borrower V.S. Repayment_intervals</a>

# In[ ]:


f, ax = plt.subplots(figsize=(15, 5))
print("Genders count with repayment interval monthly\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'monthly'].value_counts())
print("Genders count with repayment interval weekly\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'weekly'].value_counts())
print("Genders count with repayment interval bullet\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'bullet'].value_counts())
print("Genders count with repayment interval irregular\n",kiva_loans_data['sex_borrowers'][kiva_loans_data['repayment_interval'] == 'irregular'].value_counts())

sns.countplot(x="sex_borrowers", hue='repayment_interval', data=kiva_loans_data).set_title('sex borrowers with repayment_intervals');


# * There are **more Females** with **monthly** reapyment_interval than **Males**.
# * There are **more Males** with **irregular** reapyment_interval than **Females**.

# ## <a id='kf'>5.7 Kiva Field partner name V.S. Funding count</a>

# In[ ]:


#Distribution of Kiva Field Partner Names with funding count
print("Top Kiva Field Partner Names with funding count : ", len(loan_themes_by_region_data["Field Partner Name"].unique()))
print(loan_themes_by_region_data["Field Partner Name"].value_counts().head(10))
lender = loan_themes_by_region_data['Field Partner Name'].value_counts().head(40)
plt.figure(figsize=(15,8))
sns.barplot(lender.index, lender.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=14)
plt.xlabel('Field Partner Name', fontsize=18)
plt.ylabel('Funding count', fontsize=18)
plt.title("Top Kiva Field Partner Names with funding count", fontsize=25)
plt.show()


# * There are total **302 Kiva Field Partner.**
# * Out of these, **Alalay sa Kaunlaran (ASKI)** did **higher** number of funding followed by **SEF International** and **Gata Daku Multi-purpose Cooperative (GDMPC)**.

# ## <a id='tcwfa'>5.8 Top Countries with funded_amount(Dollar value of loan funded on Kiva.org)</a>

# In[ ]:


countries_funded_amount = kiva_loans_data.groupby('country').mean()['funded_amount'].sort_values(ascending = False)
print("Top Countries with funded_amount(Dollar value of loan funded on Kiva.org)(Mean values)\n",countries_funded_amount.head(10))


# In[ ]:


data = [dict(
        type='choropleth',
        locations= countries_funded_amount.index,
        locationmode='country names',
        z=countries_funded_amount.values,
        text=countries_funded_amount.index,
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Top Countries with funded_amount(Mean value)'),
)]
layout = dict(title = 'Top Countries with funded_amount(Dollar value of loan funded on Kiva.org)',
             geo = dict(
            showframe = False,
            #showcoastlines = False,
            projection = dict(
                type = 'Mercatorodes'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# **Top** country **Cote D'Ivoire** which is **more loan** funded on Kiva.org follwed by **Mauritania**.

# ## <a id='tmrwa'>5.9 Top mpi_regions with amount(Dollar value of loans funded in particular LocationName)</a>
# 

# In[ ]:


mpi_region_amount = round(loan_themes_by_region_data.groupby('mpi_region').mean()['amount'].sort_values(ascending = False))
print("Top mpi_region with amount(Dollar value of loans funded in particular LocationName)(Mean values)\n",mpi_region_amount.head(10))


# In[ ]:


data = [dict(
        type='choropleth',
        locations= mpi_region_amount.index,
        locationmode='country names',
        z=mpi_region_amount.values,
        text=mpi_region_amount.index,
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Top mpi_regions with amount(Mean value)'),
)]
layout = dict(title = 'Top mpi_regions with amount(Dollar value of loans funded in particular LocationName)',
             geo = dict(
            showframe = False,
            #showcoastlines = False,
            projection = dict(
                type = 'Mercatorodes'
            )
        ),)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# * **Top mpi_regions** who got more funding is **Itasy, Madagascar** followed by **Kaduna, Nigeria**

# ## <a id='vpito'>5.10 Various popularities in terms of loan amount</a>

# ### <a id='plsit'>5.10.a Popular loan sector  in terms of loan amount</a>

# In[ ]:



plt.figure(figsize=(15,8))
count = round(kiva_loans_data.groupby(['sector'])['loan_amount'].mean().sort_values(ascending=False))
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Average loan amount in Dollar', fontsize=20)
plt.ylabel('Loan sector', fontsize=20)
plt.title('Popular loan sector in terms of loan amount', fontsize=24)


# * **Entertainment** sector is taking more loan followed by **Wholesale**.

# ### <a id='plait'>5.10.b Popular loan activity in terms of loan amount</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = round(kiva_loans_data.groupby(['activity'])['loan_amount'].mean().sort_values(ascending=False).head(20))
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Average loan amount in Dollar', fontsize=20)
plt.ylabel('Loan sector', fontsize=20)
plt.title('Popular loan activity in terms of loan amount', fontsize=24)


# * The most popular activities are **Technology** and **Landscaping/Gardening** in terms of loans amount followed by **Communications**.

# ### <a id='pcito'>5.10.c Popular countries in terms of loan amount</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = round(kiva_loans_data.groupby(['country'])['loan_amount'].mean().sort_values(ascending=False).head(20))
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Average loan amount in Dollar', fontsize=20)
plt.ylabel('Countries', fontsize=20)
plt.title('Popular countries in terms of loan amount', fontsize=24)


# **Cote D'lvoire** is More popular country who is taking more amount of loans  followed by **Mauritania**.

# ### <a id='prito'>5.10.d Popular regions(locations within countries) in terms of loan amount</a>

# In[ ]:


plt.figure(figsize=(15,8))
count = round(kiva_loans_data.groupby(['region'])['loan_amount'].mean().sort_values(ascending=False).head(20))
sns.barplot(count.values, count.index, )
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Average loan amount in Dollar', fontsize=20)
plt.ylabel('regions(locations within countries)', fontsize=20)
plt.title('Popular regions(locations within countries) in terms of loan amount', fontsize=24)


# Regions(locations within countries) i.e, **Juba, Tsihombe, Musoma, Cerrik, Kolia, Parakou and Simeulue** are most **popular regions** who are taking more loans.

# ## <a id='wcfcn'>5.11 Wordcloud for Country Names</a>

# In[ ]:


from wordcloud import WordCloud

names = kiva_loans_data["country"][~pd.isnull(kiva_loans_data["country"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for country Names", fontsize=35)
plt.axis("off")
plt.show() 


# ## <a id='mlbmy'>5.12 Mean loan by month-year with repayment intervals</a>

# In[ ]:


kiva_loans_data['date'] = pd.to_datetime(kiva_loans_data['date'])
kiva_loans_data['date_month_year'] = kiva_loans_data['date'].dt.to_period("M")
plt.figure(figsize=(8,10))
g1 = sns.pointplot(x='date_month_year', y='loan_amount', 
                   data=kiva_loans_data, hue='repayment_interval')
g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
g1.set_title("Mean Loan by Month Year", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Loan Amount", fontsize=12)
plt.show()


# Repayment intervals **bullet** had taken more loan amount throught out the years.

# ## <a id='ydoco'>5.13 Yearwise distribution of count of loan availed by each country</a>

# In[ ]:


kiva_loans_data['Century'] = kiva_loans_data.date.dt.year
loan = kiva_loans_data.groupby(['country', 'Century'])['loan_amount'].mean().unstack()
loan = loan.sort_values([2017], ascending=False)
f, ax = plt.subplots(figsize=(15, 20)) 
loan = loan.fillna(0)
temp = sns.heatmap(loan, cmap='Reds')
plt.show()


# In **2017,** **Cote D'lvoire** and **Benin** had taken more amount of loan and in **2016**, **South sudan** had taken.

# ## <a id='cmah'>5.14 Correlation Matrix and Heatmap</a>

# ### <a id='saric'>5.14.a Sectors and Repayment Intervals correlation</a>

# In[ ]:


sector_repayment = ['sector', 'repayment_interval']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(kiva_loans_data[sector_repayment[0]], kiva_loans_data[sector_repayment[1]]).style.background_gradient(cmap = cm)


# * **Agriculture Sector** had **higher** number of **monthly** repayment interval followed by **food sector** had **higher** **irregilar** repayment interval.

# ### <a id='caric'>5.14.b country and Repayment Intervals correlation</a>

# In[ ]:


sector_repayment = ['country', 'repayment_interval']
cm = sns.light_palette("red", as_cmap=True)
pd.crosstab(kiva_loans_data[sector_repayment[0]], kiva_loans_data[sector_repayment[1]]).style.background_gradient(cmap = cm)


# * Weekly repayment interval loan had taken by only Kenya country.
# * **Phillippines** had higher number of **monthly repayment interval** than others.

# ### <a id='cmaho'>5.14.c Correlation Matrix and Heatmap of kiva_loans_data</a>

# In[ ]:


#Correlation Matrix
corr = kiva_loans_data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='cubehelix', square=True)
plt.title('Correlation between different features')
corr


# * As we can see **loan_amount** and **funded_amount** are highly correlated.

# ## <a id='timri'>5.15 Term_In_Months V.S. Repayment_Interval </a>

# In[ ]:


fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'monthly'] , color='b',shade=True, label='monthly')
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'weekly'] , color='r',shade=True, label='weekly')
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'irregular'] , color='g',shade=True, label='irregular')
ax=sns.kdeplot(kiva_loans_data['term_in_months'][kiva_loans_data['repayment_interval'] == 'bullet'] , color='y',shade=True, label='bullet')
plt.title('Term in months(Number of months over which loan was scheduled to be paid back) vs Repayment intervals')
ax.set(xlabel='Terms in months', ylabel='Frequency')


# Repayment Interval **monthly** having **higher frequency** than others repayment intervals

# ## <a id='ltcsf'>5.16 Loan theme created specifically for Kiva or not ?</a>

# In[ ]:


temp = loan_themes_by_region_data['forkiva'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Loan theme specifically for Kiva V.S. Loan theme not specifically for Kiva')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# * In only **16 % loans**, loan theme was **specifically created for Kiva**.

# ## <a id='twfa'>5.17 Time Series Analysis</a>

# ### <a id='toaafa'>5.17.1 Trend of loan amount V.S. funded amount</a>

# In[ ]:


kiva_loans_data.posted_time = pd.to_datetime(kiva_loans_data['posted_time'])
kiva_loans_data.disbursed_time = pd.to_datetime(kiva_loans_data['disbursed_time'])
kiva_loans_data.funded_time = pd.to_datetime(kiva_loans_data['funded_time'])


# In[ ]:


kiva_loans_data.index = pd.to_datetime(kiva_loans_data['posted_time'])
plt.figure(figsize = (12, 8))
ax = kiva_loans_data['loan_amount'].resample('w').sum().plot()
ax = kiva_loans_data['funded_amount'].resample('w').sum().plot()
ax.set_ylabel('Amount ($)')
ax.set_xlabel('month-year')
ax.set_xlim((pd.to_datetime(kiva_loans_data['posted_time'].min()), 
             pd.to_datetime(kiva_loans_data['posted_time'].max())))
ax.legend(["loan amount", "funded amount"])
plt.title('Trend of loan amount V.S. funded amount')

plt.show()


# ### <a id='toaaf'>5.17.2 Trend of unfunded amount V.S. funded amount</a>

# In[ ]:


kiva_loans_data.index = pd.to_datetime(kiva_loans_data['posted_time'])

kiva_loans_data['unfunded_amount'] = kiva_loans_data['loan_amount'] - kiva_loans_data['funded_amount']
plt.figure(figsize = (12, 8))
ax = kiva_loans_data['unfunded_amount'].resample('w').sum().plot()
ax = kiva_loans_data['funded_amount'].resample('w').sum().plot()
ax.set_ylabel('Amount ($)')
ax.set_xlabel('month-year')
ax.set_xlim((pd.to_datetime(kiva_loans_data['posted_time'].min()), 
             pd.to_datetime(kiva_loans_data['posted_time'].max())))
ax.legend(["unfunded amount", "funded amount"])
plt.title('Trend of unfunded amount V.S. funded amount')

plt.show()


# ### <a id='tdbpf'>5.17.3 Trend of Disbursed to borrower V.S. Posted on kiva.org V.S. Funded on kiva.org</a>

# In[ ]:


temp_data = kiva_loans_data.copy()
temp_data['count']= 1  #add 1 to each row so we can count number of loans 
disbursed = temp_data.set_index(temp_data['disbursed_time'])
#disbursed.head()
disbursed = disbursed.resample('10D').sum()

posted = temp_data.set_index(temp_data['posted_time'])
posted = posted.resample('10D').sum()

funded = temp_data.set_index(temp_data['funded_time'])
funded = funded.resample('10D').sum()

plt.figure(figsize=(15,8))
plt.plot(disbursed['count'], color='green', label='Disbursed to borrower', marker='o')
plt.plot(posted['count'], color='red', label='Posted on kiva.org', marker='o')
plt.plot(funded['count'], color='blue', label='Funded on kiva.org', marker='o')
plt.legend(loc='down right')
plt.title("Number of loans, in 10-day intervals")
plt.ylabel("Number of loans")
plt.show()


# * Most of the loan amount were Disbursed to borrower at 2nd month of year ie. 2nd month of 2014, 2015, 2016 and 2017 despite not being fully funded yet.

# ### <a id='i'>5.18 India</a>

# ### <a id='tluii'>5.18.a Top 13 loan uses in India</a>

# In[ ]:


loan_use_in_india = kiva_loans_data['use'][kiva_loans_data['country'] == 'India']
percentages = round(loan_use_in_india.value_counts() / len(loan_use_in_india) * 100, 2)[:13]
trace = go.Pie(labels=percentages.keys(), values=percentages.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))
data = [trace]
layout = go.Layout(width=800, height=800, title='Top 13 loan uses in India',titlefont= dict(size=20), 
                   legend=dict(x=0.1,y=-5))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)


# **Top use of loan** in **india** is to buy a smokeless stove followed by to expand her tailoring business by purchasing cloth materials and a sewing machine.

# ### <a id='tfrii'>5.18.b Top 7 funded regions in India</a>

# **The table contains only India data**

# In[ ]:


temp = pd.DataFrame(loan_themes_by_region_data[loan_themes_by_region_data["country"]=='India'])
temp.head()


# In[ ]:


# A table to show top 7 regions in India with higher funded amount
print("The top 7 regions in India with higher funded amount(Descending Order)")
top_cities = temp.sort_values(by='amount',ascending=False)
top7_cities=top_cities.head(7)
top7_cities


# In[ ]:


# Plotting these Top 7 funded regions on India map. Circles are sized according to the 
# regions of the india

plt.subplots(figsize=(20, 15))
map = Basemap(width=4500000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top7_cities['lon'])
lt=array(top7_cities['lat'])
pt=array(top7_cities['amount'])
nc=array(top7_cities['region'])

x, y = map(lg, lt)
population_sizes = top7_cities["amount"].apply(lambda x: int(x / 3000))
plt.scatter(x, y, s=population_sizes, marker="o", c=population_sizes, alpha=0.9)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=20, fontweight='bold')

plt.title('Top 7 funded regions in India',fontsize=30)


# * **Top 7 funded regions in India :**
#     1. Surendranagar
#     2. Nadia
#     3. Dahod
#     4. Falakata
#     5. Khurda 
#     6. Jaipur
#     7. Rayagada

# ### <a id='mdfpn'>5.18.c Most dominant field partner names in India</a>

# In[ ]:


temp = pd.DataFrame(loan_themes_by_region_data[loan_themes_by_region_data["country"]=='India'])
plt.figure(figsize=(15,8))
count = temp['Field Partner Name'].value_counts().head(10)
sns.barplot(count.values, count.index)
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Most dominant field partner names in India', fontsize=12)
plt.title("Most dominant field partner names in India with count", fontsize=20)


# * **Most dominant field partner** in India is **Milaap**.

# ### <a id='tolii'>5.18.d Trend of loans in India</a>

# In[ ]:


kiva_loans_data.index = pd.to_datetime(kiva_loans_data['funded_time'])
plt.figure(figsize = (12, 8))
ax = kiva_loans_data[kiva_loans_data["country"]=='India']['funded_time'].resample('w').count().plot()
ax.set_ylabel('count')
ax.set_xlabel('month-year')
plt.title('Trend of loans in India')

plt.show()


# ### <a id='tdbpfi'>5.18.e Trend of Disbursed to borrower V.S. Posted on kiva.org V.S. Funded on kiva.org in India</a>

# In[ ]:


temp_data = kiva_loans_data.copy()
temp_data = temp_data[temp_data.country=='India']
temp_data['count']= 1  #add 1 to each row so we can count number of loans 
disbursed = temp_data.set_index(temp_data['disbursed_time'])
#disbursed.head()
disbursed = disbursed.resample('10D').sum()

posted = temp_data.set_index(temp_data['posted_time'])
posted = posted.resample('10D').sum()

funded = temp_data.set_index(temp_data['funded_time'])
funded = funded.resample('10D').sum()

plt.figure(figsize=(15,8))
plt.plot(disbursed['count'], color='green', label='Disbursed to borrower', marker='o')
plt.plot(posted['count'], color='red', label='Posted on kiva.org', marker='o')
plt.plot(funded['count'], color='blue', label='Funded on kiva.org', marker='o')
plt.legend(loc='down right')
plt.title("Number of loans, in 10-day intervals(India)")
plt.ylabel("Number of loans")
plt.show()


# ##  <a id='p'>5.19 Philippines</a>

# ### <a id='tluip'>5.19.a Top 13 loan uses in Philippines</a>

# In[ ]:


loan_use_in_Philippines = kiva_loans_data['use'][kiva_loans_data['country'] == 'Philippines']
percentages = round(loan_use_in_Philippines.value_counts() / len(loan_use_in_Philippines) * 100, 2)[:13]
trace = go.Pie(labels=percentages.keys(), values=percentages.values, hoverinfo='label+percent', 
                textfont=dict(size=18, color='#000000'))
data = [trace]
layout = go.Layout(width=800, height=800, title='Top 13 loan uses in Philippines',titlefont= dict(size=20), 
                   legend=dict(x=0.1,y=-5))

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False)


# **Top use of loan** in **Philippines** is to to build a sanitary toilet for her family.

# ### <a id='tfrip'>5.19.b Top 7 funded regions in Philippines</a>

# **The table contains only Philippines data**

# In[ ]:


temp = pd.DataFrame(loan_themes_by_region_data[loan_themes_by_region_data["country"]=='Philippines'])
temp.head()


# In[ ]:


# A table to show top 7 regions in Philippines with higher funded amount
print("The top 7 regions in India with higher funded amount(Descending Order)")
top_cities = temp.sort_values(by='amount',ascending=False)
top7_cities=top_cities.head(7)
top7_cities


# * **Top 7 funded regions in Philippines :**
#     1. Bais
#     2. Kabankalan
#     3. Quezon
#     4. Dumaguete
#     5. Himamaylan
#     6. Brookes 
#     7. Concepcion

# ### <a id='mdfpnp'>5.19.c Most dominant field partner names in Philippines</a>

# In[ ]:


temp = pd.DataFrame(loan_themes_by_region_data[loan_themes_by_region_data["country"]=='Philippines'])
plt.figure(figsize=(15,8))
count = temp['Field Partner Name'].value_counts().head(10)
sns.barplot(count.values, count.index)
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=19)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Most dominant field partner names in Philippines', fontsize=12)
plt.title("Most dominant field partner names in Philippines with count", fontsize=20)


# * **Most dominant field partner** in Philippines is **ASKI**.

# ### <a id='tolip'>5.19.d Trend of loans in Philippines</a>

# In[ ]:


kiva_loans_data.index = pd.to_datetime(kiva_loans_data['funded_time'])
plt.figure(figsize = (12, 8))
ax = kiva_loans_data[kiva_loans_data["country"]=='Philippines']['funded_time'].resample('w').count().plot()
ax.set_ylabel('count')
ax.set_xlabel('month-year')
plt.title('Trend of loans in Philippines')

plt.show()


# ### <a id='tdbpfp'>5.19.e Trend of Disbursed to borrower V.S. Posted on kiva.org V.S. Funded on kiva.org in Philippines</a>

# In[ ]:


temp_data = kiva_loans_data.copy()
temp_data = temp_data[temp_data.country=='Philippines']
temp_data['count']= 1  #add 1 to each row so we can count number of loans 
disbursed = temp_data.set_index(temp_data['disbursed_time'])
#disbursed.head()
disbursed = disbursed.resample('10D').sum()

posted = temp_data.set_index(temp_data['posted_time'])
posted = posted.resample('10D').sum()

funded = temp_data.set_index(temp_data['funded_time'])
funded = funded.resample('10D').sum()

plt.figure(figsize=(15,8))
plt.plot(disbursed['count'], color='green', label='Disbursed to borrower', marker='o')
plt.plot(posted['count'], color='red', label='Posted on kiva.org', marker='o')
plt.plot(funded['count'], color='blue', label='Funded on kiva.org', marker='o')
plt.legend(loc='down right')
plt.title("Number of loans, in 10-day intervals(Philippines)")
plt.ylabel("Number of loans")
plt.show()


# ## <a id='sols'>5.20 status of loans</a>

# In[ ]:


temp = loans_data['status'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='status of loans')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# * 95 % loan is fully funded. Only 4 % expired.

# ## <a id='dmols'>5.21 Distribution models of loans</a>

# In[ ]:


temp = loans_data['distribution_model'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title=' Distribution models of loans')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# * 99 % loans were disrtributed through **field partners**.

# ## <a id='trtgl'>5.22 Top Reasons to give loan</a>

# In[ ]:


import re
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# lowercase text  
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text    
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text    
    temp = [s.strip() for s in text.split() if s not in STOPWORDS]# delete stopwords from text
    new_text = ''
    for i in temp:
        new_text +=i+' '
    text = new_text
    return text.strip()


# In[ ]:


# remove null value from column "loan_because"
temp_data = lenders_data.dropna(subset=['loan_because'])
# convertinginto lowercase
temp_data['loan_because'] = temp_data['loan_because'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['loan_because'] = temp_data['loan_because'].map(text_prepare)


from wordcloud import WordCloud

#names = kiva_loans_data["country"][~pd.isnull(kiva_loans_data["country"])]
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['loan_because'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top reasons to give loan", fontsize=35)
plt.axis("off")
plt.show() 


# ## <a id='uplob'>5.23 Understanding Poverty levels of Borrowers</a>

# ### <a id="mdpia">5.23.1 Multi-dimensional Poverty Index(MPI) for different regions</a>

# *** Multidimensional Poverty Index (MPI**) is an international measure of acute poverty covering over 100 developing countries
# ### **Formula** :
# The MPI is calculated as follows :
# 
#       MPI =  H * A
# 
# * H : Percentage of people who are MPI poor (incidence of poverty)
# * A : Average intensity of MPI poverty across the poor (%)
# ![aaaaaa](http://hdr.undp.org/sites/default/files/mpi.png)
# Fot detailed information : https://en.wikipedia.org/wiki/Multidimensional_Poverty_Index

# In[ ]:


data = [ dict(
        type = 'scattergeo',
        lat = kiva_mpi_locations_data['lat'],
        lon = kiva_mpi_locations_data['lon'],
        text = kiva_mpi_locations_data['LocationName'],
        marker = dict(
             size = 10,
             line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            cmin = 0,
            color = kiva_mpi_locations_data['MPI'],
            cmax = kiva_mpi_locations_data['MPI'].max(),
            colorbar=dict(
                title="Multi-dimenstional Poverty Index"
            )
        ))]
layout = dict(title = 'Multi-dimensional Poverty Index for different regions')
fig = dict( data=data, layout=layout )
py.iplot(fig)


# If you want more details abount MPI please zoom out this map

# ### <a id='tcwhh'>5.23.2 Top 10 countries with higher Human Development Index(HDI) </a>

# **The underlying principle behind the Human Development Index** :

# ![ssssssssssssss](https://upload.wikimedia.org/wikipedia/en/2/2b/HDI_explained_the_best_way.png)

# **For detailed information :** https://en.wikipedia.org/wiki/Human_Development_Index

# In[ ]:


print("Top 10 countries with higher Human Development Index(HDI) \n")
temp = country_stats_data.sort_values(by =['hdi'], ascending = False)
temp[['country_name','hdi']].head(10)


# In[ ]:


data = [dict(
        type='choropleth',
        locations= country_stats_data['country_name'],
        locationmode='country names',
        z=country_stats_data['hdi'],
        text=country_stats_data['country_name'],
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Human Development Index(HDI)'),
)]
layout = dict(title = 'Human Development Index(HDI) for different countries',)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# ### <a id='pbpl'>5.23.3 Country level Analysis</a>

# ### <a id='pbplc'>5.23.3.a Population below poverty line for different countries in %</a>

# In[ ]:


temp = country_stats_data.sort_values(by =['population_below_poverty_line'], ascending = False)
temp2 = temp[['country_name','population_below_poverty_line']]
temp1 = temp2.head(10)
plt.figure(figsize = (15, 10))
sns.barplot(temp1['population_below_poverty_line'], temp1['country_name'])
for i, v in enumerate(temp1['population_below_poverty_line']):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xlabel('population_below_poverty_line in %', fontsize=12)
plt.ylabel('country_name', fontsize=12)
plt.title("population below poverty line for different countries in % ", fontsize=20)


# In[ ]:


data = [dict(
        type='choropleth',
        locations= country_stats_data['country_name'],
        locationmode='country names',
        z=country_stats_data['population_below_poverty_line'],
        text=country_stats_data['country_name'],
        colorscale='Red',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='population_below_poverty_line in %'),
)]
layout = dict(title = 'Population below poverty line for different countries in % ',)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# * Out of all countries, Top most country is **Syria** who is population below poverty line followed by Zimbabwe. 

# ### <a id='pphdi'>2.23.3.b Population below poverty line V.S. HDI</a>

# In[ ]:


populated_countries = country_stats_data.sort_values(by='population', ascending=False)[:25]

data = [go.Scatter(
    y = populated_countries['hdi'],
    x = populated_countries['population_below_poverty_line'],
    mode='markers+text',
    marker=dict(
        size= np.log(populated_countries.population) - 2,
        color=populated_countries['hdi'],
        colorscale='Portland',
        showscale=True
    ),
    text=populated_countries['country_name'],
    textposition=["top center"]
)]
layout = go.Layout(
    title='population below poverty line V.S. HDI',
    xaxis= dict(title='population below poverty line in %'),
    yaxis=dict(title='HDI')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### <a id='ledc'>2.23.3.c  Life_expectancy for different countries</a>

# * **Life expectancy** is a statistical measure of the average time an organism is expected to live, based on the year of their birth, their current age and other demographic factors including gender.
# * ** Life expectancy** is the primary factor in determining an individual's risk factor.
# * ** Detailed information** : https://en.wikipedia.org/wiki/Life_expectancy

# In[ ]:


temp = country_stats_data.sort_values(by =['life_expectancy'], ascending = False)
temp2 = temp[['country_name','life_expectancy']]
temp1 = temp2.head(20)
plt.figure(figsize = (15, 10))
sns.barplot(temp1['life_expectancy'], temp1['country_name'])
for i, v in enumerate(temp1['life_expectancy']):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xlabel('life_expectancy in years', fontsize=12)
plt.ylabel('country_name', fontsize=12)
plt.title("life_expectancy for different countries in years ", fontsize=20)


# * Life expectancy of **Hong kong** is higher than other countries followed by **japan**
# 
# -------------------------------------
# ## **Question :**  Can an application be declined when the loan term exceeds the applicant's life expectancy?

# ### <a id='tcrmup'>2.23.3.d  Top countries with higher rural MPI's along with urban MPI's</a>

# In[ ]:


mpi_national_data = mpi_national_data.sort_values(by=['MPI Rural'], ascending=False).head(30)

mpi_national_urban = mpi_national_data[['Country', 'MPI Urban']]
mpi_national_urban.rename(columns={'MPI Urban':'MPI'}, inplace=True)
mpi_national_urban['sign'] = 'MPI Urban'

mpi_national_rural = mpi_national_data[['Country', 'MPI Rural']]
mpi_national_rural.rename(columns={'MPI Rural':'MPI'}, inplace=True)
mpi_national_rural['sign'] = 'MPI Rural'

mpi_urban_rural = mpi_national_urban.append(mpi_national_rural)
#mpi_urban_rural.head()
sns.factorplot(x='Country', y='MPI', hue='sign', data=mpi_urban_rural, kind='bar', size=8, aspect=2.5)
plt.xticks(rotation=90, size = 20)
plt.title("Top 30 countries with higher rural MPI's along with urban MPI's", size = 30)
plt.xlabel('Country', fontsize=30)
plt.ylabel('MPI', fontsize=30)


# * Top most country is **Niger** with** higher Urban MPI.**

# ### <a id='tcrhruhr'>2.23.3.e  Top countries with higher rural Headcount Ratio along with urban Headcount Ratio</a>

# * **The Head count ratio (HCR)** is the proportion of a population that exists, or lives, below the poverty line.
# * **A head count ratio** simply measures the number of poor below the poverty line. 

# In[ ]:


mpi_national_data = mpi_national_data.sort_values(by=['Headcount Ratio Rural'], ascending=False).head(30)

mpi_national_urban = mpi_national_data[['Country', 'Headcount Ratio Rural']]
mpi_national_urban.rename(columns={'Headcount Ratio Rural':'Headcount Ratio'}, inplace=True)
mpi_national_urban['sign'] = 'Headcount Ratio Rural'

mpi_national_rural = mpi_national_data[['Country', 'Headcount Ratio Urban']]
mpi_national_rural.rename(columns={'Headcount Ratio Urban':'Headcount Ratio'}, inplace=True)
mpi_national_rural['sign'] = 'Headcount Ratio Urban'

mpi_urban_rural = mpi_national_urban.append(mpi_national_rural)
#mpi_urban_rural.head()
sns.factorplot(x='Country', y='Headcount Ratio', hue='sign', data=mpi_urban_rural, kind='bar', size=8, aspect=2.5)
plt.xticks(rotation=90, size = 20)
plt.title("Top 30 countries with higher rural Headcount Ratio along with urban Headcount Ratio", size = 30)
plt.xlabel('Country', fontsize=30)
plt.ylabel('Headcount Ratio', fontsize=30)


# * Top most country is **Somalia** with **higher rural headcount ratio**.

# ### <a id='tcriduid'>2.23.3.f  Top countries with higher rural Intensity of Deprivation along with urban Intensity of Deprivation</a>

# *** Intensity of deprivation** :
#   * Average percentage of deprivation experienced by people in multidimensional poverty.
#  

# In[ ]:


mpi_national_data = mpi_national_data.sort_values(by=['Intensity of Deprivation Rural'], ascending=False).head(30)

mpi_national_urban = mpi_national_data[['Country', 'Intensity of Deprivation Rural']]
mpi_national_urban.rename(columns={'Intensity of Deprivation Rural':'Intensity of Deprivation'}, inplace=True)
mpi_national_urban['sign'] = 'Intensity of Deprivation Rural'

mpi_national_rural = mpi_national_data[['Country', 'Intensity of Deprivation Urban']]
mpi_national_rural.rename(columns={'Intensity of Deprivation Urban':'Intensity of Deprivation'}, inplace=True)
mpi_national_rural['sign'] = 'Intensity of Deprivation Urban'

mpi_urban_rural = mpi_national_urban.append(mpi_national_rural)
#mpi_urban_rural.head()
sns.factorplot(x='Country', y='Intensity of Deprivation', hue='sign', data=mpi_urban_rural, kind='bar', size=8, aspect=2.5)
plt.xticks(rotation=90, size = 20)
plt.title("Top 30 countries with higher Intensity of Deprivation along with urban Intensity of Deprivation", size = 30)
plt.xlabel('Country', fontsize=30)
plt.ylabel('Intensity of Deprivation', fontsize=30)


# * Top most country is **Niger** with **higher rural Intensity of deprivation.**

# # <a id='s'>6. Summary :</a>
# -------------------------------
# * **Agriculture Sector** is more frequent in terms of number of loans followed by **Food**.
# * Types of **interval payments** monthly, irregular, bullet and weekly. Out of which **monthly** is **more** frequent and **weekly** is **less** frequent.
# * **Philippines** is **most** frequent countries who got more loans followed by **Kenya**.
# * **Weekly repayment interval loan** had taken by **only Kenya** country.
# * In world region, **sub-Saharan Africa** got **more** number of loans.
# * Number of lenders contributing to loan(**lender_count**) is 8 whose count is high followed by 7 and 9.
# * **Top 2 loan activity** which got more number of funded are **Farming** and **general Store**.
# * Out of **302 Kiva Field Partners** ,  **Alalay sa Kaunlaran (ASKI)** did **higher** number of funding followed by **SEF International** and **Gata Daku Multi-purpose Cooperative (GDMPC)**.
# * **14 months** over which loan was scheduled to be paid back have taken higher times followed by 8 and 11.
# * The average amount is **funded** **more** by **Male** than Female.
# * Approx. **80 % borrower are Female** and approx. **20 % borrowers are Male**. 
# * There are **more Females** with **monthly** reapyment_interval than **Males**.
# * There are **more Males** with **irregular** reapyment_interval than **Females**.
# * **Entertainment sector** is taking **more** loan followed by **Wholesale**.
# * The **most popular activities** are **Technology** and **Landscaping/Gardening** in terms of loans amount followed by **Communications**.
# * **Cote D'lvoire** is **More popular country** who is taking more amount of loans followed by **Mauritania**.
# * Regions(locations within countries) i.e, **Juba, Tsihombe, Musoma, Cerrik, Kolia, Parakou and Simeulue** are most **popular regions** who are taking more loans.
# * Repayment intervals **bullet** had taken more loan amount throught out the years.
# * In **2017,** **Cote D'lvoire** and **Benin** had taken more amount of loan and in **2016**, **South sudan** had taken.
# * **Top mpi_regions** who got more funding is **Itasy, Madagascar** followed by **Kaduna, Nigeria**.
# * In only **16 % loans**, loan theme was **specifically created for Kiva**.
# * **Most popupar use of loan** is **to buy a water filter to provide safe drinking water for their family**.
# * Most of the loan amount were Disbursed to borrower at 2nd month of year ie. 2nd month of 2014, 2015, 2016 and 2017 despite not being fully funded yet.
# * **India** :
#     * **Top use of loan** in **india** is **to buy a smokeless stove** followed **by to expand her tailoring business       by purchasing cloth materials and a sewing machine.**
#     * **Most dominant field partner** in India is **Milaap**.
#     * **Top 3 funded regions in India :**
#         1. Surendranagar
#         2. Nadia
#         3. Dahod
# * **Philippines** :
#     * **Top use of loan** in **Philippines** is to to build a sanitary toilet for her family.
#     * **Most dominant field partner** in Philippines is **ASKI**.
#     * **Top 3 funded regions in Philippines :**
#         1. Bais
#         2. Kabankalan
#         3. Quezon
# * 99 % loans were disrtributed through **field partners**.
# * **Country level Analysis for understanding povertry level :**
#      * **Top 3 countries **with higher Human Development Index(**HDI**) are **Norway**, **Switzerland** and **Australia** according to this data.
#      * Out of all countries, Top most country is **Syria** who is population below poverty line followed by Zimbabwe. 
#      * Life expectancy of **Hong kong** is higher than other countries followed by **japan**
#      * Top most country is **Niger** with** higher Urban MPI ** and **Intensity of Deprivation**.(Focused on rural area)
#      * Top most country is **Somalia** with **higher rural headcount ratio**.(Focused on rural area)

# # More To Come.Stay Tuned.!! (If you find useful please Upvote and/or leaves a comment :) )
