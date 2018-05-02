
# coding: utf-8

# ## Please share your feedback and Add Your Vote on the Top Right Corner :-)

# # Kiva - Beginner Guide to EDA and Data Visuaization

# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Kiva.org_logo_2016.svg/640px-Kiva.org_logo_2016.svg.png" width="300" height="100" />

# ## 1. Introduction

# In their own words - "*Kiva is an international nonprofit, founded in 2005 and based in San Francisco, with a mission to connect people through lending to alleviate poverty. We celebrate and support people looking to create a better future for themselves, their families and their communities*."

# By lending as little as $25 on Kiva, anyone can help a borrower start or grow a business, go to school, access clean energy or realize their potential. For some, it’s a matter of survival, for others it’s the fuel for a life-long ambition.
# 
# 100% of every dollar you lend on Kiva goes to funding loans. Kiva covers costs primarily through optional donations, as well as through support from grants and sponsors.

# ### 1.1 Kiva by the numbers
# 
# * 2.7M Borrowers
# * 1.7M Lenders
# * 83 Countries
# * $1.11B Loans funded through Kiva
# * 97.0% Repayment rate
# * 81% Of Kiva borrowers are women
# * A Kiva loan is funded every 2 min

# <img src="https://www-kiva-org.global.ssl.fastly.net/cms/page/images/hp-slideshow-r1-xxl-std.jpg"  width="600" />

# ## 2 Basic Data Overview

# ### 2.1 Load Libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ### 2.2 Load Data

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


kiva_loans_df = pd.read_csv('../input/kiva_loans.csv')
kiva_mpi_df = pd.read_csv('../input/kiva_mpi_region_locations.csv')
kiva_theme_id_df = pd.read_csv('../input/loan_theme_ids.csv')
kiva_theme_region_df = pd.read_csv('../input/loan_themes_by_region.csv')


# ### 2.3 Shape and Features in DataFrames

# In[ ]:


print("Shape of kiva_loans_df -> {}".format(kiva_loans_df.shape))
print("\nFeatures in kiva_loans_df -> {}".format(kiva_loans_df.columns.values))


# In[ ]:


# Multidimensional Poverty Index (MPI)
print("Shape of kiva_mpi_df -> {}".format(kiva_mpi_df.shape))
print("\nFeatures in kiva_mpi_df -> {}".format(kiva_mpi_df.columns.values))


# In[ ]:


print("Shape of kiva_theme_id_df -> {}".format(kiva_theme_id_df.shape))
print("\nFeatures in kiva_theme_id_df -> {}".format(kiva_theme_id_df.columns.values))


# In[ ]:


print("Shape of kiva_theme_region_df -> {}".format(kiva_theme_region_df.shape))
print("\nFeatures in kiva_theme_region_df -> {}".format(kiva_theme_region_df.columns.values))


# ### 2.4 Basic Info of Data

# In[ ]:


kiva_loans_df.info()


# In[ ]:


kiva_mpi_df.info()


# In[ ]:


kiva_theme_id_df.info()


# In[ ]:


kiva_theme_region_df.info()


# ### 2.5 Sample Data from DataFrames

# In[ ]:


kiva_loans_df.head()


# In[ ]:


kiva_mpi_df.head()


# In[ ]:


kiva_theme_id_df.head()


# In[ ]:


kiva_theme_region_df.head()


# ## 3 Data Analysis

# ### 3.1 Data Analysis - kiva_loans_df
# 
# Below we will try to gather few details about the features present in the kiva_loans_df. Some of the important features that we would be looking at include:
# 
# 1. loan_amount
# 2. funded_amount
# 3. activity
# 4. sector
# 5. country
# 6. region
# 7. term_in_months
# 8. lender_count
# 9. borrower_genders
# 10. repayment_interval

# #### 3.1.1 EDA on Loan Amount

# Below we will plot a Histogram of the Loan Amount to check the distribution.

# In[ ]:


_ = plt.hist(kiva_loans_df['loan_amount'])


# Looks like most of the data is spread between `$25` (the lowest amount) and `$10,000`. We will plot a BoxPlot below to see if there are any outliers and how spread is the data.

# In[ ]:


plt.figure(figsize=(15,3))
_ = plt.boxplot(kiva_loans_df['loan_amount'], vert=False)


# - The BoxPlot shows that there is a potential outlier of $100,000 and also we see that the data is highly centred around the lowest value. We will see a BoxPlot by removing the outlier.
# - We will use the describe method on the DataFrame to get better insights of the data.

# In[ ]:


# Detail of the borrower who is looking for the highest loan amount
kiva_loans_df[kiva_loans_df.loan_amount == 100000]


# We can see from the above entry that the highest Loan Amount is indeed funded. The highest loan is taken to `create more than 300 jobs for women and farmer...`

# In[ ]:


plt.figure(figsize=(15,3))
_ = plt.boxplot(kiva_loans_df[kiva_loans_df.loan_amount < 100000].loan_amount, vert=False)


# Detail of the borrowers who are looking for loan amount greater than $50,000 and who got funded

# In[ ]:


# Detail of the borrowers who are looking for loan amount greater than $50,000 and who got funded
kiva_loans_df[(kiva_loans_df.loan_amount >= 50000) & (kiva_loans_df.funded_amount >= 50000)]


# In[ ]:


print(kiva_loans_df['loan_amount'].describe())


# - Looks like the data is heavily `Right Skewed`. We can see that the max amount is `$100,000` but the **Q3** (75%) lies at `$1000` only.
# - Below we will plot a histogram until thrice the SD to check the distribution.

# In[ ]:


plt.figure(figsize=(10,5))
_ = plt.hist(kiva_loans_df['loan_amount'], range=(25, np.std(kiva_loans_df['loan_amount'])*3), bins = 50)


# As can be seen from the above histrogram most of the data lies below `$500`. Below we will see a one last histogram for the data between `$25` and `$500`.

# In[ ]:


plt.figure(figsize=(10,5))
_ = plt.hist(kiva_loans_df['loan_amount'], range=(25, 500), bins = 10)


# #### 3.1.2 EDA on Funded Amount

# Below we will plot a Histogram of the Funded Amount to check the distribution.

# In[ ]:


_ = plt.hist(kiva_loans_df['funded_amount'])


# Looks like most of the data is spread between `$25` (the lowest amount) and `$10,000`. We will plot a BoxPlot below to see if there are any outliers and how spread is the data.

# In[ ]:


plt.figure(figsize=(15,3))
_ = plt.boxplot(kiva_loans_df['funded_amount'], vert=False)


# - The BoxPlot shows that there is a potential outlier of $100,000 and also we see that the data is highly centred around the lowest value. We will see a BoxPlot by removing the outlier.
# - We will use the describe method on the DataFrame to get better insights of the data.

# In[ ]:


# Detail of the borrower who took the highest loan amount
kiva_loans_df[kiva_loans_df.funded_amount == 100000]


# In[ ]:


plt.figure(figsize=(15,3))
_ = plt.boxplot(kiva_loans_df[kiva_loans_df.funded_amount < 100000].funded_amount, vert=False)


# In[ ]:


# Detail of the borrowers who took loan amount greater than $50,000
kiva_loans_df[kiva_loans_df.funded_amount >= 50000]


# In[ ]:


print(kiva_loans_df['funded_amount'].describe())


# Looks like the data is heavily `Right Skewed`. We can see that the max amount is `$100,000` but the **Q3** (75%) lies at `$900` only.

# Below we will plot a histogram until thrice the SD to check the distribution.

# In[ ]:


plt.figure(figsize=(10,5))
_ = plt.hist(kiva_loans_df['funded_amount'], range=(25, np.std(kiva_loans_df['funded_amount'])*3), bins = 50)


# As can be seen from the above histrogram most of the data lies below `$500`. Below we will see a one last histogram for the data between `$25` and `$500`.

# In[ ]:


plt.figure(figsize=(10,5))
_ = plt.hist(kiva_loans_df['funded_amount'], range=(25, 500), bins = 10)


# #### 3.1.3 EDA on Activity

# Below we will see the different activites that are supported by **Kiva**

# In[ ]:


print(kiva_loans_df['activity'].unique())


# As there are many `Activities` we will print a table below.

# In[ ]:


kiva_loans_df[['activity']].groupby(kiva_loans_df.activity)                            .count()                            .sort_values('activity', ascending=False)


# From the above table we can see that there are a total of 163 different activities. Below we will visualize the activity count for the **top 50 activities**.

# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_loans_df['activity'], order = kiva_loans_df['activity'].value_counts().iloc[0:50].index)
plt.title("Kiva Activity Count", fontsize=20)
plt.xlabel('Kiva Activity', fontsize=18)
plt.ylabel('Activity Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# As we can see above **`Farming`** has the highest activity count. Below are the top 10 activites from the snapshot data:
# 
# 1. Farming
# 2. General Store
# 3. Personal Housing Expenses
# 4. Food Production/Sales
# 5. Agriculture
# 6. Pigs
# 7. Retail
# 8. Clothing Sales
# 9. Home Appliances
# 10. Higher education costs
# 

# #### 3.1.4 EDA on Sector

# Below we will see the different sectors that are supported by **Kiva**

# In[ ]:


print(kiva_loans_df['sector'].unique())


# Below we will see the count for each sector.

# In[ ]:


kiva_loans_df[['sector']].groupby(kiva_loans_df.sector)                            .count()                            .sort_values('sector', ascending=False)


# In[ ]:


plt.figure(figsize=(8,4))
sns.countplot(kiva_loans_df['sector'], order = kiva_loans_df['sector'].value_counts().index)
plt.title("Kiva Sector Count", fontsize=20)
plt.xlabel('Kiva Sector List', fontsize=18)
plt.ylabel('Sector Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# As we can see in the above visualization, `Agriculture` sector is predominantly funded by Kiva. Following Agriculture, `Food` and `Retail` occupy the next slots.

# #### 3.1.5 EDA on Country

# In[ ]:


print(kiva_loans_df['country'].unique())


# In[ ]:


kiva_loans_df[['country']].groupby(kiva_loans_df.country)                            .count()                            .sort_values('country', ascending=False)


# We can see that Kiva is funding to users in 87 countries with Philippines, Kenya and El Salvador topping the list.

# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_loans_df['country'], order = kiva_loans_df['country'].value_counts().iloc[0:50].index)
plt.title("Kiva Funding Countries", fontsize=20)
plt.xlabel('Funded Countries', fontsize=18)
plt.ylabel('Country Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# From the above visualization we can see that the total number of fundings for `Philippines` is nearly double compared to its next neighbour `Kenya`.

# #### 3.1.6 EDA on Region

# In[ ]:


print("Total Regions Funded by Kiva - {}".format(np.count_nonzero(kiva_loans_df['region'].unique())))


# As there are more than 12k regions in the 87 countries. We will just print the top 50 regions funded by Kiva.

# In[ ]:


kiva_loans_df[['region']].groupby(kiva_loans_df.region)                            .count()                            .sort_values('region', ascending=False)                            .iloc[0:49]


# #### 3.1.7 EDA on Term in Months

# In[ ]:


print("Total Funding Terms on Kiva - {}".format(np.count_nonzero(kiva_loans_df['term_in_months'].unique())))


# In[ ]:


print("Funding Terms on Kiva Ranges From {} Months and {} Months ".format(np.min(kiva_loans_df['term_in_months'].unique()), np.max(kiva_loans_df['term_in_months'].unique())))


# In[ ]:


kiva_loans_df[['term_in_months']].groupby(kiva_loans_df.term_in_months)                            .count()                            .sort_values('term_in_months', ascending=False)


# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_loans_df['term_in_months'], order = kiva_loans_df['term_in_months'].value_counts().iloc[0:50].index)
plt.title("Kiva Funding Term in Months", fontsize=20)
plt.xlabel('Term in Months', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# From the above visualization we can see that most of the funding is given for a tenure of `14 Months` followed by `8 Months` and `11 Months`.

# #### 3.1.8 EDA on Lender Count

# In[ ]:


print("Total Lender Counts On Kiva - {}".format(np.count_nonzero(kiva_loans_df['lender_count'].unique())))


# In[ ]:


print("Lender Count on Kiva Ranges From {} Lenders and {} Lenders".format(np.min(kiva_loans_df['lender_count'].unique()), np.max(kiva_loans_df['lender_count'].unique())))


# Looks like there are some entries with `0 Lenders`. We will see what those entries are below by printing some sample.

# In[ ]:


kiva_loans_df[kiva_loans_df.lender_count == 0].sample(5)


# From the above table it is clear that `0 Lenders` corresponds to those entries where the borrowers are still looking for a lender who can give them a Loan. Looks Fair!!

# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_loans_df['lender_count'], order = kiva_loans_df['lender_count'].value_counts().iloc[0:50].index)
plt.title("Kiva Funding Lender Count", fontsize=20)
plt.xlabel('Lender Count', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# From above visualization we can see that most of the Lending Groups are of `8 Lenders` followed by `7 Lenders` and `9 Lenders`.

# #### 3.1.9 EDA on Borrower Gender

# In[ ]:


print("Gender Counts - {}".format(np.count_nonzero(kiva_loans_df['borrower_genders'].unique())))


# `11229 Gender Counts`, looks suspicious. Let's see whats happening.

# In[ ]:


kiva_loans_df[['borrower_genders']].groupby(kiva_loans_df.borrower_genders)                            .count()                            .sort_values('borrower_genders', ascending=False)                            .iloc[0:9]


# Looks like the Gender information is capturing the details of all the Borrowers and not the main Borrower.

# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_loans_df['borrower_genders'], order = kiva_loans_df['borrower_genders'].value_counts().iloc[0:5].index)
plt.title("Kiva Funding Borrower Count by Gender", fontsize=20)
plt.xlabel('Borrower Gender', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# From the above visualization we can clearly see that `Women` are the main Borrowers from the `Kiva` platform.

# #### 3.1.10 EDA on Repayment Interval

# In[ ]:


print(kiva_loans_df['repayment_interval'].unique())


# In[ ]:


kiva_loans_df[['repayment_interval']].groupby(kiva_loans_df.repayment_interval)                            .count()                            .sort_values('repayment_interval', ascending=False)


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(kiva_loans_df['repayment_interval'], order = kiva_loans_df['repayment_interval'].value_counts().iloc[0:49].index)
plt.title("Kiva Repayment Interval", fontsize=20)
plt.xlabel('Repayment Interval', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# We see that the most of the users are making their payment `Monthly` and there are substantial number of users who are maling `Irregular` payments.

# ### 3.2 Data Analysis - kiva_mpi_df
# 
# Below we will try to gather few details about the features present in the kiva_mpi_df. Some of the important features that we would be looking at include:
# 
# 1. MPI

# In[ ]:


print("Country with Highest MPI ({:.5f}) - {}".format(kiva_mpi_df['MPI'].max(), kiva_mpi_df[kiva_mpi_df.MPI == kiva_mpi_df['MPI'].max()].country.iloc[0]))
print("Country with Lowest MPI ({:.5f}) - {}".format(kiva_mpi_df['MPI'].min(), kiva_mpi_df[kiva_mpi_df.MPI == kiva_mpi_df['MPI'].min()].country.iloc[0]))


# ### 3.3 Data Analysis - kiva_theme_id_df
# 
# Below we will try to gather few details about the features present in the kiva_theme_id_df. Some of the important features that we would be looking at include:
# 
# 1. Loan Theme Type

# In[ ]:


print(kiva_theme_id_df['Loan Theme Type'].unique())


# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_theme_id_df['Loan Theme Type'], order = kiva_theme_id_df['Loan Theme Type'].value_counts().iloc[0:50].index)
plt.title("Kiva Loan Theme Type", fontsize=20)
plt.xlabel('Loan Theme Type', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# As can be seen from the above visualization, most of the Funds are spent for `General`, `Underserved` and `Agriculture` followed by `Rural Inclusion` and `Water`

# ### 3.4 Data Analysis - kiva_theme_region_df
# 
# Below we will try to gather few details about the features present in the kiva_theme_region_df. Some of the important features that we would be looking at include:
# 
# 1. Field Partner Name
# 2. sector
# 3. Loan Theme Type
# 4. country

# #### 3.4.1 EDA on Field Partner Name

# In[ ]:


print(kiva_theme_region_df['Field Partner Name'].unique())


# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_theme_region_df['Field Partner Name'], order = kiva_theme_region_df['Field Partner Name'].value_counts().iloc[0:50].index)
plt.title("Kiva Field Partner Name", fontsize=20)
plt.xlabel('Field Partner Name', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# #### 3.4.2 EDA on Sector

# In[ ]:


print(kiva_theme_region_df['sector'].unique())


# In[ ]:


kiva_theme_region_df[['sector']].groupby(kiva_theme_region_df.sector)                            .count()                            .sort_values('sector', ascending=False)


# In[ ]:


plt.figure(figsize=(8,4))
sns.countplot(kiva_theme_region_df['sector'], order = kiva_theme_region_df['sector'].value_counts().index)
plt.title("Kiva Funding Sector", fontsize=20)
plt.xlabel('Funding Sector', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# #### 3.4.3 EDA on Loan Theme Type

# In[ ]:


print(kiva_theme_region_df['Loan Theme Type'].unique())


# In[ ]:


kiva_theme_region_df[['Loan Theme Type']].groupby(kiva_theme_region_df['Loan Theme Type'])                            .count()                            .sort_values('Loan Theme Type', ascending=False)


# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_theme_region_df['Loan Theme Type'], order = kiva_theme_region_df['Loan Theme Type'].value_counts().iloc[0:50].index)
plt.title("Kiva Loan Theme Type", fontsize=20)
plt.xlabel('Loan Theme Type', fontsize=18)
plt.ylabel('Funding Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# #### 3.4.4 EDA on Country

# In[ ]:


print(kiva_theme_region_df['country'].unique())


# In[ ]:


kiva_theme_region_df[['country']].groupby(kiva_theme_region_df.country)                            .count()                            .sort_values('country', ascending=False)


# In[ ]:


plt.figure(figsize=(13,4))
sns.countplot(kiva_theme_region_df['country'], order = kiva_theme_region_df['country'].value_counts().iloc[0:50].index)
plt.title("Kiva Loan Themes by Country", fontsize=20)
plt.xlabel('Loan Themes by Country', fontsize=18)
plt.ylabel('Loan Count', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# ### 3.5 Funded Amount by Country

# In[ ]:


funded_amnt_by_cntry_df = pd.DataFrame(kiva_loans_df.groupby('country').sum()['funded_amount'].sort_values(ascending=False)).reset_index()


# In[ ]:


print(funded_amnt_by_cntry_df[:5])


# In[ ]:


plt.figure(figsize=(13,4))
sns.barplot(x='country', y='funded_amount', data=funded_amnt_by_cntry_df[:25])
plt.title("Kiva Funded Amount by Country", fontsize=20)
plt.xlabel('Funded Amount by Country', fontsize=18)
plt.ylabel('Funded Amount', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# ### 3.6 Loan Amount by Country

# In[ ]:


loan_amnt_by_cntry_df = pd.DataFrame(kiva_loans_df.groupby('country').sum()['loan_amount'].sort_values(ascending=False)).reset_index()


# In[ ]:


print(loan_amnt_by_cntry_df[:5])


# In[ ]:


plt.figure(figsize=(13,4))
sns.barplot(x='country', y='loan_amount', data=loan_amnt_by_cntry_df[:25])
plt.title("Kiva Loan Amount by Country", fontsize=20)
plt.xlabel('Loan Amount by Country', fontsize=18)
plt.ylabel('Loan Amount', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# In[ ]:


loan_funded_amnt_by_cntry_df = pd.DataFrame(kiva_loans_df.groupby('country').sum())[['loan_amount', 'funded_amount']].sort_values(by=['loan_amount', 'funded_amount'], ascending=False).reset_index()


# In[ ]:


loan_funded_amnt_by_cntry_df[:5]


# In[ ]:


fig = plt.figure(figsize=(13,4))
ax = fig.add_subplot(111)
ax_cpy = ax.twinx()
width = 0.4

loan_funded_amnt_by_cntry_df.set_index('country').loan_amount[:25].plot(kind='bar', color='DarkOrange', ax=ax, width=width, position=1)
loan_funded_amnt_by_cntry_df.set_index('country').funded_amount[:25].plot(kind='bar', color='Gray', ax=ax_cpy, width=width, position=0)

plt.title("Kiva Loan Amount vs Funded Amount by Country", fontsize=20)
ax.set_xlabel('Loan Amount vs Funded Amount by Country', fontsize=18)
ax.set_ylabel('Loan Amount', fontsize=18)
ax_cpy.set_ylabel('Funded Amount', fontsize=18)
plt.xticks(fontsize=14, rotation=90)
plt.show()


# # To be Continued .....

# ## Please share your feedback and Add Your Vote on the Top Right Corner :-)
