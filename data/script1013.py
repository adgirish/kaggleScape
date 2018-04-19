
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from numpy import log10, ceil, ones
from numpy.linalg import inv 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # prettier graphs
import matplotlib.pyplot as plt # need dis too
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML # for da youtube memes
import itertools # let's me iterate stuff
from datetime import datetime # to work with dates
import geopandas as gpd
from fuzzywuzzy import process
from shapely.geometry import Point, Polygon
import shapely.speedups
shapely.speedups.enable()
import fiona 
from time import gmtime, strftime
from shapely.ops import cascaded_union

sns.set_style('darkgrid') # looks cool, man
import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# "![](https://i.imgflip.com/13cqor.jpg)
# <center>**Warning:**  Newb alert, dirty python incoming</center>
# 
# 
# ![](https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png)This is my first Kernel, first use of python, and its attempt was to be somewhat exploratory and stream of conscious, so apologies on the organization.  In the end I think some of the charts or outliers I discovered are of interest or are actionable.  I tried to distinguish some of these points in particuarly with an image to call attention to them.
# 
# # Exploratory Data Analysis
# * [1. Introduction and My Personal Experience](#intro)
#   * [1.1 Approach (Data and Visual)](#approach)
#   * [1.2 Unexplored Themes](#unexplored)
# * [2. Data Prep](#dataprep)
#   * [2.1 Create Base Superset](#data_createset)
#   * [2.2 Create New Fields](#data_createnew)
#   * [2.3 Update Fields](#data_update)
#   * [2.4 Bad Seeming Data](#data_bad)
#   * [2.5 Completed Data](#data_complete)
# * [3. Distributions and Contributions](#dist)
#   * [3.1 Distribution of Loan Amount](#dist_loan)
#   * [3.2 Distribution of Funded Amount](#dist_fund)
#   * [3.3 Average Kiva Member Contribution](#avg_cont)
# * [4. Top Sectors and Activities](#tsa)
#   * [4.1 Top Sectors](#ts)
#   * [4.2 Absolute Top 30 Overall Activities, by Sector](#act30)
#   * [4.3 Funding Speed by World Region by Sector](#speed_wr)
#   * [4.4 Funding Speed by Group Type by Sector](#speed_gt)
# * [5. Loan Count by Gender and Group](#group)
# * [6. What's #Trending Kiva?](#trend)
# * [7. Loan Theme Types](#themes)
# * [8. Exploring Currency](#curr)
#   * [8.1 Currency Usage](#curr_usg)
#   * [8.2 Mean Loan by Currency for Top 20 Currencies](#curr_avg)
#   * [8.3 What's going on in Lebanon?](#curr_leb)
#   * [8.4 Lebanese Field Partners](#leb_fld)
# * [9. Bullet Loans for Agriculture](#bullet_agg)
#   * [9.1 El Salvador Investigation by Activity](#sal1)
#   * [9.2 El Salvador Loan Count Over Time](#sal2)
#   * [9.3 El Salvador Animal Loans](#sal3)
# * [10. Unfunded Loans](#unfunded)
# * [11. Is the Philippines Really the Country with the Most Kiva Activity?](#phil)
# 
# # Kiva Poverty Targeting System
# * [12. Current Poverty Targeting System](#kiva_poverty)
#   * [12.1 Current Kiva Scoring - National MPI](#curr_nat_mpi)
#     * [12.1.1 Problem 1: Missing rural percentages](#prob1)
#     * [12.1.2 Problem 2: Can populated values be trusted over time, or even currently?](#prob2)
#     * [12.1.3 National MPI with World Bank Rural Percentage Population Data](#wb1)
#     * [12.1.4 Manual Assessment of Current Methodology National MPI](#manual)
#   * [12.2 Current Kiva Scoring - Sub-National MPI](#kiva_sub)
#     * [12.2.1 Problem 3: Mix of Methodology](#prob3)
#     * [12.2.2 Problem 4: Misaligned Regions](#prob4)
#     * [12.2.3 Question - Should weights be by loan count or dollar amount?](#weights)
# * [13. Philippines Local Analysis (National MPI 0.052)](#13)
#   * [13.1 Leveraging Geospatial Data](#13_1)
#   * [13.2 Reassigning MPI Regions Based on Loan (Point) in Region (Polygon)](#13_2)
#     * [13.2.1 Doing the Point in Polygon Work](#13_2_1)
#     * [13.2.2 Apparent Coastal Exception Problem via Method](#13_2_2)
#     * [13.2.3 Use Existing Kiva Data To Fill Holes](#13_2_3)
#     * [13.2.4 Have the Intern Manually Adjust Remaining Holes](#13_2_4)
#     * [13.2.5 Reassignment Results](#13_2_5)
#     * [13.2.6 Field Partner MPI - Existing (National, Sub-National) vs. Amended](#13_2_6)
#     * [13.2.7 Field Partner MPI Method Comparison](#13_2_7)
# * [14. Mozambique Local Analysis (National MPI 0.389)](#14)
#     * [14.1 Field Partner MPI - Existing (National, Sub-National) vs. Amended](#14_1)
# * [15. Rwanda Local Analysis (National MPI 0.259)](#15)
# * [16. Sierra Leone Local Analysis (National MPI 0.464)](#16)
# * [17. El Salvador Local Analysis (National MPI 0.026)](#17)
# 

# <a id=intro>
# # 1. Introduction and My Personal Experience 
# 
# Kiva is a non-profit micro-funding/loan capitalization website.  Users from around the world come to lend money to those in need around the world.  Often these loans are for small entrepeneurs.  Kiva lenders receive no return on capital and are subject to loan defaults and currency exchange losses.  Loans are distributed by Kiva field partners to borrowers to improve their lives and help facilitiate their growth out of poverty.  Kiva field partner lenders do charge a local market interest rate.  Kiva funders insure capital is available and loans have funding to be made.  The default rate for these loans is very low in comparison [to other default rates](https://www.federalreserve.gov/releases/chargeoff/delallsa.htm).  If you have never used it I [invite you to give it a try!](https://www.kiva.org/invitedby/mikedev10)  17.8% of my loans were to the Philippines, I worked there a few weeks in 2003 and have worked from there in January 2016/17/18 ditching the Chicago winter.  Kiva users will find the Philippines and women come up often in the data, likely even more often in my own as I made a conscious effort to lend to both.  For a long time I only lent to women.  I also like groups.  I lend heavily more towards business type use vs. personal investment or personal use.
# 
# ![](https://www.doyouevendata.com/wp-content/uploads/2018/03/kiva.jpg)

# <a id=approach></a>
# ## 1.1 Approach (Data and Visual) 
# 
# I'm willing to bet my approach is not the best in regards to best practices in working with data, probably in part due to memory usage.  However I'm looking to make my life easier here and the dataset all fits, so we're going to roll with it.  I've attempted to tie kiva provided data together as best I can, along with additional MPI data to get a richer view of the areas and how they experience poverty, to create one large set to work with.  Seaborn produces a lot of pretty color plots, but I will *only be leveraging color when it has meaning* so as to avoid confusion.
# <a id=unexplored></a>
# ## 1.2 Unexplored Themes 
# potentially available to explore with additional data:
# 1. relation to education attainment (general)
# 2. relation to education attainment in areas of the world where girls have less rights or the cultural expectation to stay at home while their husband works, regardless of their education attainment
# 3. relation to educational attainment in areas of the world boys are more subject to violence ([Girls in the Middle East do better than boys in school by a greater margin than almost anywhere else in the world: a case study in motivation, mixed messages, and the condition of boys everywhere.](https://www.theatlantic.com/education/archive/2017/09/boys-are-not-defective/540204/))
# 4. relation to loan reporting and and interest with regards to religion (Buddhism - it doesn't seem expected to have some of the kiva reporting strings as part of lending; Islam - interest is not charged in Sharia complaint finance, although perhaps it kind of is, [it's a bit confusing](https://en.wikipedia.org/wiki/Riba).)
# 5. relation with weaker or stronger property rights (perhaps some metrics could be leveraged from Cato's [Human Freedom Index](https://www.cato.org/human-freedom-index))

# <a id=dataprep></a>
# # 2. Data Prep 
# <a id=data_createset></a>
# ## 2.1 Data Prep - Create Base Superset 
# 
# Let's take a look at the data in the sets we've got to work with.
# **MPI Poverty Metrics** (2 external, 1 kiva)

# In[ ]:


df_mpi_ntl = pd.read_csv("../input/mpi/MPI_national.csv")
df_mpi_ntl.shape


# In[ ]:


df_mpi_ntl[(df_mpi_ntl['ISO'] == 'AFG') | (df_mpi_ntl['ISO'] == 'ARM')]


# In[ ]:


df_mpi_subntl = pd.read_csv("../input/mpi/MPI_subnational.csv")
df_mpi_subntl.shape


# In[ ]:


df_mpi_subntl.head()


# In[ ]:


df_kv_mpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
df_kv_mpi.shape


# In[ ]:


df_kv_mpi.head()


# Why does kiva have so many more records, from the same root datasource?  Well, the file has junk in it.

# In[ ]:


df_kv_mpi.tail()


# Let's combine this into a set of useful superset geographic poverty data.  Note the kiva provided MPI is the same as MPI Regional in the richer MPI data.  We'll take the geo stuff from there though.

# In[ ]:


df_mpi = pd.merge(df_mpi_ntl[['ISO', 'Country', 'MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban',
                         'MPI Rural', 'Headcount Ratio Rural', 'Intensity of Deprivation Rural']], 
              df_mpi_subntl[['ISO country code', 'Sub-national region', 'World region', 'MPI National', 'MPI Regional',
                            'Headcount Ratio Regional', 'Intensity of deprivation Regional']], how='left', left_on='ISO', right_on='ISO country code')
df_mpi.drop('ISO country code', axis=1, inplace=True)
df_mpi = df_mpi.merge(df_kv_mpi[['ISO', 'LocationName', 'region', 'geo', 'lat', 'lon']], left_on=['ISO', 'Sub-national region'], right_on=['ISO', 'region'])
df_mpi.drop('Sub-national region', axis=1, inplace=True)

#cols = df_mpi.columns.tolist()
#reorder it a bit more to my liking
cols = ['ISO', 'Country', 'MPI Urban', 'Headcount Ratio Urban', 'Intensity of Deprivation Urban', 'MPI Rural', 'Headcount Ratio Rural', 'Intensity of Deprivation Rural', 
        'region', 'World region', 'LocationName', 'MPI National', 'MPI Regional', 'Headcount Ratio Regional', 'Intensity of deprivation Regional', 'geo', 'lat', 'lon']
df_mpi = df_mpi[cols]
df_mpi.shape 


# In[ ]:


df_mpi[df_mpi['ISO'] == 'AFG'].head()


# In[ ]:


#df_mpi['LocationName'].value_counts().head()
df_mpi.shape


# Great, we didn't lose everything and should have a nice set of MPI data now.  Let's check out the loan data and make a superset to play with for visualization.

# In[ ]:


df_kv_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_kv_loans.shape


# In[ ]:


df_kv_loans.head()


# In[ ]:


df_kv_theme = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
df_kv_theme.shape


# In[ ]:


df_kv_theme.head()


# In[ ]:


df_kv_theme_rgn = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
df_kv_theme_rgn.shape


# In[ ]:


df_kv_theme_rgn.head()


# Some of this data...  does not line up well.  :/  We only have 6 MPI regions for Pakistan...  but 127 regions in our themes, and 146 regions in our loans.  What's going on here?

# In[ ]:


len(df_mpi[df_mpi['ISO'] == 'PAK']['region'].unique())


# In[ ]:


len(df_kv_theme_rgn[df_kv_theme_rgn['country'] == 'Pakistan']['region'].unique())


# In[ ]:


len(df_kv_loans[df_kv_loans['country'] == 'Pakistan']['region'].unique())


# In[ ]:


print("loan themes by region has " + str(len(df_kv_theme_rgn['region'].unique())) + " distinct values and " 
      + str(len(df_kv_theme_rgn['region'].str.lower().unique())) + " distinct lowered values.")


# In[ ]:


print("kiva loans has " + str(len(df_kv_loans['region'].str.lower().unique())) + " distinct values and " 
       + str(len(df_kv_loans['region'].str.lower().str.lower().unique())) + " distinct lowered values.")


# In[ ]:


print("mpi regions has " + str(len(df_mpi['region'].unique())) + " values.")


# In[ ]:


# Youtube
HTML('<h3>How do we get all these different values to join??</h3><iframe width="560" height="315" src="https://www.youtube.com/embed/tpD00Q4N6Jk?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


# It seems like the best best is to join from kiva loans to kiva themes, then kiva themes to theme regions; leveraging on country and region.  Then using mpi_region (not fully populated and per dataset notes, I assume is set by some kind of geo proximity) join it to the mpi data.

# In[ ]:


#left join required, some data missing loan themes 671205 - 671199 = 6 missing
df_all_kiva = pd.merge(df_kv_loans, df_kv_theme, how='left', on='id')
df_all_kiva = df_all_kiva.merge(df_kv_theme_rgn, how='left', on=['Partner ID', 'Loan Theme ID', 'country', 'region'])
#df_all_kiva = df_all_kiva.merge(df_kv_mpi, how='left', on=['country', 'region'])
#df_all_kiva.head()
df_all_kiva = df_all_kiva.merge(df_mpi, how='left', left_on=['ISO', 'mpi_region'], right_on=['ISO', 'LocationName'])
#try cleaning this up a bit
df_all_kiva.drop('country_code', axis=1, inplace=True)
df_all_kiva.drop('Loan Theme Type_y', axis=1, inplace=True)
df_all_kiva.drop('geocode_old', axis=1, inplace=True)
df_all_kiva.drop('geo_y', axis=1, inplace=True)
df_all_kiva.drop('sector_y', axis=1, inplace=True)
df_all_kiva.drop(['LocationName_y', 'Country'], axis=1, inplace=True)
df_all_kiva = df_all_kiva.rename(index=str, columns={'region_x': 'region_kiva', 'region_y': 'region_mpi', 'Loan Theme Type_x': 'Loan Theme Type',
                                      'LocationName_x': 'LocationName_kiva', 'geocode': 'geocode_kiva', 'mpi_region': 'LocationName_kiva_mpi',
                                      'mpi_geo': 'geo_mpi', 'lat_y': 'lat_mpi', 'lon_y': 'lon_mpi',
                                       'geo_x': 'geo_kiva', 'lat_x': 'lat_kiva', 'lon_x': 'lon_kiva',
                                       'sector_x': 'sector', 'region_x': 'region_kiva', 'region_y': 'region_mpi',
                                       'partner_id': 'partner_id_loan', 'Partner ID': 'partner_id_loan_theme'
                                      })
#useful but dupey weird on this table; we can in theory aggregate to these anyway for our loans
#df_all_kiva.drop(['number', 'amount'], axis=1, inplace=True)
df_all_kiva.head()


# Also grabbing some population data from another dataset.

# In[ ]:


df_world_pop = pd.read_csv('../input/world-population/WorldPopulation.csv')
df_world_pop[['Country', '2016']].head()


# <a id=data_createnew></a>
# ## 2.2 Data Prep - Create New Fields
# 
# I also wanted to make a distinction about groups.  As a lender, I felt these were lower risk loans, as I believed the community would help eachother both to execute what the loan was for successfully, as well as if trouble, even help eachother in paying it back.  I broke the data up into somewhat arbitrary group sizes.  I wasn't sure how to deal with the NaN values either, so with some googling this is what I came up with to assign my group categories and mark the NaNs as well.  It is probably not the most elegant nor efficient python ever written.  However, it gets the job done.  

# In[ ]:


def group_type(genders):

    try:
        float(genders)
        return np.nan

    except ValueError:

        grp = ''

        male_cnt = genders.split(', ').count('male')
        female_cnt = genders.split(', ').count('female')

        if(male_cnt + female_cnt == 0):
            return 'unknown'
        elif(male_cnt + female_cnt == 1):
            if(male_cnt == 1):
                return 'individual male'
            else:
                return 'individual female'
        elif(male_cnt == 1 & female_cnt == 1):
            return 'male + female pair'
        elif(male_cnt == 2 & female_cnt == 0):
            return 'male pair'
        elif(male_cnt == 0 & female_cnt == 2):
            return 'female pair'
        else:
            if(male_cnt == 0):
                grp = 'all female '
            elif(female_cnt == 0):
                grp = 'all male '
            else:
                grp = 'mixed gender '

        if(male_cnt + female_cnt > 5):
            grp = grp + 'large group (>5)'
        else:
            grp = grp + 'small group (3 to 5)'

        return grp


# In[ ]:


df_all_kiva['group_type'] = df_all_kiva['borrower_genders'].apply(group_type)
df_all_kiva[['borrower_genders', 'group_type']].head()


# Let's add the hashtags with their own columns as well.  This part takes a while to chooch through, kinda the most expensive single step even though it's not particularly insightful...

# In[ ]:


def tag_hashtag(t, hashtag):
    
    try:
        float(t)
        return np.nan

    except ValueError:

        if(hashtag in t):
            return 1
        else:
            return 0

s = df_all_kiva['tags']
unq_tags = pd.unique(s.str.split(pat=', ', expand=True).stack())
unq_tags = [s for s in unq_tags if '#' in s]

for tag in unq_tags:
    df_all_kiva[tag] = df_all_kiva['tags'].apply(tag_hashtag, args=(tag,))
    
df_all_kiva[~df_all_kiva['tags'].isnull()][['#Parent', '#Woman Owned Biz', '#Elderly', '#Animals', '#Repeat Borrower', 'tags']].head()


# In exploring the data below, I found a loan for 100k.  Was this a real loan?  I figured it would be news if it was, so I google it; indeed it was a real loan.  [The kiva link is here](https://www.kiva.org/lend/1398161).  In fact, this also brings up the interesting point that kiva provided us with the real loan ids, so we can indeed go check out the actual loan page for any of these loans, at the URL: https://www.kiva.org/lend/ID- very cool!  Why not make that a field too...  Try going to the URLs to check them out!

# In[ ]:


df_all_kiva[df_all_kiva['loan_amount'] == df_all_kiva['loan_amount'].max()]


# In[ ]:


df_all_kiva['loan_URL'] = df_all_kiva['id'].apply(lambda x: 'https://www.kiva.org/lend/' + str(x))
df_all_kiva['loan_URL'].head()


# <a id=data_update></a>
# ## 2.3 Data Prep - Update Fields
# 
# World Region isn't set now in many a place where joins failed, although it's easy enough to update, as it is simply based on country.  Since India doesn't have MPI regions, it's not in the data anywhere, but I'm going to set the World Region to South Asia for them as well (same as Pakistan).  So let's do that with some ugleh python I wrote.  After that we'll get the dataframe to use proper datetimes for our timestamps as well.

# In[ ]:


df_all_kiva[['country', 'World region']].head(5)


# In[ ]:


# we'll do this in multiple lines to make it more readable
assoc_df = df_all_kiva[['country', 'World region']].merge(df_mpi_subntl[['Country', 'World region']].drop_duplicates(), how='left', left_on=['country'], right_on=['Country'])

df_all_kiva['World region_y'] = assoc_df.iloc[:,3].values
df_all_kiva['World region'] = df_all_kiva['World region'].fillna(df_all_kiva['World region_y'])
df_all_kiva.drop('World region_y', axis=1, inplace=True)
#df_all_kiva[df_all_kiva['country'] == 'India']['World region'] = 'South Asia'
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'India', 'South Asia', df_all_kiva['World region'])
df_all_kiva[['country', 'World region']].head(5)


# Setting some dates with the code below...

# In[ ]:


df_all_kiva['date'] = pd.to_datetime(df_all_kiva['date'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva['posted_time'] = pd.to_datetime(df_all_kiva['posted_time'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva['funded_time'] = pd.to_datetime(df_all_kiva['funded_time'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva['disbursed_time'] = pd.to_datetime(df_all_kiva['disbursed_time'], format='%Y-%m-%d %H:%M:%S')
df_all_kiva[['date', 'posted_time', 'funded_time', 'disbursed_time']].head()


# Setting MPI National since I have some NaN still and can get the value... also making a display string as well...  same with ISO code.  We can fill world region too...

# In[ ]:


df_all_kiva = df_all_kiva.merge(df_all_kiva[['country', 'MPI National']].drop_duplicates().dropna(axis=0, how='any'), on='country', how='left')
df_all_kiva = df_all_kiva.rename(index=str, columns={'MPI National_y': 'MPI National'})
df_all_kiva.drop('MPI National_x', axis=1, inplace=True)

df_all_kiva['MPI National str'] = df_all_kiva['MPI National'].astype(float).round(3).astype(str).fillna('?')
df_all_kiva['country_mpi'] = df_all_kiva['country'] + ' - ' + df_all_kiva['MPI National str']
df_all_kiva.drop('MPI National str', axis=1, inplace=True)

df_all_kiva = df_all_kiva.merge(df_all_kiva[['country', 'ISO']].drop_duplicates().dropna(axis=0, how='any'), on='country', how='left')
df_all_kiva = df_all_kiva.rename(index=str, columns={'ISO_y': 'ISO'})
df_all_kiva.drop('ISO_x', axis=1, inplace=True)

df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Iraq', 'IRQ', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Chile', 'CHL', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Kosovo', 'XKX', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Congo', 'COG', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Mauritania', 'MRT', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Vanuatu', 'VUT', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Panama', 'PAN', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Virgin Islands', 'VIR', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Saint Vincent and the Grenadines', 'VCT', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Guam', 'GUM', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == 'Puerto Rico', 'PRI', df_all_kiva['ISO'])
df_all_kiva['ISO'] = np.where(df_all_kiva['country'] == "Cote D'Ivoire", 'CIV', df_all_kiva['ISO'])

df_all_kiva = df_all_kiva.merge(df_all_kiva[['country', 'World region']].drop_duplicates().dropna(axis=0, how='any'), on='country', how='left')
df_all_kiva = df_all_kiva.rename(index=str, columns={'World region_y': 'World region'})
df_all_kiva.drop('World region_x', axis=1, inplace=True)

df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Paraguay', 'Latin America and Caribbean', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Lebanon', 'Arab States', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Armenia', 'Europe and Central Asia', df_all_kiva['World region'])

df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Palestine', 'Arab States', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Samoa', 'XKX', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'United States', 'CHL', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Kyrgyzstan', 'Europe and Central Asia', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Mexico', 'CHL', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Georgia', 'Europe and Central Asia', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Azerbaijan', 'Europe and Central Asia', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Albania', 'Europe and Central Asia', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Turkey', 'CHL', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Costa Rica', 'XKX', df_all_kiva['World region'])

df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Kosovo', 'Europe and Central Asia', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Ukraine', 'Europe and Central Asia', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Solomon Islands', 'CHL', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'South Africa', 'Sub-Saharan Africa', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Moldova', 'Europe and Central Asia', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Panama', 'Latin America and Caribbean', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Israel', 'Arab States', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Thailand', 'East Asia and the Pacific', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Congo', 'Sub-Saharan Africa', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Somalia', 'Sub-Saharan Africa', df_all_kiva['World region'])

#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Puerto Rico', 'XKX', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Saint Vincent and the Grenadines', 'CHL', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Chile', 'Latin America and Caribbean', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Vanuatu', 'CHL', df_all_kiva['World region'])
#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Virgin Islands', 'XKX', df_all_kiva['World region'])

#df_all_kiva['World region'] = np.where(df_all_kiva['country'] == 'Guam', 'CHL', df_all_kiva['World region'])
df_all_kiva['World region'] = np.where(df_all_kiva['country'] == "Cote D'Ivoire", 'Sub-Saharan Africa', df_all_kiva['World region'])


# <a id=data_bad></a>
# ## 2.4 Data Prep - Bad Seeming Data
# 
# I say "seeming" in case someone points out that I've made some flaws in my logic... else I think this is bad data.
# 
# I also found some things in the data I couldn't quite resolve.  This included different partner ID values.  The first partner id and country in this example is Mexico, partner 294 - Kubo.financiero in Mexico -  from kiva_loans.csv.  However partner 199 for this loan from loan_themes_by_region.csv is for 199 - CrediCampo in El Salvador.  [Visiting the loan URL](https://www.kiva.org/lend/1340274), we find that the latter information is indeed associated with the loan.  Some of the 'use' column descriptions on these are weird, but going out to the URL they appear to all be real loans.  There are 54 records like this.  It's small enough to be considered a "don't care" I suppose in any case, among our 671,205 total; I've simply left them in untouched with no attempt to repair them for now.

# In[ ]:


#df_all_kiva[(df_all_kiva['partner_id_loan'] != df_all_kiva['partner_id_loan_theme']) & (~df_all_kiva['partner_id_loan_theme'].isnull())][['id', 'country', 'partner_id_loan', 'partner_id_loan_theme', 'loan_URL', 'region_kiva', 'region_mpi', 'use']]
#54 total
df_all_kiva[df_all_kiva['id'] == 1340274][['id', 'country', 'partner_id_loan', 'partner_id_loan_theme', 'loan_URL', 'region_kiva', 'region_mpi']]


# <a id=data_complete></a>
# ## 2.5 Data Prep - Completed Data
# 
# A reordering to my liking, along with the characteristics of the data that we will move forward with.  Time to go fishing!

# In[ ]:


cols = ['id', 'loan_amount', 'funded_amount', 'activity', 'sector', 'use', 'currency', 'lender_count', 'repayment_interval', 'term_in_months', 'date', 
 'posted_time', 'funded_time', 'disbursed_time',  'borrower_genders', 'group_type', 'Loan Theme ID', 'Loan Theme Type', 'forkiva',  'partner_id_loan', 
 'partner_id_loan_theme', 'Field Partner Name', 'rural_pct', 'World region', 'country', 'ISO', 'country_mpi', 'MPI National', 'MPI Urban', 'Headcount Ratio Urban', 
 'Intensity of Deprivation Urban', 'MPI Rural', 'Headcount Ratio Rural', 'Intensity of Deprivation Rural', 'LocationName_kiva', 'LocationName_kiva_mpi',  
 'names',  'region_kiva', 'region_mpi', 'MPI Regional', 'Headcount Ratio Regional', 'Intensity of deprivation Regional', 'geocode_kiva', 'geo_kiva', 'lat_kiva', 
 'lon_kiva', 'geo_mpi', 'lat_mpi', 'lon_mpi', 'loan_URL', 'tags', '#Elderly', '#Woman Owned Biz', '#Repeat Borrower', '#Parent', '#Vegan', '#Eco-friendly',
 '#Sustainable Ag', '#Schooling', '#First Loan', '#Low-profit FP', '#Post-disbursed', '#Health and Sanitation', '#Fabrics', '#Supporting Family', '#Single Parent',
 '#Biz Durable Asset', '#Interesting Photo', '#Single', '#Widowed', '#Inspiring Story', '#Animals', '#Refugee', '#Job Creator', '#Hidden Gem', '#Unique',
 '#Tourism', '#Orphan', '#Trees', '#Female Education', '#Technology', '#Repair Renew Replace']
#df_all_kiva.info()
df_all_kiva = df_all_kiva[cols]
df_all_kiva.info()


# I feel like this is as good as this available set is going to get.  I tried to lop off everything that appeared repeated and unusable.  The Kiva data seems to be more accurate in regards to location, with them attempting to choose the best representation of MPI, I attempted to tie in the richer MPI data, and make more use out of the original atomic Kiva data by running some functions against it to create some new columns.

# <a id=dist></a>
# # 3 Distributions and Contributions
# <a id=dist_loan></a>
# ## 3.1 Distribution of Loan Amount
# 
# Let's take a look at how requested loan amounts are distributed.
# 

# In[ ]:


sns.set_palette

sns.distplot(df_all_kiva['loan_amount'])
plt.show()


# Oh my!  Tis quite a large graph along the x axis.  I double checked to make sure the loan amount in the data description was indeed in USD, and not local currency.  Let's see how we can get this chart to be a little more useful.

# In[ ]:


for x in range(0,10):
    print('99.' + str(x) + 'th percentile loan_amount is: ' + str(df_all_kiva['loan_amount'].quantile(0.99 + x/1000)))


# Let's stick with 99th percentile for plotting this data.

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(.99) ]['loan_amount'])
plt.show()


# <a id=dist_fund></a>
# ## 3.2 Distribution of Funded Amount
# 
# What percentage of loans are actually funded?  What's the funded distribution look like?

# In[ ]:


df_all_kiva[df_all_kiva['loan_amount'] == df_all_kiva['funded_amount']]['id'].count() / df_all_kiva['id'].count()*100


# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(df_all_kiva[df_all_kiva['funded_amount'] < df_all_kiva['funded_amount'].quantile(.99) ]['funded_amount'])
plt.show()


# Let's do the same for the count of lenders.  Generally the approach is to lend 25 on each loan and mitigate risk over many different loans, and this chart to look very similar as a result.  At least... that's what I used to do on prosper.com, and that's what I expected here...

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(df_all_kiva[df_all_kiva['lender_count'] < df_all_kiva['lender_count'].quantile(.99) ]['lender_count'])
plt.show()


# The above graph is surprisingly dissimilar to the ones above it to me...  Let's take a look at the average amount lent.
# <a id=avg_cont></a>
# ## 3.3 Average Kiva Member Contribution

# In[ ]:


df_all_kiva['funded_amount'].sum() / df_all_kiva['lender_count'].sum()


# Wow - I didn't expect that at all.  It makes sense that funding the really big loans would have large values though, and there's a fair amount of those...  surely we close in on 25 per kiva user contribution as we move down to smaller loans, right?  A bit of my hack python later and...

# In[ ]:


lst1 = range(100,0,-10)
lst2 = list()

for x in range(0, 10):
    #print('at ' + str(round((1 - x/10)*100, 0)) + 'th percentile loan amount, average lender lent: ' + str(round(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['loan_amount'].sum() / df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['lender_count'].sum(), 2)) + ' with average loan ' + str(round(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['loan_amount'].mean(), 2)) + ' and average number of lenders ' + str(round(df_all_kiva[df_all_kiva['loan_amount'] < df_all_kiva['loan_amount'].quantile(1 - x/10) ]['lender_count'].mean(), 2)) )
    lst2.append(round(df_all_kiva[df_all_kiva['funded_amount'] < df_all_kiva['funded_amount'].quantile(1 - x/10) ]['funded_amount'].sum() / df_all_kiva[df_all_kiva['funded_amount'] < df_all_kiva['funded_amount'].quantile(1 - x/10) ]['lender_count'].sum(), 2))
    
dfavg = pd.DataFrame(
    {'percentile': lst1,
     'average_per_lender': lst2
    })

plt.figure(figsize=(10,5))
ax = sns.barplot(x='percentile', y='average_per_lender', data=dfavg, color='c')
ax.set_title('Average Lender Contribution by Percentile', fontsize=15)
plt.show()


# Of my *own* 997 loans, it appears I've put 50 in to 3 of them, and 25 into the rest.  I was expecting this to be pretty common, and the average contribution to be very close to 25 throughout.  However it appears that people put in $50+ much more often than I do myself!  The average across the board is higher than I expected.

# <a id=tsa></a>
# # 4 Top Sectors and Activities
# 
# <a id=ts></a>
# ## 4.1 Top Sectors

# In[ ]:


plt.figure(figsize=(15,8))
plotSeries = df_all_kiva['sector'].value_counts()
ax = sns.barplot(plotSeries.values, plotSeries.index, color='pink')
ax.set_title('Top Sectors', fontsize=15)
plt.show()


# <a id=act30></a>
# ## 4.2 Absolute Top 30 Overall Activities, by Sector

# In[ ]:


#plt.figure(figsize=(15,8))
fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)
#plotSeries = df_all_kiva['activity'].value_counts().head(20)
df_plot = df_all_kiva.groupby('sector')['activity'].value_counts()
df_plot = df_plot.to_frame()
df_plot.columns = ['count']
df_plot.reset_index(level=1, inplace=True)
df_plot.reset_index(level=0, inplace=True)
df_plot = df_plot.sort_values('count', ascending=False).head(30)
sectors = df_plot['sector'].unique()
palette = itertools.cycle(sns.color_palette('hls', len(sectors)))

for s in sectors:
    df_plot['graphcount'] = np.where(df_plot['sector'] == s, df_plot['count'], 0)
    sns.barplot(x='graphcount', y='activity', data=df_plot,
            label=s, color=next(palette))
    
ax.legend(ncol=2, loc='best', frameon=True)
ax.set_xlabel('count of loans')
leg = ax.get_legend()
new_title = 'Sector'
leg.set_title(new_title)
ax.set_title('Top Overall Activities', fontsize=15)
plt.show()


# Here we also see Agriculture, Retail, and Food strongly represented in the top activities for loans.
# <a id=speed_wr></a>
# ## 4.3 Funding Speed by World Region by Sector
# 
# I tried to calculate a funding speed here, how quickly loans were funded by world region and sector.  The output was too skewed so I ended up graphing the log version of it.  I coupled it with an absolute count of loans as well.
# 1. Arab States - Health funds extremely fast.  However we can also see this was not very many loans.
# 2. We see a lot of similar rates, although Arab States overall seem to fund quickly.  Perhaps this is attributable to a boost from religious lenders?
# 3. The absolute chart shows us pretty similar information to what we already knew from above about hot sectors.  It does allow us to see it's a few regions really driving this.

# In[ ]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'country', 'sector']]
#df_funding.head()

df_funding['days_to_fund'] = (df_funding['funded_time'] - df_funding['posted_time'])
df_funding['days_to_fund'] = df_funding['days_to_fund'].apply(lambda x: x.total_seconds()/60/60/24)
#df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
df_funding = df_funding.groupby(['World region', 'sector']).sum()
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)
df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
#df_funding.head()
#df_funding.groupby('')
#df_plot = df_all_kiva.groupby('sector')['activity'].value_counts()
#df_funding['num_days'] = (datetime.strptime(df_funding['funded_time'].split('+')[0], date_format) - datetime.strptime(df_funding['posted_time'].split('+')[0], date_format)).total_seconds()/60/60/24
#df_funding.groupby(['country']['sector'])
df_heat = df_funding[['World region', 'sector', 'funding_speed']]
f, ax = plt.subplots(figsize=(18, 8))
df_heat['funding_speed'] = np.log10(df_heat['funding_speed'])
#df_heat.pivot('World region', 'sector', 'funding_speed').info()
sns.heatmap(df_heat.pivot('World region', 'sector', 'funding_speed'), annot=True, linewidths=.5, ax=ax)
plt.show()


# In[ ]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'country', 'sector']]
df_funding = df_funding[['World region', 'sector', 'funded_amount']].groupby(['World region', 'sector']).agg('count')
#df_funding
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)

df_heat = df_funding[['World region', 'sector', 'funded_amount']]
df_heat = df_heat.rename(index=str, columns={'funded_amount': 'count'})
f, ax = plt.subplots(figsize=(18, 8))
#df_heat.pivot('World region', 'sector', 'count').info()
sns.heatmap(df_heat.pivot('World region', 'sector', 'count'), annot=True, fmt='d', linewidths=.5, ax=ax)
plt.show()


# <a id=speed_gt></a>
# ## 4.4 Funding Speed by Group Type by Sector
# 
# This is the same idea, although now looking at borrower gender demographics.  We can see individuals in Sub-Saharan Africa take the longest to fund - although we can also see that is because they are in competition with a large amount of people.  Perhaps organizing or joining a group could help if obtaining funding is a problem.  Some of these loan counts are very low and arguably I should be excluding these groups, perhaps a revisit in the future, this was in part to test my learning python abilities, unfortunately nothing too big to draw from here.
# 

# In[ ]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'country', 'group_type']]
#df_funding.head()

df_funding['days_to_fund'] = (df_funding['funded_time'] - df_funding['posted_time'])
df_funding['days_to_fund'] = df_funding['days_to_fund'].apply(lambda x: x.total_seconds()/60/60/24)
#df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
df_funding = df_funding.groupby(['World region', 'group_type']).sum()
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)
df_funding['funding_speed'] = df_funding['funded_amount'] / df_funding['days_to_fund']
#df_funding.head()
#df_funding.groupby('')
#df_plot = df_all_kiva.groupby('sector')['activity'].value_counts()
#df_funding['num_days'] = (datetime.strptime(df_funding['funded_time'].split('+')[0], date_format) - datetime.strptime(df_funding['posted_time'].split('+')[0], date_format)).total_seconds()/60/60/24
#df_funding.groupby(['country']['sector'])
df_heat = df_funding[['World region', 'group_type', 'funding_speed']]
f, ax = plt.subplots(figsize=(18, 8))
df_heat['funding_speed'] = np.log10(df_heat['funding_speed'])
#df_heat.pivot('World region', 'sector', 'funding_speed').info()
sns.heatmap(df_heat.pivot('World region', 'group_type', 'funding_speed'), annot=True, linewidths=.5, ax=ax)
plt.show()


# In[ ]:


df_funding = df_all_kiva[~df_all_kiva['funded_time'].isnull()][['posted_time','funded_time', 'funded_amount', 'World region', 'group_type', 'country', 'sector']]
df_funding = df_funding[['World region', 'group_type', 'funded_amount']].groupby(['World region', 'group_type']).agg('count')
#df_funding
df_funding.reset_index(level=1, inplace=True)
df_funding.reset_index(level=0, inplace=True)

df_heat = df_funding[['World region', 'group_type', 'funded_amount']]
df_heat = df_heat.rename(index=str, columns={'funded_amount': 'count'})
f, ax = plt.subplots(figsize=(18, 8))
df_heat['count'].fillna(0)
df_heat = df_heat.pivot('World region', 'group_type', 'count')

sns.heatmap(df_heat, annot=True, fmt='g', linewidths=.5, ax=ax)
plt.show()


# <a id=group></a>
# # 5 Loan Count by Gender and Group

# In[ ]:


df_stacked = df_all_kiva[['group_type', 'id']].groupby(['group_type']).agg('count')

df_stacked.reset_index(level=0, inplace=True)
df_stacked = df_stacked.rename(index=str, columns={'id': 'count'})

df_stacked = df_stacked.sort_values('count', ascending=False)
groups = df_stacked['group_type'].unique()

fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)

for gt in groups:
    df_stacked['graphcount'] = np.where(df_stacked['group_type'] == gt, df_stacked['count'], 0)

    if ((gt == 'individual female') | ('all female' in gt)):
        if(gt == 'individual female'):
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='women', color='#f36cee')
        else:
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='_nolegend_', color='#f36cee')
    elif ((gt == 'male + female pair') | ('mixed' in gt)):
        if(gt == 'male + female pair'):
            sns.barplot(x='graphcount', y='group_type', data=df_stacked,
                label='mixed', color='#8f0e87')
        else:
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='_nolegend_', color='#8f0e87')
    else:
        if(gt == 'individual male'):
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='men', color='#08b1e7')
        else:
            sns.barplot(x='graphcount', y='group_type', data=df_stacked, 
                label='_nolegend_', color='#08b1e7')
    
ax.set_xlabel('count of loans')
ax.legend(ncol=1, loc='best', frameon=True)

leg = ax.get_legend()
new_title = 'Gender'
leg.set_title(new_title)
ax.set_title('Group Size and Gender Mix', fontsize=15)
plt.show()


# Women totally dominate Kiva, followed by individual men and pairs of men.  After that we're back in full force with women's groups both small and large, as well as mixed groups.  I have done a fair amount of group loans as I am of the mind they may both be lower risk and may help with the borrower achieving local success with the power of the group to help them through any stumbling points.
# <a id=trend></a>
# # 6 What's #Trending Kiva?
# 
# Let's take a look at the #most #popular #hashtags. 

# In[ ]:


df_sum_tags = pd.DataFrame()
for tag in unq_tags:
        s = df_all_kiva[tag].sum()
        df_sum_tags = df_sum_tags.append(pd.DataFrame(s, index=[tag], columns=['count']))
        
df_sum_tags

#plt.figure(figsize=(15,9))
fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)
df_sum_tags.sort_values('count', inplace=True, ascending=False)
#df_sum_tags.sort_index(inplace=True)
sns.barplot(y=df_sum_tags.index, x=df_sum_tags['count'], color='#c44e52')
ax.set_xlabel('#count #of #loans')
ax.set_ylabel('#hashtag')
ax.set_title('#All #of #the #Hashtags', fontsize=15)
plt.show()


# <a id=themes></a>
# # 7 Loan Theme Types
# 
# Let's take a look at loan theme type.  Rather than count our specific subset of loans, it seems it may be more useful for us to simply use the number column from the loan_themes_by_region data, which contains the total number of loans the partner has made for that theme.  **NOTE:** The highest is General, which has both forkiva set as yes and no - however it isn't particularly interesting and skews the chart.  Thus in the code I have chosen to omit it.

# In[ ]:


df_themes = df_kv_theme_rgn.groupby(['Loan Theme Type', 'forkiva'])['number'].sum()
df_themes = df_themes.to_frame()

df_themes.reset_index(level=1, inplace=True)
df_themes.reset_index(level=0, inplace=True)
df_themes = df_themes.pivot(index='Loan Theme Type', columns='forkiva', values='number')

df_themes['No'] = df_themes['No'].fillna(0)
df_themes['Yes'] = df_themes['Yes'].fillna(0)

df_themes['total'] = df_themes['No'].fillna(0) + df_themes['Yes'].fillna(0)
df_themes = df_themes.sort_values(by='total', ascending=False).head(40)
df_themes.reset_index(level=0, inplace=True)

s_force_order = df_themes[df_themes['Loan Theme Type'] != 'General'].sort_values('total', ascending=False)['Loan Theme Type'].head(40)

# Initialize the matplotlib figure
fig, ax = plt.subplots(figsize=(15, 10))

sns.barplot(x='total', y='Loan Theme Type', data=df_themes[df_themes['Loan Theme Type'] != 'General'],
            label='No', color='#8ed3f4', order=s_force_order)

sns.barplot(x='Yes', y='Loan Theme Type', data=df_themes[df_themes['Loan Theme Type'] != 'General'],
            label='Yes', color='#0abda0', order=s_force_order)

ax.legend(ncol=2, loc='best', frameon=True)
ax.set(ylabel='Loan Theme Type',
       xlabel='number of loans')

leg = ax.get_legend()
new_title = 'for kiva?'
leg.set_title(new_title)
ax.set_title('Top Loan Theme Types (Excluding General) by forkiva', fontsize=15)
plt.show()


# Without General skewing the graph, we can see the relative amounts of these themes pretty well.  Kiva appears to be doing a really good job pushing some of their themes!
# <a id=curr></a>
# # 8 Exploring Currency
# 
# <a id=curr_usg></a>
# ## 8.1 Currency Usage
# 
# Let's take a look at countries used in multiple countries.  I've included a loan count for currencies that are used in 2 or more countries.  However sometimes this only meant a single loan, and the graph was big.  I only ended up plotting those countries which have had loans in multiple currencies where their were more than 20 loans in a currency.  ILS and JOD (Israel and Jordan) thus only show one bar, although in Palestine < 20 loans in each currency was lent (4 and 8, respecitively).  Developing countries sometimes end up with monetary systems pegged to more stable countries or straight out use the money of those countries as a result of lack of trust or poor management of their state currency.  Some notes:
# 1. The *Central African CFA Franc* **XAF** is pegged to the Euro at 1 Euro = 655.957 XAF. It is the currency for six independent states in central Africa: Cameroon, Central African Republic, Chad, Republic of the Congo, Equatorial Guinea and Gabon.
# 2. The *West African CFA Franc* **XOF** is pegged the same way.  It is the currency for Benin, Burkina Faso, Guinea-Bissau, Ivory Coast, Mali, Niger, Senegal, and Togo.
# 3. The following countries outside the US *only* use **USD**: Ecuador, East Timor, El Salvador, Marshall Islands, Micronesia, Palau, Turks and Caicos, British Virgin Islands, and Zimbabwe.  Cambodia has the Cambodien Riel (KHR) however foreign debit cards disburse USD at ATMs, and 90% of the country uses US Dollars, with the local currency generally used for change or anything worth less than a dollar.  We can see the vast amount of Palestine's loan counts are in USD as well.
# 4. Note Congo and The Democratic Republic of the Congo are different countries.

# In[ ]:


min_num_loans = 21
df_currencies = df_all_kiva[['country', 'currency']].drop_duplicates().groupby('currency').count()
df_currencies.reset_index(level=0, inplace=True)
s_currencies = df_currencies[df_currencies['country'] > 1]['currency']
df_currencies = df_all_kiva[df_all_kiva['currency'].isin(s_currencies)].groupby(['country', 'currency'])['id'].count()
df_currencies = pd.Series.to_frame(df_currencies)
df_currencies.reset_index(level=1, inplace=True)
df_currencies.reset_index(level=0, inplace=True)
df_currencies.sort_values(['currency', 'id'], inplace=True)
df_currencies = df_currencies.rename(index=str, columns={'id': 'count'})

s_force_order = df_currencies[df_currencies['count'] >= min_num_loans].sort_values(['currency', 'count'], ascending=False).drop_duplicates()['country']

fig, ax = plt.subplots(1, 1, figsize=(15, 12), sharex=True)

currencies = df_currencies['currency'].unique()
palette = itertools.cycle(sns.color_palette('hls', len(currencies)))

df_piv = df_currencies[df_currencies['count'] >= min_num_loans].pivot(index='country', columns='currency', values='count')
df_piv.reset_index(level=0, inplace=True)

for c in currencies:
    sns.barplot(x=c, y='country', data=df_piv,
            label=c, color=next(palette), order=s_force_order)
    
ax.legend(ncol=2, loc='best', frameon=True)
ax.set_xlabel('count of loans')
leg = ax.get_legend()
new_title = 'Currency'
leg.set_title(new_title)
ax.set_title('Currencies Used Across Multiple Countries', fontsize=15)
plt.show()


# <a id=curr_avg></a>
# ## 8.2 Mean Loan by Currency for Top 20 Currencies
# 
# This is a bit of a tricky one...  I'm showing the average loan, and I'm showing the percentage of it by currency within the stacked bar.  Ie. if a country is only in USD the loan is all the single USD color.  If a country has three 100 USD loans and one 200 USD disbursed in PHP loan, it would show a total bar length of 500/4 = $125.  60% of the bar would be in USD color and 40% of the bar in PHP color.  To try and keep the graph slightly less busy than plotting all of the data, only the amount disbursed in the top 20 currencies by amount are shown.  The minimum mean loan amount for a country is 217; so only means >= 200 are shown - which keeps off countries who have a fraction of loans in a top 20 currency, but the remainder in a local currency.  Countries also must have at least 100 total loans to be shown.

# In[ ]:


num_top_currencies = 20
min_mean = 200
num_min_country_loans = 100

# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
disc_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', 
               '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080', '#FFFFFF', '#000000']
sns.set_palette(disc_colors)


df_currencies = df_all_kiva[['country', 'currency', 'loan_amount', 'id']]
df_currencies = df_currencies.groupby(['country', 'currency']).agg({'loan_amount':'sum', 'id':'count'})
df_currencies = df_currencies.rename(index=str, columns={'id': 'count'})
#df_currencies.shape  #137
df_currencies.reset_index(level=1, inplace=True)
df_currencies.reset_index(level=0, inplace=True)

df_currencies_tot = df_currencies.groupby('country').sum()
df_currencies_tot.reset_index(level=0, inplace=True)
df_currencies_tot = df_currencies_tot.rename(index=str, columns={'loan_amount': 'sum_loan_amount'})
df_currencies_tot = df_currencies_tot.rename(index=str, columns={'count': 'sum_count'})



df_currencies = df_currencies.merge(df_currencies_tot, on='country')
#avg_loan_cur = average loan times ratio of disbursed currency of total currency
df_currencies['avg_loan_cur'] = (df_currencies['sum_loan_amount'] / df_currencies['sum_count']) * (df_currencies['loan_amount'] / df_currencies['sum_loan_amount'])
df_currencies = df_currencies[df_currencies['sum_count'] >= num_min_country_loans ]

# get top x many used currencies
df_limit_cur = df_currencies.groupby('currency')['sum_loan_amount'].sum().to_frame().sort_values('sum_loan_amount', ascending=False)
df_limit_cur.reset_index(level=0, inplace=True)
s_limit_cur = df_limit_cur['currency'].head(num_top_currencies)

df_currencies = df_currencies.pivot(index='country', columns='currency', values='avg_loan_cur')
#currencies = df_currencies.columns.tolist()
#currencies = ['USD', 'PHP', 'XAF']

df_currencies = df_currencies[s_limit_cur]
df_currencies.reset_index(level=0, inplace=True)
df_currencies.dropna(axis=0, how='all')

df_currencies['total_cur'] = 0
for c in s_limit_cur:
    df_currencies[c] = df_currencies[c].fillna(0)
    df_currencies['total_cur'] = df_currencies['total_cur'] + df_currencies[c]
    
df_currencies = df_currencies[df_currencies['total_cur'] > min_mean]

fig, ax = plt.subplots(1, 1, figsize=(19, 12), sharex=True)

palette = itertools.cycle(sns.color_palette(palette=disc_colors, n_colors=22))

df_currencies.sort_values('total_cur', inplace=True, ascending=False)

for c in s_limit_cur:
    sns.barplot(x='total_cur', y='country', data=df_currencies,
            label=c, color=next(palette))
    df_currencies['total_cur'] = df_currencies['total_cur'] - df_currencies[c]

ax.legend(ncol=2, loc='best', frameon=True)
ax.set_xlabel('mean loan')
leg = ax.get_legend()
new_title = 'Currency'
leg.set_title(new_title)
ax.set_title('Mean Loan - By Disbursed Currency Percentage', fontsize=15)
plt.show()


# Ultimately, this took a lot of my hack python effort; and although the final data set is small, it seems the more currencies asked for the much more computationally expensive plotting the graph is.  Did it tell us anything interesting?
# 
# 1. A country using multiple currencies uses USD as the second currency almost exclusively.
# 2. Israel, Paraguay, Rwanda, Senegal, and Bolivia all disburse in fairly high average loan sizes, and within their own currency.  This means their currency is probably relatively stable and/or this is the result of government regulation requiring it.  It's interesting to see the currency breakdown overall as we go down the graph to me!
# 
# The initial version of this graph was stacked incorrectly and only showed Lebanon with multiple currencies, thus the research into it next below; despite that it is no longer an outlier.  It's still a bit interesting so I chose to keep it.
# <a id=curr_leb></a>
# ## 8.3 What's going on in Lebanon?

# In[ ]:


df_lebanon = df_all_kiva[df_all_kiva['country'] == 'Lebanon'][['currency', 'loan_amount', 'disbursed_time']]
df_lebanon['disbursed_time_month'] = df_lebanon['disbursed_time'] + pd.offsets.MonthBegin(-1)

df_lebanon = df_lebanon.groupby(['currency', 'disbursed_time_month']).sum()
df_lebanon.reset_index(level=1, inplace=True)
df_lebanon.reset_index(level=0, inplace=True)



fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_lebanon[df_lebanon['currency'] == 'LBP']['disbursed_time_month'], df_lebanon[df_lebanon['currency'] == 'LBP']['loan_amount'])
plt.plot(df_lebanon[df_lebanon['currency'] == 'USD']['disbursed_time_month'], df_lebanon[df_lebanon['currency'] == 'USD']['loan_amount'])
plt.legend(['LBP', 'USD'], loc='upper left')
ax.set_title('Loan Distribution by Currency in Lebanon', fontsize=15)
plt.show()


# <img align=center src=https://www.doyouevendata.com/wp-content/uploads/2018/03/lebanon.jpg>

# I plotted some monthly data and also found a graph of inflation for the Lebanese Pound (LBP) - they were actually experiencing deflation between the end of 2014 and mid-year 2016.  This means the currency was actually gaining purchasing power.  This seems to account for the decline in USD and rise in LBP.  It is curious as to who is making the distinction, ie. were borrowers asking for LBP or were field partners offering/pushing it?  Inflation is more of the borrower's friend than deflation - the borrower would likely not reap any benefits of increased purchasing power as they are likely borrowing to buy something right away.  Inflation, however, means the borrower is paying back their loan with "cheaper" currency in that the currency now has less purchasing power.  In theory the interest rate the money is lent at accounts for both a small profit for the field partner as well as a factor to hedge against inflation risk.  Can we see anything interesting if we look at the field partners?
# <a id=leb_fld></a>
# ## 8.4 Lebanese Field Partners

# In[ ]:


df_lebanon = df_all_kiva[df_all_kiva['country'] == 'Lebanon'][['id', 'currency', 'loan_amount', 'disbursed_time', 'Field Partner Name', 'sector', 'activity']]
print ('USD Partner loan count:')
print(df_lebanon[df_lebanon['currency'] == 'USD']['Field Partner Name'].value_counts().head(15))
print ('LBP Partner loan count:')
print(df_lebanon[df_lebanon['currency'] == 'LBP']['Field Partner Name'].value_counts().head(15))


# In[ ]:


df_lebanon = df_all_kiva[(df_all_kiva['country'] == 'Lebanon') & (df_all_kiva['currency'] == 'LBP')][['Field Partner Name', 'loan_amount', 'disbursed_time']]
df_lebanon['disbursed_time_month'] = df_lebanon['disbursed_time'] + pd.offsets.MonthBegin(-1)

df_lebanon = df_lebanon.groupby(['Field Partner Name', 'disbursed_time_month']).sum()
df_lebanon.reset_index(level=1, inplace=True)
df_lebanon.reset_index(level=0, inplace=True)

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_lebanon[df_lebanon['Field Partner Name'] == 'Al Majmoua']['disbursed_time_month'], df_lebanon[df_lebanon['Field Partner Name'] == 'Al Majmoua']['loan_amount'])
plt.plot(df_lebanon[df_lebanon['Field Partner Name'] == 'Ibdaa Microfinance']['disbursed_time_month'], df_lebanon[df_lebanon['Field Partner Name'] == 'Ibdaa Microfinance']['loan_amount'])
plt.legend(['Al Majmoua', 'Ibdaa Microfinance'], loc='upper left', frameon=True)
ax.set_title('**LBP Only** Loans by Field Partner in Lebanon', fontsize=15)
plt.show()


# Digging deeper into the data, it looks like Lebanon has only two field partners.  Al Majmoua does most of their lending in USD, and was very little until the deflation period started.  It did increase but is tracking back towards very low numbers again.  Ibdaa Microfinance only deals in LBP and appears to be on quite the roller coaster in regards to lending to Kiva borrowers.
# <a id=bullet_agg></a>
# # 9 Bullet Loans for Agriculture
# 
# Per [Kiva Labs - Financing Agriculture](https://www.kiva.org/about/impact/labs/financingagriculture) there is a drive towards extending bullet type loans in the Agriculture sector as a solution proposed to the uncertainty of farming life, whether raising crops or rearing animals.  These loans allow for the majority of pay back to made in in a lump sum at the end of the loan life - timing well with the farmer actually selling their crop they have tended to for a farming cycle, or an animal they have raised for years.  This graph contains all the countries where at least 100 loans total have been made.  It is ordered in descending order of MPI National, which may not be the best proxy for food security where these loans might really tend to help, but it's what we've got to roll with.

# In[ ]:


def isbullet(ri):

    if 'bullet' in ri:
        return 'bullet'
    else:
        return 'not bullet'
#df_bullet = df_all_kiva[df_all_kiva['sector'] == 'Agriculture'][['country', 'repayment_interval', 'MPI National']]
df_bullet = df_all_kiva[df_all_kiva['sector'] == 'Agriculture'][['country_mpi', 'repayment_interval', 'MPI National']]
df_bullet['loan_type'] = df_bullet['repayment_interval'].apply(isbullet)

#df_bullet = df_bullet.groupby(['country', 'loan_type', 'MPI National']).count()
df_bullet = df_bullet.groupby(['country_mpi', 'loan_type', 'MPI National']).count()

df_bullet = df_bullet.rename(index=str, columns={'repayment_interval': 'count'})

df_bullet.reset_index(level=2, inplace=True)
df_bullet.reset_index(level=1, inplace=True)
df_bullet.reset_index(level=0, inplace=True)

num_min_country_loans = 10

df_bullet = df_bullet[df_bullet['count'] >= num_min_country_loans ]

#df_bullet['MPI National str'] = df_bullet['MPI National'].astype(float).round(3).astype(str).fillna('?')
#df_bullet['country_mpi'] = df_bullet['country'] + ' - ' + df_bullet['MPI National str']

s_force_order = df_bullet[['MPI National', 'country_mpi']].sort_values('MPI National', ascending=False).drop_duplicates()['country_mpi']

df_piv = df_bullet.pivot(index='country_mpi', columns='loan_type', values='count')
df_piv.reset_index(level=0, inplace=True)
df_piv['total'] = df_piv['bullet'].fillna(0) + df_piv['not bullet'].fillna(0)

fig, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

bts = ['bullet', 'not bullet']

palette = itertools.cycle(sns.color_palette('hls', len(bts)))
sns.barplot(x='total', y='country_mpi', data=df_piv,
        label='not bullet', color=next(palette), order=s_force_order)

sns.barplot(x='bullet', y='country_mpi', data=df_piv,
        label='bullet', color=next(palette), order=s_force_order)

ax.legend(ncol=1, loc='upper right', frameon=True)
ax.set_xlabel('count of loans')
ax.set_ylabel('country - mpi national')
leg = ax.get_legend()
new_title = 'Loan Type'
leg.set_title(new_title)
ax.set_title('Count of Agricultural Loans by Loan Type, Ordered by National MPI Descending', fontsize=15)
plt.show()


# We do have some takeaways here;
# 1. As a general rule, **anything in red on this chart is open for improvement!**
# 2. **Countries near the top of the chart** may have some of the strongest impact in reducing poverty as they are generally more impoverished.
# 3. Countries with many loans in red have a large opportunity to make an impact as well - many loans are already being made through field partners, just not of the bullet type.
# 4. Mali and Nigeria were specifically mentioned in the Kiva article, and here we can see the vast majority of their loans are bullet loans - but there's also some other countries and field partners doing a great job with this too, like Myanmar and Colombia!
# 5. We don't have default data - perhaps this is lower for Agriculture bullet loans and Kiva could use this as a point to sell to field partners currently not offering them?
# 6. Some countries have no bullet loans - perhaps Kiva could concentrate limited corporate resources there, whereas in countries where field partners are already leveraging bullet loans, encourage the field partners to spread the good news to other field partners?
# 7. The list might be better sorted in order of food insecurity if some national level data is available.

# In[ ]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>It appears Kiva has significant open opportunity to spreading the beneficial bullet loans for Agriculture borrowers, and can prioritize them by potential market size changes and poverty dimensions.')


# <a id=sal1></a>
# ## 9.1 El Salvador Investigation by Activity
# Let's take a look at El Salvador, why is it split between bullet and non bullet?  There's only a few field partners, and the majority of loans really comes from one, so that didn't appear to make a meaningful difference.

# In[ ]:


df_els = df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture')][['activity', 'repayment_interval']]
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['activity', 'loan_type']).count()
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'repayment_interval': 'count'})
df_piv = df_els.pivot(index='activity', columns='loan_type', values='count')
df_piv.reset_index(level=0, inplace=True)
df_piv['total'] = df_piv['bullet'].fillna(0) + df_piv['not bullet'].fillna(0)

s_force_order = df_piv[['activity', 'total']].sort_values('total', ascending=False).drop_duplicates()['activity']

fig, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

palette = itertools.cycle(sns.color_palette('hls', len(bts)))
sns.barplot(x='total', y='activity', data=df_piv,
        label='not bullet', color=next(palette), order=s_force_order)

sns.barplot(x='bullet', y='activity', data=df_piv,
        label='bullet', color=next(palette), order=s_force_order)

ax.legend(ncol=1, loc='center right', frameon=True)
ax.set_xlabel('count of loans')
ax.set_ylabel('country - mpi national')
leg = ax.get_legend()
new_title = 'Loan Type'
leg.set_title(new_title)
ax.set_title('El Salvador Agriculture Loans by Activity', fontsize=15)
plt.show()


# <a id=sal2></a>
# ## 9.2 El Salvador Loan Count Over Time
# Is the trend at least going up?

# In[ ]:


df_els = df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture')][['activity', 'repayment_interval', 'disbursed_time']]
df_els['disbursed_time_month'] = df_els['disbursed_time'] + pd.offsets.MonthBegin(-1)
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['loan_type', 'disbursed_time_month'])[['activity']].count()
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'activity': 'count'})

fig, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

plt.plot(df_els[df_els['loan_type'] == 'bullet']['disbursed_time_month'], df_els[df_els['loan_type'] == 'bullet']['count'], color='#67c5cb')
plt.plot(df_els[df_els['loan_type'] == 'not bullet']['disbursed_time_month'], df_els[df_els['loan_type'] == 'not bullet']['count'], color='#cb6d67')

plt.legend(['bullet', 'not bullet'], loc='upper left', frameon=True)
ax.set_title('El Salvador Loan Count', fontsize=15)
plt.show()


# It doesn't appear to be.
# <a id=sal3></a>
# ## 9.3 El Salvador Animal Loans
# What if we dig in deeper and just look at the activities Livestock, Cattle, Poultry, Pigs?

# In[ ]:


animals = ['Livestock', 'Cattle', 'Poultry', 'Pigs']
df_els = df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture') 
                    & (df_all_kiva['activity'].isin(animals))][['activity', 'repayment_interval', 'disbursed_time']]
df_els['disbursed_time_month'] = df_els['disbursed_time'] + pd.offsets.MonthBegin(-1)
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['loan_type', 'disbursed_time_month', 'activity'])[['repayment_interval']].count()
df_els.reset_index(level=2, inplace=True)
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'repayment_interval': 'count'})

linestyles = ['-', '--', '-.', ':']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True)

i = 0
for a in animals:
    if a in ['Livestock', 'Cattle']:
        ax1.plot(df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['count'], color='#67c5cb', linestyle=linestyles[i], label=a + ' ' + 'bullet', linewidth=3)
        ax1.plot(df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['count'], color='#cb6d67', linestyle=linestyles[i], label=a + ' ' + 'not bullet', linewidth=3)
    else:
        ax2.plot(df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'bullet') & (df_els['activity'] == a)]['count'], color='#67c5cb', linestyle=linestyles[i], label=a + ' ' + 'bullet', linewidth=3)
        ax2.plot(df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['disbursed_time_month'], 
                 df_els[(df_els['loan_type'] == 'not bullet') & (df_els['activity'] == a)]['count'], color='#cb6d67', linestyle=linestyles[i], label=a + ' ' + 'not bullet', linewidth=3)
    
    i = i + 1

ax1.legend(loc='upper left', frameon=True)
ax2.legend(loc='upper left', frameon=True)
ax1.set_title('El Salvador Loan Count by Animal Activity', fontsize=15)
plt.show()


# Interestingly enough it appears Livestock has been on a big non-seasonal decline, meanwhile cattle has gone up - perhaps Kiva borrowers are switching stocks noting some kind of overall trend?  Whereas livestock can produce commodities (meat, milk, eggs, fur, etc.), cattle is generally primarily raised for meat, and we'll note that a separate (ungraphed) Activity value exists for the category Dairy.  Countries generally consume more beef as they get wealthier, although I didn't notice a major upswing on GDP or anything from a quick google.  Per https://www.export.gov/article?id=El-Salvador-Agricultural-Sector we have this quote *"Dairy production is increasing due to government incentives and sanitary regulations that provide protection against contraband cheese from Nicaragua and Honduras."*  Perhaps the bump we are seeing is a result of this; although not fully categorized properly by the field partner?

# In[ ]:


df_all_kiva[(df_all_kiva['country'] == 'El Salvador') & (df_all_kiva['sector'] == 'Agriculture') 
                    & (df_all_kiva['activity'] == 'Cattle')][['loan_URL', 'disbursed_time']].sort_values('disbursed_time', ascending=False)['loan_URL'].head()


# In[ ]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>I manually sampled the 5 most recent Cattle loans in El Salvador above - 3 appeared to be for beef, 1 was a loan for auto repair to bring milk to market, and 1 was for dairy production.  Kiva should ensure field partners are coding things properly so that they are able to assess impacts and make decisions with cleaner data. It appears some of this Cattle bump should really be filed under Dairy.')


# Unfortunately we don't see much of a rising trend vs. history for Pigs or Poultry; one seems there, but small.  Maybe these activities aren't very suitable for these type of loans?  Let's see what is done not so far away in Colombia, are bullet loans popular for these specific activities there?

# In[ ]:


df_els = df_all_kiva[(df_all_kiva['country'] == 'Colombia') & (df_all_kiva['sector'] == 'Agriculture')][['activity', 'repayment_interval']]
df_els['loan_type'] = df_els['repayment_interval'].apply(isbullet)
df_els = df_els.groupby(['activity', 'loan_type']).count()
df_els.reset_index(level=1, inplace=True)
df_els.reset_index(level=0, inplace=True)
df_els = df_els.rename(index=str, columns={'repayment_interval': 'count'})
df_piv = df_els.pivot(index='activity', columns='loan_type', values='count')
df_piv.reset_index(level=0, inplace=True)
df_piv['total'] = df_piv['bullet'].fillna(0) + df_piv['not bullet'].fillna(0)

s_force_order = df_piv[['activity', 'total']].sort_values('total', ascending=False).drop_duplicates()['activity']
fig, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True)

palette = itertools.cycle(sns.color_palette('hls', len(bts)))
sns.barplot(x='total', y='activity', data=df_piv,
        label='not bullet', color=next(palette), order=s_force_order)

sns.barplot(x='bullet', y='activity', data=df_piv,
        label='bullet', color=next(palette), order=s_force_order)

ax.legend(ncol=1, loc='center right', frameon=True)
ax.set_xlabel('count of loans')
ax.set_ylabel('country - mpi national')
leg = ax.get_legend()
new_title = 'Loan Type'
leg.set_title(new_title)
ax.set_title('Colombia Agriculture Loans by Activity', fontsize=15)
plt.show()


# In[ ]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>Colombia is offering a lot of bullet loans for Pigs and Poultry!  Kiva should seek to encourage their field partners to expand their bullet loan offerings to these activities in El Salvador.')


# <a id=unfunded></a>
# # 10. Unfunded Loans

# In[ ]:


#one loan overfunded... 
df_all_kiva['funded'] = np.where(df_all_kiva['funded_amount'] < df_all_kiva['loan_amount'], 'no', 'yes')
df_funded = df_all_kiva[['loan_amount', 'funded_amount', 'funded', 'World region', 'country']]
df_funded['yes'] = np.where(df_funded['funded'] == 'yes', df_funded['funded_amount'], np.nan)
df_funded['no'] = np.where(df_funded['funded'] == 'no', df_funded['funded_amount'], np.nan)
df_funded.drop(['funded_amount', 'funded'], axis=1)

fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
#ax = 
df_funded.plot(kind='scatter', x='loan_amount', y='yes', title='funded vs requested loan amount', color='g', ax=ax)
df_funded.plot(kind='scatter', x='loan_amount', y='no', color='r', ax=ax)
ax.set_ylabel('funded_amount')
plt.show()


# Unfunded vs. funded loans - we see some hard walls at 50k and 10k USD.  Let's take a look at the 10 and under as these loans are much more numerous.

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
#ax = 
df_funded[df_funded['loan_amount'] <= 10000].plot(kind='scatter', x='loan_amount', y='yes', title='funded vs requested loan amount', color='g', ax=ax)
df_funded[df_funded['loan_amount'] <= 10000].plot(kind='scatter', x='loan_amount', y='no', color='r', ax=ax)
ax.set_ylabel('funded_amount')
plt.show()


# One loan in Europe and Central Asia was funded 3400 but was only a 3000 USD loan.  How'd that happen?  Perhaps human psychology is causing 10k to be a hard wall.  Perhaps these loans would fund better were they able to be split into separate loans?  Or closer to exactly what was needed vs. a general line of credit they may have been requested as?  Let's take a look at the World region breakdown of these loans.  Note some drop out (Samoa and the US being the largest, even after World region cleanups I made in the data prep stage).  It is likely much of the 10k wall was actually the US.

# In[ ]:


maxloan = 10000

fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
#ax = 
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'East Asia and the Pacific')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', title='funded vs requested loan amount, unfunded loans <= 10000', color='orange', ax=ax, label='East Asia and the Pacific')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Sub-Saharan Africa')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='brown', ax=ax, alpha=0.5, label='Sub-Saharan Africa')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Latin America and Caribbean')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='purple', ax=ax, alpha=0.5, label='Latin America and Caribbean')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'South Asia')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='green', ax=ax, alpha=0.5, label='South Asia')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Europe and Central Asia')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='cyan', ax=ax, alpha=0.5, label='Europe and Central Asia')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Arab States')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='red', ax=ax, alpha=0.5, label='Arab States')
ax.set_ylabel('funded_amount')

ax.legend(ncol=2, loc='best', frameon=True)
leg = ax.get_legend()
leg.set_title('World region')

plt.show()


# Interestingly enough, a hard Latin America and Caribbean wall appears at 5000.  It turns out that it is majority Bolivia (python not shown).  What about loans for 3000 or less?  Pretty busy in the above graph.

# In[ ]:


maxloan = 3000

fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
#ax = 
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'East Asia and the Pacific')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', title='funded vs requested loan amount, unfunded loans <= 3000', color='orange', ax=ax, label='East Asia and the Pacific')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Sub-Saharan Africa')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='brown', ax=ax, alpha=0.5, label='Sub-Saharan Africa')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Latin America and Caribbean')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='purple', ax=ax, alpha=0.5, label='Latin America and Caribbean')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'South Asia')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='green', ax=ax, alpha=0.5, label='South Asia')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Europe and Central Asia')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='cyan', ax=ax, alpha=0.5, label='Europe and Central Asia')
df_funded[(df_funded['loan_amount'] <= maxloan) 
          & (df_funded['World region'] == 'Arab States')
          & (df_funded['funded'] == 'no')].plot(kind='scatter', x='loan_amount', y='no', color='red', ax=ax, alpha=0.5, label='Arab States')
ax.set_ylabel('funded_amount')

ax.legend(ncol=2, loc='best', frameon=True)
leg = ax.get_legend()
leg.set_title('World region')

plt.show()


# Top 30 most frequently unfunded country - sector combinations:

# In[ ]:


df_fund = df_all_kiva[['funded', 'country', 'sector', 'id']].groupby(['country', 'funded', 'sector'])['id'].count().to_frame()
df_fund.rename(index=str, columns={'id': 'count'}, inplace=True)
df_fund.reset_index(level=2, inplace=True)
df_fund.reset_index(level=1, inplace=True)
df_fund.reset_index(level=0, inplace=True)
df_fund['combo'] = df_fund['country'] + '-' + df_fund['sector']
df_fund = df_fund.pivot(index='combo', columns='funded', values='count')
df_fund['no'] = df_fund['no'].fillna(0)
df_fund['yes'] = df_fund['yes'].fillna(0)
df_fund['total'] = df_fund['no'] + df_fund['yes']
df_fund['no_pct'] = df_fund['no'] / df_fund['total'] * 100
df_fund[df_fund['total'] > 10].sort_values('no_pct', ascending=False).head(25)

plt.figure(figsize=(15,8))
#plotSeries = df_all_kiva['sector'].value_counts()
#ax = sns.barplot(plotSeries.values, plotSeries.index, color='pink')
#ax.set_title('Top Sectors', fontsize=15)
#plt.show()
df_fund.reset_index(level=0, inplace=True)
ax = sns.barplot(x='no_pct', y='combo', color='#bba5ea', data=df_fund[df_fund['total'] > 10].sort_values('no_pct', ascending=False).head(30))
ax.set_xlabel('percent unfunded')
ax.set_ylabel('country - sector')
ax.set_title('Top 30 Mostly Frequently Not Funded Country - Sectors', fontsize=15)
plt.show()


# <a id=phil></a>
# # 11. Is the Philippines Really the Country with the Most Kiva Activity?
# These are the top 15 countries by loan count.

# In[ ]:


plt.figure(figsize=(15,8))
plotSeries = df_all_kiva['country'].value_counts().head(15)
ax = sns.barplot(plotSeries.values, plotSeries.index, color='c')
ax.set_title('Top 15 Countries by Loan Count', fontsize=15)
ax.set_xlabel('count of loans')
plt.show()


# However, are they the countries with the most lending activity going on?  The Philippines is pretty big; Ecuador rather small in comparison.  What if we make a per capita adjustment?  Keeping life simple I'm just going to use 2016 population data.  Let's break it down by sector too.

# In[ ]:


min_loans_per_million = 100
# make countries by sector set to display
df_countries = df_all_kiva[['id', 'country', 'country_mpi', 'sector', 'MPI National']].groupby(['country', 'country_mpi', 'sector', 'MPI National'])['id'].count().to_frame()
df_countries.rename(index=str, columns={'id': 'count'}, inplace=True)
df_countries.reset_index(level=3, inplace=True)
df_countries.reset_index(level=2, inplace=True)
df_countries.reset_index(level=1, inplace=True)
df_countries.reset_index(level=0, inplace=True)
sectors = df_countries['sector'].unique()
#df_countries['MPI National str'] = df_countries['MPI National'].astype(float).round(3).astype(str).fillna('?')
#df_countries['country_mpi'] = df_countries['country'] + ' - ' + df_countries['MPI National str']
# adjust by population
df_countries = df_countries.merge(df_world_pop[['Country', '2016']], left_on=['country'], right_on=['Country'])
df_countries.drop('Country', axis=1, inplace=True)
df_countries['loans_per_mil'] = df_countries['count'] / df_countries['2016'] * 1000000
# get total loans per population
df_total_per_mil = df_all_kiva[['id', 'country']].groupby(['country'])['id'].count().to_frame()
df_total_per_mil.rename(index=str, columns={'id': 'count'}, inplace=True)
df_total_per_mil.reset_index(level=0, inplace=True)
df_total_per_mil = df_total_per_mil.merge(df_world_pop[['Country', '2016']], left_on=['country'], right_on=['Country'])
df_total_per_mil.drop('Country', axis=1, inplace=True)
df_total_per_mil['total_loans_per_mil'] = df_total_per_mil['count'] / df_total_per_mil['2016'] * 1000000
#restrict output to at least s many loans per million
#df_countries[df_countries['country'].isin(df_total_per_mil[df_total_per_mil['loans_per_mil'] >= min_loans_per_million]['country'])]
df_countries = df_countries.merge(df_total_per_mil[df_total_per_mil['total_loans_per_mil'] >= min_loans_per_million][['country', 'total_loans_per_mil']], on='country')
s_force_order = df_countries[['total_loans_per_mil', 'country_mpi']].sort_values('total_loans_per_mil', ascending=False).drop_duplicates()['country_mpi']

df_piv = df_countries.pivot(index='country_mpi', columns='sector', values='loans_per_mil')
df_piv.reset_index(level=0, inplace=True)
for s in sectors:
    df_piv[s] = df_piv[s].fillna(0)
    
#don't know how to keep this so i'm just going to make it again
df_piv['total_loans_per_mil'] = 0
for s in sectors:
    df_piv['total_loans_per_mil'] = df_piv['total_loans_per_mil'] + df_piv[s]
    
disc_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', 
               '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080', '#808080', '#FFFFFF', '#000000']
sns.set_palette(disc_colors)
palette = itertools.cycle(sns.color_palette(palette=disc_colors, n_colors=22))

fig, ax = plt.subplots(1, 1, figsize=(15, 9), sharex=True)

for s in sectors:    
    sns.barplot(x='total_loans_per_mil', y='country_mpi', data=df_piv,
            label=s, color=next(palette), order=s_force_order)
    #print('sector: ' + s + 'total_loans_per_mil: ' + str(df_piv[df_piv['country_mpi'] == 'Philippines - 0.052']['total_loans_per_mil']))
    df_piv['total_loans_per_mil'] = df_piv['total_loans_per_mil'] - df_piv[s]
    
ax.legend(ncol=2, loc='lower right', frameon=True)
ax.set_xlabel('count of loans')
leg = ax.get_legend()
new_title = 'Sector'
leg.set_title(new_title)
ax.set_title('Loans Per Million People', fontsize=15)
plt.show()


# On a per capita basis, the Philippines is actually 7th!  El Salvador on the other hand has a tooon of usage - well over double the 2nd most popular country for Kiva loans!!  It also has a huge amount for housing.  Cambodia has an exceptional amount for personal use.  What's going on in there?

# In[ ]:


df_all_kiva[(df_all_kiva['country'] == 'Cambodia') & (df_all_kiva['sector'] == 'Personal Use')]['activity'].value_counts()


# Looks like Cambodia is big on Kiva loans for home appliances!
# <a id=kiva_poverty></a>
# # 12. Current Poverty Targeting System
# Annalie has a good kernel on the [current targeting system](https://www.kaggle.com/annalie/kivampi), with a good follow up by Elliott [here](https://www.kaggle.com/elliottc/kivampi).  Annalie provides an additional metric for financial inclusion [here](https://www.kaggle.com/annalie/metric-for-financial-inclusion).  Overall I think their metrics and approaches sound very good and are challenging to improve upon, particularly because most readily available statistics or data are at the country level, while it seems the most value would be derived in digging deeper into regional or much more local levels for field partners and borrowers.  On top of that, we are generally working with countries that would be among the most challenging to get this type of data from.  Aside from more local/accurate data, there's also a challenge in placing the lender in the correct local area, should that information even be available.
# 
# Things to keep in mind: higher MPI = generally poorer area across the multiple dimensions MPI tries to represent.  We're working with blended MPI data provided at the National as well as administrative regions (Sub-National) within a country; Kiva is looking to improve their accuracy using this data and other external datasets to really understand their borrowers and to make sure limited efforts and funds are being routed to where they can best serve people.  The brunt of this section is looking at current MPI calculations and how they might be improved.
# 
# ![](http://hdr.undp.org/sites/default/files/mpi.png)
# 
# <a id=curr_nat_mpi></a>
# ## 12.1 Current Kiva Scoring - National MPI
# Per the above referenced kernels, 
# 
# "**Nation-level MPI Scores** are broken into rural and urban scores. So Kiva's broadest measure simply assigns each field partner an average of these two numbers, weighted by rural_pct. About a dozen field partners also serve multiple countries, in which case we take a volume-weighted average."
# 
# **NOT** taking into account Field Partner scores that go across multiple countries, let's take a look at Mozambique (MOZ) scores using the existing methodology.  

# In[ ]:


LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv") #.set_index([''])
MPI = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','MPI Urban','MPI Rural']].set_index("ISO")
LT = LT.join(MPI,how='left',on="ISO")[['Partner ID','Field Partner Name','ISO','MPI Rural','MPI Urban','rural_pct','amount', 'mpi_region']].dropna()

LT['rural_pct'] /= 100
#~ Compute the MPI Score for each loan theme
LT['MPI Score'] = LT['rural_pct']*LT['MPI Rural'] + (1-LT['rural_pct'])*LT['MPI Urban']

LT[LT['ISO'] == 'MOZ'][['Partner ID', 'ISO', 'MPI Rural', 'MPI Urban', 'rural_pct', 'mpi_region', 'MPI Score']].drop_duplicates()


# <a id=prob1></a>
# ### 12.1.1 Problem 1: Missing rural percentages
# Note region partners missing rural percentages are dropped from this calc.  I am under the impression rural percentage is self-reported by the field partner as the percent of rural borrowers within that region for them, which is why partner 210 has a different MPI score than partner 23, as the latter has loaned 85% to rural borrowers vs. the former's 20%.  How many partners have a value?

# In[ ]:


df_kiva_rural = df_kv_theme_rgn[['Partner ID', 'rural_pct']].drop_duplicates()
df_kiva_rural['populated'] = np.where(df_kiva_rural['rural_pct'].isnull(), 'No', 'Yes')
df_kiva_rural['populated'].value_counts()


# A full **39.7% of partners are missing values completely**, which is pretty huge.  It should also be understood that the granularity of the file is partner-region-theme level, BUT the rural_percentage exists only at the ***PARTNER level***.
# <a id=prob2></a>
# ### 12.1.2 Problem 2: Can populated values be trusted over time, or even currently?
# What's the distribution of the values we actually have?

# In[ ]:


plt.figure(figsize=(12,6))
sns.distplot(df_kiva_rural[~df_kiva_rural['rural_pct'].isnull()]['rural_pct'], bins=30)
plt.show()


# In[ ]:


df_kiva_rural = df_kv_theme_rgn[['Partner ID', 'rural_pct']].drop_duplicates().groupby('rural_pct')[['Partner ID']].count()
#df_kiva_rural = df_kiva_rural.groupby('rural_pct')[['Partner ID']].count()
df_kiva_rural.reset_index(level=0, inplace=True)
df_kiva_rural = df_kiva_rural.rename(columns={'Partner ID': 'count'})
df_kiva_rural['rural_pct'] = df_kiva_rural['rural_pct'].astype(str).apply(lambda x: str.split(x, '.')[0])
s_force_order = df_kiva_rural.sort_values('count', ascending=False).head(30)['rural_pct']

plt.figure(figsize=(15,8))
ax = sns.barplot(x='count', y='rural_pct', color='#ffa06c', data=df_kiva_rural.sort_values('count', ascending=False).head(30), orient='h', order=s_force_order)
ax.set_ylabel('field partner provided rural percent')
ax.set_xlabel('frequency')
ax.set_title('frequency of specified rural_pct (top 30)', fontsize=15)
plt.show()


# There's a whole bunch of very nice whole round numbers in there - 95, 90, 85, 80, 75 - so what's the likelihood that the numbers are even real; is this data actually kept track of, or is it a gut feel?  What about changes over time - maybe a partner goes from 50% rural to 80% rural - since the attribute appears to live across all partner levels, doesn't this skew current or past analysis?

# In[ ]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>Rural percentages are heavily used in this calculation - however many are missing, and those that exist, seem they may not be particularly trustworthy.  Is data this noisy actually providing value when comparing peers?  Kiva should add a <b>loan level attribute</b> of whether the borrower is living in a rural or urban area.  Note the definition may differ between governments (and people) - however this would allow Kiva themselves to calculate the amount, over a period of time, and for all partners.')


# <a id=wb1></a>
# ### 12.1.3 National MPI with World Bank Rural Percentage Population Data
# I [added a dataset](https://www.kaggle.com/doyouevendata/world-bank-rural-population) which has rural population percentage by country from the world bank.  It was missing Kosovo.  However per this [2016 presentation](http://www.efgs.info/wp-content/uploads/conferences/efgs/2016/S8-1_presentationV1_IdrizShala_EFGS2016.pdf), I will manually add the value 54.7% into the set.
# 
# Then we'll look at all the existing partners in Mozambique without dropping nulls.  We'll create MPI Nat Calc, but this time we'll simply use the 2016 data from the World Bank in rural population percentage.  How does this look?  To keep it simple we'll only look at field partner 23 as well.

# In[ ]:


#df_all_kiva.drop('rural_pct_2016', axis=1, inplace=True)
df_rural = pd.read_csv('../input/world-bank-rural-population/rural_pop.csv')[['ISO','2016']]
df_all_kiva = df_all_kiva.merge(df_rural, how='left', left_on='ISO', right_on='ISO')
df_all_kiva = df_all_kiva.rename(columns={'2016': 'rural_pct_2016'})
df_all_kiva['rural_pct_2016'] = np.where(df_all_kiva['country'] == 'Kosovo', 54.7, df_all_kiva['rural_pct_2016'])

df_moz = df_all_kiva[df_all_kiva['country'] == 'Mozambique'][['ISO', 'country', 'partner_id_loan_theme', 'LocationName_kiva', 'LocationName_kiva_mpi', 'region_mpi', 'rural_pct',                                    
    'MPI National', 'MPI Regional', 'rural_pct_2016', 'MPI Urban','MPI Rural']].drop_duplicates().sort_values('region_mpi')
df_moz['MPI Nat Calc'] = df_moz['rural_pct_2016'] / 100 * df_moz['MPI Rural'] + (100 - df_moz['rural_pct_2016']) / 100 * df_moz['MPI Urban']
df_moz.sort_values(['partner_id_loan_theme', 'region_mpi'])
df_moz[df_moz['partner_id_loan_theme'] == 23][['ISO', 'partner_id_loan_theme', 'LocationName_kiva', 'LocationName_kiva_mpi', 'region_mpi', 'rural_pct',                                    
    'MPI National', 'MPI Regional', 'rural_pct_2016', 'MPI Nat Calc']]


# What do we see?
# 1. MPI Nat Calc is very close to MPI National - close enough to argue the same methodology may very well produce this value, with the difference simply being that rural percentage from another year was used.
# 2. The Kiva methodology produced a value that seems very likely quite out of whack for this case - which I actually chose rather randomly as field partner 23 was a low number and at the top of the list.  Maputo is the capital of Mozambique and a [pretty urban appearing area](https://www.google.co.th/maps/place/Maputo,+Mozambique/@-25.8962418,32.5406432,12z/data=!3m1!4b1!4m5!3m4!1s0x1ee69723b666da55:0x42497f579a6bb442!8m2!3d-25.969248!4d32.5731746); yet the field partner stating 85% of loans were rural in the region is registering a quite large 0.43635 for the area; even the one with 20% is hitting 0.2472.  These are high numbers in comparison to not only MPI National and even MPI Urban rates for the region, but in fact very high compared to the supplied MPI Regional rate of a much much lower 0.043.  Elliott mentioned the data can be noisy, but in this case it looks likely to be extremely off.
# 3. Boane, Maputo, Mozambique is not actually a part of Maputo Cidade (Regional MPI 0.043) but is in fact a part of the Maputo Province (Maputo ProvÃ°ncia in my Excel file...) with a Regioinal MPI of 0.133.  So there is also a question of *how off*, given that the regional assignment is also imperfect.
# 
# Along the lines of number 3 - were many loans misattributed?  Looking at partner 23 loan theme region stats:
# <a id=manual></a>
# ### 12.1.4 Manual Assessment of Current Methodology National MPI Assigned to Field Partner

# In[ ]:


df_kv_theme_rgn[(df_kv_theme_rgn['ISO'] == 'MOZ') & (df_kv_theme_rgn['Partner ID'] == 23)].groupby(['region', 'mpi_region'])[['number', 'amount']].sum().sort_values('amount', ascending=False)


# In fact, a great many loans were misattributed here.  I actually picked Boane at random but it ended up being the largest region.  Machava-15 appears to be a neighborhood in Matola within the province as well, but quite close to Maputo Cidade.  Namaacha is also in the Maputo Province.  In fact, the regions ending in ", Maputo" are all part of Maputo Province.
# 
# This all makes it **highly unlikely** that this field partner's score should be anywhere near the currently calculated value of 0.43635.
# 
# **Question** - what does Kiva do with the peer compared information; ie. the unfortunate question I'm ferreting around is, do field partners have any incentive to exaggerate their rural percentage/poverty numbers?  Hopefully not, but even in the absence of this, the self-reporting of rural percentage and the current methodology seems to really exaggerate the MPI, particularly in the case where most field partner loans are going to urbanized areas - even if the people who lived there were living in a rural part of that urbanized region.  Ie. rural in city suburbs is likely not anywhere near as bad as rural out in nowhere.

# In[ ]:


HTML('<img style="margin: 0px 20px" align=left src=https://www.doyouevendata.com/wp-content/uploads/2018/03/attn.png>Current National MPI Field Partner methodology has cases that appear to very heavily exaggerate MPI due to the methodology and, in particular, the rural percentage value of a field partner as input.')


# <a id=kiva_sub></a>
# ## 12.2 Current Kiva Scoring - Sub-National MPI
# Per the above Kiva team kernels, Regional MPI is calculated as follows:
# 1. Merge in the region's MPI (For partners/countries without sub-national MPI scores, merge in the country-level rural/urban MPI Scores from before)
# 2. Take the average across all regions (weighted by volume) for a given loan theme. (This is the Loan Theme MPI Score)
# 3. Further aggregate to get Field Partner MPI Scores (if we're interested)
# <a id=prob3></a>
# ### 12.2.1 Problem 3: Mix of Methodology
# In step 1, we re-use the calculated National MPI, meaning MPI National numbers *weighted by provided rural_percentage*.  In step 2, for those loans that have an MPI Regional value, we use it by *weighting it with loans*.  **Is it appropriate to mix the methodologies like this?**  It seems that we could be skewing country to country comparisons as well as regional/partner/theme comparisons.
# 
# Also, a point of clarity - Annalie has provided a method for calcluating **Field Partner MPI**, let's look at partner 23:

# In[ ]:


MPIsubnat = pd.read_csv("../input/mpi/MPI_subnational.csv")[['Country', 'Sub-national region', 'World region', 'MPI National', 'MPI Regional']]
# Create new column LocationName that concatenates the columns Country and Sub-national region
MPIsubnat['LocationName'] = MPIsubnat[['Sub-national region', 'Country']].apply(lambda x: ', '.join(x), axis=1)

LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")[['Partner ID', 'Loan Theme ID', 'region', 'mpi_region', 
                                                                                                'ISO', 'number', 'amount', 'LocationName', 'names']]

# Merge dataframes
LT = LT.merge(MPIsubnat, left_on='mpi_region', right_on='LocationName', suffixes=('_lt', '_mpi'))[['Partner ID', 'Loan Theme ID', 'Country', 'ISO', 
                                                                                                   'MPI National',
                                                                                                   'mpi_region', 'MPI Regional', 'number', 'amount']]

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LT.groupby(['Partner ID', 'Loan Theme ID', 'Country', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_lt = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_regional_scores = LS.groupby(level='Partner ID').apply(weighted_avg_lt)

LS.loc[23]


# Elliott, on the other hand, provided a method for calculating **Field Partner Theme MPI**, again, 23:

# In[ ]:


# Load data
MPI = pd.read_csv("../input/mpi/MPI_subnational.csv")[['Country', 'Sub-national region', 'World region', 'MPI Regional']]
MPInat = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','Country','MPI Rural', 'MPI Urban']].set_index('ISO')
LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")[['country','Partner ID', 'Loan Theme ID', 'region', 'mpi_region', 'ISO', 'number', 'amount','rural_pct', 'LocationName', 'Loan Theme Type']]
# Create new column mpi_region and join MPI data to Loan themes on it
MPI['mpi_region'] = MPI[['Sub-national region', 'Country']].apply(lambda x: ', '.join(x), axis=1)
MPI = MPI.set_index('mpi_region')
LT = LT.join(MPI, on='mpi_region', rsuffix='_mpi') #[['country','Partner ID', 'Loan Theme ID', 'Country', 'ISO', 'mpi_region', 'MPI Regional', 'number', 'amount','Loan Theme Type']]
#~ Pull in country-level MPI Scores for when there aren't regional MPI Scores
LT = LT.join(MPInat, on='ISO',rsuffix='_mpinat')
LT['Rural'] = LT['rural_pct']/100        #~ Convert rural percentage to 0-1
LT['MPI Natl'] = LT['Rural']*LT['MPI Rural'] + (1-LT['Rural'])*LT['MPI Urban']
LT['MPI Regional'] = LT['MPI Regional'].fillna(LT['MPI Natl'])
#~ Get "Scores": volume-weighted average of MPI Region within each loan theme.
Scores = LT.groupby('Loan Theme ID').apply(lambda df: np.average(df['MPI Regional'], weights=df['amount'])).to_frame()
Scores.columns=["MPI Score"]
#~ Pull loan theme details
LT = LT.groupby('Loan Theme ID').first()[['country','Partner ID','Loan Theme Type','MPI Natl','Rural','World region']].join(Scores)#.join(LT_['MPI Natl'])
LT[LT['Partner ID'] == 23]


# OPHI Provided values put MPI National at 0.389.  Most of partner 23's loans are in Maputo Cidade (0.043) or Maputo Province (0.133) - these make up 91% of the lent dollars by partner 23.
# 
# However, we see the Kiva National value calculated at 0.43635.  Using Regional values, 0.043.  Using themes, 0.094960 to 0.247239.  This all seems very dirty in trying to compare countries, regions, and field partners, unfortunately.  These numbers are really all across the board.
# <a id=prob4></a>
# ### 12.2.2 Problem 4: Misaligned Regions
# Part of this problem is due to that metnioned above and in section 12.1.4 - the MPI Region assigned to this field partner is incorrect for these locations.
# <a id=weights></a>
# ### 12.2.3 Question - Should weights be by loan count or dollar amount?
# Just an open question - a rare, but out there, very large loan (10k 50k) can certainly touch multiple lives - however I generally expect the greatest poverty reduction to come as a result of time and improved government policies in the aggregate, and think of Kiva as a way to help be a part of helping individuals improve their lives and helping individuals alleviate the burdens and challenges of poverty.  So I think a fundamental question to pose - which I don't have an answer for - is should we weight by people/loans or by dollars?  Dollars are absolutely understandable - but it seems arguable that number of lives touched would make a good metric too.  Note the logic by counting loans/people is a little more in line with the rural_percentage methodology used today as well, as it is strictly focusing on the number and location of borrowers, regardless of the amount.
# 
# <a id=13></a>
# # 13. Philippines Local Analysis (National MPI 0.052)
# 
# <a id=13_1></a>
# ## 13.1 Leveraging Geospatial Data
# MPI Subnational has 17 regions for the Philippines.  I think this is 2017 data, although in 2015, the Phililppines created the new Negros Island region.  The site [PhilGIS](http://philgis.org/) has geospatial data for these regions, in multiple formats, and to deeper depths than just the regions.  Unfortunately, as new as I am to playing with python, I'm even newer to touching anything geospatial.  Thankfully, I found this excellent work by [dcdabbler](https://dcdabbler.wordpress.com/2016/08/26/using-geopandas-to-build-updated-philippine-regions-shape-file-in-python/) to help guide me through just what I think I need in this process.  Thanks also to Mithrillion for [this dataset](https://www.kaggle.com/mithrillion/kiva-challenge-coordinates/data) providing many loan coordinates.  We've got region shapes, we've got loan points, now we can draw some maps!

# In[ ]:


#merge in loan coordinates
df_kiv_loc = pd.read_csv("../input/kiva-challenge-coordinates/kiva_locations.csv", sep='\t', error_bad_lines=False)
df_all_kiva = df_all_kiva.merge(df_kiv_loc, how='left', left_on=['country', 'region_kiva'], right_on=['country', 'region'])
df_all_kiva.drop('region', axis=1)

#let's make a philippines only dataframe.  we gonna turn this panda into a geopanda
df_phil = df_all_kiva[df_all_kiva['country'] == 'Philippines']
df_phil['geometry'] = df_phil.apply(lambda row: Point(row['lng'], row['lat']), axis=1)
df_phil = gpd.GeoDataFrame(df_phil, geometry='geometry')
df_phil.crs = {"init":'3123'}
#df_phil.crs = {"init": "epsg:4326"}

# read in shapes, assign MPI subnational names where exist, including 2 missing from existing data
gdf_regions = gpd.GeoDataFrame.from_file('../input/philippines-geospatial-administrative-regions/ph-regions-2015.shp')
gdf_regions['region_mpi'] = gdf_regions['REGION']
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('ARMM'), 'Armm', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Ilocos'), 'Ilocos Region', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Cagaya'), 'Cagayan Valley', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Central Luz'), 'Central Luzon', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('CALAB'), 'Calabarzon', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Bicol'), 'Bicol Region', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Western Vi'), 'Western Visayas', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Central Vi'), 'Central Visayas', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Eastern Vi'), 'Eastern Visayas', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Region IX'), 'Zamboanga Peninsula', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Northern Min'), 'Northern Mindanao', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Davao'), 'Davao Peninsula', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('SOCC'), 'Soccsksargen', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Caraga'), 'CARAGA', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('Cordill'), 'Cordillera Admin Region', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('NCR'), 'National Capital Region', gdf_regions['region_mpi'])
gdf_regions['region_mpi'] = np.where(gdf_regions['REGION'].str.contains('MIMAROPA'), 'Mimaropa', gdf_regions['region_mpi'])

fig, ax = plt.subplots(1, figsize=(12,12))
gdf_regions.plot(ax=ax, color='blue')
df_phil[df_phil['lat'] != -999].plot(ax=ax, markersize=5, color='red')
ax.set_title('Kiva Loans in the Philippines')
plt.show()


# Even better, we can draw much more detailed maps.

# In[ ]:


# create a series with the region numbers/abbreviations to indicate map locations
def make_map_text(s):
    #text = name.split(' ')
    #return text[1] if name.startswith('Region') else text[0]
    return s[s.find("(")+1:s.find(")")]

gdf_regions['region_sort'] = gdf_regions['REGION'].apply(lambda x: make_map_text(x))
#mess with this to hack the colors next to eachother a bit
gdf_regions['region_sort'] = np.where(gdf_regions['region_sort'] == 'NIR', 'Region Va', gdf_regions['region_sort'])
gdf_regions['region_sort'] = np.where(gdf_regions['region_sort'] == 'Region IX', 'Region X', gdf_regions['region_sort'])

gdf_regions = gdf_regions.sort_values('region_sort')

regions = gdf_regions['region_mpi']

palette = itertools.cycle(sns.color_palette('pastel', len(regions)))

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['region_mpi'] == r].plot(ax=ax, color=next(palette), label=r)

df_phil[df_phil['lat'] != -999].plot(ax=ax, markersize=5, color='red')

#for i, point in gdf_regions.centroid.iteritems():
    #reg_n = gdf_regions.iloc[i]['region_mpi']
#    reg_n = gdf_regions.loc[i, 'region_mpi']
#    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    
ax.text(s='Ilocos Region', x=119, y=16.65, fontsize='large')   
ax.text(s='CAR', x=120.85, y=17.5, fontsize='large') 
ax.text(s='Cagayan Valley', x=122.6, y=17, fontsize='large') 
ax.text(s='Central Luzon', x=118.5, y=15, fontsize='large') 
ax.text(s='NCR', x=120.8, y=14.5, fontsize='large') 
ax.text(s='Calabarzon', x=121.4, y=14.4, fontsize='large') 
ax.text(s='Mimaropa', x=119.7, y=12.7, fontsize='large') 
ax.text(s='Bicol Region', x=124.5, y=13.75, fontsize='large') 
ax.text(s='Western Visayas', x=120.4, y=11, fontsize='large') 
ax.text(s='Central Visayas', x=123.5, y=10.4, fontsize='large') 
ax.text(s='Eastern Visayas', x=125.6, y=11.5, fontsize='large') 
ax.text(s='Negros Island Region', x=120.3, y=9.8, fontsize='large') 
ax.text(s='Northern Mindanao', x=123.3, y=8.75, fontsize='large') 
ax.text(s='Davao Peninsula', x=125.6, y=6, fontsize='large') 
ax.text(s='Soccsksargen', x=123.8, y=5.3, fontsize='large') 
ax.text(s='CARAGA', x=126.3, y=9, fontsize='large') 
ax.text(s='Armm', x=123.2, y=6.8, fontsize='large') 
ax.text(s='Zamboanga Peninsula', x=119.8, y=7.5, fontsize='large') 

ax.set_title('Kiva Loans in Philippines Administrative Regions')
#ax.set_axis_off()

plt.show()


# Before 2015, Negros Island was half Western Visayas and half Central Visayas.  It is now it's own administration region.  I'm not sure how old the MPI data is.  One could argue I should simply use the pre-2015 regions, although, I'm not going to do that; I'm just going to assign Negros one.  Western and Central Visayas are both 0.055.  So that's what Negros Island is getting.  Then we'll plot a regional map by MPI.  I applied the same adjustment to all regions to show some color disparity, although there's probably a smarter way to do it than I did (mutliply MPI by 6 for color code) - nonetheless it works!

# In[ ]:


gdf_regions = gdf_regions.merge(df_mpi_subntl[df_mpi_subntl['Country'] == 'Philippines'][['Sub-national region', 'MPI Regional']], how='left', left_on='region_mpi', right_on='Sub-national region')
gdf_regions['MPI Regional'] = np.where(gdf_regions['REGION'].str.contains('NIR'), 0.055, gdf_regions['MPI Regional'])
gdf_regions


# In[ ]:


Blues = plt.get_cmap('Blues')
df_phil.crs = {"init":'3123'}
gdf_regions.crs = {"init":'3123'}
fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['region_mpi'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['region_mpi'] == r]['MPI Regional']*6), label=r)

df_phil[df_phil['lat'] != -999].plot(ax=ax, markersize=5, color='red')

#for i, point in gdf_regions.centroid.iteritems():
    #reg_n = gdf_regions.iloc[i]['region_mpi']
#    reg_n = gdf_regions.loc[i, 'region_mpi']
#    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    
ax.text(s='Ilocos Region', x=119, y=16.65, fontsize='large')   
ax.text(s='CAR', x=120.85, y=17.5, fontsize='large') 
ax.text(s='Cagayan Valley', x=122.6, y=17, fontsize='large') 
ax.text(s='Central Luzon', x=118.5, y=15, fontsize='large') 
ax.text(s='NCR', x=120.8, y=14.5, fontsize='large') 
ax.text(s='Calabarzon', x=121.4, y=14.4, fontsize='large') 
ax.text(s='Mimaropa', x=119.7, y=12.7, fontsize='large') 
ax.text(s='Bicol Region', x=124.5, y=13.75, fontsize='large') 
ax.text(s='Western Visayas', x=120.4, y=11, fontsize='large') 
ax.text(s='Central Visayas', x=123.5, y=10.4, fontsize='large') 
ax.text(s='Eastern Visayas', x=125.6, y=11.5, fontsize='large') 
ax.text(s='Negros Island Region', x=120.3, y=9.8, fontsize='large') 
ax.text(s='Northern Mindanao', x=123.3, y=8.75, fontsize='large') 
ax.text(s='Davao Peninsula', x=125.6, y=6, fontsize='large') 
ax.text(s='Soccsksargen', x=123.8, y=5.3, fontsize='large') 
ax.text(s='CARAGA', x=126.3, y=9, fontsize='large') 
ax.text(s='Armm', x=123.2, y=6.8, fontsize='large') 
ax.text(s='Zamboanga Peninsula', x=119.8, y=7.5, fontsize='large') 

ax.set_title('Kiva Loans in Philippines Administrative Regions by MPI\nDarker = Higher MPI.  Range from 0.026 (NCR) to 0.140 (ARMM)')
#ax.set_axis_off()

plt.show()


# It should be noted that at 0.140, Autonomous Region of Muslim Mindanao (ARMM) isn't the worst MPI out there, however it's easily the worst in the Philippines.  [It should also be noted](http://conflictalert.info/news-group/press-releases/press-release-explosion-violence-muslim-mindanao-2016/) this is a somewhat violent area as a result of drug activity/policy, as well as having some rebels and extremist groups.  Violence also spills over into Northern Mindanao.  I am not sure how compatible Kiva microfinance is with Islamic banking or how much it is leveraged in the region; note that the [concept of interest](https://en.wikipedia.org/wiki/Riba) is a bit different in Islamic finance.
# 
# There are certainly multiple regions here which appear underserved.
# <a id=13_2></a>
# ## 13.2 Reassigning MPI Regions Based on Loan (Point) in Region (Polygon)
# 
# <a id=13_2_1></a>
# ### 13.2.1 Doing the Point in Polygon Work
# 
# The easier way to do this is with sjoin, although I couldn't get it to work as per https://www.kaggle.com/product-feedback/53008 on Kaggle.  The below is probably a lot more compute intensive and brute force as I scan all points within each polygon using a loop.  It takes over 2 hours and the loop may be commented out in favor of a saved result set read.  UPDATE: I have now simply saved the results and piggybacked them into an intermediate dataset for use to myself as a helper input.

# In[ ]:


#sjoin won't work - see https://www.kaggle.com/product-feedback/53008
#result = gpd.tools.sjoin(df_phil[df_phil['lat'] != -999], gdf_regions, how='left')

##### the below code produced this data i'm going to read in - i'm just going to save 2.5 hours and reference
##### this available intermediate/helper dataset i snuck into another dataset i made
#df_phil['region_mpi_new'] = np.NaN
#for i in range(0, 18):
#    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#    df_phil['r_map'] = df_phil.within(gdf_regions['geometry'][i])
#    df_phil['region_mpi_new'] = np.where(df_phil['r_map'], gdf_regions['region_mpi'][i], df_phil['region_mpi_new'])
#df_phil[['id', 'region_mpi_new']].to_csv('phil_pip_output.csv', index = False)
#### end point in polygon area

#### start pip algo helper output
df_phil_pip = pd.read_csv("../input/philippines-geospatial-administrative-regions/phil_pip_output.csv")
df_phil = df_phil.merge(df_phil_pip, on='id', how='left')
#### end pip algo helper


df_phil[['region_mpi', 'region_mpi_new']].head()


# <a id=13_2_2></a>
# ### 13.2.2 Apparent Coastal Exception Problem via Method
# Unfortunately, this method itself has some problems, in particular those points which appear to be coastal on the outside of the polygons.  Example below for Cordova, Cebu and Talibon, Bohol - which did not get properly assigned Central Visayas.

# In[ ]:


df_phil[(df_phil['lat'] != -999) & ((df_phil['id'] == 654958) | (df_phil['id'] == 1326525))][['id', 'region', 'region_mpi', 'geometry', 'region_mpi_new']]


# In[ ]:


fig, ax = plt.subplots(1, figsize=(12,12))

gdf_regions[gdf_regions['region_mpi'] == 'Central Visayas'].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['region_mpi'] == r]['MPI Regional']*6), label=r)

df_phil[(df_phil['lat'] != -999) & ((df_phil['id'] == 654958) | (df_phil['id'] == 1326525))].plot(ax=ax, markersize=5, color='red')

plt.show()


# It's not doing that bad against the existing method I suppose.  The original dataset is missing 12,008 mpi_region values for the same set of data, the geospatial output produced 8,863 missing.  For whatever reason my attempt to set the column as np.NaN turned it into a string "nan" in some cases, but we'll just roll with that for now and I'll code around actual NaN and value nan...

# In[ ]:


df_phil['region_mpi'].fillna('nan').value_counts()


# In[ ]:


df_phil['region_mpi_new'].fillna('nan').value_counts()


# <a id=13_2_3></a>
# ### 13.2.3 Use Existing Kiva Data To Fill Holes
# For anything with a valid Point that was not found within any Polygon, let's leverage the output of the existing Kiva method.  Note this will potentially place some Negros Island region loans into Western/Central Visayas - although they have the same Regional MPI, so we should be good on our data analysis anyway for the country, field partners, and themes.  This gets us down to 3,131 missing.

# In[ ]:


df_phil['region_mpi_new'] = np.where(((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())) & (df_phil['lat'] != -999), df_phil['region_mpi'], df_phil['region_mpi_new'])
df_phil['region_mpi_new'].fillna('nan').value_counts()


# <a id=13_2_4></a>
# ### 13.2.4 Have the Intern Manually Adjust Remaining Holes
# Far less now but still some remain.  Good old fashioned manual labor can help place these.  Instead of just coding I think it's coding and getting your intern to do it?  Best off would be controlling the input application so that only things with valid programmatically retrievable lat/lon could be input into the interface.  The below manual work will get us down to only 735 missing.

# In[ ]:


df_phil['region_kiva'] = df_phil['region_kiva'].str.lower()
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('cagayan')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('nueva vizcay')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('neuva vizcay')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('nueva vizaya')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('aurora')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Luzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('misamis occidental')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('mis occ')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('mis. occ')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('mis.occ')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('sinacaban')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('siquijor')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('siquior')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('cavite')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Calabarzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('isabela')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('isablea')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('kalinga')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cordillera Admin Region', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('oro. city')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('oro.city')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('bohol')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('ifugao')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cordillera Admin Region', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('oroquieta city')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('lopez jaena')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('tangub city')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('buguey')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('lala ldn')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('ilo-ilo')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Western Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('iloilo')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Western Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('larena')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('ozamiz city')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('negros occidental')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Western Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('tarlac')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Luzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('tangub')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('nueva ecija')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Luzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('cebu')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('tuguegarao')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('bukidnon')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('tudela')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('leyte')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Eastern Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('tacloban')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Eastern Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('calamba')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Calabarzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('sapang dalaga')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('bulacan')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Luzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('negros occicental')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Western Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('negros oriental')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('laguna')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Calabarzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('quirino')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('misamis oriental')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('lanao del norte')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('misami occidental')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('plaridel')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Central Luzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('santiago city')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('pangasinan')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Ilocos Region', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('pontevedra')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Western Visayas', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('oroq')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('ozamiz')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('ozamis')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('cauayan')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Cagayan Valley', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains(', ldn')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Northern Mindanao', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('san mateo')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Calabarzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'] = np.where((df_phil['region_kiva'].str.contains('san pablo')) & ((df_phil['region_mpi_new'] == 'nan') | (df_phil['region_mpi_new'].isnull())), 'Calabarzon', df_phil['region_mpi_new'])
df_phil['region_mpi_new'].fillna('nan').value_counts()


# <a id=13_2_5></a>
# ### 13.2.5 Reassignment Results
# We now have a nice solid group of regions assigned.  What has changed?

# In[ ]:


df_new = df_phil['region_mpi_new'].fillna('nan').value_counts().to_frame()
df_old = df_phil['region_mpi'].fillna('nan').value_counts().to_frame()
df_new.reset_index(level=0, inplace=True)
df_old.reset_index(level=0, inplace=True)
df_compare = df_new.merge(df_old, on='index', how='left')
df_compare['change'] = df_compare['region_mpi_new'] - df_compare['region_mpi'].fillna(0)
df_compare.sort_values('change', ascending=False)[['index', 'region_mpi', 'region_mpi_new', 'change']]


# Above we can see many changes.  NIR region is new and took +44,560 loans out of Central and Western Visayas, who had a change of -48,551.  Some larger changes here are we can see NCR went from 0 to 3,687 loans - this is the Metro Manila capital area of the Philippines which is 100% urban.  Zamboanga Peninsula is another region with previously *no representation* while we now see it has 7,285 loans.  These regions only exist here as a result of the geospatial placement and manual massaging.  Zamboanga is only touching Northern Mindanao - are all 7,285 of its loans from there?  No - we can see Northern Mindanao only lost 2,362 loans.  It looks like we have more accurate regional assignment now, and will thus be able to calculate more accurate MPI for our kiva borrowers.

# <a id=13_2_6></a>
# ### 13.2.6 Field Partner MPI - Existing (National, Sub-National) vs. Amended
# We can now leverage Annalie's code to calculate Field Partner MPI.  None of these field partners lend outside of the Philippines.  **Existing National MPI** method:

# In[ ]:


LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv") #.set_index([''])
MPI = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','MPI Urban','MPI Rural']].set_index("ISO")
LT = LT.join(MPI,how='left',on="ISO")[['Partner ID','Field Partner Name','ISO','MPI Rural','MPI Urban','rural_pct','amount']].dropna()
#~ Convert rural percentage to 0-1
LT['rural_pct'] /= 100
#~ Compute the MPI Score for each loan theme
LT['MPI Score'] = LT['rural_pct']*LT['MPI Rural'] + (1-LT['rural_pct'])*LT['MPI Urban']

#~ Need a volume-weighted average for mutli-country partners. 
weighted_avg = lambda df: np.average(df['MPI Score'],weights=df['amount'])
#~ Get total volume & average MPI Score for each partner country 
FP = LT.groupby(['Partner ID','ISO']).agg({'MPI Score': np.mean,'amount':np.sum})
#~ and get weighted average over countries. Done!
Scores = FP.groupby(level='Partner ID').apply(weighted_avg)
FP.reset_index(level=1, inplace=True)
FP[FP['ISO'] == 'PHL']


# **Existing Sub-National** method:

# In[ ]:


MPIsubnat = pd.read_csv("../input/mpi/MPI_subnational.csv")[['Country', 'Sub-national region', 'World region', 'MPI National', 'MPI Regional']]
# Create new column LocationName that concatenates the columns Country and Sub-national region
MPIsubnat['LocationName'] = MPIsubnat[['Sub-national region', 'Country']].apply(lambda x: ', '.join(x), axis=1)

LTsubnat = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")[['Partner ID', 'Loan Theme ID', 'region', 'mpi_region', 'ISO', 'number', 'amount', 'LocationName', 'names']]

# Merge dataframes
LTsubnat = LTsubnat.merge(MPIsubnat, left_on='mpi_region', right_on='LocationName', suffixes=('_LTsubnat', '_mpi'))[['Partner ID', 'Loan Theme ID', 'Country', 'ISO', 'mpi_region', 'MPI Regional', 'number', 'amount']]

#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LTsubnat.groupby(['Partner ID', 'Loan Theme ID', 'Country', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_LTsubnat = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_regional_scores = LS.groupby(level=['Partner ID', 'ISO']).apply(weighted_avg_LTsubnat)
MPI_regional_scores = MPI_regional_scores.to_frame()
MPI_regional_scores.reset_index(level=1, inplace=True)
MPI_regional_scores = MPI_regional_scores.rename(index=str, columns={0: 'MPI Score'})

df_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == 'PHL']
df_subnat.reset_index(level=0, inplace=True)
df_subnat['method'] = 'Existing Sub-National'


# In[ ]:


df_subnat


# How about using a similar dollar weighted regional methodology for field partners and where their actual loans stand?  Note I am no longer using the loan count/amount from them regions input, but the actual funded_amount on each individual loan in our dataset.  Aside from the region assignment change here, this is also taking that change into account - and there appear to be a lot more dollars involved when using this level.  I do not know if all loans were disbursed so I am including all loans.  Just because Kiva does not fund the loan, that does not mean it does not get disbursed by the lender, who may use their own capital to do so.

# In[ ]:


df_phil = df_phil.merge(df_phil[['region_mpi_new', 'MPI Regional']].drop_duplicates().dropna(axis=0, how='any'), on='region_mpi_new', how='left')
df_phil = df_phil.rename(index=str, columns={'MPI Regional_y': 'MPI Regional'})
df_phil.drop('MPI Regional_x', axis=1, inplace=True)
df_phil = df_phil[(df_phil['region_mpi_new'] != 'nan') & (~df_phil['region_mpi_new'].isnull())]
df_mpi_fld = df_phil[['partner_id_loan_theme', 'funded_amount', 'MPI Regional', 'region_mpi_new']].dropna(axis=0, how='any').groupby(
    ['partner_id_loan_theme', 'MPI Regional', 'region_mpi_new'])[['funded_amount']].sum()
df_mpi_fld.reset_index(level=2, inplace=True)
df_mpi_fld.reset_index(level=1, inplace=True)
df_mpi_fld.reset_index(level=0, inplace=True)
df_mpi_fld_tot = df_mpi_fld.groupby('partner_id_loan_theme')[['funded_amount']].sum()
df_mpi_fld_tot.reset_index(level=0, inplace=True)
df_mpi_fld = df_mpi_fld.merge(df_mpi_fld_tot, on='partner_id_loan_theme', how='left')
df_mpi_fld = df_mpi_fld.rename(index=str, columns={'funded_amount_x': 'funded_amount', 'funded_amount_y': 'total_amount'})
df_mpi_fld['mpi_part'] = df_mpi_fld['funded_amount'] / df_mpi_fld['total_amount'] * df_mpi_fld['MPI Regional']
results = df_mpi_fld.groupby('partner_id_loan_theme')[['mpi_part', 'funded_amount']].sum()
results.reset_index(level=0, inplace=True)
results = results.merge(df_kv_theme_rgn[['Partner ID', 'Field Partner Name']].drop_duplicates(), 
              how='left', left_on='partner_id_loan_theme', right_on='Partner ID')[['partner_id_loan_theme', 'mpi_part', 'funded_amount', 'Field Partner Name']]
results


# <a id=13_2_7></a>
# ### 13.2.7 Field Partner MPI Method Comparison
# Graphing the differences, we see:

# In[ ]:


df_curr_fld_mpi = FP[FP['ISO'] == 'PHL']
df_curr_fld_mpi.reset_index(level=0, inplace=True)
df_curr_fld_mpi['method'] = 'Existing National'
df_res = results[['partner_id_loan_theme', 'mpi_part']]
df_res['method'] = 'Amended Sub-National'
df_res = df_res.rename(index=str, columns={'partner_id_loan_theme': 'Partner ID', 'mpi_part': 'MPI Score'})
frames = (df_curr_fld_mpi[['Partner ID', 'MPI Score', 'method']], df_subnat[['Partner ID', 'MPI Score', 'method']], df_res)

df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

ax.legend(ncol=1, loc='best', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)
ax.set_title('Existing vs. Amended Field Partner MPI - Philippines', fontsize=15)
plt.show()


# As noted in [Section 12.1.4](#manual) the National methodology (blue) seems super dirty (vs. green or red) with trusting the rural percentage, and we also lose field partners.  The number can also be drastically off from attempts to measure more accurately via regional values.  Shifts between Existing and Amended Sub-National calculations (green vs. red) are based on how well the existing algrorithm assigned locations.
# 
# Let's map some of these partners for a sanity check.

# In[ ]:


Blues = plt.get_cmap('Blues')
df_phil.crs = {"init":'3123'}
gdf_regions.crs = {"init":'3123'}
fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['region_mpi'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['region_mpi'] == r]['MPI Regional']*6), label=r)

df_phil[(df_phil['lat'] != -999) & (df_phil['partner_id_loan_theme'] == 123)].plot(ax=ax, markersize=5, color='red', label='123')
df_phil[(df_phil['lat'] != -999) & (df_phil['partner_id_loan_theme'] == 126)].plot(ax=ax, markersize=5, color='orange', label='126')
df_phil[(df_phil['lat'] != -999) & (df_phil['partner_id_loan_theme'] == 136)].plot(ax=ax, markersize=5, color='green', label='136')
df_phil[(df_phil['lat'] != -999) & (df_phil['partner_id_loan_theme'] == 409)].plot(ax=ax, markersize=5, color='yellow', label='409')

#for i, point in gdf_regions.centroid.iteritems():
    #reg_n = gdf_regions.iloc[i]['region_mpi']
#    reg_n = gdf_regions.loc[i, 'region_mpi']
#    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    
ax.text(s='Ilocos Region', x=119, y=16.65, fontsize='large')   
ax.text(s='CAR', x=120.85, y=17.5, fontsize='large') 
ax.text(s='Cagayan Valley', x=122.6, y=17, fontsize='large') 
ax.text(s='Central Luzon', x=118.5, y=15, fontsize='large') 
ax.text(s='NCR', x=120.8, y=14.5, fontsize='large') 
ax.text(s='Calabarzon', x=121.4, y=14.4, fontsize='large') 
ax.text(s='Mimaropa', x=119.7, y=12.7, fontsize='large') 
ax.text(s='Bicol Region', x=124.5, y=13.75, fontsize='large') 
ax.text(s='Western Visayas', x=120.4, y=11, fontsize='large') 
ax.text(s='Central Visayas', x=123.5, y=10.4, fontsize='large') 
ax.text(s='Eastern Visayas', x=125.6, y=11.5, fontsize='large') 
ax.text(s='Negros Island Region', x=120.3, y=9.8, fontsize='large') 
ax.text(s='Northern Mindanao', x=123.3, y=8.75, fontsize='large') 
ax.text(s='Davao Peninsula', x=125.6, y=6, fontsize='large') 
ax.text(s='Soccsksargen', x=123.8, y=5.3, fontsize='large') 
ax.text(s='CARAGA', x=126.3, y=9, fontsize='large') 
ax.text(s='Armm', x=123.2, y=6.8, fontsize='large') 
ax.text(s='Zamboanga Peninsula', x=119.8, y=7.5, fontsize='large') 

ax.set_title('Kiva Loans in Philippines Administrative Regions by MPI\nDarker = Higher MPI.  Range from 0.026 (NCR) to 0.140 (ARMM)')
#ax.set_axis_off()
ax.legend(loc='best', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)
plt.show()


# This all looks like it checks out.  123 does lend in a lower MPI area.  Similar partners 126 and 136 both lend in a higher MPI area, serving Zamboanga and Northern Mindanao.  Partner 409 is a new higher MPI entrant, and we can see that is because they have few loans, although they are in Northern Mindanao.
# 
# Does Negros Women for Tomorrow Foundation (NWTF) live/lend up to it's name?  Let's see where they are lending.

# In[ ]:


Blues = plt.get_cmap('Blues')
df_phil.crs = {"init":'3123'}
gdf_regions.crs = {"init":'3123'}
fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    gdf_regions[gdf_regions['region_mpi'] == r].plot(ax=ax, color=Blues(gdf_regions[gdf_regions['region_mpi'] == r]['MPI Regional']*6), label=r)

df_phil[(df_phil['lat'] != -999) & (df_phil['partner_id_loan_theme'] == 145)].plot(ax=ax, markersize=5, color='red', label='145')

#for i, point in gdf_regions.centroid.iteritems():
    #reg_n = gdf_regions.iloc[i]['region_mpi']
#    reg_n = gdf_regions.loc[i, 'region_mpi']
#    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    
ax.text(s='Ilocos Region', x=119, y=16.65, fontsize='large')   
ax.text(s='CAR', x=120.85, y=17.5, fontsize='large') 
ax.text(s='Cagayan Valley', x=122.6, y=17, fontsize='large') 
ax.text(s='Central Luzon', x=118.5, y=15, fontsize='large') 
ax.text(s='NCR', x=120.8, y=14.5, fontsize='large') 
ax.text(s='Calabarzon', x=121.4, y=14.4, fontsize='large') 
ax.text(s='Mimaropa', x=119.7, y=12.7, fontsize='large') 
ax.text(s='Bicol Region', x=124.5, y=13.75, fontsize='large') 
ax.text(s='Western Visayas', x=120.4, y=11, fontsize='large') 
ax.text(s='Central Visayas', x=123.5, y=10.4, fontsize='large') 
ax.text(s='Eastern Visayas', x=125.6, y=11.5, fontsize='large') 
ax.text(s='Negros Island Region', x=120.3, y=9.8, fontsize='large') 
ax.text(s='Northern Mindanao', x=123.3, y=8.75, fontsize='large') 
ax.text(s='Davao Peninsula', x=125.6, y=6, fontsize='large') 
ax.text(s='Soccsksargen', x=123.8, y=5.3, fontsize='large') 
ax.text(s='CARAGA', x=126.3, y=9, fontsize='large') 
ax.text(s='Armm', x=123.2, y=6.8, fontsize='large') 
ax.text(s='Zamboanga Peninsula', x=119.8, y=7.5, fontsize='large') 

ax.set_title('Kiva Loans in Philippines Administrative Regions by MPI\nDarker = Higher MPI.  Range from 0.026 (NCR) to 0.140 (ARMM)')
#ax.set_axis_off()
ax.legend(loc='best', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)
plt.show()


# Interesting!  I thought these may all be loans limited to the Negros Island Region.  It's the one between Western and Central Visayas, shaped like, well, you know.  They have in fact lent all over the Visayas regions, and even far out into the Mimaropa area.

# <a id=14></a>
# # 14. Mozambique Local Analysis (National MPI 0.389)
# 
# <a id=14_1></a>
# ## 14.1 Field Partner MPI - Existing (National, Sub-National) vs. Amended
# 
# Well let's do this with Mozambique now, since the original Kiva methodology and location looked all over the board and the regions were definitely part of the problem.  This geospatial source is actually at the district level, although with some googling I was able to bring them up to a regional level.

# In[ ]:


#read data
gdf_regions = gpd.GeoDataFrame.from_file('../input/mozambique-geospatial-regions/moz_polbnda_adm2_districts_wfp_ine_pop2012_15_ocha.shp')
gdf_regions['PROVINCE'] = np.where(gdf_regions['PROVINCE'].str.contains('Zamb'), 'Zambézia', gdf_regions['PROVINCE'])

#make some regions
moz_regions = {}

provinces = gdf_regions['PROVINCE'].drop_duplicates()
for p in provinces:
    polys = gdf_regions[gdf_regions['PROVINCE'] == p]['geometry']
    u = cascaded_union(polys)
    moz_regions[p] = u
    
#make a geodataframe for the regions    
s = pd.Series(moz_regions, name='geometry')
s.index.name = 'region_mpi_new'
s.reset_index()
df_moz_regions = gpd.GeoDataFrame(s, geometry='geometry')
df_moz_regions.crs = {"init":'42106'}
df_moz_regions.reset_index(level=0, inplace=True)

#assign regional MPI to regions
df_moz_regions = df_moz_regions.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == 'MOZ'][['Sub-national region', 'MPI Regional']], how='left', 
                                      left_on='region_mpi_new', right_on='Sub-national region')
#manual updates due to character or spelling differences
df_moz_regions['MPI Regional'] = np.where(df_moz_regions['region_mpi_new'] == 'Zambézia', 0.528, df_moz_regions['MPI Regional'])
df_moz_regions['MPI Regional'] = np.where(df_moz_regions['region_mpi_new'] == 'Maputo', 0.133, df_moz_regions['MPI Regional'])
df_moz_regions['MPI Regional'] = np.where(df_moz_regions['region_mpi_new'] == 'Maputo City', 0.043, df_moz_regions['MPI Regional'])

#make a geodataframe for the loans
df_moz = df_all_kiva[df_all_kiva['ISO'] == 'MOZ']
df_moz['geometry'] = df_moz.apply(lambda row: Point(row['lng'], row['lat']), axis=1)
df_moz = gpd.GeoDataFrame(df_moz, geometry='geometry')
df_moz.crs = {"init":'42106'}

#assign loans to regions - this is much faster than the philippines, far less loans to go through.
df_moz['region_mpi_new'] = np.NaN
for i in range(0, 11):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    df_moz['r_map'] = df_moz.within(df_moz_regions['geometry'][i])
    df_moz['region_mpi_new'] = np.where(df_moz['r_map'], df_moz_regions['region_mpi_new'][i], df_moz['region_mpi_new'])
df_moz[['id', 'region_mpi_new']].to_csv('moz_pip_output.csv', index = False)

df_moz_regions


# This resulted in a large shift, which was expected from what we observed in [Section 12.2.2](#prob4).  Notably the current methodology which placed many loans within Maputo City actually were for loans within the Manputo *Province*.  We also had 315 nulls drop to 284.  Unlike the Philippines, none of the Kiva provided info can be used to fill the polygon created assignment; in this case, the polygon simply worked much better than the existing method.

# In[ ]:


df_moz['region_mpi'].fillna('nan').value_counts()


# In[ ]:


df_moz['region_mpi_new'].fillna('nan').value_counts()


# Let's take a look at all the partners for Mozambique across an MPI map.  I multiplied Regional MPI by only 1.3 on this one to try and show the map differences.  Note this means areas are more impoverished than the Philippines in general here, which is why the multiplier is lower (and blues between PHL and MOZ maps should not be compared.

# In[ ]:


Blues = plt.get_cmap('Blues')
df_phil.crs = {"init":'3123'}

regions = df_moz_regions['region_mpi_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    df_moz_regions[df_moz_regions['region_mpi_new'] == r].plot(ax=ax, color=Blues(df_moz_regions[df_moz_regions['region_mpi_new'] == r]['MPI Regional']*1.3))

df_moz[(df_moz['lat'] != -999) & (df_moz['partner_id_loan_theme'] == 23)].plot(ax=ax, markersize=10, color='red', label='23')
df_moz[(df_moz['lat'] != -999) & (df_moz['partner_id_loan_theme'] == 261)].plot(ax=ax, markersize=10, color='orange', label='261')
df_moz[(df_moz['lat'] != -999) & (df_moz['partner_id_loan_theme'] == 366)].plot(ax=ax, markersize=10, color='green', label='366')
df_moz[(df_moz['lat'] != -999) & (df_moz['partner_id_loan_theme'] == 468)].plot(ax=ax, markersize=10, color='yellow', label='468')
df_moz[(df_moz['lat'] != -999) & (df_moz['partner_id_loan_theme'] == 210)].plot(ax=ax, markersize=10, color='brown', label='210')
df_moz[(df_moz['lat'] != -999) & (df_moz['partner_id_loan_theme'] == 492)].plot(ax=ax, markersize=10, color='purple', label='492')


for i, point in df_moz_regions.centroid.iteritems():
    reg_n = df_moz_regions.iloc[i]['region_mpi_new']
    reg_n = df_moz_regions.loc[i, 'region_mpi_new']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    

ax.set_title('Loans across Mozambique by Field Partner\nDarker = Higher MPI.  Range from 0.043 (Maputo City) to 0.528 (Zambézia)')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()


# In[ ]:


df_moz = df_moz.merge(df_moz_regions[['region_mpi_new', 'MPI Regional']].drop_duplicates().dropna(axis=0, how='any'), on='region_mpi_new', how='left')
df_moz = df_moz.rename(index=str, columns={'MPI Regional_y': 'MPI Regional'})
df_moz.drop('MPI Regional_x', axis=1, inplace=True)
df_moz = df_moz[(df_moz['region_mpi_new'] != 'nan') & (~df_moz['region_mpi_new'].isnull())]
df_mpi_fld = df_moz[['partner_id_loan_theme', 'funded_amount', 'MPI Regional', 'region_mpi_new']].dropna(axis=0, how='any').groupby(
    ['partner_id_loan_theme', 'MPI Regional', 'region_mpi_new'])[['funded_amount']].sum()
df_mpi_fld.reset_index(level=2, inplace=True)
df_mpi_fld.reset_index(level=1, inplace=True)
df_mpi_fld.reset_index(level=0, inplace=True)
df_mpi_fld_tot = df_mpi_fld.groupby('partner_id_loan_theme')[['funded_amount']].sum()
df_mpi_fld_tot.reset_index(level=0, inplace=True)
df_mpi_fld = df_mpi_fld.merge(df_mpi_fld_tot, on='partner_id_loan_theme', how='left')
df_mpi_fld = df_mpi_fld.rename(index=str, columns={'funded_amount_x': 'funded_amount', 'funded_amount_y': 'total_amount'})
df_mpi_fld['mpi_part'] = df_mpi_fld['funded_amount'] / df_mpi_fld['total_amount'] * df_mpi_fld['MPI Regional']
results = df_mpi_fld.groupby('partner_id_loan_theme')[['mpi_part', 'funded_amount']].sum()
results.reset_index(level=0, inplace=True)
results = results.merge(df_kv_theme_rgn[['Partner ID', 'Field Partner Name']].drop_duplicates(), 
              how='left', left_on='partner_id_loan_theme', right_on='Partner ID')[['partner_id_loan_theme', 'mpi_part', 'funded_amount', 'Field Partner Name']]

#get national
df_curr_fld_mpi = FP[FP['ISO'] == 'MOZ']
df_curr_fld_mpi.reset_index(level=0, inplace=True)
df_curr_fld_mpi['method'] = 'Existing National'

#get subnational
df_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == 'MOZ']
df_subnat.reset_index(level=0, inplace=True)
df_subnat['method'] = 'Existing Sub-National'


#get amended
df_res = results[['partner_id_loan_theme', 'mpi_part']]
df_res['method'] = 'Amended Sub-National'
df_res = df_res.rename(index=str, columns={'partner_id_loan_theme': 'Partner ID', 'mpi_part': 'MPI Score'})
frames = (df_curr_fld_mpi[['Partner ID', 'MPI Score', 'method']], df_subnat[['Partner ID', 'MPI Score', 'method']], df_res)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

ax.legend(ncol=1, loc='best', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)
ax.set_title('Existing vs. Amended Field Partner MPI - Mozambique', fontsize=15)
plt.show()


# Using National MPI we are again dropping the most partners, and yielding extremely dirty results (blue vs. green/red).  It is a very untrustworthy metric, and it should be kept in mind that it is reused as input for the Sub-National scoring in the existing method (green) when a country does not contain Sub-National data.  The green vs. red gaps here are as a result of being able to accurately assign loans to regions.  The problem with 23 for example is that the existing data is placing all the loans in the city - while there are a significant amount outside of the city and well into the province.
# 
# Using the existing value, we would mistakenly conclude that partners 210, 23, 366, and 492 are addressing impoverished areas equally.  However we see 23 and 366 are in fact reaching out much further into impoverished areas.
# 
# Ok now that we've got these comparisons down, we could move onto something else, although it seems we might see some more interesting things and assess the differences better by looking at some more countries.  So let's see if I can find some other countries of interest and their accompanying geospatial data.
# <a id=15></a>
# # 15. Rwanda Local Analysis (National MPI 0.259)
# Rwanda is 15th in loans per capita per Section 12.3.  It has a higher MPI than most as well.
# I'm going to omit partners 319 and 493 as they serve multiple countries (Tanzania and Uganda respectively) - which requires a weighting and correct goespatial data to be available and used for those countries as well.  For simplicity's sake, doing this one country at a time analysis, I am simply going to omit those lenders.  In the real world you'd do all countries in a large batch beforehand, and then calculate the MPI for all of them after you trusted the amended region placement data.

# In[ ]:


ISO = 'RWA'
#read data
gdf_regions = gpd.GeoDataFrame.from_file('../input/rwanda-2006-geospatial-administrative-regions/Province_Boundary_2006.shp')
gdf_regions['Prov_Name'] = np.where(gdf_regions['Prov_Name'] == 'Southern Province', 'South', gdf_regions['Prov_Name'])
gdf_regions['Prov_Name'] = np.where(gdf_regions['Prov_Name'] == 'Western Province', 'West', gdf_regions['Prov_Name'])
gdf_regions['Prov_Name'] = np.where(gdf_regions['Prov_Name'] == 'Eastern Province', 'East', gdf_regions['Prov_Name'])
gdf_regions['Prov_Name'] = np.where(gdf_regions['Prov_Name'] == 'Northern Province', 'North', gdf_regions['Prov_Name'])
gdf_regions = gdf_regions.rename(index=str, columns={'Prov_Name': 'region_mpi_new'})

df_cnt_regions = gpd.GeoDataFrame(gdf_regions, geometry='geometry')
df_cnt_regions.crs = {"init":'1199'}
df_cnt_regions.reset_index(level=0, inplace=True)

#assign regional MPI to regions
df_cnt_regions = df_cnt_regions.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == ISO][['Sub-national region', 'MPI Regional']], how='left', 
                                      left_on='region_mpi_new', right_on='Sub-national region')

#make a geodataframe for the loans
df_cnt = df_all_kiva[df_all_kiva['ISO'] == ISO]
df_cnt['geometry'] = df_cnt.apply(lambda row: Point(row['lng'], row['lat']), axis=1)
df_cnt = gpd.GeoDataFrame(df_cnt, geometry='geometry')
df_cnt.crs = {"init":'1199'}

#assign loans to regions - this is much faster than the philippines, far less loans to go through.
df_cnt['region_mpi_new'] = np.NaN
for i in range(0, df_cnt_regions['region_mpi_new'].size):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    df_cnt['r_map'] = df_cnt.within(df_cnt_regions['geometry'][i])
    df_cnt['region_mpi_new'] = np.where(df_cnt['r_map'], df_cnt_regions['region_mpi_new'][i], df_cnt['region_mpi_new'])
df_cnt[['id', 'region_mpi_new']].to_csv(ISO + '_pip_output.csv', index = False)


# Before and after; the original data had a ton of nulls, plus too many placed in the lowest MPI region of Kigali City.

# In[ ]:


df_cnt['region_mpi'].fillna('nan').value_counts()


# In[ ]:


df_cnt['region_mpi_new'].fillna('nan').value_counts()


# In[ ]:


Blues = plt.get_cmap('Blues')
df_phil.crs = {"init":'1199'}

regions = df_cnt_regions['region_mpi_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    df_cnt_regions[df_cnt_regions['region_mpi_new'] == r].plot(ax=ax, color=Blues(df_cnt_regions[df_cnt_regions['region_mpi_new'] == r]['MPI Regional']*2))

df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 117)].plot(ax=ax, markersize=10, color='red', label='117')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 161)].plot(ax=ax, markersize=10, color='orange', label='161')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 212)].plot(ax=ax, markersize=10, color='green', label='212')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 271)].plot(ax=ax, markersize=10, color='yellow', label='271')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 528)].plot(ax=ax, markersize=10, color='brown', label='528')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 441)].plot(ax=ax, markersize=10, color='purple', label='441')

for i, point in df_cnt_regions.centroid.iteritems():
    reg_n = df_cnt_regions.iloc[i]['region_mpi_new']
    reg_n = df_cnt_regions.loc[i, 'region_mpi_new']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    

ax.set_title('Loans across Rwanda by Field Partner\nDarker = Higher MPI.  Range from 0.118 (Kigali City) to 0.295 (South)')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()


# In[ ]:


df_cnt = df_cnt.merge(df_cnt_regions[['region_mpi_new', 'MPI Regional']].drop_duplicates().dropna(axis=0, how='any'), on='region_mpi_new', how='left')
df_cnt = df_cnt.rename(index=str, columns={'MPI Regional_y': 'MPI Regional'})
df_cnt.drop('MPI Regional_x', axis=1, inplace=True)
df_cnt = df_cnt[(df_cnt['region_mpi_new'] != 'nan') & (~df_cnt['region_mpi_new'].isnull())]
df_mpi_fld = df_cnt[['partner_id_loan_theme', 'funded_amount', 'MPI Regional', 'region_mpi_new']].dropna(axis=0, how='any').groupby(
    ['partner_id_loan_theme', 'MPI Regional', 'region_mpi_new'])[['funded_amount']].sum()
df_mpi_fld.reset_index(level=2, inplace=True)
df_mpi_fld.reset_index(level=1, inplace=True)
df_mpi_fld.reset_index(level=0, inplace=True)
df_mpi_fld_tot = df_mpi_fld.groupby('partner_id_loan_theme')[['funded_amount']].sum()
df_mpi_fld_tot.reset_index(level=0, inplace=True)
df_mpi_fld = df_mpi_fld.merge(df_mpi_fld_tot, on='partner_id_loan_theme', how='left')
df_mpi_fld = df_mpi_fld.rename(index=str, columns={'funded_amount_x': 'funded_amount', 'funded_amount_y': 'total_amount'})
df_mpi_fld['mpi_part'] = df_mpi_fld['funded_amount'] / df_mpi_fld['total_amount'] * df_mpi_fld['MPI Regional']
results = df_mpi_fld.groupby('partner_id_loan_theme')[['mpi_part', 'funded_amount']].sum()
results.reset_index(level=0, inplace=True)
results = results.merge(df_kv_theme_rgn[['Partner ID', 'Field Partner Name']].drop_duplicates(), 
              how='left', left_on='partner_id_loan_theme', right_on='Partner ID')[['partner_id_loan_theme', 'mpi_part', 'funded_amount', 'Field Partner Name']]

#get national
df_curr_fld_mpi = FP[FP['ISO'] == ISO]
df_curr_fld_mpi.reset_index(level=0, inplace=True)
df_curr_fld_mpi['method'] = 'Existing National'

#get subnational
df_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == ISO]
df_subnat.reset_index(level=0, inplace=True)
df_subnat['method'] = 'Existing Sub-National'


#get amended
df_res = results[['partner_id_loan_theme', 'mpi_part']]
df_res['method'] = 'Amended Sub-National'
df_res = df_res.rename(index=str, columns={'partner_id_loan_theme': 'Partner ID', 'mpi_part': 'MPI Score'})
frames = (df_curr_fld_mpi[['Partner ID', 'MPI Score', 'method']], df_subnat[['Partner ID', 'MPI Score', 'method']], df_res)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
sns.set_palette('muted')

#hide multi-country partners
df_compare = df_compare[(df_compare['Partner ID'] != '319') & (df_compare['Partner ID'] != '493')]

sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

#hide multi-country partners


ax.legend(ncol=1, loc='best', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)
ax.set_title('Existing vs. Amended Field Partner MPI - Rwanda', fontsize=15)
plt.show()


# We can see the extreme noise from the National methodology again.  With correctly placed regions, we see partners 117 and 161 are lending to much more impoverished areas than previously thought.  271 quite a bit as well.  441, on the other hand, dropped a ton.
# <a id=16></a>
# # 16. Sierra Leone Analysis (National MPI 0.464)

# In[ ]:


ISO = 'SLE'
#read data
gdf_regions = gpd.GeoDataFrame.from_file('../input/sierra-leone-geospatial-administrative-regions/sle_admbnda_adm2_1m_gov_ocha.shp')
gdf_regions['admin2Name'] = np.where(gdf_regions['admin2Name'] == 'Western Area Rural', 'Western Rural', gdf_regions['admin2Name'])
gdf_regions['admin2Name'] = np.where(gdf_regions['admin2Name'] == 'Western Area Urban', 'Western Urban', gdf_regions['admin2Name'])
gdf_regions = gdf_regions.rename(index=str, columns={'admin2Name': 'region_mpi_new'})

df_cnt_regions = gpd.GeoDataFrame(gdf_regions, geometry='geometry')
df_cnt_regions.crs = {"init":'2161'}
df_cnt_regions.reset_index(level=0, inplace=True)

#assign regional MPI to regions
df_cnt_regions = df_cnt_regions.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == ISO][['Sub-national region', 'MPI Regional']], how='left', 
                                      left_on='region_mpi_new', right_on='Sub-national region')

#make a geodataframe for the loans
df_cnt = df_all_kiva[df_all_kiva['ISO'] == ISO]
df_cnt['geometry'] = df_cnt.apply(lambda row: Point(row['lng'], row['lat']), axis=1)
df_cnt = gpd.GeoDataFrame(df_cnt, geometry='geometry')
df_cnt.crs = {"init":'2161'}

#assign loans to regions - this is much faster than the philippines, far less loans to go through.
df_cnt['region_mpi_new'] = np.NaN
for i in range(0, df_cnt_regions['region_mpi_new'].size):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    df_cnt['r_map'] = df_cnt.within(df_cnt_regions['geometry'][i])
    df_cnt['region_mpi_new'] = np.where(df_cnt['r_map'], df_cnt_regions['region_mpi_new'][i], df_cnt['region_mpi_new'])
df_cnt[['id', 'region_mpi_new']].to_csv(ISO + '_pip_output.csv', index = False)


# Before and after location results:

# In[ ]:


df_cnt['region_mpi'].fillna('nan').value_counts()


# In[ ]:


df_cnt['region_mpi_new'].fillna('nan').value_counts()


# In[ ]:


Blues = plt.get_cmap('Blues')
df_phil.crs = {"init":'2161'}

regions = df_cnt_regions['region_mpi_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    df_cnt_regions[df_cnt_regions['region_mpi_new'] == r].plot(ax=ax, color=Blues(df_cnt_regions[df_cnt_regions['region_mpi_new'] == r]['MPI Regional']*1))

df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 57)].plot(ax=ax, markersize=10, color='red', label='57')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 183)].plot(ax=ax, markersize=10, color='orange', label='183')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 148)].plot(ax=ax, markersize=10, color='green', label='148')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 504)].plot(ax=ax, markersize=10, color='yellow', label='504')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 531)].plot(ax=ax, markersize=10, color='purple', label='531')

for i, point in df_cnt_regions.centroid.iteritems():
    reg_n = df_cnt_regions.iloc[i]['region_mpi_new']
    reg_n = df_cnt_regions.loc[i, 'region_mpi_new']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    

ax.set_title('Loans across Sierra Leone by Field Partner\nDarker = Higher MPI.  Range from 0.191 (Western Urban) to 0.601 (Koinadugu)')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()


# In[ ]:


df_cnt = df_cnt.merge(df_cnt_regions[['region_mpi_new', 'MPI Regional']].drop_duplicates().dropna(axis=0, how='any'), on='region_mpi_new', how='left')
df_cnt = df_cnt.rename(index=str, columns={'MPI Regional_y': 'MPI Regional'})
df_cnt.drop('MPI Regional_x', axis=1, inplace=True)
df_cnt = df_cnt[(df_cnt['region_mpi_new'] != 'nan') & (~df_cnt['region_mpi_new'].isnull())]
df_mpi_fld = df_cnt[['partner_id_loan_theme', 'funded_amount', 'MPI Regional', 'region_mpi_new']].dropna(axis=0, how='any').groupby(
    ['partner_id_loan_theme', 'MPI Regional', 'region_mpi_new'])[['funded_amount']].sum()
df_mpi_fld.reset_index(level=2, inplace=True)
df_mpi_fld.reset_index(level=1, inplace=True)
df_mpi_fld.reset_index(level=0, inplace=True)
df_mpi_fld_tot = df_mpi_fld.groupby('partner_id_loan_theme')[['funded_amount']].sum()
df_mpi_fld_tot.reset_index(level=0, inplace=True)
df_mpi_fld = df_mpi_fld.merge(df_mpi_fld_tot, on='partner_id_loan_theme', how='left')
df_mpi_fld = df_mpi_fld.rename(index=str, columns={'funded_amount_x': 'funded_amount', 'funded_amount_y': 'total_amount'})
df_mpi_fld['mpi_part'] = df_mpi_fld['funded_amount'] / df_mpi_fld['total_amount'] * df_mpi_fld['MPI Regional']
results = df_mpi_fld.groupby('partner_id_loan_theme')[['mpi_part', 'funded_amount']].sum()
results.reset_index(level=0, inplace=True)
results = results.merge(df_kv_theme_rgn[['Partner ID', 'Field Partner Name']].drop_duplicates(), 
              how='left', left_on='partner_id_loan_theme', right_on='Partner ID')[['partner_id_loan_theme', 'mpi_part', 'funded_amount', 'Field Partner Name']]

#get national
df_curr_fld_mpi = FP[FP['ISO'] == ISO]
df_curr_fld_mpi.reset_index(level=0, inplace=True)
df_curr_fld_mpi['method'] = 'Existing National'

#get subnational
df_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == ISO]
df_subnat.reset_index(level=0, inplace=True)
df_subnat['method'] = 'Existing Sub-National'


#get amended
df_res = results[['partner_id_loan_theme', 'mpi_part']]
df_res['method'] = 'Amended Sub-National'
df_res = df_res.rename(index=str, columns={'partner_id_loan_theme': 'Partner ID', 'mpi_part': 'MPI Score'})
frames = (df_curr_fld_mpi[['Partner ID', 'MPI Score', 'method']], df_subnat[['Partner ID', 'MPI Score', 'method']], df_res)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')



ax.legend(ncol=1, loc='best', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)
ax.set_title('Existing vs. Amended Field Partner MPI - Sierra Leone', fontsize=15)
plt.show()


# Noisy National data, and in all cases, it seems the existing data and methodology is measuring a bit to a good chunk more high on the more accurate Sub-National level.
# 
# <a id=17></a>
# # 17. El Salvador Local Analysis (National MPI 0.026)
# El Salvador was our country with the most loans per capita [(from Section 11)](#phil), by far!

# In[ ]:


ISO = 'SLV'
#read data
gdf_regions = gpd.GeoDataFrame.from_file('../input/el-salvador-administrative-geospatial-data/SLV_adm1.shp')
gdf_regions = gdf_regions.rename(index=str, columns={'NAME_1': 'region_mpi_new'})
gdf_regions['region_mpi_new'] = np.where(gdf_regions['region_mpi_new'] == 'Ahuachapán', 'Ahuachapan', gdf_regions['region_mpi_new'])
gdf_regions['region_mpi_new'] = np.where(gdf_regions['region_mpi_new'] == 'Cabañas', 'Cabanas', gdf_regions['region_mpi_new'])
gdf_regions['region_mpi_new'] = np.where(gdf_regions['region_mpi_new'] == 'Cuscatlán', 'Cuscatlan', gdf_regions['region_mpi_new'])
gdf_regions['region_mpi_new'] = np.where(gdf_regions['region_mpi_new'] == 'La Unión', 'La Union', gdf_regions['region_mpi_new'])
gdf_regions['region_mpi_new'] = np.where(gdf_regions['region_mpi_new'] == 'Morazán', 'Morazan', gdf_regions['region_mpi_new'])
gdf_regions['region_mpi_new'] = np.where(gdf_regions['region_mpi_new'] == 'Usulután', 'Usulutan', gdf_regions['region_mpi_new'])

df_cnt_regions = gpd.GeoDataFrame(gdf_regions, geometry='geometry')
df_cnt_regions.crs = {"init":'5460'}
df_cnt_regions.reset_index(level=0, inplace=True)

#assign regional MPI to regions
df_cnt_regions = df_cnt_regions.merge(df_mpi_subntl[df_mpi_subntl['ISO country code'] == ISO][['Sub-national region', 'MPI Regional']], how='left', 
                                      left_on='region_mpi_new', right_on='Sub-national region')

#make a geodataframe for the loans
df_cnt = df_all_kiva[df_all_kiva['ISO'] == ISO]
df_cnt['geometry'] = df_cnt.apply(lambda row: Point(row['lng'], row['lat']), axis=1)
df_cnt = gpd.GeoDataFrame(df_cnt, geometry='geometry')
df_cnt.crs = {"init":'5460'}

#assign loans to regions - this is much faster than the philippines, far less loans to go through.
df_cnt['region_mpi_new'] = np.NaN
for i in range(0, df_cnt_regions['region_mpi_new'].size):
    print('i is: ' + str(i) + ' at ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    df_cnt['r_map'] = df_cnt.within(df_cnt_regions['geometry'][i])
    df_cnt['region_mpi_new'] = np.where(df_cnt['r_map'], df_cnt_regions['region_mpi_new'][i], df_cnt['region_mpi_new'])
df_cnt[['id', 'region_mpi_new']].to_csv(ISO + '_pip_output.csv', index = False)


# Before and after location results, 24,123 nulls reduced down to 1!  Partner 225 lends across 9 countries so I'm going to omit them from the rest of this, as I have not done a region correction across all 9 to weight them properly.

# In[ ]:


df_cnt['region_mpi'].fillna('nan').value_counts()


# In[ ]:


df_cnt['region_mpi_new'].fillna('nan').value_counts()


# In[ ]:


Blues = plt.get_cmap('Blues')
df_phil.crs = {"init":'5460'}

regions = df_cnt_regions['region_mpi_new']

fig, ax = plt.subplots(1, figsize=(12,12))

for r in regions:
    df_cnt_regions[df_cnt_regions['region_mpi_new'] == r].plot(ax=ax, color=Blues(df_cnt_regions[df_cnt_regions['region_mpi_new'] == r]['MPI Regional']*5))

df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 81)].plot(ax=ax, markersize=10, color='red', label='81')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 167)].plot(ax=ax, markersize=10, color='orange', label='167')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 199)].plot(ax=ax, markersize=10, color='green', label='199')
df_cnt[(df_cnt['lat'] != -999) & (df_cnt['partner_id_loan_theme'] == 333)].plot(ax=ax, markersize=10, color='yellow', label='333')

for i, point in df_cnt_regions.centroid.iteritems():
    reg_n = df_cnt_regions.iloc[i]['region_mpi_new']
    reg_n = df_cnt_regions.loc[i, 'region_mpi_new']
    ax.text(s=reg_n, x=point.x, y=point.y, fontsize='large')
    

ax.set_title('Loans across El Salvador by Field Partner\nDarker = Higher MPI.  Range from 0.01 (San Salvador) to 0.054 (La Union)')
ax.legend(loc='upper left', frameon=True)
leg = ax.get_legend()
new_title = 'Partner ID'
leg.set_title(new_title)

plt.show()


# In[ ]:


df_cnt = df_cnt.merge(df_cnt_regions[['region_mpi_new', 'MPI Regional']].drop_duplicates().dropna(axis=0, how='any'), on='region_mpi_new', how='left')
df_cnt = df_cnt.rename(index=str, columns={'MPI Regional_y': 'MPI Regional'})
df_cnt.drop('MPI Regional_x', axis=1, inplace=True)
df_cnt = df_cnt[(df_cnt['region_mpi_new'] != 'nan') & (~df_cnt['region_mpi_new'].isnull())]
df_mpi_fld = df_cnt[['partner_id_loan_theme', 'funded_amount', 'MPI Regional', 'region_mpi_new']].dropna(axis=0, how='any').groupby(
    ['partner_id_loan_theme', 'MPI Regional', 'region_mpi_new'])[['funded_amount']].sum()
df_mpi_fld.reset_index(level=2, inplace=True)
df_mpi_fld.reset_index(level=1, inplace=True)
df_mpi_fld.reset_index(level=0, inplace=True)
df_mpi_fld_tot = df_mpi_fld.groupby('partner_id_loan_theme')[['funded_amount']].sum()
df_mpi_fld_tot.reset_index(level=0, inplace=True)
df_mpi_fld = df_mpi_fld.merge(df_mpi_fld_tot, on='partner_id_loan_theme', how='left')
df_mpi_fld = df_mpi_fld.rename(index=str, columns={'funded_amount_x': 'funded_amount', 'funded_amount_y': 'total_amount'})
df_mpi_fld['mpi_part'] = df_mpi_fld['funded_amount'] / df_mpi_fld['total_amount'] * df_mpi_fld['MPI Regional']
results = df_mpi_fld.groupby('partner_id_loan_theme')[['mpi_part', 'funded_amount']].sum()
results.reset_index(level=0, inplace=True)
results = results.merge(df_kv_theme_rgn[['Partner ID', 'Field Partner Name']].drop_duplicates(), 
              how='left', left_on='partner_id_loan_theme', right_on='Partner ID')[['partner_id_loan_theme', 'mpi_part', 'funded_amount', 'Field Partner Name']]

#get national
df_curr_fld_mpi = FP[FP['ISO'] == ISO]
df_curr_fld_mpi.reset_index(level=0, inplace=True)
df_curr_fld_mpi['method'] = 'Existing National'

#get subnational
df_subnat = MPI_regional_scores[MPI_regional_scores['ISO'] == ISO]
df_subnat.reset_index(level=0, inplace=True)
df_subnat['method'] = 'Existing Sub-National'


#get amended
df_res = results[['partner_id_loan_theme', 'mpi_part']]
df_res['method'] = 'Amended Sub-National'
df_res = df_res.rename(index=str, columns={'partner_id_loan_theme': 'Partner ID', 'mpi_part': 'MPI Score'})
frames = (df_curr_fld_mpi[['Partner ID', 'MPI Score', 'method']], df_subnat[['Partner ID', 'MPI Score', 'method']], df_res)
df_compare = pd.concat(frames)
df_compare['Partner ID'] = df_compare['Partner ID'].astype(str).str.split('.', 1).str[0]


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))

#hide multi-country partners
df_compare = df_compare[(df_compare['Partner ID'] != '225')]

sns.set_palette('muted')
sns.barplot(x='Partner ID', y='MPI Score', data=df_compare, hue='method')

ax.legend(ncol=1, loc='upper center', frameon=True)
ax.set(ylabel='MPI Score',
       xlabel='Partner ID')

leg = ax.get_legend()
new_title = 'Method'
leg.set_title(new_title)
ax.set_title('Existing vs. Amended Field Partner MPI - El Salvador', fontsize=15)
plt.show()


# Here we see 333 show up, and find 167 and 81 are lending to the more impoverished areas of El Salvador.

# # Draft work down below

# In[ ]:


df_cato = pd.read_csv('../input/cato-2017-human-freedom-index/cato_2017_human_freedom_index.csv')

df_cato['Countries'] = np.where(df_cato['Countries'] == 'Yemen, Rep.', 'Yemen', df_cato['Countries'])
df_cato['Countries'] = np.where(df_cato['Countries'] == 'TimorLeste', 'Timor-Leste', df_cato['Countries'])
df_cato['Countries'] = np.where(df_cato['Countries'] == 'Congo, Democratic Republic of', 'The Democratic Republic of the Congo', df_cato['Countries'])
df_cato['Countries'] = np.where(df_cato['Countries'] == 'Kyrgyz Republic', 'Kyrgyzstan', df_cato['Countries'])
df_cato['Countries'] = np.where(df_cato['Countries'] == 'Congo, Republic of', 'Congo', df_cato['Countries'])
df_cato['Countries'] = np.where(df_cato['Countries'] == 'Laos', "Lao People's Democratic Republic", df_cato['Countries'])
df_cato['Countries'] = np.where(df_cato['Countries'] == 'Myanmar', 'Myanmar (Burma)', df_cato['Countries'])
df_cato['Countries'] = np.where(df_cato['Countries'] == "Cote d'Ivoire", "Cote D'Ivoire", df_cato['Countries'])

df_test = df_all_kiva.merge(df_cato, how='left', left_on='country', right_on='Countries')[['country', 'Countries']].drop_duplicates()
df_test = df_all_kiva.merge(df_cato, how='left', left_on='country', right_on='Countries')
df_test.head()


# In[ ]:


def read_findex(datafile=None, interpolate=False, invcov=True, variables = ["Account", "Loan", "Emergency"], norm=True):
    """
    Returns constructed findex values for each country

    Read in Findex data - Variables include: Country ISO Code, Country Name,
                          Pct with Account at Financial institution (Poor),
                          Pct with a loan from a Financial institution (Poor),
                          Pct who say they could get an emergency loan (Poor)

    Take average of 'poorest 40%' values for each value in `variables'

     If `normalize':
        Apply the normalization function to every MPI variable
    """
    if datafile == None: datafile = "../input/findex-world-bank/FINDEXData.csv"

    F = pd.read_csv(datafile)#~ [["ISO","Country Name", "Indicator Name", "MRV"]]
    
    Fcols = {'Country Name': 'Country',
        'Country Code': 'ISO',
        'Indicator Name': 'indicator',
        'Indicator Code': 'DROP',
        '2011': 'DROP',
        '2014': 'DROP',
        'MRV': 'Val'
        }
    F = F.rename(columns=Fcols).drop("DROP",1)
    F['Val'] /= 100.
    
    indicators = {"Account at a financial institution, income, poorest 40% (% ages 15+) [ts]": "Account",
        "Coming up with emergency funds: somewhat possible, income, poorest 40% (% ages 15+) [w2]": "Emergency",
        "Coming up with emergency funds: very possible, income, poorest 40% (% ages 15+) [w2]": "Emergency",
        "Borrowed from a financial institution, income, poorest 40% (% ages 15+) [ts]": "Loan"
        }

    F['Poor'] = F['indicator'].apply(lambda ind: "Poor" if "poorest" in ind else "Rich") 
    F['indicator'] = F['indicator'].apply(lambda ind: indicators.setdefault(ind,np.nan)) 
    F = F.dropna(subset=["indicator"])
    F = F.groupby(["Poor","ISO","indicator"])["Val"].sum()
    F = 1 - F.loc["Poor"]

    F = F.unstack("indicator")
    
    # fill missing values for the emergency indicator with a predicted score from OLS regression analysis 
    if interpolate:
        results = smf.ols("Emergency ~ Loan + Account",data=F).fit()
        F['Emergency_fit'] = results.params['Intercept'] + F[['Loan','Account']].mul(results.params[['Loan','Account']]).sum(1)
        F['Emergency'].fillna(F['Emergency_fit'],inplace=True)
    if invcov: F['Findex'] = invcov_index(F[variables]) #.mean(1)
    else: F['Findex'] = F[variables].mean(1,skipna=True)
        
    flatvar = flatten(F['Findex'].dropna(), use_buckets = False, return_buckets = False)
    F = F.join(flatvar,how='left',lsuffix=' (raw)')
    
    return F

def invcov_index(indicators):
    """
    Convert a dataframe of indicators into an inverse covariance matrix index
    """
    df = indicators.copy()
    df = (df-df.mean())/df.std()
    I  = np.ones(df.shape[1])
    E  = inv(df.cov())
    s1  = I.dot(E).dot(I.T)
    s2  = I.dot(E).dot(df.T)
    try:
        int(s1)
        S  = s2/s1
    except TypeError: 
        S  = inv(s1).dot(s2)
    
    S = pd.Series(S,index=indicators.index)

    return S

def flatten(Series, outof = 10., bins = 20, use_buckets = False, write_buckets = False, return_buckets = False):
    """
    NOTE: Deal with missing values, obviously!
    Convert Series to a uniform distribution from 0 to `outof'
    use_buckets uses the bucketing rule from a previous draw.
    """

    tempSeries = Series.dropna()
    if use_buckets: #~ Use a previously specified bucketing rule
        cuts, pcts = list(rule['Buckets']), np.array(rule['Values']*(100./outof))
    else: #~ Make Bucketing rule to enforce a uniform distribution
        pcts = np.append(np.arange(0,100,100/bins),[100])
        cuts = [ np.percentile(tempSeries,p) for p in pcts ]
        while len(cuts)>len(set(cuts)):
            bins -= 1
            pcts = np.append(np.arange(0,100,100/bins),[100])
            cuts = [ np.percentile(tempSeries,p) for p in pcts ]

    S = pd.cut(tempSeries,cuts,labels = pcts[1:]).astype(float)
    S *= outof/100

    buckets = pd.DataFrame({"Buckets":cuts,"Values":pcts*(outof/100)})

    if return_buckets: return S, 
    else: return S
    
F = read_findex()
F.reset_index(level=0, inplace=True)
F.head()


# In[ ]:


df_pair = df_test[['ISO', 'country', 'Rule of Law', 'Legal System & Property Rights', 
                   'Sound Money', 'Credit market regulations', 
        'Business regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'MPI National']].drop_duplicates()

df_pair = df_pair.merge(df_total_per_mil, on='country')
df_pair.drop('2016', axis=1, inplace=True)

df_pair = df_pair.merge(F, how='left', on='ISO')
df_pair



# In[ ]:


g = sns.PairGrid(df_pair[['country', #'Rule of Law', 'Legal System & Property Rights', 
    'Sound Money', 'Credit market regulations', 
    'Business regulations', 'ECONOMIC FREEDOM (Score)', 'Income per Capita', 'MPI National', 'Findex']]
    , diag_sharey=False)
g.map_lower(plt.scatter)
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
plt.show()


# In[ ]:


df_pair.sort_values('Income per Capita', ascending=False).head(20)


# In[ ]:


df_pair[df_pair['ISO'].isnull()][['ISO', 'country']]


# More to come - just wanted to get what I've got so far published out!
# 
# Look it's me in the additional data snapshot!

# In[ ]:


df_lender = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')
df_lender['lender_URL'] = df_lender['permanent_name'].apply(lambda x: 'https://www.kiva.org/lender/' + str(x))
print(df_lender[df_lender['permanent_name'] == 'mikedev10']['lender_URL'])

