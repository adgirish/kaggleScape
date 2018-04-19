
# coding: utf-8

# # Kiva Crowdfunding - Understanding Poverty
# 
# Kiva is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. More information can be found at https://www.kiva.org/.
# 
# This notebook series is my contribution to the Data Science for Good: Kiva Crowdfunding challenge. 
# The objective is to help Kiva to better understand their borrowers and build more localized models to estimate the poverty levels in the regions where Kiva has active loans.
# 
# Kive Crowdfunding notebook series:
#   - [Part I - Understanding Poverty]
#   - [Part II - Targeting Poverty at a National Level]
#   - [Part III - Targeting Poverty at a Subnational Level]
# 
# The series in broken down into three notebooks. The first notebook is an exploratory analysis of the data to get a feeling for what we are working with. The second notebook examines external datasets and looks at how MPI and other indicators can be used to get a better understanding of poverty levels of Kiva borrowers at a national level. The third notebook examines external data at a subnational level to see how Kiva can get a more accurate prediction of poverty level based on location.
# 
# This is the first notebook of the series. There are already many execellent and more in-depth kernels covering EDA, therefore this one is kept relatively brief and focussed on the most interesting features. It was developed mainly to aid the author's own understanding. Some EDA kernels from which inspiration has been drawn are:
#  - [Simple Exploration Notebook - Kiva]
#  - [A Very Extensive Kiva Exploratory Analysis]
#  - [Kiva Data Analysis w/ Naive Poverty Metric]
#  
# [Part I - Understanding Poverty]: https://www.kaggle.com/taniaj/kiva-crowdfunding-understanding-poverty
# [Part II - Targeting Poverty at a National Level]: https://www.kaggle.com/taniaj/kiva-crowdfunding-targeting-poverty-national
# [Part III - Targeting Poverty at a Subnational Level]: http://
# [Simple Exploration Notebook - Kiva]: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-kiva "Simple Exploration Notebook - Kiva"
# [A Very Extensive Kiva Exploratory Analysis]: https://www.kaggle.com/codename007/a-very-extensive-kiva-exploratory-analysis "A Very Extensive Kiva Exploratory Analysis"
# [Kiva Data Analysis w/ Naive Poverty Metric]: https://www.kaggle.com/ambarish/kiva-data-analysis-w-naive-poverty-metric "Kiva Data Analysis w/ Naive Poverty Metric"
# 
# #### Contents:
# * [Exploratory Analysis of Available Kiva Data](#eda)
#     * [Number of Loans and Loan Amounts per Country](#number_loans_amounts_per_country)
#     * [Adjustements to account for Population Size](#adjustment_pop_size)
#     * [Loans per Sector](#loans_per_sector)
#     * [Loans Amount Distribution](#loans_amount_distributon)
#     * [Repayment Intervals](#borrower_genders)
#     * [Borrower Genders](#borrower_genders)
#     * [Multi-dimensional Poverty Index (MPI) ](#mpi_eda)
#         * [MPI by Region](#mpi_by_region)
# * [Combining Kiva Loan Data with MPI](#combining_loan_mpi)
#     * [Data Preprocessing](#preprocessing)
#     * [Exploratory Analysis](#combined_loans_mpi_eda)
#     * [Relationships between Loan Features and MPI](#loan_features_mpi_correlation)
#     * [Feature Engineering](#feature_engineering)
#         * [Mapping Uses to a Keyword](#feature_eng_keyword)
#     * [Multidimensional Poverty - National](#multidimensional_poverty_national)
#     * [Multidimensional Poverty - Sub-National](#multidimensional_poverty_subnational)
# * [Conclusion](#conclusion)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from scipy.stats.mstats import gmean

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(rc={"figure.figsize": (20,10), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 }, 
        palette=sns.color_palette("OrRd_d", 20))

import warnings
warnings.filterwarnings('ignore')

get_ipython().system('cp ../input/images/regional-intensity-of-deprivation.png .')


# In[ ]:


# Original Kiva datasets
kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
kiva_mpi_locations_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_by_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")

# Additional Kiva datasets
mpi_national_df = pd.read_csv("../input/mpi/MPI_national.csv")
# The subnational Kiva data has been enhanced with lat/long data
mpi_subnational_df = pd.read_csv("../input/kiva-mpi-subnational-with-coordinates/mpi_subnational_coords.csv")

# World Bank population data
world_pop_df = pd.read_csv("../input/world-population/WorldPopulation.csv")


# In[ ]:


#kiva_loans_df = kiva_loans_df.sort_values(by=['country'])
#kiva_loans_df.country.unique()


# ## Exploratory Analysis of Available Kiva Data <a class="anchor" id="eda"></a>

# ### Number of Loans and Loan Amounts per Country <a class="anchor" id="number_loans_amounts_per_country"/>
# 
# First let us get a quick idea of where the loan requests are mostly coming from.

# In[ ]:


# Plot loans per country
sns.countplot(y="country", data=kiva_loans_df, 
              order=kiva_loans_df.country.value_counts().iloc[:20].index).set_title("Distribution of Kiva Loans by Country")
plt.ylabel('')


# - The Philippines has by far the most number of Kiva loan requests.

# In[ ]:


# Plot loans per region
sns.countplot(y="region", data=kiva_loans_df, 
              order=kiva_loans_df.region.value_counts().iloc[:20].index).set_title("Distribution of Kiva Loans by Region")
plt.ylabel('')


# In[ ]:


#kiva_loans_df.loc[kiva_loans_df['region'] == 'Kaduna'].head()


# - Kaduna, Nigeria is the region with the highest number of requests, followed by Lahore, India.
# 
# Let us plot the total number loan requests on a map to better visualise where the Kiva loan requests are mostly coming from.

# In[ ]:


countries_number_loans = kiva_loans_df.groupby('country').count()['loan_amount'].sort_values(ascending = False)
data = [dict(
        type='choropleth',
        locations= countries_number_loans.index,
        locationmode='country names',
        z=countries_number_loans.values,
        text=countries_number_loans.index,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='# Loans'),
)]
layout = dict(title = 'Number of Loans Requested by Country', 
        geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=50, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-total-map')


# Let us have a look at whether the distribution is similar for the total loan amount requested.

# In[ ]:


countries_loan_amount = kiva_loans_df.groupby('country').sum()['loan_amount'].sort_values(ascending = False)
data = [dict(
        type='choropleth',
        locations= countries_loan_amount.index,
        locationmode='country names',
        z=countries_loan_amount.values,
        text=countries_loan_amount.index,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Loan Amount'),
)]
layout = dict(title = 'Total Loan Amount Requested by Country', 
        geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=50, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-total-map')


# - While the Phlippines still leads, we also see the US and some areas in South America have more total loan amount requests than other countries. 
# 
# ### Adjustements to account for Population Size <a class="anchor" id="adjustment_pop_size"/>
# This is surprising as there are many African countries that have higher poverty levels than the Philippines.
# To get a clearer picture, Lets adjust this to get a per-capita distribution and plot that.

# In[ ]:


# Get loan_per_country data
kiva_loan_country_df = kiva_loans_df[['id', 'country']].groupby(['country'])['id'].agg({'loan_amount': ['sum','count']}).reset_index()
kiva_loan_country_df.columns = kiva_loan_country_df.columns.droplevel()
kiva_loan_country_df.columns = ['country', 'loan_amount', 'loan_count']

# Join world population data to kiva loan_per_country data
kiva_loan_country_df = kiva_loan_country_df.merge(world_pop_df[['Country', '2016']], left_on=['country'], right_on=['Country'])
kiva_loan_country_df.drop('Country', axis=1, inplace=True)

# Calculate values per million population
kiva_loan_country_df['loans_per_mil'] = kiva_loan_country_df['loan_count'] / (kiva_loan_country_df['2016'] / 1000000)
kiva_loan_country_df['loan_amount_per_mil'] = kiva_loan_country_df['loan_amount'] / (kiva_loan_country_df['2016'] / 1000000)


# In[ ]:


# Plot loans per million per country
with sns.color_palette("OrRd_d", 10), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,4))
    kiva_loan_country_df.sort_values('loans_per_mil', ascending=False, inplace=True)
    sns.barplot(kiva_loan_country_df.head(10).loans_per_mil, kiva_loan_country_df.head(10).country).set_title("Number of Loans (population adjusted) per Country")
    plt.ylabel('')


# In[ ]:


kiva_loan_country_df.head(10)


# In[ ]:


# Plot loan amount per million per country
with sns.color_palette("OrRd_d", 10), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,4))
    kiva_loan_country_df.sort_values('loan_amount_per_mil', ascending=False, inplace=True)
    sns.barplot(kiva_loan_country_df.head(10).loan_amount_per_mil, kiva_loan_country_df.head(10).country).set_title("Loan Amount (population adjusted) per Country")
    plt.ylabel('')


# - After adjustment of loan count and total amount to account for population sizes, Samoa has the largest relative number and amount of loans by far. The population of Samoa is extremely small (195125 people) so the results are somewhat amplified when using this adjustment method.
# - El Salvador, with a population of 6.3 million has the second highest adjusted number of loans and total loan amount. 
# - The Philippines, with 103.3 million people is now the 10th highest in adjusted number of loans and total loan amount. 
# 
# Lets visualise these adjusted loan amounts on the world map.

# In[ ]:


data = [dict(
        type='choropleth',
        locations= kiva_loan_country_df.country,
        locationmode='country names',
        z=kiva_loan_country_df.loan_amount_per_mil,
        text=kiva_loan_country_df.index,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Loan Amount'),
)]
layout = dict(title = 'Total Loan Amount (adjusted) Requested by Country', 
        geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=50, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-total-map')


# - The adjusted loan amounts are much more equally distributed across countries. The outlier, Samoa, is not actually visible on the map (too small?)
# 
# 
# ### Loans per Sector  <a class="anchor" id="loans_per_sector"/>
# Now a quick look at the distribution of loan requests per sector.

# In[ ]:


# Plot loans per sector
sns.countplot(y="sector", data=kiva_loans_df, 
              order=kiva_loans_df.sector.value_counts().iloc[:20].index).set_title("Distribution of Loans by Sector")
plt.ylabel('')


# - Agriculture tops the list of loan requests by sector, followed by Food and Retail.
# 
# ### Loans Amount Distribution  <a class="anchor" id="loans_amount_distributon"/>
# Let us look at the distribution of loan amounts. Note, the initial plot was unreadable due to an outlier so it has been truncated as a first step before plotting.

# In[ ]:


# Truncate outliers
percentile_99 = np.percentile(kiva_loans_df.loan_amount.values, 99)
kiva_loans_df['loan_amount_trunc'] = kiva_loans_df['loan_amount'].copy()
kiva_loans_df.loc[kiva_loans_df['loan_amount_trunc'] > percentile_99, 'loan_amount_trunc'] = percentile_99


# In[ ]:


# Plot loan amount histogram
sns.distplot(kiva_loans_df.loan_amount_trunc.values, kde=False)
plt.title("Loan Amount Distribution")
plt.xlabel('Loan Amount')
plt.ylabel('Number of Loans')


# - The loan amounts are relatively low, most under 1000 USD.
# 
# Let us visualise the repayment term and interval of the Kiva loans.

# In[ ]:


# Plot repayent term histogram
sns.distplot(kiva_loans_df.term_in_months.values, kde=False)
plt.title("Loan Term Distribution")
plt.xlabel('Loan Term')
plt.ylabel('Number of Loans')


# - The loan terms are mostly between 6 and 30 mnths.
# 
# ### Repayment Intervals <a class="anchor" id="repayment_intervals"/>

# In[ ]:


# Plot repayment interval of loans
with sns.color_palette("YlOrBr_d", 4):
    plt.figure(figsize=(6,6))
    plt.title("Repayment Interval")
    kiva_loans_df.repayment_interval.value_counts().T.plot.pie(labeldistance=1.1)
    plt.ylabel('')


# - Most loans are repaid monthly or irregularly. 
# 
# ### Borrower Genders <a class="anchor" id="borrower_genders"/>

# In[ ]:


def parse_genders(borrower_genders):
    gender_list = borrower_genders.split(",")
    gender_list = list(set(gender_list))
    gender_list = [borrower_genders.strip() for borrower_genders in gender_list]
    if len(gender_list)==2:
        if 'female' in gender_list and 'male' in gender_list:
            return "both"
        elif 'female' in gender_list:
            return "multiple female"
        elif 'male' in gender_list:
            return "multiple male"
    elif gender_list[0]=="female":
        return "single female"
    elif gender_list[0]=="male":
        return "single male"
    else:
        return "unknown"
    
# Plot loans by borrower gender
with sns.color_palette("YlOrBr_d", 8):
    plt.figure(figsize=(6,6))
    plt.title("Borrower Gender")
    kiva_loans_df.borrower_genders[kiva_loans_df.borrower_genders.isnull()]= 'unknown'
    kiva_loans_df['gender'] = kiva_loans_df.borrower_genders.apply(parse_genders)
    kiva_loans_df.gender.value_counts().plot.pie(labeldistance=1.1, explode = (0, 0.025, 0.05, 0.1, 0.3, 0.7))
    plt.ylabel('')


# - There are a lot more female borrowers
# - For loans with multiple borrowers, there are a lot more female-only borrower groups than mixed or male-only groups.

# ### Multi-dimensional Poverty Index (MPI)  <a class="anchor" id="mpi_eda"/>
# 
# So what exactly is this Multidimensional Poverty Index? 
# 
# The MPI omplements monetary measures of poverty by considering overlapping deprivations suffered by individuals at the same time. The index identifies deprivations across the same three dimensions as the HDI and shows the number of people who are multidimensionally poor (suffering deprivations in 33% or more of the weighted indicators) and the number of weighted deprivations with which poor households typically contend with. 
# 
# Note: Because the MPI aims to identify deprivations across the same three dimensions as the HDI (as I understand, using a different method to the HDI, which better captures the differences between the poorest countries), I will not do further analysis of the HDI or try to include it in any metric developed here.
# 
# References:
# 
# <http://hdr.undp.org/en/content/multidimensional-poverty-index-mpi>
# 
# <http://hdr.undp.org/sites/default/files/hdr2016_technical_notes.pdf>
# 
# These three dimensions are: **Health, Education and Standard of Living**
# 
# Exerpt from the UNDP technical notes:
# <div class="alert alert-block alert-info">
# <p/>
# <b>Education:</b><br/>
#     • School attainment: no household member has completed at least six years of schooling.<br/>
#     • School attendance: a school-age child (up to grade 8) is not attending school.<br/>
# <p/>
# <b>Health:</b><br/>
#     • Nutrition: a household member (for whom there is nutrition information) is malnourished, as measured by the body mass index for adults (women ages 15–49 in most of the surveys) and by the height-for-age z-score calculated based on World Health Organization standards for children under age 5.<br/>
#     • Child mortality: a child has died in the household within the five years prior to the survey.<br/>
# <p/>
# <b>Standard of living:</b><br/>
#     • Electricity: not having access to electricity.<br/>
#     • Drinking water: not having access to clean drinking water or having access to clean drinking water through a source that is located 30 minutes away or more by walking.<br/>
#     • Sanitation: not having access to improved sanitation facilities or having access only to shared improved sanitation facilities.<br/>
#     • Cooking fuel: using “dirty” cooking fuel (dung, wood or charcoal).<br/>
#     • Having a home with dirt, sand or dung floor.<br/>
#     • Assets: not having at least one asset related to access to information (radio, television or telephone) or having at least one asset related to information but not having at least one asset related to mobility (bike, motorbike, car, truck, animal cart or motorboat) or at least one asset related to livelihood (refrigerator, arable land or livestock).<br/>
# </div>
# 
# Firstly, a quick plot to visualise the MPI data on the world map.

# In[ ]:


# Plot Kiva MPI Locations
data = [ dict(
        type = 'scattergeo',
        lon = kiva_mpi_locations_df['lon'],
        lat = kiva_mpi_locations_df['lat'],
        text = kiva_mpi_locations_df['LocationName'],
        drawmapboundary = dict(fill_color = '#A6CAE0', linewidth = 0.1),
        mode = 'markers',
        marker = dict(
            size = 6,
            opacity = 0.9,
            symbol = 'circle',
            line = dict(width = 1, color = 'rgba(80, 80, 80)'),
            colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
            reversescale=True,
            cmin = 0,
            color = kiva_mpi_locations_df['MPI'],
            cmax = kiva_mpi_locations_df['MPI'].max(),
            colorbar=dict(title="MPI")
        ))]
layout = dict(
            title = 'Kiva MPI Locations',
            geo = dict(
            showframe = False, 
            showcoastlines = True,
            showcountries=True,
            showland = True,
            landcolor = 'rgb(245, 241, 213)',
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# It was pointed out to me that some regions are plotted in the wrong place on the map! We can easily confirm this is the case by, for example, looking at the point in southern Australia, which when hovering over it indicates it is Tanzania!
# 
# I have done a quick investigation of the errors including validating the locations against other external datasets (but discarded it from this notebook as it is off topic) and come to the conclusion that there are relatively few errors in the latitude and longitude values. As the values are currently only used in the above visualisation to get a picture of what the MPI distribution looks like, I have decided to ignore this for now. 
# Later on if Latitude and Longitude are more heavily used in this notebook I may have to do more research to get a reliable source of latitude and longitude values and do some data cleaning.

# ### MPI by World Region  <a class="anchor" id="mpi_by_region"/>

# In[ ]:


print(kiva_mpi_locations_df.shape)
kiva_mpi_locations_df.sample(5)


# Looking at the MPI dataset we notice that there are a lot of missing values to deal with. The main features we are interested in are LocationName, country, region and MPI so we will drop all the entries that don't have these values.
# (Note: LocationName is available wherever country and region are available so we will keep and use this column too.)

# In[ ]:


print("Original MPI dataset: ", kiva_mpi_locations_df.shape)
region_mpi_df = kiva_mpi_locations_df[['world_region', 'LocationName', 'country','region', 'MPI', 'lat', 'lon']]
region_mpi_df = region_mpi_df.dropna()
print("Cleaned MPI dataset: ", region_mpi_df.shape)


# Unfortunately a lot of data was discarded at this step, however we will proceed with it and may have to source region-MPI mappings from other external datasets later on.

# In[ ]:


# Plot MPI by World Region
with sns.color_palette("OrRd_d", 6), sns.plotting_context("notebook", font_scale=1.5):
    plt.subplot(211).set_title("MPI count by World Region")
    world_region_mpi_count_df = region_mpi_df.groupby(['world_region'])['MPI'].count().reset_index(name='count_mpi')
    sns.barplot(world_region_mpi_count_df.count_mpi, world_region_mpi_count_df.world_region)
    plt.ylabel('')

    plt.subplot(212).set_title("MPI average by World Region")
    world_region_mpi_mean_df = region_mpi_df.groupby(['world_region'])['MPI'].mean().reset_index(name='mean_mpi')
    sns.barplot(world_region_mpi_mean_df.mean_mpi, world_region_mpi_mean_df.world_region)
    plt.ylabel('')

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])


# - This is mostly as we would expect it. Sub-Saharan Africa has the highest number of MPI reports as well as the higest average MPI of all world regions.
# It is interesting to note that Latin America and the Caribbean have the second highest number of MPI reports but the second lowest average MPI. The opposite is true for South Asia, with a low numer of MPI reports but high average MPI.

# ## Combining Kiva Loan Data with MPI  <a class="anchor" id="combining_loan_mpi"/>
# 
# In this section we will join the Kiva Loan data with the MPI dataset to try and get an understanding of the relationship between loan requests and MPI.

# ### Data Preprocessing <a class="anchor" id="preprocessing"/>

# We will do the same removal of entries with no country or region for the Kiva Loans dataset. We will also drop a few less relevant features and then join the MPI information to the Loan information.

# In[ ]:


print("Original Kiva Loans dataset: ", kiva_loans_df.shape)

# Merging Kiva loans to MPI using loan_themes
kiva_loans_mpi_df = pd.merge(kiva_loans_df, loan_theme_ids_df, how='left', on='id')
kiva_loans_mpi_df = kiva_loans_mpi_df.merge(loan_themes_by_region_df, how='left', on=['Partner ID', 'Loan Theme ID', 'country', 'region'])
kiva_loans_mpi_df = kiva_loans_mpi_df.merge(kiva_mpi_locations_df, how='left', left_on=['country', 'mpi_region'], right_on=['country', 'LocationName'])

# Drop entries with null MPI
kiva_loans_mpi_df = kiva_loans_mpi_df.dropna(subset=['MPI'])

# Remove some information that is no longer needed
kiva_loans_mpi_df.drop('mpi_region', axis=1, inplace=True)
kiva_loans_mpi_df.drop('LocationName_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('sector_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('Loan Theme Type_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('ISO_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('region_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('geo_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('lat_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('lon_y', axis=1, inplace=True)

# Rename some columns
kiva_loans_mpi_df = kiva_loans_mpi_df.rename(index=str, columns={'region_x': 'region', 'sector_x' : 'sector', 'Loan Theme Type_x':'loan_theme_type',
                                                     'ISO_x':'ISO', 'LocationName_x':'location_name', 'geo_x':'geo', 'lat_x':'lat', 'lon_x':'lon'
                                      })

print("Merged Loans MPI dataset: ", kiva_loans_mpi_df.shape)


# ### Exploratory Analysis <a class="anchor" id="combined_loans_mpi_eda"/>

# In[ ]:


# Scatter plot of number of loans per MPI
total_loans_mpi_df = kiva_loans_mpi_df.groupby(['country','region','MPI'])['loan_amount'].count().reset_index(name='total_number_loans')
total_loans_mpi_df.sample(5)
sns.regplot(x = total_loans_mpi_df.MPI, y = total_loans_mpi_df.total_number_loans, fit_reg=False)
plt.title("Total number of Loan Requests vs. Regional MPI")
plt.show()


# - The total number of loan requests to Kiva is actually higher in the regions with a lower MPI (ie: regions that have less poverty, although note as I understand it they do have substantial poverty to even receive a MPI score). 
# - There is also one outlier with an extremely high total number of requests.

# In[ ]:


# Examine outliers
percentile_95_df = total_loans_mpi_df[total_loans_mpi_df.total_number_loans > total_loans_mpi_df.total_number_loans.quantile(.95)]
percentile_95_df.sort_values('total_number_loans', ascending=False).head(10)


# In[ ]:


# Scatter plot of total loan amount per MPI
total_loan_amt_df = kiva_loans_mpi_df.groupby(['country','region','MPI'])['loan_amount'].sum().reset_index(name='total_loan_amt')
sns.regplot(x = total_loan_amt_df.MPI, y = total_loan_amt_df['total_loan_amt'], fit_reg = False)
plt.title("Total of Loan Amount Requested vs. Regional MPI")
plt.show()


# In[ ]:


# Examine outliers
percentile_95_df = total_loan_amt_df[total_loan_amt_df.total_loan_amt > total_loan_amt_df.total_loan_amt.quantile(.95)]
percentile_95_df.sort_values('total_loan_amt', ascending=False).head(10)


# - Total loan amount exhibits a similar trend to total number of loan requests, there are quite a few higher amount requests in the regions with a lower MPI.
# 
# The trend is interesting as one would have expected the opposite - countries with higher MPI would request more loans. A possible explanation for the opposite may be that as MPI increases, the ability of these people to even request loans through Kiva decreases. For example, due to lack of knowledge about loans and the application process, lack of tools to apply for the loan in the first place, etc.
# 
# What may be important is that the people from high MPI regions who do request funds get them. Let us plot the percentage of funded loan amount according to regional MPI.

# In[ ]:


# Scatter plot of total loan amount per MPI
total_funded_amt_df = kiva_loans_mpi_df.groupby(['country','region','MPI'])['funded_amount'].sum().reset_index(name='total_funded_amt')
total_loan_amt_df= pd.merge(total_loan_amt_df, total_funded_amt_df, how='left')
sns.regplot(x = total_loan_amt_df.MPI, y = total_loan_amt_df['total_funded_amt']/total_loan_amt_df['total_loan_amt'], fit_reg = False)
plt.title("Percentage funded vs. Regional MPI")
plt.show()


# It is great to see that so many loans have been fully funded! However, it looks like there is currently no strong relationship between regional MPI and the probability of a loan funded.

# ### Relationships between Loan Features and MPI  <a class="anchor" id="loan_features_mpi_correlation"/>

# #### Sector - MPI Correlation

# In[ ]:


# Plot MPI per sector
kiva_sector_mpi = kiva_loans_mpi_df.groupby(['sector'])['MPI'].mean().reset_index(name='mean_mpi')
kiva_sector_mpi.sort_values(['mean_mpi'], ascending=False, inplace=True)
sns.barplot(x='mean_mpi', y='sector', data=kiva_sector_mpi)
plt.ylabel('')
plt.title("Average MPI per Sector")


# - Personal Use loans have the highest average MPI, followed by Wholesale and Agriculture loans.
# 
# This is not quite what I expected to see. I would have thought sectors such as Health, Food and Housing would have had a higher average MPI as they are the bare necessities for life. However, the results could be telling us that when people get to the stage where they know how to and are able to apply for a Kiva loan, they already have these basic things and are trying to get a step further with improving their lives.
# 
# Or it could be that the loans are not classified into sectors as I assumed. Let us look at some details of loans classified as Health or Food vs. Personal Use.
# 

# In[ ]:


def color_func(word, font_size, position, orientation,random_state=None, **kwargs):
    return("hsl(0,100%%, %d%%)" % np.random.randint(5,55))

# Plot word clouds
plt.subplot(221).set_title("Sector: Health")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Health'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.subplot(222).set_title("Sector: Food")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Food'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.subplot(223).set_title("Sector: Agriculture")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Agriculture'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.subplot(224).set_title("Sector: Personal Use")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Personal Use'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.suptitle("Loan Use")
plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
plt.figure(figsize=(16,20))


# We can confirm that my original idea of what the Health and Food sector loans were about was incorrect. They are not mostly about base needs but quite often for medical treatments or business uses.
# What I originally though would be classified as health seems to come under Personal Use. These are really basic needs such as safe, clean water for a family.
# Runnig similar queries for each sector shows that we can classify roughly as follows:
# 
# **Basic needs:** Personal Use, Agriculture, Housing, Education
# 
# **Basic business** needs: Retail, Clothing, Food, Wholesale, Services, Health, Construction, Manufacturing, Transportation
# 
# **Business needs / Extras:** Arts,Entertainment
# 
# There is naturally some overlap as the sector classifications are subjective.
# 
# *This outcome of this investigation is used in the ordering of the sector category in later work.

# What I originally though would be classified as health seems to come under Personal Use. These are really basic needs such as clean water for a family.
# 
# 

# #### Activity - MPI Correlation

# In[ ]:


# Plot MPI per Activity
kiva_activity_mpi = kiva_loans_mpi_df.groupby(['activity'])['MPI'].mean().reset_index(name='mean_mpi')
kiva_activity_mpi = kiva_activity_mpi.sort_values(by=['mean_mpi'], ascending=False).head(30)

with sns.color_palette("OrRd_d", 30), sns.plotting_context("notebook", font_scale=1.2):
    sns.barplot(x='mean_mpi', y='activity', data=kiva_activity_mpi)
    plt.ylabel('')
    plt.title("Average MPI per Activity")


# - There is a fair spread of MPI values per activity but no obvious relationship in the ordering of which activities have higher vs lower average MPIs. 
# 
# It is surprising that 'Pub' is the activity with the second highest mean MPI.

# #### Borrower Gender - MPI Correlation

# In[ ]:


# Plot MPI per gender
kiva_gender_mpi = kiva_loans_mpi_df.groupby(['gender'])['MPI'].mean().reset_index(name='mean_mpi')
kiva_gender_mpi.sort_values(['mean_mpi'], ascending=False, inplace=True)

with sns.color_palette("OrRd_d", 5), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,3))
    sns.barplot(x='mean_mpi', y='gender', data=kiva_gender_mpi)
    plt.ylabel('')
    plt.title("Borrower Gender vs. average MPI")


# - There is not a huge of difference between mean MPI among borrower genders and the number of borrowers (note the scale).

# #### Heatmap <a name="heatmap"/>
# 
# Using previous findings we encode some categorical features to draw a heatmap. 

# In[ ]:


# Encode some categorical features
category_mapping = {'world_region':{'Sub-Saharan Africa':1, 'South Asia':2, 'East Asia and the Pacific':3, 
                                      'Arab States':4,'Latin America and Caribbean':5,'Europe and Central Asia':6},
                    'repayment_interval':{'irregular':1, 'bullet':2, 'monthly':3, 'weekly':4, },
                    'sector':{'Personal Use':1, 'Agriculture':2, 'Housing':3, 'Education':4, 'Retail':5, 
                                'Clothing':6, 'Food':7, 'Wholesale':8, 'Services':9, 'Health':10, 
                                'Construction':11, 'Manufacturing':12, 'Transportation':13, 'Arts':14, 'Entertainment':15}}
kiva_loans_corr_df = kiva_loans_mpi_df.replace(category_mapping)

# Get dummies for gender
gender_encoded_df = pd.get_dummies(kiva_loans_corr_df['gender'], prefix='gender')
kiva_loans_corr_df = pd.concat([kiva_loans_corr_df, gender_encoded_df], axis=1, join_axes=[kiva_loans_corr_df.index])


# Plot correlation between MPI and loan factors
kiva_loans_corr = kiva_loans_corr_df[['loan_amount', 'term_in_months', 'repayment_interval', 'world_region',
                                      'sector', 'lat', 'lon', 'rural_pct', 'MPI',
                                     'gender_both', 'gender_multiple female', 'gender_multiple male', 
                                      'gender_single female', 'gender_single male']].corr()

cmap = sns.diverging_palette(50,30,sep=20, as_cmap=True)

sns.heatmap(kiva_loans_corr, 
            xticklabels=kiva_loans_corr.columns.values,
            yticklabels=kiva_loans_corr.columns.values, 
            cmap=cmap, vmin=-0.7, vmax=0.7, annot=True, square=True)
plt.title('Kiva Loan Feature Correlation')


# In[ ]:


kiva_loans_corr_df.sample()


# - There is a strong correlation between world region and MPI, not surprising.
# - There is some correlation between repayment_interval and MPI.
# - There is some correlation between rural_pct and MPI.
# - There is a weak correlation between sector and MPI.
# 
# Unfortunately, the heatmap shows overall relatively weak correlations with MPI (apart from world region). However, this lack of correlation could also be taken advantage of in the search for better indicators of poverty using these features, with the knowledge that they could potentially give us information that is not covered by the MPI.

# ### Feature Engineering <a class="anchor" id="feature_engineering"/>
#     
# #### Mapping Uses to Keyword  <a class="anchor" id="feature_eng_keyword"/>
# 
# The previous analyses seem to indicate that the Use feature may be more interesting / revealing than the broader classifications into sector or activity. Lets see what we can do to extract information here.

# In[ ]:


kiva_loans_mpi_df['use_filtered'] = kiva_loans_mpi_df.use.fillna('')
kiva_loans_mpi_df['use_filtered'] = kiva_loans_mpi_df['use_filtered'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in (STOPWORDS)]))
kiva_loans_mpi_df['use_filtered'].sample(5)


# Lets try creating a category use_word, which will be the first of the top 20 most important words (based on the frequency and some personal judgement, eg: I think it is important to include the word "business" as it indicates a different use from personal). If the use description doesn't contain any of these words, it will be classified as "other".

# In[ ]:


sanitation_words = ['water', 'filter', 'drinking', 'latrine', 'waste', 'wastes', 'toilet', 'toilets']
food_words = ['farm', 'food', 'corn', 'maize', 'rice', 'bread', 'oil', 'grain', 'meat', 'yield', 'harvest', 
              'potatoes', 'cooking', 'milk', 'sugar', 'beans', 'fruit', 'fruits' 'vegetables', 'fertilizer', 
              'seed', 'grow', 'growing', 'cultivation', 'crops', 'plant']
shelter_words = ['house', 'home', 'household', 'roof', 'repair', 'maintenance', 'broken', 'yard', 'bathroom', 'fix']
clothing_words = ['clothing', 'shoes', 'sewing', 'skirts', 'blouses']
education_words = ['university', 'tuition', 'education', 'study', 'studies', 'teach', 'teaching', 'course', 'degree']
family_words = ['family', 'child', 'children', 'daughter', 'son', 'father', 'mother',
                'provide', 'eliminating', 'pressure', 'medical']
building_words = ['supplies', 'materials', 'build', 'solar', 'cement']
improvement_words = ['buy', 'purchase', 'invest', 'improved', 'sell', 'business', 'fees', 'income', 'pay', 'store', 'yields', 'stock', 
                     'products', 'prices', 'increase', 'inputs', 'shop', 'hire', 'snacks', 'restock', 'trade']

def assign_keyword(words):
    result = assign_word_from_list(sanitation_words, "sanitation", words)
    if result != "other":
        return result
    result = assign_word_from_list(food_words, "food", words)
    if result != "other":
        return result
    result = assign_word_from_list(shelter_words, "shelter", words)
    if result != "other":
        return result
    result = assign_word_from_list(clothing_words, "clothing", words)
    if result != "other":
        return result
    result = assign_word_from_list(education_words, "education", words)
    if result != "other":
        return result
    result = assign_word_from_list(family_words, "family", words)
    if result != "other":
        return result
    result = assign_word_from_list(building_words, "building", words)
    if result != "other":
        return result
    result = assign_word_from_list(improvement_words, "improvement", words)

    return result
                 
def assign_word_from_list(category_words, keyword, words):
    result = "other"
    word_list = words.lower().split(" ")
#    print("words: ", word_list)
    for category_word in category_words:
        for word in word_list:
            if category_word == word:
                result = keyword
#                print("keyword: ", word)
                return result
    return result
            
kiva_loans_mpi_df['keyword'] =  kiva_loans_mpi_df.use_filtered.apply(assign_keyword)


# In[ ]:


kiva_loans_mpi_df.loc[kiva_loans_mpi_df['keyword'] == 'other'].sample(3)


# In[ ]:


# Plot MPI per sector
loan_keyword_mpi = kiva_loans_mpi_df.groupby(['keyword']).keyword.count().reset_index(name='keyword_count')
loan_keyword_mpi.sort_values(['keyword_count'], ascending=False, inplace=True)

with sns.color_palette("OrRd_d", 10), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,4))
    sns.barplot(x='keyword_count', y='keyword', data=loan_keyword_mpi)
    plt.ylabel('')
    plt.title("Keyword counts")


# In[ ]:


# Encode the keyword feature and check the correlation to MPI
keyword_mapping = {'keyword' : {'sanitation':1, 'food':2, 'shelter':3, 'clothing':4, 'education':5, 
                'family':6, 'building':7, 'improvement':8, 'other':9}}
keyword_mpi_corr_df = kiva_loans_mpi_df.replace(keyword_mapping)

keyword_mpi_corr_df['keyword'].corr(keyword_mpi_corr_df['MPI'])


# -0.11, no better than using repayment_interval or sector alone (ref: [Heatmap](#heatmap)).
# 
# This approach may be relying too heavily on intuition and what I expect based on my (maybe flawed) understanding. Let us take another approach and try dummes encoding to see if there are any clearly outstanding correlations.

# In[ ]:


# correlation using dummy encoding
encoded_feature = pd.get_dummies(kiva_loans_mpi_df.sector)
corr = encoded_feature.corrwith(kiva_loans_mpi_df.MPI)
corr.sort_values(ascending=False)


# I have run the above code for all the categorical features (excluding location as we know there is very high correlation there) in the original Kiva loan dataset with the following results:
#     - The highest MPI correlation in the Sector classification is Agriculture with 0.11.
#     
#     - The highest correlation for Activity is Farming with 0.21. The second highest activity is Pigs with -0.11.
#     
#     - The highest correlation for gender is Single Female at -0.20 followed by Single Male at 0.15.
#     
#     - The highest correlation for our new Keyword feature is Food with 0.16, followed by Building with -0.10. 
#      
# So in conclusion, the keyword feature doesn't seem to bring any value and the exitsting features also seem to have rather low correlation with MPI.

# ### Multidimensional Poverty - National Level <a class="anchor" id="multidimensional_poverty_national"/>

# Lets have a look at the Kaggle Multidimensional Poverty Measures dataset to see what features it can give us which were not present / easily derived from the original Kiva datasets. 
# 
# Firstly we have MPI already broken down by country, split into urban MPI and rural MPI. We also have MPI split into its components - headcount ratio and intensity. 
# 
# <div class="alert alert-block alert-info">
# The MPI value is the product of two measures: the multidimensional poverty headcount ratio and the intensity of poverty.
# <p/>
#     MPI = H . A
# </div>
# 
# Lets get a quick visualisation of these.

# In[ ]:


data = [dict(
        type='choropleth',
        locations= mpi_national_df.Country,
        locationmode='country names',
        z=mpi_national_df['MPI Urban'],
        text=mpi_national_df.Country,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
)]
layout = dict(
            title = 'Urban MPI, Country Level',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# In[ ]:


data = [dict(
        type='choropleth',
        locations= mpi_national_df.Country,
        locationmode='country names',
        z=mpi_national_df['MPI Rural'],
        text=mpi_national_df.Country,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
)]
layout = dict(
            title = 'Rural MPI, Country Level',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# - Rural MPI is overall higher than Urban MPI.
# - Africa seems to be the worse affected area.

# In[ ]:


# Sort
mpi_national_20_df = mpi_national_df.sort_values(by=['MPI Rural'], ascending=False).head(20)

# Transform the dataframe
mpi_national_urban = mpi_national_20_df[['Country', 'MPI Urban']]
mpi_national_urban.rename(columns={'MPI Urban':'value'}, inplace=True)
mpi_national_urban['indicator'] = 'MPI Urban'

mpi_national_rural = mpi_national_20_df[['Country', 'MPI Rural']]
mpi_national_rural.rename(columns={'MPI Rural':'value'}, inplace=True)
mpi_national_rural['indicator'] = 'MPI Rural'

mpi_urban_rural = mpi_national_urban.append(mpi_national_rural)

# Plot the urban and rural MPI per country (top 20)
with sns.color_palette("OrRd_d", 4), sns.plotting_context("notebook", font_scale=2):
    sns.factorplot(x='Country', y='value', hue='indicator', data=mpi_urban_rural, 
                   kind='bar', legend_out=False,  size=12, aspect=2)
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title("Urban and Rural MPI per country (top 20)")
    #plt.savefig('urban_rural_mpi.png');


# - Niger has the higest rural MPI, while South Sudan has the highest urban MPI
# 
# Note: These are just the countries with the highest 15 Rural MPIs but I have verified that South Sudan does also have the highest Urban MPI out of all the countries listed.

# In[ ]:


# Sort
mpi_national_20_df = mpi_national_df.sort_values(by=['Headcount Ratio Rural'], ascending=False).head(20)

# Transform the dataframe
mpi_national_hr_urban = mpi_national_20_df[['Country', 'Headcount Ratio Urban']]
mpi_national_hr_urban.rename(columns={'Headcount Ratio Urban':'value'}, inplace=True)
mpi_national_hr_urban['indicator'] = 'Headcount Ratio Urban'

mpi_national_hr_rural = mpi_national_20_df[['Country', 'Headcount Ratio Rural']]
mpi_national_hr_rural.rename(columns={'Headcount Ratio Rural':'value'}, inplace=True)
mpi_national_hr_rural['indicator'] = 'Headcount Ratio Rural'

mpi_hr_urban_rural  = mpi_national_hr_urban.append(mpi_national_hr_rural)

# Plot the urban and rural Headcount Ratio per country (top 20)
with sns.color_palette("OrRd_d", 4), sns.plotting_context("notebook", font_scale=2):
    sns.factorplot(x='Country', y='value', hue='indicator', data=mpi_hr_urban_rural, 
                   kind='bar', legend_out=False, size=12, aspect=2)
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title("Urban and Rural Headcount Ratio per country (top 20)")


# - Somalia has the highest rural headcount ratio of people in poverty.
# - South Sudan has the highest urban headcount ratio of people in poverty.

# In[ ]:


# Sort
mpi_national_20_df = mpi_national_df.sort_values(by=['Intensity of Deprivation Rural'], ascending=False).head(20)

# Transform the dataframe
mpi_national_id_urban = mpi_national_20_df[['Country', 'Intensity of Deprivation Urban']]
mpi_national_id_urban.rename(columns={'Intensity of Deprivation Urban':'value'}, inplace=True)
mpi_national_id_urban['indicator'] = 'Intensity of Deprivation Urban'

mpi_national_id_rural = mpi_national_20_df[['Country', 'Intensity of Deprivation Rural']]
mpi_national_id_rural.rename(columns={'Intensity of Deprivation Rural':'value'}, inplace=True)
mpi_national_id_rural['indicator'] = 'Intensity of Deprivation Rural'

mpi_id_urban_rural  = mpi_national_id_urban.append(mpi_national_id_rural)

# Plot the urban and rural Intensity of Deprivation per country (top 20)
with sns.color_palette("OrRd_d", 4), sns.plotting_context("notebook", font_scale=2):
    sns.factorplot(x='Country', y='value', hue='indicator', data=mpi_id_urban_rural, 
                   kind='bar', legend_out=False, size=12, aspect=2)
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title("Urban and Rural Intensity of Deprivation per country (top 20)")


# - Niger has the highest rural intensity of deprivation.
# - South Sudan has the highest urban intensity of deprivation.
# 
# - The difference in intensity of deprivation (the proportion of the weighted component indicators in which, on average, poor people are deprived) between urban and rural areas seems to be less than the difference in headcount ratio (proportion of the multidimensionally poor in the population) between urban rural areas. That is, the ratio of poor people is much higher in rural than urban areas. The intensity of deprivation is also higher in rural than urban areas but the difference is less.
# 
# 
# ### Multidimensional Poverty - Sub-National Level <a class="anchor" id="multidimensional_poverty_subnational"/>
# 
# Lets have a look at the result at a lower level (subnational). Here, the Googlemaps Geocodes Service has been used to get latitude and longitude values for Sub-national regions listed in the Kiva mpi_subnational dataset. This work has been done separately (currently in a priate kernel) and the csv output is just read in for this kernel.
# 
# This time to get a slightly different perspective, (since we need to plot markers by lat/long instead of filling countries anyway) we will plot the MPI in its two separated components - Headcount ratio and Deprivation Intensity. The marker size represents headcount ratio and the colour represents intensity.

# In[ ]:


data = [ dict(
        type = 'scattergeo',
        lon = mpi_subnational_df['lng'],
        lat = mpi_subnational_df['lat'],
        text = mpi_subnational_df['Sub-national region'],
        #drawmapboundary = dict(fill_color = '#A6CAE0', linewidth = 0.1),
        mode = 'markers',
        marker = dict(
            symbol = 'circle',
            sizemode = 'diameter',
            opacity = 0.7,
            line = dict(width=0),
            sizeref = 5,
            size= mpi_subnational_df['Headcount Ratio Regional'],
            color = mpi_subnational_df['Intensity of deprivation Regional'],
            colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
            reversescale=True,
            cmin = 0,
            cmax = mpi_subnational_df['Intensity of deprivation Regional'].max(),
            colorbar=dict(title="Intensity")
        ))]
layout = dict(
            title = 'Regional Headcount Ratio and Intensity of deprivation',
            geo = dict(
            showframe = False, 
            showwater=True, 
            showcountries=True,
            showland = True,
            landcolor = 'rgb(245, 241, 213)',
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)


# This give another perspective on just how bad poverty is especially in Africa. You may need to zoom in and pan across the map to see some of the markers in other areas, which are now relatively small or lightly coloured to illustrate the big differences in poverty between regions. 
# 
# Separating the MPI into headcount ratio and intensity of deprivation illustrates nicely, the inequality among the poor. 

# There are no surprises here.
# 
# - Gender development is the lowest in Africa and parts of South Asia.
# 
# - Gender inequality still exists in all countries (white ones indicate missing data, not utopia).
# - Gender inequality is the highest in many African countries.
# - Relatively high gender inequality aso exists in parts of South Asia and South America.
# 
# We can conclude that gender inequality, especially in the regions were poverty is rampant, is significant. The MPI does not take into account gender in any way so we may well be able to build a better indicator of poverty by taking into account the gender of a borrower applying for a loan. 

# ## Conclusion  <a class="anchor" id="conclusion"/>
# 
# *To be continued.*
