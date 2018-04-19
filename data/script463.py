
# coding: utf-8

# <div class="image123">
#     <div class="imgContainer">
#         <img src="https://eighty-thousand-hours-wp-production.s3.amazonaws.com/2014/09/og-logo_0.png" height="1700" width="1000"/>
#     </div>
# 
# 
# # Effective Altruism & Kiva
# ---
# 
# > **"Effective altruism is about answering one simple question: how can we use our resources to help others the most?  
# Rather than just doing what feels right, we use evidence and careful analysis to find the very best causes to work on."**  
# *[From the Official EA Website](https://www.effectivealtruism.org/)*
# 
# 
# In 2012, *80,000 Hours* (a popular Effective Altruism site) [wrote](http://80000hours.org/2012/11/is-microcredit-mostly-hype/) about their skepticism on the benefit of microcredit - directly naming Kiva.
# > "A lot more research is needed. In the meantime, it seems that microcredit charities would have more impact by shifting towards consumer loans and microsavings."  
# *Benjamin Todd, Executive Director of 80,000 Hours*
# 
# ### Personal Initiative and Impetus
# My hope is that by having Kiva's data publicly available for analysis some of the critiques brought forth by Todd can be addressed by the Kaggle Data Science community.  
# I intend to explore some of those critiques here in addition to Kiva's Problem Statement.  
# 
# ---
# 
# # Problem Statement
# For the locations in which Kiva has active loans, your objective is to pair Kiva's data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics.
# 
# A good solution would connect the features of each loan or product to one of several poverty mapping datasets, which indicate the average level of welfare in a region on as granular a level as possible. Many datasets indicate the poverty rate in a given area, with varying levels of granularity. Kiva would like to be able to disaggregate these regional averages by gender, sector, or borrowing behavior in order to estimate a Kiva borrower’s level of welfare using all of the relevant information about them. Strong submissions will attempt to map vaguely described locations to more accurate geocodes.
# 
# Kernels submitted will be evaluated based on the following criteria:
# 
# 1. Localization - How well does a submission account for highly localized borrower situations? Leveraging a variety of external datasets and successfully building them into a single submission will be crucial.
# 
# 2. Execution - Submissions should be efficiently built and clearly explained so that Kiva’s team can readily employ them in their impact calculations.
# 
# 3. Ingenuity - While there are many best practices to learn from in the field, there is no one way of using data to assess welfare levels. It’s a challenging, nuanced field and participants should experiment with new methods and diverse datasets.

# ### Table of Contents  
# 
# <a href='#data_sources'>Additional Data Sources</a>  
# <ul>
#   <li><a href='#desgin'> 1. Experimental Design</a>
#     <ul>
#     <li><a href='#world'>1.1 Kiva's Presence in the World</a>  </li>
#     <li><a href='#proportion'>1.2 Proportion Among Countries</a>    </li>
#     <li><a href='#overall'>1.3 Overall Funding Distributions</a></li>
#     <li><a href='#ave'>1.4 Sector Averages</a></li>
#     <li><a href='#boxplots'>1.5 Sector Distributions</a>  </li>
#     </ul>
#     <li><a href='#mpi'>2. Country MPI Across Time</a>
#   </li>
#   <li><a href='#lastly'>Stay Tuned! This Section will Say What I'm Working On</a> </li>
# </ul>
# 
# 
# 
# 
# 
#  

# <a id='data_sources'></a>
# # Additional Data Sources
# I've also searched through Kaggle to include a working set of relevant data sources. Feel free to put this into your own Notebook :)  
# **Please upvote if you find this useful!**

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')

# Data Munging
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from IPython.display import HTML

# Data Visualizations
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import squarify
# Plotly has such beautiful graphs
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

# Beluga's Idea
competition_data_dir = '../input/data-science-for-good-kiva-crowdfunding/'
additional_data_dir = '../input/additional-kiva-snapshot/'

# Create DataFrames of the 4 given datasets
kiva_loans = pd.read_csv(competition_data_dir + 'kiva_loans.csv')
theme_region = pd.read_csv(competition_data_dir + 'loan_themes_by_region.csv')
theme_ids = pd.read_csv(competition_data_dir + 'loan_theme_ids.csv')
# What year is this MPI? 2017?
kiva_mpi = pd.read_csv(competition_data_dir + 'kiva_mpi_region_locations.csv')
# Addtional Snapshot Data - Beluga
all_kiva_loans = pd.read_csv(additional_data_dir + 'loans.csv')


kiva_loans_df = kiva_loans.copy()
kiva_loans_df['loan_amount_trunc'] = kiva_loans_df['loan_amount'].copy()
ulimit = np.percentile(kiva_loans_df.loan_amount.values, 99)
llimit = np.percentile(kiva_loans_df.loan_amount.values, 1)
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']>ulimit] = ulimit
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']<llimit] = llimit

kiva_loans_df['funding_amount_trunc'] = kiva_loans_df['funded_amount'].copy()
upper_limit = np.percentile(kiva_loans_df.funded_amount.values, 99)
lower_limit = np.percentile(kiva_loans_df.funded_amount.values, 1)
kiva_loans_df['funding_amount_trunc'].loc[kiva_loans_df['funded_amount']>upper_limit] = upper_limit
kiva_loans_df['funding_amount_trunc'].loc[kiva_loans_df['funded_amount']<lower_limit] = lower_limit

# Joining my dataset with Kiva's subregional dataset
# mpi_time['LocationName'] = mpi_time['region'] + ', ' + mpi_time['country']
# ez_mpi_join = pd.merge(kiva_mpi, mpi_time, on='LocationName')
# del ez_mpi_join['Unnamed: 0']
# del ez_mpi_join['country_y']
# del ez_mpi_join['region_y']
# ez_mpi_join = ez_mpi_join.rename(columns={'country_x': 'country', 'region_x': 'region'})


# In[2]:


# Multidimensional Poverty Index (MPI) - The Dataset I uploaded
mpi_time = pd.read_csv('../input/multidimensional-poverty-measures/subnational_mpi_across_time.csv')
# Multidimensional Poverty Index (MPI)
national_mpi = pd.read_csv('../input/mpi/MPI_national.csv')
subnational_mpi = pd.read_csv('../input/mpi/MPI_subnational.csv')
# Google API Location Data
google_locations = pd.read_csv('../input/kiva-challenge-coordinates/kiva_locations.csv', sep='\t')


# In[3]:


kiva_dates = pd.to_datetime(kiva_loans_df['disbursed_time'])
print("From the partial dataset from Kiva:")
print("The first loan was disbursed on ", kiva_dates.min())
print("The last loan was disbursed on ", kiva_dates.max())

snapshot_dates = pd.to_datetime(all_kiva_loans['disburse_time'])
print('\n')
print("From the additional dataset (Beluga upload):")
print("The first loan was disbursed on ", snapshot_dates.min())
print("The last loan was disbursed on ", snapshot_dates.max())


# <a id='design'></a>
# # Experimental Design and Accounting for Bias
# ---
# The given data set from Kiva ranges from early January 2014 to late July 2017. The addtional dataset provided from the Kaggle community via Kiva's website shows the first loan to be on April 2005 (the year Kiva began) and the most recent loan being disbured March 2018.  
# 
# Kiva does have a self-selection bias into which regions it gives funding. A popular way to account for this bias is to **use a Propensity Score Matching** to measure differences in regions.   
# For example, Kiva gives loans to set of regions in some country with a starting level of measurable poverty while other regions in that same country do not receive loans with their own respective measure of poverty. Then, the "experiment" compares the starting levels of poverty for each region to the ending levels of poverty for each region. It is important to consider the initial differences between the cohorts and **why** one cohort did not receive lending facilities. 
# 
# ### Putting theory into practice
# Given the additional dataset, a proper "experiement" can compare regions where Kiva has allocated funds and regions without treatment. At a high level, the process should be:
# 1. Explore Poverty Metrics by region
#     - Metrics: MPI, Poverty Headcount, HDI etc.
#     - Regions: Of 80 countries, codify regions
# 2. Given those regions, bucket regions into two groups
#     - A: Received Funding from Kiva
#     - B: Did **Not** Receive Funding From Kiva
# 3. Establish baseline metrics between two groups for pre-2005
# 4. Depending on the Poverty Measures available, set up treatment times
# 5. Establish statistical significance

# <a id='world'></a>
# ### Kiva's Presence in the World
# The above map shows the number of loans in each country they provide loans. As expected, they are primarily in Sub-Saharan Africa, South America, and Southeast Asia. It would be interesting to see the proportion of loans among all other countries.

# In[ ]:


con_df = pd.DataFrame(kiva_loans_df['country'].value_counts()).reset_index()
con_df.columns = ['country', 'num_loans']
con_df = con_df.reset_index().drop('index', axis=1)

#Find out more at https://plot.ly/python/choropleth-maps/
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_loans'],
        text = con_df['country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.85,"rgb(40, 60, 190)"],[0.9,"rgb(70, 100, 245)"],\
            [0.94,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
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
    title = 'Number of Loans by Country',
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


# <a id='proportion'></a>
# ### Country Proportionality
# A lot of other kernels have shown this information using a bar graph where Philippines vastly skews the remaining part of the distribution - very similar to a power law distribution.  
# And although the Philippines is the dominant leader, Kiva has still given a lot of loans to the remaining countries they serve.  

# In[ ]:


plt.figure(figsize=(15,8))
count = kiva_loans['country'].value_counts()
squarify.plot(sizes=count.values, label=count.index, value=count.values)
plt.title('Distribution per Country')


# <a id='overall'></a>
# ###  The One Percents
# Now let's imagine how much Kiva _normally_ funds by removing the top and bottom 1% on the **remaining charts**.
# 
# ### Overall Funding Distribution
# After omitting outliers, their loan fundings range from small amounts to  ~5,000 USD with the large majority under 1,000 USD.  
# 
# Let's examine what sectors generally receive higher amounts of funding by sorting by their averages.

# In[ ]:


# Credit to SRK
plt.figure(figsize=(12,8))
sns.distplot(kiva_loans_df.funding_amount_trunc.values, bins=50, kde=False)
plt.xlabel('Funding Amount - USD', fontsize=12)
plt.title("Funding Amount Histogram after Outlier Truncation")
plt.show()


# <a id='ave'></a>
# ### Sector Averages
# Wholesale, Entertainment, and Clothing have the largest averages amoung all the sectors (omitting outliers).  
# 
# But what about the _distriubtion_ for each sector?

# In[ ]:


# Credit goes to Niyamat Ullah
# https://www.kaggle.com/niyamatalmass/who-takes-the-loan
plot_df_sector_popular_loan = pd.DataFrame(kiva_loans_df.groupby(['sector'])['funding_amount_trunc'].mean().sort_values(ascending=False)[:20]).reset_index()
plt.subplots(figsize=(15,7))
sns.barplot(x='sector',
            y='funding_amount_trunc',
            data=plot_df_sector_popular_loan,
            palette='RdYlGn_r',
            edgecolor=sns.color_palette('dark',7))
plt.ylabel('Average Funding Amount - USD', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Loan sector', fontsize=20)
plt.title('Popular Sectors by Average Funding Amount (omitting outliers)', fontsize=24)
plt.savefig('popular_sectors.png')
plt.show()


# <a id='boxplots'></a>
# ### Sector Distributions
# The majority of the sectors hover around 1,000 USD just like the overall distribution.

# In[ ]:


trace = []
for name, group in kiva_loans_df.groupby("sector"):
    trace.append ( 
        go.Box(
            x=group["funding_amount_trunc"].values,
            name=name
        )
    )
layout = go.Layout(
    title='Loan Amount Distribution by Sector',
    width = 800,
    height = 800
)
#data = [trace0, trace1]
fig = go.Figure(data=trace, layout=layout)
py.iplot(fig, filename="LoanAmountSector")


# <a id='mpi'></a>
# # Country MPI & Kiva Sums for a Given Year
# Let's look at how much funding each country has received from Kiva at the time of the first MPI calculation for that country.  
# Then, we'll normalize the sub-regional variables to understand the national MPI at a given year.
# Next, let's sum the total amount of funding that matches the year when the MPI survey was conducted.
# 
# Lastly, we'll compare the changes in Poverty with the changes of Funding from Kiva. It's key to understand the relationship between Kiva funding loans to impoverished areas.
# 

# ## Understanding MPI

# <div class="image123">
#     <div class="imgContainer">
#         <img src="http://hdr.undp.org/sites/default/files/mpi.png" height="1200" width="800"/>
#     </div>
# 

# In[167]:


# Clean up the all_kiva_loans dataframe
clean_df = all_kiva_loans.dropna(subset = ['disburse_time'])
clean_df['disburse_time'] = pd.to_datetime(clean_df['disburse_time'])
clean_df['cleaned_disburse_time'] = pd.DatetimeIndex(clean_df['disburse_time']).normalize()
clean_df['year'] = clean_df['cleaned_disburse_time'].dt.year

# Clean, merge and create new dataframe for country level MPI analysis over time
df1 = mpi_time.groupby(['country', 'year1']).agg({'total_population_year1': 'sum',
                                      'nb_poor_year1': 'sum',
                                      'poverty_intensity_year1': 'mean'}).reset_index()
df2 = mpi_time.groupby(['country', 'year2']).agg({'total_population_year2': 'sum',
                                      'nb_poor_year2': 'sum',
                                      'poverty_intensity_year2': 'mean'}).reset_index()
country_mpi_time = df1.merge(df2, left_on='country', right_on='country')
country_mpi_time =country_mpi_time[country_mpi_time['year1'] != country_mpi_time['year2']].reset_index()
del country_mpi_time['index']
country_mpi_time['country_mpi_year1'] = (country_mpi_time['nb_poor_year1'] / country_mpi_time['total_population_year1']) * (country_mpi_time['poverty_intensity_year1'] / 100.0)
country_mpi_time['country_mpi_year2'] = (country_mpi_time['nb_poor_year2'] / country_mpi_time['total_population_year2']) * (country_mpi_time['poverty_intensity_year2'] / 100.0)

# Find the unique set of ['country', 'year'] combinations
year_combo1 = country_mpi_time[['country', 'year1']].rename(columns={'year1': 'year'}).drop_duplicates()
year_combo2 = country_mpi_time[['country', 'year2']].rename(columns={'year2': 'year'}).drop_duplicates()
year_combo = year_combo1.append(year_combo2).drop_duplicates()

# Append country_sums to year_combos
list_of_ctry_sums = list()
for i, r in year_combo.iterrows():
        country_here, year_here = r['country'], r['year']
        yr_ctry_sum = clean_df[(clean_df['country_name']==country_here) & (clean_df['year'] <= year_here)].funded_amount.sum()
        list_of_ctry_sums.append(yr_ctry_sum)
year_combo['country_sum'] = list_of_ctry_sums

new_df1 = country_mpi_time.merge(year_combo, left_on=['country', 'year1'], right_on=['country', 'year']).rename(columns={'country_sum': 'country_kiva_funded_sum_year1'})
new_df2 = country_mpi_time.merge(year_combo, left_on=['country', 'year2'], right_on=['country', 'year']).rename(columns={'country_sum': 'country_kiva_funded_sum_year2'})

temp_df1 = new_df1[['country', 'year1', 'year2', 'country_kiva_funded_sum_year1']]
temp_df2 = new_df2[['country', 'year1', 'year2', 'country_kiva_funded_sum_year2']]
new_df = temp_df1.merge(temp_df2, left_on=['country', 'year1', 'year2'], right_on=['country', 'year1', 'year2'])

df = country_mpi_time.merge(new_df, left_on=['country', 'year1', 'year2'], right_on=['country', 'year1', 'year2'])
df = df.drop_duplicates(subset=['country', 'year1'], keep='last')
df = df.drop_duplicates(subset=['country', 'year2'], keep='first')

df['mpi_diff'] = df['country_mpi_year2'] - df['country_mpi_year1']
df['kiva_diff'] = df['country_kiva_funded_sum_year2'] - df['country_kiva_funded_sum_year1']
df['log_kiva_diff'] = np.log1p(df['kiva_diff'].values)


# In[189]:


sns.jointplot(data=df[df['kiva_diff'] > 0.0][['mpi_diff', 'kiva_diff']], x='kiva_diff', y='mpi_diff', kind="kde")


# # Graph
# The y-axis is the difference between MPI for survey year 2 and survey year 1. Remember the higher the MPI, the _more_ impoverished that area is. **Therefore, the negative numbers are a good sign! Pun intended.**
# 
# The x-axis is the difference between the sum of Kiva Loans funded matching up with survey years. 
# 
# # No Correlation
# Interestingly enough, the arrow of causation cannot be interpretted for the change of funding and the change of poverty.
# 
# For example, if poverty increases in an area then Kiva would seek partnerships to disburse loans in that part of the world.
# Or, Kiva disbursed loans to that impoverished area yet made the problem worse in that country.

# In[106]:


loan_coords = pd.read_csv(additional_data_dir + 'loan_coords.csv')
loans = pd.read_csv(additional_data_dir + 'loans.csv')
loans_with_coords = loans[['loan_id', 'country_name', 'town_name']].merge(loan_coords, how='left', on='loan_id')

kiva_loans = kiva_loans.set_index("id")
themes = pd.read_csv(competition_data_dir + "loan_theme_ids.csv").set_index("id")
keys = ['Loan Theme ID', 'country', 'region']
locations = pd.read_csv(competition_data_dir + "loan_themes_by_region.csv",
                        encoding = "ISO-8859-1").set_index(keys)
loc_df  = kiva_loans.join(themes['Loan Theme ID'], how='left').join(locations, on=keys, rsuffix = "_")
matched_pct = 100 * loc_df['geo'].count() / loc_df.shape[0]
print("{:.1f}% of loans in kiva_loans.csv were successfully merged with loan_themes_by_region.csv".format(matched_pct))
print("We have {} loans in kiva_loans.csv with coordinates.".format(loc_df['geo'].count()))


# <a id='lastly'></a>
# # Stay tuned
# We'll begin to combine data sets and see where there is significant overlap of regional data to infer where Kiva has measurably made the biggest difference.  
# Also, we'll visualize MPI, HDI, GDP per capita and more acroynms.  
# 
# **Make sure to upvote if you find this useful or helpful :)**
