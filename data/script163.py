
# coding: utf-8

# # Exploratory Data Analysis
# 
# I'm exploring the data a bit to get a feel of where loans are more commonly being disbursed, where larger loans are being disbursed, how various factors affect loan amount. After this initial analysis, I will go into borrower profiles and how Kiva can use these to make decisions about disbursing loans.
# 
# 1. Number of Loans By Country
# 2. Most popular sectors in which loans are taken
# 3.  Distribution of Loan duration
# 4. Distribution of number of lenders
# 5. Gender of borrowers
# 6. Distribution of Loan Amount
#     * 6.1 Analysis of loan amount below \$2000
#     * 6.2 Analysis of loan amount  \$2,000 - \$20,000
#     * 6.3 Analysis of loan amount \$20,000 to \$60,000
#     * 6.4 Loan amount above \$60,000
#     * 6.5 Loan amount by Sector
#     * 6.6 Loan Amount by Gender
#     * 6.7 Loan Amount by Country
# 7. Time taken to fund loans
#     * 7.1 Maximum time taken for a loan to be funded
#     * 7.2 Distribution of time taken for a loan to be funded
#     * 7.3 Distribution of time taken for a loan to be funded greater than 100 days
#     * 7.4 Is there any difference in the time taken to fund a loan based on the gender of the borrower?
#     
#     
# # Analysis of Borrower Profiles in India
# 1. Number of loans
# 2. Sector
# 3. Loans by region in India
#     * 3.1 Top 10 regions receiving loans
#     * 3.2 Getting in the latitude and longitude data
#     * 3.3 Visualizing the regions in india in which loans are disbursed
# 
# 

# * * *
# 

# # Exploratory Data Analysis
# - Python is being used as the tool for this analysis, with pandas helping with data-frame operations, and also with plotting
# - The following data sources are used:
#     1. Original data set posted by Kiva [here](https://www.kaggle.com/gaborfodor/additional-kiva-snapshot/)
#     2. Additional data posted by @beluga [here](https://www.kaggle.com/gaborfodor/additional-kiva-snapshot): Currently, I am only using the loans_coords.csv file to map loans to their geo-locations
# - The analysis is done on a snapshot of the data having 671 unique loans. It would be interesting to repeat this analysis on the larger dataset provided in the second source above
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
import seaborn as sns
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# In[2]:


loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv", parse_dates=['disbursed_time', 'funded_time', 'posted_time'])
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
loan_coords_df = pd.read_csv("../input/additional-kiva-snapshot/loan_coords.csv")


# In[3]:


loan_coords_df.columns = ['id', 'latitude', 'longitude']


# In[4]:


loans_df.shape


# In[5]:


loans_df.head()


# In[6]:


# From: https://deparkes.co.uk/2016/11/04/sort-pandas-boxplot/
def boxplot_sorted(df, by, column):
    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    # find and sort the median values in this new dataframe
    meds = df2.median().sort_values()
    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    return df2[meds.index].plot(kind='box', logy=True)


# ## 1. Number of loans by country
# - Since Kiva extends services to financially excluded people around the globe, it makes sense that the countries where the most loans are given out are developing nations like the Phillipines and Kenya

# In[7]:


pylab.rcParams['figure.figsize'] = (8.0, 25.0)
plt.style.use('fivethirtyeight')
loans_df.groupby(loans_df.country).id.count().sort_values().plot.barh(color='cornflowerblue');
plt.ylabel('Loan Count')
plt.title("Loan Count by Country");


# ## 2. Most popular sectors in which loans are taken
# - Agriculture, food and retail are the most popular sectors in which loans are taken as food production is one of the biggest challenges for developing nations

# In[8]:


pylab.rcParams['figure.figsize'] = (6.0, 6.0)
loans_df.groupby(loans_df.sector).id.count().sort_values().plot.bar(color='cornflowerblue');
plt.ylabel('Loan Count')
plt.title("Loan Count by Sector");


# ## 3. Distribution of Loan duration
# Most loans are short term loans of less than 24 months(two years)

# In[9]:


loans_df.term_in_months.plot.hist(bins=100);
plt.ylabel('Loan Count')
plt.title("Loan Count by Loan Duration");


# ## 4. Distribution of number of lenders
# - Most loans have between 1 to 150 lenders. Only 0.6% of all loans have more than 150 lenders.
# - For loans with fewer than 150 lenders, the distribution is skewed to the right, with the median 13!
# - There are outliers having a large numbers of lenders with the maximum number of lenders being 2986
# - There is a single loan which required 2986 lenders to fund which was for an amount of $100,000 for creating more than 300 Agricultural jobs in Haiti

# In[10]:


loans_df.lender_count.plot(kind='box', logy=True);
plt.title("Distribution of Number of Lenders per loan");
plt.xlabel("Number of lenders in powers of 10");


# In[11]:


loans_df[loans_df.lender_count > 150].shape[0]/loans_df.shape[0]


# In[12]:


axes = plt.gca()
axes.set_xlim([0,150])
loans_df.lender_count.plot.hist(bins=1000);
plt.xlabel('Number of Lenders')
plt.title("Distribution of Number of Lenders where number < 150");


# In[13]:


max(loans_df.lender_count)


# In[14]:


loans_df[loans_df.lender_count == max(loans_df.lender_count)]


# ## 5. Gender of borrowers
# - Female only borrowers(single or group) are significantly more than male only borrowers(single or group) and mixed groups

# In[15]:


def process_gender(x):
    
    if type(x) is float and np.isnan(x):
        return "nan"
    genders = x.split(",")
    male_count = sum(g.strip() == 'male' for g in genders)
    female_count = sum(g.strip() == 'female' for g in genders)
    
    if(male_count > 0 and female_count > 0):
        return "MF"
    elif(female_count > 0):
        return "F"
    elif (male_count > 0):
        return "M"


# In[16]:


loans_df.borrower_genders = loans_df.borrower_genders.apply(process_gender)


# In[17]:


loans_df.borrower_genders.value_counts().plot.bar(color='cornflowerblue');
plt.xlabel('Borrower Group/Individual Gender')
plt.ylabel('Count')
plt.title("Loan Count by Gender of Borrower");


# ## 6. Distribution of Loan Amount
# - We will consider the **funded_amount** variable as this is the amount which is disbursed to the borrower by the field agent
# - As all amounts are in USD, no currency conversion is required
# - Most of the values are below \$2000, with only 8 % of all loans lying above this value
# - There is the outlier of \$100,000 for Agriculture in Haiti which we can ignore

# In[18]:


loans_df.funded_amount.plot(kind='box', logy=True);
plt.title("Distribution of Loan Funded Amount");


# In[19]:


loans_df.funded_amount.describe()


# In[20]:


# Q3 + 1.5 * IQR
IQR = loans_df.funded_amount.quantile(0.75) - loans_df.funded_amount.quantile(0.25)
upper_whisker = loans_df.funded_amount.quantile(0.75) + 1.5 * IQR
loans_above_upper_whisker = loans_df[loans_df.funded_amount > upper_whisker]
loans_above_upper_whisker.shape


# In[21]:


# percentage of loans above upper whisker
loans_above_upper_whisker.shape[0]/loans_df.shape[0]


# ### Analysis of loan amount \$0
# - A **funded_amount** of 0 implies that the loan was not able to attract full funding after having being posted to Kiva
# - A total of 3383(0.5%) loans were not funded. We can look investigate more about the kind of loans that went unfunded in another section

# In[22]:


loans_zero = loans_df[loans_df.funded_amount == 0]
print("Number of unfunded loans", loans_zero.shape)
print("% of unfunded loans", loans_zero.shape[0]/loans_df.shape[0])


# ### 6.1 Analysis of loan amount below \$2000
# - The distribution is skewed to the right with higher loan amounts being less common

# In[23]:


loans_below_upper_whisker = loans_df[loans_df.funded_amount < upper_whisker]


# In[24]:


loans_below_upper_whisker.funded_amount.plot.hist();
plt.xlabel('Funded Amount')
plt.title("Distribution of Loan Funded amount < $2000");


# ### 6.2 Analysis of loan amount  \$2,000 - \$20,000
# - Most of the outliers lie in this range

# In[25]:


df = loans_above_upper_whisker[loans_above_upper_whisker.funded_amount < 20000]
df.funded_amount.plot.hist();
plt.xlabel('Funded Amount')
plt.title("Distribution of Loan Funded Amount between \$2,000 and \$20,000");
df.shape


# ### 6.3 Analysis of loan amount \$20,000 to \$60,000
# - A few values lie in this range
# - Most of the high value loans are disbursed for Agriculture and Retail

# In[26]:


df = loans_above_upper_whisker[(loans_above_upper_whisker.funded_amount > 20000) & (loans_above_upper_whisker.funded_amount < 60000)]
df.funded_amount.plot.hist()
plt.xlabel('Funded Amount')
plt.title("Distribution of Loan Funded Amount between \$20,000 and \$60,000");


# In[27]:


df.sector.value_counts().sort_values().plot.bar(color='cornflowerblue');
plt.ylabel('Count')
plt.xlabel('Sector')
plt.title("Loan Count by Sector for Loan Amount between \$20,000 and \$60,000");


# ### 6.4 Loan amount above \$60,000
# - There is only a single loan amount with a value of \$100,000 in this range distributed for Agriculture in Haiti
# 

# In[28]:


loans_df[loans_df.funded_amount > 60000]


# ### 6.5 Loan amount by Sector
# - Not much observable difference in distributions of loan amount by sector except that loan amounts for the personal use sector tends to be on the lower side

# In[29]:


pylab.rcParams['figure.figsize'] = (16.0, 8.0)
boxplot_sorted(loans_df[loans_df.funded_amount < 10000], by=["sector"], column="funded_amount");
plt.xticks(rotation=90);
plt.ylabel('Funded Amount')
plt.xlabel('Sector')
plt.title('Funded Amount by Sector');


# ### 6.6 Loan Amount by Gender
# - The distribution of loan amount show slightly lower amounts for female only borrowers than for male only borrowers. I will dig into this a bit more after exploring the breakdown by country. 
# - The distribution for mixed gender groups of borrowers is much more widespread

# In[30]:


pylab.rcParams['figure.figsize'] = (6.0, 6.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["borrower_genders"], column="funded_amount");
plt.title('Funded Amount by Gender')
plt.ylabel('Funded Amount')
plt.xlabel('Gender')


# In[31]:


loan_amount_values = loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")].groupby("borrower_genders").loan_amount
loan_amount_values.median()


# In[32]:


loan_amount_values.quantile(0.75) - loan_amount_values.quantile(0.25)


# ### 6.7 Loan Amount by Country
# - There's a lot going on here, but some countries that are clearly on the higher end of the loan amount spectrum are Afhanistan, Congo, Chile. It will be interesting to see what kind of sectors the loans in these countries were made for
# - In Afghanistan, there were only 2 loans disbursed with amounts between \$6,000 and \$8,000 and both of them were for Textile activity.
# - On the other end of the spectrum are countries like Nigeria which has the lowest distribution. A possible explanation for Nigeria is that the value of the dollar against the Nigerian naira is so high that even small loan amounts in dollar value go a long way

# In[33]:


pylab.rcParams['figure.figsize'] = (24.0, 8.0)
boxplot_sorted(loans_df[(loans_df.funded_amount < 10000) & (loans_df.borrower_genders != "nan")], by=["country"], column="funded_amount");
plt.xticks(rotation=90);
plt.title('Funded Amount by Country')
plt.ylabel('Funded Amount')
plt.xlabel('Country');


# In[34]:


loans_df[loans_df.country == 'Afghanistan']


# In[35]:


pylab.rcParams['figure.figsize'] = (6.0, 6.0)
loans_df[loans_df.country == 'Chile'].sector.value_counts().plot.bar(color='cornflowerblue');
plt.title("Loan Count by Sector in Chile")
plt.xlabel("Sector")
plt.ylabel("Loan Count")


# ## 7. Time taken to fund loans
# The loan disbursal process works like this in Kiva: the field agent disburses the loan to the borrower at the **disbursed_time**. The agent then posts the loan to Kiva at the **posted_time**. Lenders are then able to see the loan on Kiva and make contributions towards it. The time at which the loan has been completely funded is the **funded_time**
# 
# We can then consider the time taken to completed fund a loan to be **funded_time** - **posted_time**. There is only one case where the posted time is 17 days greater than the funded time. This is probably an error which I need to look into further. (**UPDATE** - The process I described above, where the field agent issues the funds to the borrower before posting it on Kiva is called *Pre-Disbursal* and although it is followed for *most* loans, it is not necessarily followed for all loans)
# 
# - The longest time it took for a loan to get funded was 1 year and 2 months
# - Most of the loans are funded within a 100 days. Only 0.1% of all loans in the entire sample take more than a 100 days to be funded

# In[36]:


time_to_fund = (loans_df.funded_time - loans_df.posted_time)
time_to_fund_in_days = (time_to_fund.astype('timedelta64[s]')/(3600 * 24))
loans_df = loans_df.assign(time_to_fund=time_to_fund)
loans_df = loans_df.assign(time_to_fund_in_days=time_to_fund_in_days)



# ### 7.1 Maximum time taken for a loan to be funded

# In[37]:


max(time_to_fund_in_days)


# ### 7.2 Distribution of time taken for a loan to be funded

# In[38]:


lower = loans_df.time_to_fund_in_days.quantile(0.01)
upper = loans_df.time_to_fund_in_days.quantile(0.99)
loans_df[(loans_df.time_to_fund_in_days > lower)].time_to_fund_in_days.plot.hist();
plt.title('Loan Count by Time taken to fund')
plt.xlabel('Time taken to fund')
plt.ylabel('Loan Count')


# ### 7.3 Distribution of time taken for a loan to be funded greater than 100 days

# In[39]:


loans_df[(loans_df.time_to_fund_in_days > 100)].shape


# In[40]:


loans_df[(loans_df.time_to_fund_in_days > 100)].shape[0]/loans_df.shape[0]


# In[41]:


loans_df[(loans_df.time_to_fund_in_days > 100)].time_to_fund_in_days.plot.hist();


# ### 7.4 Is there any difference in the time taken to fund a loan based on the gender of the borrower?
#  - It looks like female only borrower/borrower groups take *slightly* less time to get funded. We would need to investigate if this is significant.

# In[42]:


pylab.rcParams['figure.figsize'] = (8.0, 8.0)
boxplot_sorted(loans_df[loans_df.borrower_genders != 'nan'], by=["borrower_genders"], column="time_to_fund_in_days");


# ### 7.5 Is there any difference in the time taken to fund a loan based on the country of the borrower?
# - Countries are sorted in the increasing order of their median time to fund
# - Countries like Afghanistan and Chile are on the lower end of the spectrum. 
# - It's surprising to see the United States on the higher end of the spectrum.
# 

# In[43]:


pylab.rcParams['figure.figsize'] = (24.0, 8.0)
#loans_df[["time_to_fund_in_days", "country"]].boxplot(by="country");
axes = boxplot_sorted(loans_df, by=["country"], column="time_to_fund_in_days")
axes.set_title("Time to Fund by country in days")
plt.xticks(rotation=90);


# * * *
# 

# # Analysis of Borrower Profiles in India
# - More than 11k loans have been disbursed in India, putting India in the 13th place for the total number of loans disbursed via Kiva
# - The sector with the highest number of loans is Agriculture
# - The sectors in which loans are disbursed in India are similar to the global distribution for the top 4 categories, except for the housing category. In India, housing falls at #3 in terms of number of loans disbursed. It makes sense that with a booming population, housing is a concern and one of the top sectors, next to agriculture and food
# 

# In[44]:


df_india = loans_df[loans_df.country == 'India']


# ## 1. Number of loans

# In[45]:


df_india.shape


# ## 2. Sector

# In[46]:


pylab.rcParams['figure.figsize'] = (8.0, 8.0)
df_india.groupby('sector').id.count().sort_values().plot.bar(color='cornflowerblue');


# ## 3. Loans by region in India
# - The top ten regions receiving loans were analyzed and 8 out of them had the highest count of loans for agriculture. An exception was the city of Jaipur for which the largest number of loans were disbursed for the Arts sector. Jaipur is famous for it's art and craft shops and past rulers of the city also patronized these trades. 

# ### 3.1 Top 10 regions receiving loans

# In[47]:


pylab.rcParams['figure.figsize'] = (8.0, 8.0)
df_india.groupby('region').id.count().sort_values(ascending=False).head(20).plot.bar(color='cornflowerblue');


# In[48]:


df_india_top_ten = df_india.groupby('region').id.count().sort_values(ascending=False).head(10)


# In[49]:


df_india_top_ten


# In[50]:


df_india_top_ten.plot.bar(color='cornflowerblue');


# In[51]:


for region in df_india_top_ten.index:
    plt.title(region)
    df_india[df_india.region==region].groupby('sector').id.count().sort_values(ascending=False).plot.bar(color='cornflowerblue')
    plt.show()


# ## 3.2 Getting in the latitude and longitude data
# - I'm making use of [this](https://www.kaggle.com/gaborfodor/additional-kiva-snapshot/data) additional data source to obtain the latitude and longitude for each loan's location. The  file `loan_coords.csv` contains the mapping from loan_id to latitude-longitude pair
# - 99% of the loans disbursed in India have mappings to their latitude and longitude, so this is something we can work with

# In[52]:


pd.Series(list(set(df_india.id) & set(loan_coords_df.id))).shape[0]/df_india.shape[0]


# In[53]:


df_india = df_india.merge(loan_coords_df, on='id', how='inner')


# ## 3.3 Visualizing the regions in india in which loans are disbursed
# - There is a large concentration of loans in the East of India, especially in the state of Odisha. It would be interesting to plot the MPI(multi-dimensional poverty index) per state and see if the current allocation of loans

# In[54]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# In[55]:


longitudes = list(df_india.longitude)
latitudes = list(df_india.latitude)


# In[56]:


plt.figure(figsize=(14, 8))
earth = Basemap(projection='lcc',
                resolution='h',
                llcrnrlon=67,
                llcrnrlat=5,
                urcrnrlon=99,
                urcrnrlat=37,
                lat_0=28,
                lon_0=77
)
earth.drawcoastlines()
earth.drawcountries()
earth.drawstates(color='#555566', linewidth=1)
earth.drawmapboundary(fill_color='#46bcec')
earth.fillcontinents(color = 'white',lake_color='#46bcec')
# convert lat and lon to map projection coordinates
longitudes, latitudes = earth(longitudes, latitudes)
plt.scatter(longitudes, latitudes, 
            c='red',alpha=0.5, zorder=10)
plt.savefig('Loans Disbursed in India', dpi=350)


# ## 3.4 Visualizing the MPI per region in India
# 
# ### MPI from the OPHI data
# - The code for effectively joining all data sources has been reused from Elliot's notebook here: https://www.kaggle.com/elliottc/kivampi. Elliot is from the Kiva impact team and has provided this code as an example of how Kiva currently uses MPI. However, `MPI_subnational.csv` doesn't contain subnational MPI information for Indian regions, so I will need to look at another poverty index for India. 
# - I will consider using the state-wise Tendulkar Poverty Estimate from [here](http://niti.gov.in/state-statistics#) (COMING SOON!)

# In[57]:


MPI = pd.read_csv("../input/mpi/MPI_subnational.csv")
MPI[MPI.Country=='India'].shape


# In[58]:


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
LT = LT.groupby('Loan Theme ID').first().join(Scores)#.join(LT_['MPI Natl'])


# In[59]:


LT['Loan Theme ID'] = LT.index


# In[60]:


loans_with_mpi_df = loans_df.merge(loan_theme_ids_df, on='id').merge(LT, on='Loan Theme ID')
loans_with_mpi_india_df = loans_with_mpi_df[loans_with_mpi_df.Country == 'India']


# In[61]:


loans_with_mpi_india_df.shape


# ### Simple State-Wise Poverty Rate in India
# - Data source: [Wikipedia](https://www.wikiwand.com/en/List_of_Indian_states_and_union_territories_by_poverty_rate)
# - Let's look at the state-wise poverty rate in India (The percentage of people below the poverty line). Although this is a unidimensional poverty index looking only at income as a factor, it will give us some idea of the relative poverty of Indian states
# - Here, we see that the Poverty Rate is quite high in the north-eastern states (shown in dark red) which could justify the large number of loans concentrated in these states. We'll have to dig further with the MPI measure for each state

# In[72]:


df_poverty_rate = pd.read_csv("../input/poverty-rate-of-indian-states/IndiaPovertyRate.csv", encoding = "ISO-8859-1")
latitudes = list(df_poverty_rate.Latitude)
longitudes = list(df_poverty_rate.Longitude)
poverty_rate = list(df_poverty_rate.PovertyRate)


# In[73]:


plt.figure(figsize=(14, 8))
earth = Basemap(projection='lcc',
                resolution='h',
                llcrnrlon=67,
                llcrnrlat=5,
                urcrnrlon=99,
                urcrnrlat=37,
                lat_0=28,
                lon_0=77
)
earth.drawcoastlines()
earth.drawcountries()
earth.drawstates(color='#555566', linewidth=1)
earth.drawmapboundary(fill_color='#46bcec')
earth.fillcontinents(color = 'white',lake_color='#46bcec')
# convert lat and lon to map projection coordinates
longitudes, latitudes = earth(longitudes, latitudes)
plt.scatter(longitudes, latitudes, 
            c=poverty_rate, zorder=10, cmap='bwr')
plt.savefig('Loans Disbursed in India', dpi=350)

