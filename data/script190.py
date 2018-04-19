
# coding: utf-8

# # Kiva's Current Poverty Targeting System
# 
# ## The Challenge
# Kiva.org provides zero-interest, risk-tolerant capital to microfinance institutions and businesses around the world with the goal of supporting financial services to poor and financially excluded borrowers. Because we reach the borrowers on our [Lending Page](http://www.kiva.org/lend) through field partners, we don't usually have a chance to directly assess their level of wealth or poverty. Having a more accurate or precise estimate of each borrower's level of poverty would be valuable to us for several reasons. For example:
# 
# Making sure new funding sources target the poorest Kiva borrowers.
# Allowing lenders to select low-income borrowers when that's their goal.
# Assessing new potential field partners
# 
# While we don't have direct measures, we do have a variety of informative variables that (we suspect) could jointly predict levels of income or financial access given the right training data. We're working with Kaggle because we'd like to build such a model. Our primary criteria for a good predictive model of poverty will be accuracy and coverage. Improvements on our current global model would be welcome progress, but equally welcome would be a highly accurate country-level model, especially in a country with lots of borrowers like Kenya, the Philippines, Senegal, Paraguay, or many others.

# ## Kiva's Current Poverty Targeting System
# Now I'll introduce you to Kiva's current poverty targeting system, which assigns scores to field partners and loan themes based on their location using the [Multi-dimensional Poverty Index (MPI)](http://ophi.org.uk/multidimensional-poverty-index/global-mpi-2017/mpi-data/) and the Global Findex dataset for financial inclusion.

# ## MPI Scores
# MPI scores are assigned at two levels of granularity, national and sub-national.
# 
# ### National MPI Scores
# Nation-level MPI Scores are broken into rural and urban scores. So Kiva's broadest measure simply assigns each field partner an average of these two numbers, weighted by rural_pct. About a dozen field partners also serve multiple countries, in which case we take a volume-weighted average.

# In[1]:


# Load libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from numpy import log10, ceil, ones
from numpy.linalg import inv 
from matplotlib import pyplot as plt
from seaborn import regplot
import statsmodels.formula.api as smf

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


# Load data
LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv") #.set_index([''])
MPI = pd.read_csv("../input/mpi/MPI_national.csv")[['ISO','MPI Urban','MPI Rural']].set_index("ISO")
LT = LT.join(MPI,how='left',on="ISO")[['Partner ID','Field Partner Name','ISO','MPI Rural','MPI Urban','rural_pct','amount']].dropna()

LT.head()


# #### Compute the MPI score for each partner: 
# The MPI score is a weighted average of the (cumulative?) amount that a partner has received given the partners presence in rural and urban areas. 
# 
# These rural and mpi scores are an indication of the poverty level of the borrowers. However, the goal of this challenge is to give a more precise indication of the poverty level of a borrower given his or her precise location. 

# In[3]:


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


# So we can see that the scores in this case follow a predictable distribution with a long tail. The MPI is clearly suited to differentiating one poor country from another, and does less to differentiate among middle-income countries. You can also see that rual MPI scores are (pretty much) always higher than urban scores, as you might expect.

# In[4]:


fig, ax = plt.subplots(1, 3,figsize=(12,4))
Scores.plot(kind='hist', bins=30,ax=ax[0], title= "Rural-weighted MPI Scores by Field Parnter")
MPI['MPI Rural'].plot(kind='hist', bins=30,ax=ax[1], title="Rural MPI Scores by Country")
MPI.plot(kind='scatter',x = 'MPI Rural', y = 'MPI Urban', title = "Urban vs. Rural MPI Scores by Country\n(w/ y=x line)", ax=ax[2])
ax[2].plot(ax[2].get_xlim(),ax[2].get_ylim())
#regplot('MPI Rural','MPI Urban', data=MPI, ax=ax[2]) ; ax[2].set_title("Urban vs. Rural MPI Scores by Country")
plt.tight_layout()


# ### Sub-National MPI Scores
# A primary reason for chosing OPHI's MPI as our poverty index was that it is disaggregated at the administrative region level, so that we can account for targeting within countries to some extent. Here I'll demonstrate the current system for assigning a given loan or loan theme to a given MPI region. Those scores then get aggregated as a volume-weighted average.

# In[5]:


# Load data

MPIsubnat = pd.read_csv("../input/mpi/MPI_subnational.csv")[['Country', 'Sub-national region', 'World region', 'MPI National', 'MPI Regional']]
# Create new column LocationName that concatenates the columns Country and Sub-national region
MPIsubnat['LocationName'] = MPIsubnat[['Sub-national region', 'Country']].apply(lambda x: ', '.join(x), axis=1)

LT = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")[['Partner ID', 'Loan Theme ID', 'region', 'mpi_region', 'ISO', 'number', 'amount', 'LocationName', 'names']]

# Merge dataframes
LT = LT.merge(MPIsubnat, left_on='mpi_region', right_on='LocationName', suffixes=('_lt', '_mpi'))[['Partner ID', 'Loan Theme ID', 'Country', 'ISO', 'mpi_region', 'MPI Regional', 'number', 'amount']]

LT.head()


# In[6]:


#~ Get total volume and average MPI Regional Score for each partner loan theme
LS = LT.groupby(['Partner ID', 'Loan Theme ID', 'Country', 'ISO']).agg({'MPI Regional': np.mean, 'amount': np.sum, 'number': np.sum})
#~ Get a volume-weighted average of partners loanthemes.
weighted_avg_lt = lambda df: np.average(df['MPI Regional'], weights=df['amount'])
#~ and get weighted average for partners. 
MPI_regional_scores = LS.groupby(level='Partner ID').apply(weighted_avg_lt)


# If we compare the Rural/Urban weighted MPI Scores of a field partner with the Regional weighted MPI Scores, we see a peak between an MPI score of 0.3 and 0.4 if we use Regional-weighted MPI Scores. So if Kiva uses a more detailed score for poverty, we see that we lend to borrowers that are actually in poorer areas than the Rural weighted scores show us.   

# In[7]:


fig, ax = plt.subplots(1, 2, figsize=(12,4))
MPI_regional_scores.plot(kind='hist', bins=30, ax=ax[0], title= "Regional-weighted MPI Scores by Field Partner")
Scores.plot(kind='hist', bins=30, ax=ax[1], title= "Rural-weighted MPI Scores by Field Partner")
plt.tight_layout()


# Although Kiva gives larger loan amounts to field partners that have a lower MPI score, this doesn't necessarily mean that there are also more loans given to borrowers of these field partners.  

# In[8]:


# plot amount per partner by regional partner MPI score
A = LS.groupby(level='Partner ID').agg({'amount':np.sum})
N = LS.groupby(level='Partner ID').agg({'number':np.sum})

fig,ax = plt.subplots(1,2, figsize=(12,4))
pd.concat([A, MPI_regional_scores], axis=1).plot(kind='scatter', x=0, y='amount', title="Amount invested by Field Partner MPI score", ax=ax[0])
pd.concat([N, MPI_regional_scores], axis=1).plot(kind='scatter', x=0, y='number', title="Number of loans invested by Field Partner MPI score", ax=ax[1])
plt.tight_layout()
for ax in ax.flat:
    ax.set(xlabel='MPI score')


# ## Financial Inclusion
# 
# Apart from measuring poverty levels, Kiva is concerned with financial exlusive areas. The [Global Findex ](http://www.worldbank.org/en/programs/globalfindex) database published by the World Bank is an example of a dataset that measures financial inclusion per country.
# This database provides in-depth data on how individuals save, borrow, make payments, and manage risks.
# 
# In this example we show how you could incorporate this data to come up with a metric for financial exclusion per country.
# 
# The Global Findex data is stored in the 'findex-world-bank' folder and is called 'FINDEXData.csv'.

# ### Findex Score
# 
# To measure a country's financial inclusion we chose three indicators from the Global Findex dataset that capture the essence of financial inclusion: 
# * Having a savingsaccount at a formal financial institution
# * Being able to borrow from a formal financial institution
# * Having access to a fund in the case of an emergency
# 
# For these indicators we used the data that corresponds to the answers of the population whose income is lower than the mean income.   

# In[23]:


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


# In[24]:


F = read_findex()
F.head()


# ### Compare Findex Score and MPI Score
# 
# From here you can merge the Findex Scores with MPI Scores and compare the metric for financial inclusion with the multidimensional poverty score. 
# The below figure shows that countries with high poverty levels often score high on the financial inclusion index. However we don't see a clear correlation between the two metrics, which means that the Findex data measures something that is unrelated to poverty.  

# In[25]:


# Compare scores on a country level
MPI.join(F).plot(kind='scatter', x='MPI Rural', y='Findex', title="MPI Rural Score versus Findex Score");


# ## Further Analysis
# 
# More importantly, how would these metrices and other relevant data sources define poverty and financial inclusion?
# How detailed can we go? 
# 
# Do you have an idea of a general metric that is scalable for **each** country? How would you fill missing values for countries? 
# 
# Or would you rather focus on one country in detail?
