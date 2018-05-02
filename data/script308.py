
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import datetime as dt
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 99

# Visualization
from wordcloud import WordCloud
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# NLP
import string
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


# # Summary <a class="anchor" id="summary"></a>
# ### About kiva.org
# 
# [Kiva.org](https://www.kiva.org/) is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical.
# 
# ### Objective of the challenge
# The objective of the [Data Science for Good challenge](https://www.kaggle.com/kiva/data-science-for-good-kiva-crowdfunding), is to help Kiva to build more localized models to estimate the poverty/welfare levels of residents in the regions where Kiva has active loans. The main goal is to pair the provided data with additional data sources to estimate the welfare level of borrowers in specific regions, based on shared economic and demographic characteristics. A good solution would connect the features of each loan to one of several poverty mapping datasets, which indicate the average level of welfare in a region on as granular a level as possible.
# 
# ### Motivation
# We already have quite a lot excellent EDA kernels for the original competition data set.
# Here I try to focus more on the main objective to enhance the original dataset with addititional varying granularity [data](https://www.kaggle.com/gaborfodor/additional-kiva-snapshot).
# 
# ** [Loan level Information](#loan)**
#   * Additional information from kiva.org (more loans, detailed description, lenders and loan-lender connection)
# 
# **[Region Level Information](#region)**
#   * Global Gridded Geographically Based Economic Data with (lat, lon) coords
#   * Latitude and Longitude for each region to help to join other external datasets
# 
# **[Country Level Information](#region)**:
#   * Public statistics merged manually to fix country name differences. (Population, HDI, Population below Poverty)
# 
# 
# Feel free to fork the kernel or use the dataset!
# 
# Most of the code is hidden by default, click on the dark code buttons to show it.
# 
# Most of the graphs are generated with plotly so they are interactive.

# In[ ]:


start = dt.datetime.now()
display.Image(filename='../input/additional-kiva-snapshot/cover_v2.png', width=1200) 
get_ipython().system('cp ../input/additional-kiva-snapshot/cover_v2.png .')


# You can add new datasets to your kernel and read them from subdirectories.
# 
# Please note that after adding a new dataset the original competition data directory changes from *../input/* to *../input/data-science-for-good-kiva-crowdfunding/*.

# In[ ]:


competition_data_dir = '../input/data-science-for-good-kiva-crowdfunding/'
additional_data_dir = '../input/additional-kiva-snapshot/'
print(os.listdir(competition_data_dir))
print(os.listdir(additional_data_dir))


# # Loan level <a class="anchor" id="loan"></a>
# ### Loans
# 
# It has more rows (1.4 M) and more columns. It is easy to join to the original *kiva_loans.csv*.
# 
# Some of the new columns has different name but the same content (e.g. activity and activity_name, region and town_name)

# In[ ]:


loans = pd.read_csv(additional_data_dir + 'loans.csv')
kiva_loans = pd.read_csv(competition_data_dir + 'kiva_loans.csv')
merged_loans = pd.merge(kiva_loans, loans, how='left', left_on='id', right_on='loan_id')

print('Loans provided for the challenge: {}'.format(kiva_loans.shape))
print('Loans from the additional snapshot: {}'.format(loans.shape))
print('Match ratio {:.3f}%'.format(100 * merged_loans.loan_id.count() / merged_loans.id.count()))

loans.head(2)


# ### Lenders
# More than 2M lenders. You need to work with lots of missing values.

# In[ ]:


lenders = pd.read_csv(additional_data_dir + 'lenders.csv')
lenders.head(4)
lenders.shape


# ### Loans - Lenders
# Connections between loans and lenders. It probably does not help directly the goal of the competition. Though it allows to use Network Analysis or try out Recommendation Systems.

# In[ ]:


loans_lenders = pd.read_csv(additional_data_dir + 'loans_lenders.csv')
loans_lenders.head(4)
loans_lenders.count()
loans_lenders.shape


# ### Free text fields available for NLP
# 
# We have descriptions for each loan. Most of them are in English some of the different languages have translated versions as well. The original competion set already has gender. Parsing these descriptions you could add other demographic features (e.g. age, marital status, number of children, household size, etc. ) for most of the loans.
# 
# Some of the lenders have provided reason why do they provide loans. While it is not essential given the goal of the competition it might be interesting.

# In[ ]:


stop = set(stopwords.words('english'))
def tokenize(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    text = regex.sub(" ", text)
    tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
    tokens = []
    for token_by_sent in tokens_:
        tokens += token_by_sent
    tokens = list(filter(lambda t: t.lower() not in stop, tokens))
    filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
    filtered_tokens = [w.lower() for w in filtered_tokens if len(w) >= 3]
    return " ".join(filtered_tokens)


# In[ ]:


lenders_reason = lenders[~pd.isnull(lenders['loan_because'])][['loan_because']]
lenders_reason['tokens'] = lenders_reason['loan_because'].map(tokenize)
lenders_reason_string = " ".join(lenders_reason.tokens.values)
lenders_reason_wc = WordCloud(background_color='white', max_words=2000, width=3200, height=2000)
_ = lenders_reason_wc.generate(lenders_reason_string)

lenders_reason.head()
lenders_reason.shape


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
plt.imshow(lenders_reason_wc)
plt.axis("off")
plt.title('Reason to give loan', fontsize=24)
plt.show();


# In[ ]:


loan_descriptions = loans[loans['original_language'] == 'English'][['description']]
loan_descriptions = loan_descriptions[~pd.isnull(loan_descriptions['description'])]
loan_description_sample = loan_descriptions.sample(frac=0.2)

loan_description_sample['tokens'] = loan_description_sample['description'].map(tokenize)
loan_description_string = " ".join(loan_description_sample.tokens.values)
loan_description_wc = WordCloud(background_color='white', max_words=2000, width=3200, height=2000)
_ = loan_description_wc.generate(loan_description_string)

print(loan_descriptions.shape, loan_description_sample.shape)
loan_description_sample.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 10))
plt.imshow(loan_description_wc)
plt.axis("off")
plt.title('Loan description', fontsize=24)
plt.show();


# # Country level <a class="anchor" id="country"></a>
# 
# ## Standardized country names
# 
# With the additional snapshot we can show the top countries based on the number of loans and lenders as well. To help to join other datasets standardized country names, codes, continents, region are provided.
# 

# In[ ]:


country_stats = pd.read_csv(additional_data_dir + 'country_stats.csv')
loan_country_cnt = loans.groupby(['country_code']).count()[['loan_id']].reset_index()
loan_country_cnt.columns = ['country_code', 'loan_cnt']
loan_country_cnt = loan_country_cnt.sort_values(by='loan_cnt', ascending=False)
loan_country_cnt.head()

lender_country_cnt = lenders.groupby(['country_code']).count()[['permanent_name']].reset_index()
lender_country_cnt.columns = ['country_code', 'lender_cnt']
lender_country_cnt = lender_country_cnt.sort_values(by='lender_cnt', ascending=False)
lender_country_cnt.head()

country_count = pd.merge(loan_country_cnt, lender_country_cnt, how='outer', on='country_code')
country_count = country_count.merge(country_stats[['country_code']])
threshold = 10
country_count.loc[country_count.loan_cnt < threshold, 'loan_cnt'] = 0
country_count.loc[country_count.lender_cnt < threshold, 'lender_cnt'] = 0
country_count = country_count.fillna(0)


# ## Country level statistics
# We can merge public poverty, HDI statistics on country level.
# 
# - **population** [source][3]
# - **population_below_poverty_line**:    Percentage [source][1]
# - **hdi**: Human Development Index [source][2]
# - **life_expectancy**: Life expectancy at birth [source][2]
# - **expected_years_of_schooling**: Expected years of schooling [source][2]
# - **mean_years_of_schooling**: Mean years of schooling [source][2]
# - **gni**: Gross national income (GNI) per capita [source][2]
# 
#   [1]: https://www.cia.gov/library/publications/the-world-factbook/fields/2046.html
#   [2]: http://hdr.undp.org/en/composite/HDI
#   [3]: https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)

# In[ ]:


country_stats = pd.read_csv(additional_data_dir + 'country_stats.csv')
country_stats = pd.merge(country_count, country_stats, how='inner', on='country_code')
country_stats['population_in_poverty'] = country_stats['population'] * country_stats['population_below_poverty_line'] / 100.

country_stats.shape
country_stats.head()


# On the right upper corner we can see that the Human Development Index and its components are highly correlated. As a highlevel check we can see that poverty has negative correlation with HDI.
# 
# Development indicators have positive correlation with the number of lenders while poverty has weaker positive correlation with the number of loans.

# In[ ]:


cols = ['loan_cnt', 'lender_cnt', 'population', 'population_below_poverty_line', 'population_in_poverty',
        'hdi', 'life_expectancy', 'expected_years_of_schooling', 'mean_years_of_schooling', 'gni']
corr = country_stats[cols].dropna().corr(method='spearman').round(1)
xcols = [c.replace('_', ' ') for c in corr.index]
ycols = [c.replace('_', ' ') for c in corr.index]

layout = dict(
    title = 'Country level correlations',
    width = 900,
    height = 900,
    margin=go.Margin(l=200, r=50, b=50, t=250, pad=4),
)
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=list(xcols),
    y=list(ycols),
    colorscale='Portland',
    reversescale=True,
    showscale=True,
    font_colors = ['#efecee', '#3c3636'])
fig['layout'].update(layout)
py.iplot(fig, filename='Country Correlations')


# In[ ]:


data = [dict(
        type='choropleth',
        locations=country_stats['country_name'],
        locationmode='country names',
        z=np.log10(country_stats['loan_cnt'] + 1),
        text=country_stats['country_name'],
        colorscale='Reds',
        reversescale=False,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Loans'),
)]
layout = dict(
    title = 'Number of loans by Country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-world-map')


# In[ ]:


data = [dict(
        type='choropleth',
        locations=country_stats['country_name'],
        locationmode='country names',
        z=np.log10(country_stats['lender_cnt'] + 1),
        text=country_stats['country_name'],
        colorscale='Greens',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Lenders'),
)]
layout = dict(
    title = 'Number of lenders by Country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='lenders-world-map')


# In[ ]:


data = [dict(
        type='choropleth',
        locations=country_stats['country_name'],
        locationmode='country names',
        z=country_stats['hdi'],
        text=country_stats['country_name'],
        colorscale='Portland',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='HDI'),
)]
layout = dict(
    title = 'Human Development Index',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='hdi-world-map')


# Even collecting reliable country level statistics about poverty could be difficult. There are obvious differences between countries so the same absolute poverty definition (daily 1.9 USD ) certainly has disadvantages. The CIA World Factbook tries to apply national poverty lines.
# 
# Please note that the methodology varies a lot between countries and there certainly some strange data points. E.g. China and Brazil has 3% poverty or Germany has higher poverty ratio than Russia.

# In[ ]:


largest_countries = country_stats.sort_values(by='population', ascending=False).copy()[:30]

data = [go.Scatter(
    y = largest_countries['hdi'],
    x = largest_countries['population_below_poverty_line'],
    mode='markers+text',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size= 3 * (np.log(largest_countries.population) - 10),
        color=largest_countries['hdi'],
        colorscale='Portland',
        reversescale=True,
        showscale=True)
    ,text=largest_countries['country_name']
    ,textposition=["top center"]
)]
layout = go.Layout(
    autosize=True,
    title='Poverty vs. HDI',
    hovermode='closest',
    xaxis= dict(title='Poverty%', ticklen= 5, showgrid=False, zeroline=False, showline=False),
    yaxis=dict(title='HDI', showgrid=False, zeroline=False, ticklen=5, gridwidth=2)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter_hdi_poverty')


#  The following plot shows the poverty population and the loan counts in each country.  The color indicatess the HDI.

# In[ ]:


kiva_loan_country_stats = country_stats[country_stats['loan_cnt'] > 0]
data = [go.Scatter(
    y = np.log10(kiva_loan_country_stats['loan_cnt'] + 1),
    x = np.log10(kiva_loan_country_stats['population_in_poverty'] + 1),
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size= 5 * (np.log(kiva_loan_country_stats.population) - 10),
        color = kiva_loan_country_stats['hdi'],
        colorscale='Portland',
        reversescale=True,
        showscale=True)
    ,text=kiva_loan_country_stats['country_name']
)]
layout = go.Layout(
    autosize=True,
    title='Population in poverty vs. Kiva.org loan count',
    hovermode='closest',
    xaxis= dict(title='Population in poverty (log10 scale)', ticklen= 5, showgrid=False, zeroline=False, showline=False),
    yaxis=dict(title='Loan count (log10 scale)',showgrid=False, zeroline=False, ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter_countries')


# ### Countries without kiva loans
# 
# 

# In[ ]:


country_stats_wo_kiva_loans = country_stats[np.logical_and(country_stats['loan_cnt'] == 0, country_stats['hdi'] < 0.8)]
country_stats_wo_kiva_loans = country_stats_wo_kiva_loans.sort_values(by='population_in_poverty', ascending=False)
country_stats_wo_kiva_loans = country_stats_wo_kiva_loans[['country_name', 'population', 'population_below_poverty_line', 'population_in_poverty', 'hdi', 'life_expectancy', 'gni']]
country_stats_wo_kiva_loans.head(10)


# # Regional level <a class="anchor" id="region"></a>
# 
# ### Global Gridded Geographically Based Economic Data
# 
# [Source](http://sedac.ciesin.columbia.edu/data/set/spatialecon-gecon-v4)
# 
# * **COUNTRY**: Name of country or region as of 2005
# * **LAT**: Latitude
# * **LONGITUDE**: Longitude
# * **MER2005_40**: Gross cell product, 2005 USD at market exchange rates, 2005
# * **POPGPW_2005_40**: Grid cell population, 2005 
# * **PPP2005_40**: Gross cell product, 2005 USD at purchasing power parity exchange rates, 2005

# In[ ]:


cols = ['AREA', 'COUNTRY', 'LAT', 'LONGITUDE', 'LONG_NAME', 'MER2005_40', 'NEWCOUNTRYID',
        'POPGPW_2005_40','PPP2005_40', 'QUALITY', 'RIG', 'QUALITY_REVISION', 'DATE OF LAST', 'GCP']
gecon = pd.read_csv(additional_data_dir + 'GEconV4.csv', sep=';')
gecon['MER2005_40'] = pd.to_numeric(gecon['MER2005_40'], errors='coerce')
gecon['PPP2005_40'] = pd.to_numeric(gecon['PPP2005_40'], errors='coerce')
gecon = gecon.dropna()
gecon['GCP'] = gecon['PPP2005_40'] / (gecon['POPGPW_2005_40'] + 1) * 10**6

gecon[cols].head()
gecon[cols].describe()


# In[ ]:


data = [dict(
    type='scattergeo',
    lon = gecon['LONGITUDE'],
    lat = gecon['LAT'],
    text = gecon['COUNTRY'],
    mode = 'markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 3,
        color = np.clip(gecon['GCP'].values, 0, 30),
        opacity = 0.7,
        line = dict(width=0),
        colorscale='Portland',
        reversescale=True,
        showscale=True
    ),
)]
layout = dict(
    title = 'Regional GDP per capita (2005 Thousand US$ purchase power parity)',
    hovermode='closest',
    geo = dict(showframe=False, countrywidth=1, showcountries=True,
               showcoastlines=True, projection=dict(type='Mercator'))
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='region-GDP')


# ### Provided GPS coords

# In[ ]:


kiva_loans = kiva_loans.set_index("id")
themes = pd.read_csv(competition_data_dir + "loan_theme_ids.csv").set_index("id")
keys = ['Loan Theme ID', 'country', 'region']
locations = pd.read_csv(competition_data_dir + "loan_themes_by_region.csv",
                        encoding = "ISO-8859-1").set_index(keys)
kiva_loans  = kiva_loans.join(themes['Loan Theme ID'], how='left').join(locations, on=keys, rsuffix = "_")
matched_pct = 100 * kiva_loans['geo'].count() / kiva_loans.shape[0]
print("{:.1f}% of loans in kiva_loans.csv were successfully merged with loan_themes_by_region.csv".format(matched_pct))
print("We have {} loans in kiva_loans.csv with coordinates.".format(kiva_loans['geo'].count()))


# In[ ]:


regional_counts = kiva_loans.groupby(['country', 'region', 'lat', 'lon']).count()[['funded_amount']].reset_index()
regional_counts.columns = ['country', 'region', 'lat', 'lon', 'loan_cnt']
regional_counts = regional_counts.sort_values(by='loan_cnt', ascending=False)
regional_counts.head(10)


# In[ ]:


data = [dict(
    type='scattergeo',
    lon = regional_counts['lon'],
    lat = regional_counts['lat'],
    text = regional_counts['region'],
    mode = 'markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size= 5 * (np.log10(regional_counts.loan_cnt + 1) - 1),
        color = np.log10(regional_counts['loan_cnt'] + 1),
        opacity = 0.7,
        line = dict(width=0),
        colorscale='Greens',
        reversescale=True,
        showscale=True
    ),
)]
layout = dict(
    title = 'Number of Loans by Region',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='Mercator'))
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='region-loans')


# ### Additional GPS coords
# The **full_loans.csv** has more loans and more distinct regions. Vast majority of these regions could be matched with
# the Google Maps Geolocation API. To provide an easy option to match the *loan_coords.csv* is as simple as possible.
# 
# **Locations.csv** has more details, but please note it also contains duplications if the API returned multiple records.
# 
# You could find more information about how these additional GPS ccords were created in my other kernel:
# [Process locations with Google Maps Geocoding API](https://www.kaggle.com/gaborfodor/process-locations-with-google-maps-geocoding-api)
# 

# In[ ]:


loan_coords = pd.read_csv(additional_data_dir + 'loan_coords.csv')
loan_coords.head(3)
loans_with_coords = loans[['loan_id', 'country_name', 'town_name']].merge(loan_coords, how='left', on='loan_id')
matched_pct = 100 * loans_with_coords['latitude'].count() / loans_with_coords.shape[0]
print("{:.1f}% of loans in loans.csv were successfully merged with loan_coords.csv".format(matched_pct))
print("We have {} loans in loans.csv with coordinates.".format(loans_with_coords['latitude'].count()))


# In[ ]:


town_counts = loans_with_coords.groupby(['country_name', 'town_name', 'latitude', 'longitude']).count()[['loan_id']].reset_index()
town_counts.columns = ['country_name', 'town_name', 'latitude', 'longitude', 'loan_cnt']
town_counts = town_counts.sort_values(by='loan_cnt', ascending=False)
town_counts.shape
town_counts.head()


# In[ ]:


data = [dict(
    type='scattergeo',
    lon = town_counts['longitude'],
    lat = town_counts['latitude'],
    text = town_counts['town_name'],
    mode = 'markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size= 5 * (np.log10(town_counts.loan_cnt + 1) - 1),
        color = np.log10(town_counts['loan_cnt'] + 1),
        opacity = 0.7,
        line = dict(width=0),
        colorscale='Reds',
        reversescale=False,
        showscale=True
    ),
)]
layout = dict(
    title = 'Number of Loans by Region (Enhanced)',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=1, projection=dict(type='Mercator'))
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='all-town-loans')


# Thanks for the upvotes!

# In[ ]:


lb = [
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',189,'2018-04-03'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',169,'2018-04-03'],
['SRK','Simple Exploration Notebook - Kiva',165,'2018-04-03'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',97,'2018-04-03'],
['arjundas','SimpleAnalysis for KIVA',82,'2018-04-03'],
['beluga','External Data for Kiva Crowdfunding',80,'2018-04-03'],
['Mhamed Jabri','Kivme a loan',78,'2018-04-03'],
['Niyamat Ullah','Who takes the loan?',69,'2018-04-03'],
['ddigges','Exploratory Data Analysis + Indias story',68,'2018-04-03'],
['Gabriel Preda','Kiva Data Exploration',58,'2018-04-03'],
['Abhi Hazra','Story of Kiva',53,'2018-04-03'],
['Poonam Ligade','Kiva in 2 minutes Animated Story',46,'2018-04-03'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',40,'2018-04-03'],
['beluga','Kiva - Kernel Leaderboard Progress',34,'2018-04-03'],
['doyouevendata','Kiva Exploration by a Kiva Lender and Python Newb',31,'2018-04-03'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',31,'2018-04-03'],
['Bono','Kiva EDA (extended)',31,'2018-04-03'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',30,'2018-04-03'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',29,'2018-04-03'],
['Andrea','Kiva Borrowers Welfare',27,'2018-04-03'],
['Sudhir Kumar','Kiva Welfare Loan',27,'2018-04-03'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',22,'2018-04-03'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-04-03'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',21,'2018-04-03'],
['wayward artisan','Kiva Crowdfunding - Targeting Poverty',20,'2018-04-03'],
['Tim W','Exploring Motivations for Kiva Loans',19,'2018-04-03'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-04-03'],
['Annalie','KivaMPI',17,'2018-04-03'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',17,'2018-04-03'],
['Reuben Pereira','Visualizing Poverty w/ Satellite Data',16,'2018-04-03'],
['Michael Johnson','Kiva: An Exploration in Crowdsourcing Data',14,'2018-04-03'],
['Chaitanya Gokale','Philippines: Borrower segments and poverty',14,'2018-04-03'],
['beluga','Process locations with Google Maps Geocoding API',13,'2018-04-03'],
['Bono','Kiva EDA (RMarkdown)',13,'2018-04-03'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-04-03'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',186,'2018-04-02'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',169,'2018-04-02'],
['SRK','Simple Exploration Notebook - Kiva',165,'2018-04-02'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',97,'2018-04-02'],
['arjundas','SimpleAnalysis for KIVA',79,'2018-04-02'],
['beluga','External Data for Kiva Crowdfunding',78,'2018-04-02'],
['Mhamed Jabri','Kivme a loan',70,'2018-04-02'],
['Niyamat Ullah','Who takes the loan?',65,'2018-04-02'],
['Gabriel Preda','Kiva Data Exploration',57,'2018-04-02'],
['Poonam Ligade','Kiva in 2 minutes',41,'2018-04-02'],
['ddigges','Exploratory Data Analysis + Indias story',39,'2018-04-02'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',39,'2018-04-02'],
['beluga','Kiva - Kernel Leaderboard Progress',34,'2018-04-02'],
['Bono','Kiva EDA (extended)',31,'2018-04-02'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',30,'2018-04-02'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',30,'2018-04-02'],
['doyouevendata','Kiva Exploration by a Kiva Lender and Python Newb',29,'2018-04-02'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',29,'2018-04-02'],
['Abhi Hazra','Story of Kiva',28,'2018-04-02'],
['Andrea','Kiva Borrowers Welfare',27,'2018-04-02'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',22,'2018-04-02'],
['Sudhir Kumar','Kiva Welfare Loan (+HDI report)',22,'2018-04-02'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-04-02'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',21,'2018-04-02'],
['Tim W','Exploring Motivations for Kiva Loans',19,'2018-04-02'],
['wayward artisan','Kiva Crowdfunding - Understanding Poverty Levels',18,'2018-04-02'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-04-02'],
['Reuben Pereira','Visualizing Poverty w/ Satellite Data',16,'2018-04-02'],
['Annalie','KivaMPI',16,'2018-04-02'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',16,'2018-04-02'],
['beluga','Process locations with Google Maps Geocoding API',13,'2018-04-02'],
['Chaitanya Gokale','Philippines: Borrower segments and poverty',13,'2018-04-02'],
['Bono','Kiva EDA (RMarkdown)',13,'2018-04-02'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-04-02'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',186,'2018-04-01'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',169,'2018-04-01'],
['SRK','Simple Exploration Notebook - Kiva',164,'2018-04-01'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',97,'2018-04-01'],
['beluga','External Data for Kiva Crowdfunding',78,'2018-04-01'],
['Mhamed Jabri','Kivme a loan',70,'2018-04-01'],
['arjundas','SimpleAnalysis for KIVA',65,'2018-04-01'],
['Niyamat Ullah','Who takes the loan?',65,'2018-04-01'],
['Gabriel Preda','Kiva Data Exploration',57,'2018-04-01'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',38,'2018-04-01'],
['Poonam Ligade','Kiva in 2 minutes',37,'2018-04-01'],
['beluga','Kiva - Kernel Leaderboard Progress',34,'2018-04-01'],
['ddigges','Exploratory Data Analysis + Indias story',33,'2018-04-01'],
['Bono','Kiva EDA (extended)',30,'2018-04-01'],
['doyouevendata','Kiva Exploration by a Kiva Lender and Python Newb',29,'2018-04-01'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',29,'2018-04-01'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',29,'2018-04-01'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',28,'2018-04-01'],
['Andrea','Kiva Borrowers Welfare',25,'2018-04-01'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',22,'2018-04-01'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-04-01'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',20,'2018-04-01'],
['Sudhir Kumar','Kiva Welfare Loan (+HDI report)',20,'2018-04-01'],
['Tim W','Exploring Motivations for Kiva Loans',19,'2018-04-01'],
['wayward artisan','Kiva Crowdfunding - Understanding Poverty Levels',18,'2018-04-01'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-04-01'],
['Annalie','KivaMPI',16,'2018-04-01'],
['Abhi Hazra','Story of Kiva',15,'2018-04-01'],
['Reuben Pereira','Visualizing Poverty w/ Satellite Data',15,'2018-04-01'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-04-01'],
['beluga','Process locations with Google Maps Geocoding API',13,'2018-04-01'],
['Chaitanya Gokale','Philippines: Borrower segments and poverty',13,'2018-04-01'],
['Bono','Kiva EDA (RMarkdown)',13,'2018-04-01'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-04-01'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',180,'2018-03-30'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',168,'2018-03-30'],
['SRK','Simple Exploration Notebook - Kiva',163,'2018-03-30'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',94,'2018-03-30'],
['beluga','External Data for Kiva Crowdfunding',76,'2018-03-30'],
['Mhamed Jabri','Kivme a loan',67,'2018-03-30'],
['arjundas','SimpleAnalysis for KIVA',63,'2018-03-30'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',58,'2018-03-30'],
['Gabriel Preda','Kiva Data Exploration',53,'2018-03-30'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',36,'2018-03-30'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',32,'2018-03-30'],
['beluga','Kiva - Kernel Leaderboard Progress',30,'2018-03-30'],
['Bono','Kiva EDA (extended)',30,'2018-03-30'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',29,'2018-03-30'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',29,'2018-03-30'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',28,'2018-03-30'],
['doyouevendata','Kiva Exploration by a Kiva Lender and Python Newb',26,'2018-03-30'],
['ddigges','Kiva Loans Analysis',25,'2018-03-30'],
['Andrea','Kiva Borrowers Welfare',24,'2018-03-30'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-30'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-30'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',20,'2018-03-30'],
['Tim W','Exploring Motivations for Kiva Loans',19,'2018-03-30'],
['Sudhir Kumar','Kiva Welfare Loan (+Human development report)',18,'2018-03-30'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-30'],
['wayward artisan','Kiva Crowdfunding - Understanding Poverty Levels',17,'2018-03-30'],
['Annalie','KivaMPI',16,'2018-03-30'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-30'],
['Reuben Pereira','Visualizing Poverty w/ Satellite Data',14,'2018-03-30'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-30'],
['Bono','Kiva EDA (RMarkdown)',13,'2018-03-30'],
['beluga','Process locations with Google Maps Geocoding API',13,'2018-03-30'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',179,'2018-03-29'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',167,'2018-03-29'],
['SRK','Simple Exploration Notebook - Kiva',163,'2018-03-29'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',90,'2018-03-29'],
['beluga','External Data for Kiva Crowdfunding',73,'2018-03-29'],
['Mhamed Jabri','Kivme a loan',65,'2018-03-29'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',57,'2018-03-29'],
['arjundas','SimpleAnalysis for KIVA',55,'2018-03-29'],
['Gabriel Preda','Kiva Data Exploration',52,'2018-03-29'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',36,'2018-03-29'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',32,'2018-03-29'],
['Bono','Kiva EDA (extended)',30,'2018-03-29'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',29,'2018-03-29'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',29,'2018-03-29'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',28,'2018-03-29'],
['beluga','Kiva - Kernel Leaderboard Progress',26,'2018-03-29'],
['Andrea','Kiva Borrowers Welfare',24,'2018-03-29'],
['ddigges','Kiva Loans Analysis',24,'2018-03-29'],
['doyouevendata','Kiva Exploration by a Kiva Lender and Python Newb',23,'2018-03-29'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-29'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-29'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',20,'2018-03-29'],
['Tim W','Exploring Motivations for Kiva Loans',20,'2018-03-29'],
['Sudhir Kumar','Kiva Welfare Loan (+Human development report)',18,'2018-03-29'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-29'],
['wayward artisan','Kiva Crowdfunding - Understanding Poverty Levels',16,'2018-03-29'],
['Annalie','KivaMPI',15,'2018-03-29'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-29'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-29'],
['Bono','Kiva EDA (RMarkdown)',13,'2018-03-29'],
['beluga','Process locations with Google Maps Geocoding API',13,'2018-03-29'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',172,'2018-03-28'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',162,'2018-03-28'],
['SRK','Simple Exploration Notebook - Kiva',162,'2018-03-28'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',89,'2018-03-28'],
['beluga','External Data for Kiva Crowdfunding',73,'2018-03-28'],
['Mhamed Jabri','Kivme a loan',64,'2018-03-28'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',56,'2018-03-28'],
['Gabriel Preda','Kiva Data Exploration',51,'2018-03-28'],
['arjundas','SimpleAnalysis for KIVA',50,'2018-03-28'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',36,'2018-03-28'],
['Bono','Kiva EDA (extended)',30,'2018-03-28'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',28,'2018-03-28'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',28,'2018-03-28'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',28,'2018-03-28'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',27,'2018-03-28'],
['Andrea','Kiva Borrowers Welfare',24,'2018-03-28'],
['ddigges','Kiva Loans Analysis',24,'2018-03-28'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-28'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-28'],
['Tim W','Exploring Motivations for Kiva Loans',20,'2018-03-28'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',19,'2018-03-28'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',18,'2018-03-28'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-28'],
['beluga','Kiva - Kernel Leaderboard Progress',17,'2018-03-28'],
['doyouevendata','Kiva Exploration by a Kiva Lender and Python Newb',17,'2018-03-28'],
['Annalie','KivaMPI',15,'2018-03-28'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-28'],
['wayward artisan','Kiva Crowdfunding - Understanding Poverty Levels',14,'2018-03-28'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-28'],
['Bono','Kiva EDA (RMarkdown)',13,'2018-03-28'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',169,'2018-03-27'],
['SRK','Simple Exploration Notebook - Kiva',160,'2018-03-27'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',156,'2018-03-27'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',88,'2018-03-27'],
['beluga','External Data for Kiva Crowdfunding',73,'2018-03-27'],
['Mhamed Jabri','Kivme a loan',63,'2018-03-27'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',56,'2018-03-27'],
['Gabriel Preda','Kiva Data Exploration',47,'2018-03-27'],
['arjundas','SimpleAnalysis for KIVA',44,'2018-03-27'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',36,'2018-03-27'],
['Bono','Kiva EDA (extended)',30,'2018-03-27'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',28,'2018-03-27'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',28,'2018-03-27'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',27,'2018-03-27'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',27,'2018-03-27'],
['Andrea','Kiva Borrowers Welfare',23,'2018-03-27'],
['ddigges','Kiva Loans Analysis',23,'2018-03-27'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-27'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-27'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',19,'2018-03-27'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',18,'2018-03-27'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-27'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-27'],
['Annalie','KivaMPI',15,'2018-03-27'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-27'],
['doyouevendata','Kiva Exploration by a Kiva Lender and Python Newb',14,'2018-03-27'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-27'],
['wayward artisan','Kiva Crowdfunding - Understanding Poverty Levels',13,'2018-03-27'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',166,'2018-03-26'],
['SRK','Simple Exploration Notebook - Kiva',160,'2018-03-26'],
['Bukun','Kiva Data Analysis w/ Naive Poverty Metric',148,'2018-03-26'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',86,'2018-03-26'],
['beluga','External Data for Kiva Crowdfunding',73,'2018-03-26'],
['Mhamed Jabri','Kivme a loan',59,'2018-03-26'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',55,'2018-03-26'],
['Gabriel Preda','Kiva Data Exploration',47,'2018-03-26'],
['arjundas','SimpleAnalysis for KIVA',44,'2018-03-26'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',35,'2018-03-26'],
['Bono','Kiva EDA (extended)',30,'2018-03-26'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',28,'2018-03-26'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',27,'2018-03-26'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',27,'2018-03-26'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',26,'2018-03-26'],
['Andrea','Kiva Borrowers Welfare',23,'2018-03-26'],
['ddigges','Kiva Loans Analysis',23,'2018-03-26'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-26'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-26'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',19,'2018-03-26'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-26'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-26'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',16,'2018-03-26'],
['Annalie','KivaMPI',15,'2018-03-26'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-26'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-26'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',163,'2018-03-25'],
['SRK','Simple Exploration Notebook - Kiva',159,'2018-03-25'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',143,'2018-03-25'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',83,'2018-03-25'],
['beluga','External Data for Kiva Crowdfunding',71,'2018-03-25'],
['Mhamed Jabri','Kivme a loan',59,'2018-03-25'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',54,'2018-03-25'],
['Gabriel Preda','Kiva Data Exploration',47,'2018-03-25'],
['arjundas','SimpleAnalysis for KIVA',41,'2018-03-25'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',31,'2018-03-25'],
['Bono','Kiva EDA (extended)',30,'2018-03-25'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',28,'2018-03-25'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',27,'2018-03-25'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',27,'2018-03-25'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',24,'2018-03-25'],
['Andrea','Kiva Borrowers Welfare',23,'2018-03-25'],
['ddigges','Kiva Loans Analysis',23,'2018-03-25'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-25'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-25'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',19,'2018-03-25'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-25'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-25'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',16,'2018-03-25'],
['Annalie','KivaMPI',15,'2018-03-25'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-25'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-25'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',161,'2018-03-24'],
['SRK','Simple Exploration Notebook - Kiva',159,'2018-03-24'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',141,'2018-03-24'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',80,'2018-03-24'],
['beluga','External Data for Kiva Crowdfunding',71,'2018-03-24'],
['Mhamed Jabri','Kivme a loan',59,'2018-03-24'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',53,'2018-03-24'],
['Gabriel Preda','Kiva Data Exploration',46,'2018-03-24'],
['arjundas','SimpleAnalysis for KIVA',38,'2018-03-24'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',30,'2018-03-24'],
['Bono','Kiva EDA (extended)',30,'2018-03-24'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',28,'2018-03-24'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',27,'2018-03-24'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',27,'2018-03-24'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',24,'2018-03-24'],
['Andrea','Kiva Borrowers Welfare',23,'2018-03-24'],
['ddigges','Kiva Loans Analysis',23,'2018-03-24'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-24'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-24'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',19,'2018-03-24'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-24'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-24'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',16,'2018-03-24'],
['Annalie','KivaMPI',15,'2018-03-24'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-24'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-24'],
['SRK','Simple Exploration Notebook - Kiva',158,'2018-03-23'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',150,'2018-03-23'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',136,'2018-03-23'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',77,'2018-03-23'],
['beluga','External Data for Kiva Crowdfunding',70,'2018-03-23'],
['Mhamed Jabri','Kivme a loan',56,'2018-03-23'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',50,'2018-03-23'],
['Gabriel Preda','Kiva Data Exploration',46,'2018-03-23'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',30,'2018-03-23'],
['Bono','Kiva EDA (extended)',30,'2018-03-23'],
['arjundas','SimpleAnalysis for KIVA',29,'2018-03-23'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',28,'2018-03-23'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',27,'2018-03-23'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',27,'2018-03-23'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',24,'2018-03-23'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-23'],
['Andrea','Kiva Borrowers Welfare',21,'2018-03-23'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-23'],
['ddigges','Kiva Loans Analysis',20,'2018-03-23'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-23'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-23'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',16,'2018-03-23'],
['Annalie','KivaMPI',15,'2018-03-23'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-23'],
['cesarjaitmanlabaton','Data Science for Good: Kiva Crowdfunding',13,'2018-03-23'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-23'],
['SRK','Simple Exploration Notebook - Kiva',155,'2018-03-22'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',142,'2018-03-22'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',131,'2018-03-22'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',77,'2018-03-22'],
['beluga','External Data for Kiva Crowdfunding',67,'2018-03-22'],
['Mhamed Jabri','Kivme a loan',52,'2018-03-22'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',50,'2018-03-22'],
['Gabriel Preda','Kiva Data Exploration',42,'2018-03-22'],
['Bono','Kiva EDA (extended)',30,'2018-03-22'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',29,'2018-03-22'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',28,'2018-03-22'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',27,'2018-03-22'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',27,'2018-03-22'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',24,'2018-03-22'],
['arjundas','SimpleAnalysis for KIVA',24,'2018-03-22'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-22'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',21,'2018-03-22'],
['Andrea','Kiva Borrowers Welfare',18,'2018-03-22'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-22'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-22'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',16,'2018-03-22'],
['ddigges','Kiva Loans Analysis',15,'2018-03-22'],
['Annalie','KivaMPI',15,'2018-03-22'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-22'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-22'],
['SRK','Simple Exploration Notebook - Kiva',152,'2018-03-21'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',125,'2018-03-21'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',123,'2018-03-21'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',74,'2018-03-21'],
['beluga','External Data for Kiva Crowdfunding',65,'2018-03-21'],
['Mhamed Jabri','Kivme a loan',50,'2018-03-21'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',46,'2018-03-21'],
['Gabriel Preda','Kiva Data Exploration',42,'2018-03-21'],
['Bono','Kiva EDA (extended)',29,'2018-03-21'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',27,'2018-03-21'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',27,'2018-03-21'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',26,'2018-03-21'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',26,'2018-03-21'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',24,'2018-03-21'],
['arjundas','SimpleAnalysis for KIVA',24,'2018-03-21'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-21'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',20,'2018-03-21'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-21'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-21'],
['Andrea','Kiva Borrowers Welfare',17,'2018-03-21'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',16,'2018-03-21'],
['ddigges','Kiva Loans Analysis',15,'2018-03-21'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-21'],
['Annalie','KivaMPI',14,'2018-03-21'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-21'],
['SRK','Simple Exploration Notebook - Kiva',146,'2018-03-19'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',118,'2018-03-19'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',113,'2018-03-19'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',69,'2018-03-19'],
['beluga','External Data for Kiva Crowdfunding',64,'2018-03-19'],
['Mhamed Jabri','Kivme a loan',48,'2018-03-19'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',44,'2018-03-19'],
['Gabriel Preda','Kiva Data Exploration',39,'2018-03-19'],
['Bono','Kiva EDA (extended)',29,'2018-03-19'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',27,'2018-03-19'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',27,'2018-03-19'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',26,'2018-03-19'],
['arjundas','SimpleAnalysis for KIVA',24,'2018-03-19'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',23,'2018-03-19'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',23,'2018-03-19'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-19'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',20,'2018-03-19'],
['amrrs','Whats funded - KIVA Dataset Exploration',18,'2018-03-19'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-19'],
['ddigges','Kiva Loans Analysis',15,'2018-03-19'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',15,'2018-03-19'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-19'],
['Annalie','KivaMPI',14,'2018-03-19'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-19'],
['SRK','Simple Exploration Notebook - Kiva',143,'2018-03-17'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',117,'2018-03-17'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',106,'2018-03-17'],
['beluga','External Data for Kiva Crowdfunding',64,'2018-03-17'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',60,'2018-03-17'],
['Mhamed Jabri','Kivme a loan',45,'2018-03-17'],
['Gabriel Preda','Kiva Data Exploration',38,'2018-03-17'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',38,'2018-03-17'],
['Bono','Kiva EDA (extended)',29,'2018-03-17'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',27,'2018-03-17'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',27,'2018-03-17'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',25,'2018-03-17'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',23,'2018-03-17'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',23,'2018-03-17'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-17'],
['arjundas','SimpleAnalysis for KIVA',22,'2018-03-17'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',19,'2018-03-17'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',18,'2018-03-17'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-17'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',15,'2018-03-17'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-17'],
['ddigges','Kiva Loans Analysis',14,'2018-03-17'],
['Annalie','KivaMPI',14,'2018-03-17'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-17'],
['SRK','Simple Exploration Notebook - Kiva',139,'2018-03-16'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',116,'2018-03-16'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',106,'2018-03-16'],
['beluga','External Data for Kiva Crowdfunding',63,'2018-03-16'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',60,'2018-03-16'],
['Mhamed Jabri','Kivme a loan',44,'2018-03-16'],
['Gabriel Preda','Kiva Data Exploration',38,'2018-03-16'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',36,'2018-03-16'],
['Bono','Kiva EDA (extended)',28,'2018-03-16'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',27,'2018-03-16'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',27,'2018-03-16'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',25,'2018-03-16'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',23,'2018-03-16'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',22,'2018-03-16'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-16'],
['arjundas','SimpleAnalysis for KIVA',21,'2018-03-16'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',18,'2018-03-16'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',18,'2018-03-16'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-16'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',15,'2018-03-16'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-16'],
['ddigges','Kiva Loans Analysis',14,'2018-03-16'],
['Annalie','KivaMPI',14,'2018-03-16'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-16'],
['SRK','Simple Exploration Notebook - Kiva',139,'2018-03-15'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',116,'2018-03-15'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',106,'2018-03-15'],
['beluga','External Data for Kiva Crowdfunding',63,'2018-03-15'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',60,'2018-03-15'],
['Mhamed Jabri','Kivme a loan',43,'2018-03-15'],
['Gabriel Preda','Kiva Data Exploration',38,'2018-03-15'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',36,'2018-03-15'],
['Bono','Kiva EDA (extended)',28,'2018-03-15'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',27,'2018-03-15'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',27,'2018-03-15'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',25,'2018-03-15'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',22,'2018-03-15'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-15'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',22,'2018-03-15'],
['arjundas','SimpleAnalysis for KIVA',21,'2018-03-15'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',18,'2018-03-15'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',18,'2018-03-15'],
['Tim W','Exploring Motivations for Kiva Loans',17,'2018-03-15'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-15'],
['ddigges','Kiva Loans Analysis',14,'2018-03-15'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',14,'2018-03-15'],
['Annalie','KivaMPI',14,'2018-03-15'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-15'],
['SRK','Simple Exploration Notebook - Kiva',135,'2018-03-14'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',115,'2018-03-14'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',105,'2018-03-14'],
['beluga','External Data for Kiva Crowdfunding',60,'2018-03-14'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',59,'2018-03-14'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',36,'2018-03-14'],
['Gabriel Preda','Kiva Data Exploration',35,'2018-03-14'],
['Mhamed Jabri','Kivme a loan',33,'2018-03-14'],
['Bono','Kiva EDA (extended)',28,'2018-03-14'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',27,'2018-03-14'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',27,'2018-03-14'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',25,'2018-03-14'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',22,'2018-03-14'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-14'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',21,'2018-03-14'],
['arjundas','SimpleAnalysis for KIVA',21,'2018-03-14'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',18,'2018-03-14'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',17,'2018-03-14'],
['Tim W','Exploring Motivations for Kiva Loans',16,'2018-03-14'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-14'],
['Annalie','KivaMPI',14,'2018-03-14'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-14'],
['ddigges','Kiva Loans Analysis',13,'2018-03-14'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',13,'2018-03-14'],
['SRK','Simple Exploration Notebook - Kiva',133,'2018-03-12'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',106,'2018-03-12'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',102,'2018-03-12'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',59,'2018-03-12'],
['beluga','External Data for Kiva Crowdfunding',59,'2018-03-12'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',33,'2018-03-12'],
['Mhamed Jabri','Kivme a loan',29,'2018-03-12'],
['Gabriel Preda','Kiva Data Exploration',27,'2018-03-12'],
['Bono','Kiva EDA (extended)',27,'2018-03-12'],
['Pranav Pandya','Kiva Loans EDA part 1 | Interactive Visualizations',26,'2018-03-12'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',24,'2018-03-12'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',24,'2018-03-12'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-12'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',21,'2018-03-12'],
['arjundas','SimpleAnalysis for KIVA',21,'2018-03-12'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',20,'2018-03-12'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',17,'2018-03-12'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',17,'2018-03-12'],
['Tim W','Exploring Motivations for Kiva Loans',16,'2018-03-12'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-12'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-12'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',13,'2018-03-12'],
['ddigges','Kiva Loans Analysis',12,'2018-03-12'],
['SRK','Simple Exploration Notebook - Kiva',132,'2018-03-10'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',94,'2018-03-10'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',84,'2018-03-10'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',56,'2018-03-10'],
['beluga','External Data for Kiva Crowdfunding',56,'2018-03-10'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',29,'2018-03-10'],
['Bono','Kiva EDA',26,'2018-03-10'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',24,'2018-03-10'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',22,'2018-03-10'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-10'],
['Pranav Pandya','Kiva Loans | Interactive Visualizations | Part 1',21,'2018-03-10'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',21,'2018-03-10'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',20,'2018-03-10'],
['Gabriel Preda','Kiva Data Exploration',20,'2018-03-10'],
['arjundas','SimpleAnalysis for KIVA',18,'2018-03-10'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',17,'2018-03-10'],
['Tim W','Exploring Motivations for Kiva Loans',16,'2018-03-10'],
['Mhamed Jabri','Kivme a loan',16,'2018-03-10'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',15,'2018-03-10'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-10'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-10'],
['ddigges','Kiva Loans Analysis',12,'2018-03-10'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',12,'2018-03-10'],
['SRK','Simple Exploration Notebook - Kiva',132,'2018-03-09'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',92,'2018-03-09'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',77,'2018-03-09'],
['beluga','External Data for Kiva Crowdfunding',55,'2018-03-09'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',54,'2018-03-09'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',27,'2018-03-09'],
['Bono','Kiva EDA',26,'2018-03-09'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',23,'2018-03-09'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-09'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',20,'2018-03-09'],
['Gabriel Preda','Kiva Data Exploration',20,'2018-03-09'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',19,'2018-03-09'],
['Pranav Pandya','Kiva Loans | Interactive Visualizations | Part 1',19,'2018-03-09'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',19,'2018-03-09'],
['arjundas','SimpleAnalysis for KIVA',18,'2018-03-09'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',17,'2018-03-09'],
['Mhamed Jabri','Kivme a loan',16,'2018-03-09'],
['Tim W','Exploring Motivations for Kiva Loans',15,'2018-03-09'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',15,'2018-03-09'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-09'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-09'],
['ddigges','Kiva Loans Analysis',12,'2018-03-09'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',10,'2018-03-09'],
['SRK','Simple Exploration Notebook - Kiva',130,'2018-03-08'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',92,'2018-03-08'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',72,'2018-03-08'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA',53,'2018-03-08'],
['beluga','External Data for Kiva Crowdfunding',52,'2018-03-08'],
['Bono','Kiva EDA',25,'2018-03-08'],
['Niyamat Ullah','Who takes the loan? (UPDATED with new data set)',24,'2018-03-08'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',22,'2018-03-08'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',22,'2018-03-08'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',19,'2018-03-08'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',19,'2018-03-08'],
['Gabriel Preda','Kiva Data Exploration',18,'2018-03-08'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',17,'2018-03-08'],
['Pranav Pandya','Kiva Loans | Interactive Visualizations | Part 1',17,'2018-03-08'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',17,'2018-03-08'],
['Mhamed Jabri','Kivme a loan',16,'2018-03-08'],
['arjundas','SimpleAnalysis for KIVA',15,'2018-03-08'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',14,'2018-03-08'],
['Tim W','Exploring Motivations for Kiva Loans',13,'2018-03-08'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',13,'2018-03-08'],
['ddigges','Kiva Loans Analysis',12,'2018-03-08'],
['Umesh','Kiva-Detailed Analysis using Highcharter & Leaflet',12,'2018-03-08'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',10,'2018-03-08'],
['SRK','Simple Exploration Notebook - Kiva',130,'2018-03-07'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',85,'2018-03-07'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',69,'2018-03-07'],
['beluga','External Data for Kiva Crowdfunding',50,'2018-03-07'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA Guide',49,'2018-03-07'],
['Bono','Kiva EDA',25,'2018-03-07'],
['Niyamat Ullah','Who takes the loan?',23,'2018-03-07'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',22,'2018-03-07'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',21,'2018-03-07'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',19,'2018-03-07'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',18,'2018-03-07'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',17,'2018-03-07'],
['Gabriel Preda','Kiva Data Exploration',16,'2018-03-07'],
['Mhamed Jabri','Kivme a loan',15,'2018-03-07'],
['arjundas','SimpleAnalysis for KIVA',15,'2018-03-07'],
['Pranav Pandya','Kiva Loans | Interactive Visualizations | Part 1',14,'2018-03-07'],
['Pranav Pandya','Kiva Loans EDA part 2 | Statistical Analysis',14,'2018-03-07'],
['Tim W','Exploring Motivations for Kiva Loans',13,'2018-03-07'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',12,'2018-03-07'],
['SRK','Simple Exploration Notebook - Kiva',122,'2018-03-06'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',82,'2018-03-06'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis ✓✓',64,'2018-03-06'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA Guide',48,'2018-03-06'],
['beluga','External Data for Kiva Crowdfunding',46,'2018-03-06'],
['Bono','Kiva EDA',23,'2018-03-06'],
['Niyamat Ullah','Who takes the loan?',22,'2018-03-06'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',20,'2018-03-06'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',18,'2018-03-06'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',17,'2018-03-06'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',17,'2018-03-06'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',15,'2018-03-06'],
['Mhamed Jabri','Kivme a loan',14,'2018-03-06'],
['Gabriel Preda','Kiva Data Exploration',14,'2018-03-06'],
['arjundas','SimpleAnalysis for KIVA',13,'2018-03-06'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',11,'2018-03-06'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',11,'2018-03-06'],
['ddigges','Kiva Loans Analysis',10,'2018-03-06'],
['SRK','Simple Exploration Notebook - Kiva',118,'2018-03-05'],
['Bukun','Kiva Data Analysis -Maps-Time Series Modelling',79,'2018-03-05'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis[Map ✓✓]',54,'2018-03-05'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA Guide',45,'2018-03-05'],
['beluga','External Data for Kiva Crowdfunding',42,'2018-03-05'],
['Niyamat Ullah','Who takes the loan?',21,'2018-03-05'],
['Bono','Kiva EDA',20,'2018-03-05'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',19,'2018-03-05'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',16,'2018-03-05'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',16,'2018-03-05'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',15,'2018-03-05'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',14,'2018-03-05'],
['Mhamed Jabri','Kivme a loan',13,'2018-03-05'],
['Gabriel Preda','Kiva Data Exploration',11,'2018-03-05'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',10,'2018-03-05'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',9,'2018-03-05'],
['Tim W','Exploring Motivations for Kiva Loans',8,'2018-03-05'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',8,'2018-03-05'],
['arjundas','SimpleAnalysis for KIVA',8,'2018-03-05'],
['ddigges','Kiva Loans Analysis',7,'2018-03-05'],
['SRK','Simple Exploration Notebook - Kiva',115,'2018-03-04'],
['Bukun','Kiva Data Analysis -Maps- Time Series Modelling',71,'2018-03-04'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis[Map ✓✓]',46,'2018-03-04'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA Guide',44,'2018-03-04'],
['beluga','External Data for Kiva Crowdfunding',24,'2018-03-04'],
['Bono','Kiva EDA',20,'2018-03-04'],
['Niyamat Ullah','Who takes the loan?',20,'2018-03-04'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',19,'2018-03-04'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',15,'2018-03-04'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',15,'2018-03-04'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',13,'2018-03-04'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',13,'2018-03-04'],
['Mhamed Jabri','Kivme a loan',13,'2018-03-04'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',10,'2018-03-04'],
['AlgosForGood','Kiva - Understanding Poverty Levels of Borrowers',9,'2018-03-04'],
['Sudhir Kumar','Kiva Welfare Loan: EDA',7,'2018-03-04'],
['Tim W','Exploring Motivations for Kiva Loans',6,'2018-03-04'],
['SRK','Simple Exploration Notebook - Kiva',110,'2018-03-03'],
['Bukun','Kiva Data Analysis -Maps- Time Series Modelling',62,'2018-03-03'],
['Leonardo Ferreira','ExtenKiva Exploration - EDA Guide',43,'2018-03-03'],
['Ashok Lathval','A Very Extensive Kiva Exploratory Analysis',40,'2018-03-03'],
['Bono','Kiva EDA',20,'2018-03-03'],
['Niyamat Ullah','Who takes the loan?',20,'2018-03-03'],
['Chris Crawford','Kiva Crowdfunding Starter Kernel',19,'2018-03-03'],
['beluga','External Data for Kiva Crowdfunding',16,'2018-03-03'],
['Samrat','Kiva -Beginner Guide to EDA and Data Visualization',15,'2018-03-03'],
['amrrs','Funding vs Non-Funding - Kiva Data Exploring',13,'2018-03-03'],
['Mhamed Jabri','Kivme a loan',13,'2018-03-03'],
['Poonam Ligade','Kiva - Loans that Change Life(Animated Heatmap)',13,'2018-03-03'],
['Mitchell Reynolds','External Data Sets, Kiva & Effective Altruism',12,'2018-03-03'],
['the1owl','For Kaggle & Kivas Beautiful Move 37',10,'2018-03-03'],
['Tim W','Exploring Motivations for Kiva Loans',5,'2018-03-03'],
]
LB = pd.DataFrame(lb, columns=['Author', 'Title', 'Upvotes', 'Date'])
LB['one'] = 1
df = pd.DataFrame({'Date': LB.Date.unique()})
df['one'] = 1
df = df.merge(LB, on='one', suffixes=['', '_past'])
df = df[df.Date_past <= df.Date]
lines = df.groupby(['Author', 'Date']).max()[['Upvotes']].reset_index().sort_values(by='Date')
authors = lines.groupby('Author').max()[['Upvotes']].reset_index().sort_values(by='Upvotes', ascending=False)


# In[ ]:


data = []
for author in authors.Author.values[:10]:
    df = lines[lines['Author'] == author]
    trace = go.Scatter(
        x = df.Date.values,
        y = df.Upvotes.values,
        mode = 'lines',
        name = author,
        line=dict(width=4)
    )
    data.append(trace)
layout= go.Layout(
    title= 'Kiva Kernel Leaderboard',
    xaxis= dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='#Upvotes', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='lines')


# In[ ]:


end =  dt.datetime.now()
print('Total time {} sec'.format((end - start).seconds))

