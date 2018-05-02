
# coding: utf-8

# # Introduction
# > Air pollution is on of the most dangerous threat humanity ever face. Air pollution causes damage to crops, animals, forests, and bodies of water. It also contributes to the depletion of the ozone layer, which protects the Earth from the sun's UV rays. ... Some of the other environmental effects of air pollution are haze, eutrophication, and global climate change.
# ![](http://www.dw.com/image/18537610_303.jpg)
# 
# # About this notebook
# > In this notebook, I will be analyzing [OpenAQ](https://www.kaggle.com/open-aq/openaq) dataset. The dataset is stored in Google Bigquery format. Thanks to Kaggle helper library for providing such a nice API design for converting big query result into pandas dataframe. So without further talking let's dive in and see what we find!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# google bigquery library for quering data
from google.cloud import bigquery
# BigQueryHelper for converting query result direct to dataframe
from bq_helper import BigQueryHelper
# matplotlib for plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# import plotly
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')
bq_assistant = BigQueryHelper("bigquery-public-data", "openaq")


get_ipython().run_line_magic('matplotlib', 'inline')


# # TL;DR
# > A country plot of air pollution. Hover over the desire location, it will show you average value of some dangerous elements

# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as o3_avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'o3'
      AND country != 'NL'
      AND unit = 'µg/m³'
    GROUP BY country
    ORDER BY o3_avg_value ASC
        """

df_all = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as no2_avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'no2'
      AND unit = 'µg/m³'
    GROUP BY country
    ORDER BY no2_avg_value ASC
        """

df_no2 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as so2_avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'so2'
      AND unit = 'µg/m³'
    GROUP BY country
    ORDER BY so2_avg_value ASC
        """

df_so2 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as pm10_avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'pm10'
      AND unit = 'µg/m³'
    GROUP BY country
    ORDER BY pm10_avg_value ASC
        """

df_pm10 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as pm25_avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'pm25'
      AND unit = 'µg/m³'
    GROUP BY country
    ORDER BY pm25_avg_value ASC
        """

df_pm25 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


df_all['no2_avg_value'] = df_all['country'].map(df_no2.set_index('country')['no2_avg_value'])
df_all['so2_avg_value'] = df_all['country'].map(df_so2.set_index('country')['so2_avg_value'])
df_all['pm10_avg_value'] = df_all['country'].map(df_pm10.set_index('country')['pm10_avg_value'])
df_all['pm25_avg_value'] = df_all['country'].map(df_pm25.set_index('country')['pm25_avg_value'])


# In[ ]:


country_code = {
'AF':	'AFG',
'AX':	'ALA',
'AL':	'ALB',
'DZ':	'DZA',
'AS':	'ASM',
'AD':	'AND',
'AO':	'AGO',
'AI':	'AIA',
'AQ':	'ATA',
'AG':	'ATG',
'AR':	'ARG',
'AM':	'ARM',
'AW':	'ABW',
'AU':	'AUS',
'AT':	'AUT',
'AZ':	'AZE',
'BS':	'BHS',
'BH':	'BHR',
'BD':	'BGD',
'BB':	'BRB',
'BY':	'BLR',
'BE':	'BEL',
'BZ':	'BLZ',
'BJ':	'BEN',
'BM':	'BMU',
'BT':	'BTN',
'BO':	'BOL',
'BA':	'BIH',
'BW':	'BWA',
'BV':	'BVT',
'BR':	'BRA',
'VG':	'VGB',
'IO':	'IOT',
'BN':	'BRN',
'BG':	'BGR',
'BF':	'BFA',
'BI':	'BDI',
'KH':	'KHM',
'CM':	'CMR',
'CA':	'CAN',
'CV':	'CPV',
'KY':	'CYM',
'CF':	'CAF',
'TD':	'TCD',
'CL':	'CHL',
'CN':	'CHN',
'HK':	'HKG',
'MO':	'MAC',
'CX':	'CXR',
'CC':	'CCK',
'CO':	'COL',
'KM':	'COM',
'CG':	'COG',
'CD':	'COD',
'CK':	'COK',
'CR':	'CRI',
'CI':	'CIV',
'HR':	'HRV',
'CU':	'CUB',
'CY':	'CYP',
'CZ':	'CZE',
'DK':	'DNK',
'DJ':	'DJI',
'DM':	'DMA',
'DO':	'DOM',
'EC':	'ECU',
'EG':	'EGY',
'SV':	'SLV',
'GQ':	'GNQ',
'ER':	'ERI',
'EE':	'EST',
'ET':	'ETH',
'FK':	'FLK',
'FO':	'FRO',
'FJ':	'FJI',
'FI':	'FIN',
'FR':	'FRA',
'GF':	'GUF',
'PF':	'PYF',
'TF':	'ATF',
'GA':	'GAB',
'GM':	'GMB',
'GE':	'GEO',
'DE':	'DEU',
'GH':	'GHA',
'GI':	'GIB',
'GR':	'GRC',
'GL':	'GRL',
'GD':	'GRD',
'GP':	'GLP',
'GU':	'GUM',
'GT':	'GTM',
'GG':	'GGY',
'GN':	'GIN',
'GW':	'GNB',
'GY':	'GUY',
'HT':	'HTI',
'HM':	'HMD',
'VA':	'VAT',
'HN':	'HND',
'HU':	'HUN',
'IS':	'ISL',
'IN':	'IND',
'ID':	'IDN',
'IR':	'IRN',
'IQ':	'IRQ',
'IE':	'IRL',
'IM':	'IMN',
'IL':	'ISR',
'IT':	'ITA',
'JM':	'JAM',
'JP':	'JPN',
'JE':	'JEY',
'JO':	'JOR',
'KZ':	'KAZ',
'KE':	'KEN',
'KI':	'KIR',
'KP':	'PRK',
'KR':	'KOR',
'KW':	'KWT',
'KG':	'KGZ',
'LA':	'LAO',
'LV':	'LVA',
'LB':	'LBN',
'LS':	'LSO',
'LR':	'LBR',
'LY':	'LBY',
'LI':	'LIE',
'LT':	'LTU',
'LU':	'LUX',
'MK':	'MKD',
'MG':	'MDG',
'MW':	'MWI',
'MY':	'MYS',
'MV':	'MDV',
'ML':	'MLI',
'MT':	'MLT',
'MH':	'MHL',
'MQ':	'MTQ',
'MR':	'MRT',
'MU':	'MUS',
'YT':	'MYT',
'MX':	'MEX',
'FM':	'FSM',
'MD':	'MDA',
'MC':	'MCO',
'MN':	'MNG',
'ME':	'MNE',
'MS':	'MSR',
'MA':	'MAR',
'MZ':	'MOZ',
'MM':	'MMR',
'NA':	'NAM',
'NR':	'NRU',
'NP':	'NPL',
'NL':	'NLD',
'AN':	'ANT',
'NC':	'NCL',
'NZ':	'NZL',
'NI':	'NIC',
'NE':	'NER',
'NG':	'NGA',
'NU':	'NIU',
'NF':	'NFK',
'MP':	'MNP',
'NO':	'NOR',
'OM':	'OMN',
'PK':	'PAK',
'PW':	'PLW',
'PS':	'PSE',
'PA':	'PAN',
'PG':	'PNG',
'PY':	'PRY',
'PE':	'PER',
'PH':	'PHL',
'PN':	'PCN',
'PL':	'POL',
'PT':	'PRT',
'PR':	'PRI',
'QA':	'QAT',
'RE':	'REU',
'RO':	'ROU',
'RU':	'RUS',
'RW':	'RWA',
'BL':	'BLM',
'SH':	'SHN',
'KN':	'KNA',
'LC':	'LCA',
'MF':	'MAF',
'PM':	'SPM',
'VC':	'VCT',
'WS':	'WSM',
'SM':	'SMR',
'ST':	'STP',
'SA':	'SAU',
'SN':	'SEN',
'RS':	'SRB',
'SC':	'SYC',
'SL':	'SLE',
'SG':	'SGP',
'SK':	'SVK',
'SI':	'SVN',
'SB':	'SLB',
'SO':	'SOM',
'ZA':	'ZAF',
'GS':	'SGS',
'SS':	'SSD',
'ES':	'ESP',
'LK':	'LKA',
'SD':	'SDN',
'SR':	'SUR',
'SJ':	'SJM',
'SZ':	'SWZ',
'SE':	'SWE',
'CH':	'CHE',
'SY':	'SYR',
'TW':	'TWN',
'TJ':	'TJK',
'TZ':	'TZA',
'TH':	'THA',
'TL':	'TLS',
'TG':	'TGO',
'TK':	'TKL',
'TO':	'TON',
'TT':	'TTO',
'TN':	'TUN',
'TR':	'TUR',
'TM':	'TKM',
'TC':	'TCA',
'TV':	'TUV',
'UG':	'UGA',
'UA':	'UKR',
'AE':	'ARE',
'GB':	'GBR',
'US':	'USA',
'UM':	'UMI',
'UY':	'URY',
'UZ':	'UZB',
'VU':	'VUT',
'VE':	'VEN',
'VN':	'VNM',
'VI':	'VIR',
'WF':	'WLF',
'EH':	'ESH',
'YE':	'YEM',
'ZM':	'ZMB',
'ZW':	'ZWE'}


# In[ ]:


df_country = pd.DataFrame.from_dict(country_code,orient='index').reset_index()
df_country.columns = ['alpha2', 'alpha3']
df_all['country_alpha_3'] = df_all['country'].map(df_country.set_index('alpha2')['alpha3'])


# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],[0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_all['country_alpha_3'],
        z = df_all['o3_avg_value'].astype(float),
        text =  'Average value of NO: ' + df_all['no2_avg_value'].astype(str) + '<br>' + 'Average value of SO2: ' + df_all['so2_avg_value'].astype(str)+ '<br>' + 'Average value of PM10: ' + df_all['pm10_avg_value'].astype(str)+ '<br>' + 'Average value of PM2.5: ' + df_all['pm25_avg_value'].astype(str),
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Average value of O3")
        ) ]

layout = dict(
        title = 'The average value of some dangerous element in different country<br>(Hover for breakdown)',
        geo = dict(
            projection=dict( type='orthographic' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )




    
fig = dict( data=data, layout=layout )

py.iplot( fig, filename='d3-cloropleth-map' )


# <a id="ozone_gas"></a>
# # 1. Ozone gas in different country
# ![](http://eco-globe.com/wp-content/uploads/2017/05/ozone-layers.jpg)
# > This ozone forms the ozone layer, which shields everything on the planet's surface from the sun's harmful ultraviolet rays. But when ozone forms at the surface (when pollution from cars reacts with UV rays), it is a pollutant itself, and can damage forests, crops and can irritate human lungs.

# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'o3'
      AND country != 'NL'
      AND unit = 'µg/m³'
    GROUP BY country
    ORDER BY avg_value ASC
        """

df_ozone = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


plt.subplots(figsize=(15,7))
sns.barplot(x='country',y='avg_value',data=df_ozone,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Ozone gas values in µg/m³', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Country', fontsize=20)
plt.title('Average value of Ozone gas in different countries', fontsize=24)
# plt.savefig('ave_ozone.png')
plt.show()


# <a id="no2"></a>
# 
# # 2. Analyze Nitrogen dioxide
# ![](http://elte.prompt.hu/sites/default/files/tananyagok/AtmosphericChemistry/images/6e390e2b.png)
# > Carbon monoxide is harmful when breathed because it displaces oxygen in the blood and deprives the heart, brain, and other vital organs of oxygen. Large amounts of CO can overcome you in minutes without warning—causing you to lose consciousness and suffocate. ... CO poisoning can be reversed if caught in time.
# 

# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'no2'
      AND unit = 'µg/m³'
      AND country != 'NL'
      AND country != 'RS'
    GROUP BY country
    ORDER BY avg_value ASC
        """

df_no2 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


plt.subplots(figsize=(15,7))
sns.barplot(x='country',y='avg_value',data=df_no2,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('NO2 values in µg/m³', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Country', fontsize=20)
plt.title('Average value of NO2 in different countries', fontsize=24)
plt.show()


# # 3. Analyze PM10 
# > What is PM10? PM10 is particulate matter 10 micrometers or less in diameter, PM2.5 is particulate matter 2.5 micrometers or less in diameter. ... By way of comparison, a human hair is about 100 micrometres, so roughly 40 fine particles could be placed on its width.
# 
# > Why PM10 is so dangrous? Particle pollution, also called particulate matter or PM, is a mixture of solids and liquid droplets floating in the air. ... Particles less than or equal to 10 micrometers in diameter are so small that they can get into the lungs, potentially causing serious health problems.

# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'pm10'
      AND unit = 'µg/m³'
      AND country != 'NL'
      AND country != 'AU'
      AND country != 'IT'
    GROUP BY country
    ORDER BY avg_value ASC
        """

df_pm10 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


plt.subplots(figsize=(15,10))
sns.barplot(x='country',y='avg_value',data=df_pm10,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('PM10 values in µg/m³', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Country', fontsize=20)
plt.title('Average value of PM10 in different countries', fontsize=24)
plt.savefig('pm10.png')
plt.show()


# # 4. Analyze PM2.5
# PM2.5 readings are often included in air quality reports from environmental authorities and companies. ... PM2.5 refers to atmospheric particulate matter (PM) that have a diameter of less than 2.5 micrometers, which is about 3% the diameter of a human hair.
# 
# Since they are so small and light, fine particles tend to stay longer in the air than heavier particles. This increases the chances of humans and animals inhaling them into the bodies.

# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'pm25'
      AND unit = 'µg/m³'
      AND country != 'KW'
      AND country != 'ET'
      AND country != 'NP'
      AND country != 'CN'
      AND country != 'AU'
      AND country != 'NG'
      AND country != 'NL'
      AND country != 'LK'
    GROUP BY country
    ORDER BY avg_value ASC
        """

df_pm25 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:



plt.subplots(figsize=(15,10))
sns.barplot(x='country',y='avg_value',data=df_pm25,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('PM2.5 values in µg/m³', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Country', fontsize=20)
plt.title('Average value of PM2.5 in different countries', fontsize=24)
plt.show()


# # 5. Analyze SO2
# > Inhaling sulfur dioxide causes irritation to the nose, eyes, throat, and lungs. Typical symptoms include sore throat, runny nose, burning eyes, and cough. Inhaling high levels can cause swollen lungs and difficulty breathing. Skin contact with sulfur dioxide vapor can cause irritation or burns.

# In[ ]:


QUERY = """
    SELECT
        country, 
        avg(value) as avg_value
    FROM
      `bigquery-public-data.openaq.global_air_quality`
    WHERE
      pollutant = 'so2'
      AND unit = 'µg/m³'
      AND country != 'IT'
      AND country != 'NL'
      AND country != 'DK'
      AND country != 'MK'
    GROUP BY country
    ORDER BY avg_value ASC
        """

df_so2_2 = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


plt.subplots(figsize=(15,10))
sns.barplot(x='country',y='avg_value',data=df_so2_2,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('SO2 values in µg/m³', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Country', fontsize=20)
plt.title('Average value of SO2 in different countries', fontsize=24)
plt.show()


# # 4. Conclusion
# > Thanks for reading this notebook. This is the first experience with the Big Query. If you find any mistake feel free to reach me at the comment. And if you have any suggestion how to make this notebook better please tell me at the comment.
