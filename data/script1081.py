
# coding: utf-8

# This notebook is based on sample version with first 10,000,000 rows.<br><br>
# It contains two main parts:<br>
# 1. Exploration by country<br>
# 2. Exploration by State for USA data<br>
# 
# Any comments and improvements are welcome!

# ### Load "Page view" sample data

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

page_views_sample_df = pd.read_csv("../input/page_views_sample.csv", usecols=['uuid', 'geo_location'])
# Drop NAs
page_views_sample_df.dropna(inplace=True)
# Drop EU code
page_views_sample_df = page_views_sample_df.loc[~page_views_sample_df.geo_location.isin(['EU', '--']), :]
# Drop duplicates
page_views_sample_df = page_views_sample_df.drop_duplicates('uuid', keep='first')


# # Exploration by Country

# At the beginning let see distribution by country.

# In[ ]:


country = page_views_sample_df.copy()
country.columns = ['uuid', 'Country']
country.Country = country.Country.str[:2]
country.loc[:, 'UserCount'] = country.groupby('Country')['Country'].transform('count')
country = country.loc[:, ['Country', 'UserCount']].drop_duplicates('Country', keep='first')
country.sort_values('UserCount', ascending=False, inplace=True)
country.head(10)


# Adding country names to show data in a beautiful way, to generate dictionary with Country names I've used **pycountry** library.

# In[ ]:


countryCode2Name = {u'BD': u'Bangladesh', u'BE': u'Belgium', u'BF': u'Burkina Faso', u'BG': u'Bulgaria', u'BA': u'Bosnia and Herzegovina', u'BB': u'Barbados', u'WF': u'Wallis and Futuna', u'BL': u'Saint Barth\xe9lemy', u'BM': u'Bermuda', u'BN': u'Brunei Darussalam', u'BO': u'Bolivia, Plurinational State of', u'BH': u'Bahrain', u'BI': u'Burundi', u'BJ': u'Benin', u'BT': u'Bhutan', u'JM': u'Jamaica', u'BV': u'Bouvet Island', u'BW': u'Botswana', u'WS': u'Samoa', u'BQ': u'Bonaire, Sint Eustatius and Saba', u'BR': u'Brazil', u'BS': u'Bahamas', u'JE': u'Jersey', u'BY': u'Belarus', u'BZ': u'Belize', u'RU': u'Russian Federation', u'RW': u'Rwanda', u'RS': u'Serbia', u'TL': u'Timor-Leste', u'RE': u'R\xe9union', u'TM': u'Turkmenistan', u'TJ': u'Tajikistan', u'RO': u'Romania', u'TK': u'Tokelau', u'GW': u'Guinea-Bissau', u'GU': u'Guam', u'GT': u'Guatemala', u'GS': u'South Georgia and the South Sandwich Islands', u'GR': u'Greece', u'GQ': u'Equatorial Guinea', u'GP': u'Guadeloupe', u'JP': u'Japan', u'GY': u'Guyana', u'GG': u'Guernsey', u'GF': u'French Guiana', u'GE': u'Georgia', u'GD': u'Grenada', u'GB': u'United Kingdom', u'GA': u'Gabon', u'GN': u'Guinea', u'GM': u'Gambia', u'GL': u'Greenland', u'GI': u'Gibraltar', u'GH': u'Ghana', u'OM': u'Oman', u'TN': u'Tunisia', u'JO': u'Jordan', u'HR': u'Croatia', u'HT': u'Haiti', u'HU': u'Hungary', u'HK': u'Hong Kong', u'HN': u'Honduras', u'HM': u'Heard Island and McDonald Islands', u'VE': u'Venezuela, Bolivarian Republic of', u'PR': u'Puerto Rico', u'PS': u'Palestine, State of', u'PW': u'Palau', u'PT': u'Portugal', u'KN': u'Saint Kitts and Nevis', u'PY': u'Paraguay', u'IQ': u'Iraq', u'PA': u'Panama', u'PF': u'French Polynesia', u'PG': u'Papua New Guinea', u'PE': u'Peru', u'PK': u'Pakistan', u'PH': u'Philippines', u'PN': u'Pitcairn', u'PL': u'Poland', u'PM': u'Saint Pierre and Miquelon', u'ZM': u'Zambia', u'EH': u'Western Sahara', u'EE': u'Estonia', u'EG': u'Egypt', u'ZA': u'South Africa', u'EC': u'Ecuador', u'IT': u'Italy', u'VN': u'Viet Nam', u'SB': u'Solomon Islands', u'ET': u'Ethiopia', u'SO': u'Somalia', u'ZW': u'Zimbabwe', u'SA': u'Saudi Arabia', u'ES': u'Spain', u'ER': u'Eritrea', u'ME': u'Montenegro', u'MD': u'Moldova, Republic of', u'MG': u'Madagascar', u'MF': u'Saint Martin (French part)', u'MA': u'Morocco', u'MC': u'Monaco', u'UZ': u'Uzbekistan', u'MM': u'Myanmar', u'ML': u'Mali', u'MO': u'Macao', u'MN': u'Mongolia', u'MH': u'Marshall Islands', u'MK': u'Macedonia, Republic of', u'MU': u'Mauritius', u'MT': u'Malta', u'MW': u'Malawi', u'MV': u'Maldives', u'MQ': u'Martinique', u'MP': u'Northern Mariana Islands', u'MS': u'Montserrat', u'MR': u'Mauritania', u'IM': u'Isle of Man', u'UG': u'Uganda', u'TZ': u'Tanzania, United Republic of', u'MY': u'Malaysia', u'MX': u'Mexico', u'IL': u'Israel', u'FR': u'France', u'AW': u'Aruba', u'SH': u'Saint Helena, Ascension and Tristan da Cunha', u'SJ': u'Svalbard and Jan Mayen', u'FI': u'Finland', u'FJ': u'Fiji', u'FK': u'Falkland Islands (Malvinas)', u'FM': u'Micronesia, Federated States of', u'FO': u'Faroe Islands', u'NI': u'Nicaragua', u'NL': u'Netherlands', u'NO': u'Norway', u'NA': u'Namibia', u'VU': u'Vanuatu', u'NC': u'New Caledonia', u'NE': u'Niger', u'NF': u'Norfolk Island', u'NG': u'Nigeria', u'NZ': u'New Zealand', u'NP': u'Nepal', u'NR': u'Nauru', u'NU': u'Niue', u'CK': u'Cook Islands', u'CI': u"C\xf4te d'Ivoire", u'CH': u'Switzerland', u'CO': u'Colombia', u'CN': u'China', u'CM': u'Cameroon', u'CL': u'Chile', u'CC': u'Cocos (Keeling) Islands', u'CA': u'Canada', u'CG': u'Congo', u'CF': u'Central African Republic', u'CD': u'Congo, The Democratic Republic of the', u'CZ': u'Czech Republic', u'CY': u'Cyprus', u'CX': u'Christmas Island', u'CR': u'Costa Rica', u'CW': u'Cura\xe7ao', u'CV': u'Cape Verde', u'CU': u'Cuba', u'SZ': u'Swaziland', u'SY': u'Syrian Arab Republic', u'SX': u'Sint Maarten (Dutch part)', u'KG': u'Kyrgyzstan', u'KE': u'Kenya', u'SS': u'South Sudan', u'SR': u'Suriname', u'KI': u'Kiribati', u'KH': u'Cambodia', u'SV': u'El Salvador', u'KM': u'Comoros', u'ST': u'Sao Tome and Principe', u'SK': u'Slovakia', u'KR': u'Korea, Republic of', u'SI': u'Slovenia', u'KP': u"Korea, Democratic People's Republic of", u'KW': u'Kuwait', u'SN': u'Senegal', u'SM': u'San Marino', u'SL': u'Sierra Leone', u'SC': u'Seychelles', u'KZ': u'Kazakhstan', u'KY': u'Cayman Islands', u'SG': u'Singapore', u'SE': u'Sweden', u'SD': u'Sudan', u'DO': u'Dominican Republic', u'DM': u'Dominica', u'DJ': u'Djibouti', u'DK': u'Denmark', u'DE': u'Germany', u'YE': u'Yemen', u'DZ': u'Algeria', u'US': u'United States', u'UY': u'Uruguay', u'YT': u'Mayotte', u'UM': u'United States Minor Outlying Islands', u'LB': u'Lebanon', u'LC': u'Saint Lucia', u'LA': u"Lao People's Democratic Republic", u'TV': u'Tuvalu', u'TW': u'Taiwan, Province of China', u'TT': u'Trinidad and Tobago', u'TR': u'Turkey', u'LK': u'Sri Lanka', u'LI': u'Liechtenstein', u'LV': u'Latvia', u'TO': u'Tonga', u'LT': u'Lithuania', u'LU': u'Luxembourg', u'LR': u'Liberia', u'LS': u'Lesotho', u'TH': u'Thailand', u'TF': u'French Southern Territories', u'TG': u'Togo', u'TD': u'Chad', u'TC': u'Turks and Caicos Islands', u'LY': u'Libya', u'VA': u'Holy See (Vatican City State)', u'VC': u'Saint Vincent and the Grenadines', u'AE': u'United Arab Emirates', u'AD': u'Andorra', u'AG': u'Antigua and Barbuda', u'AF': u'Afghanistan', u'AI': u'Anguilla', u'IS': u'Iceland', u'IR': u'Iran, Islamic Republic of', u'AM': u'Armenia', u'AL': u'Albania', u'AO': u'Angola', u'AQ': u'Antarctica', u'AS': u'American Samoa', u'AR': u'Argentina', u'AU': u'Australia', u'AT': u'Austria', u'IO': u'British Indian Ocean Territory', u'IN': u'India', u'AX': u'\xc5land Islands', u'AZ': u'Azerbaijan', u'IE': u'Ireland', u'ID': u'Indonesia', u'UA': u'Ukraine', u'QA': u'Qatar', u'MZ': u'Mozambique', u'FX': u'France, Metropolitan', u'AN': u'Netherlands Antilles', u'A1': u'Anguilla'}
country['CountryName'] = country['Country'].map(countryCode2Name)

# Drop NAs
country.dropna(inplace=True)

country['CumulativePercentage'] = 100 * country.UserCount.cumsum()/country.UserCount.sum()
country.reset_index(drop=True, inplace=True)
country[['CountryName', 'UserCount', 'CumulativePercentage']].head(10)


# Show data on the world map

# In[ ]:


import plotly.offline as py
py.offline.init_notebook_mode()

data = [ dict(
        type = 'choropleth',
        locations = country['CountryName'],
        z = country['UserCount'],
        locationmode = 'country names',
        text = country['CountryName'],
        colorscale = [[0,"rgb(153, 241, 243)"],[0.005,"rgb(16, 64, 143)"],[1,"rgb(0, 0, 0)"]],
        autocolorscale = False,
        marker = dict(
            line = dict(color = 'rgb(58,100,69)', width = 0.6)),
            colorbar = dict(autotick = True, tickprefix = '', title = '# of Users')
            )
       ]

layout = dict(
    title = 'Total number of users by country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
        type = 'equirectangular'
        ),
    margin = dict(b = 0, t = 0, l = 0, r = 0)
            )
    )

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap')


# As we can see more than 81% of all Users are from USA. Lets see distribution by States for USA.

# # Exploration by State for USA data

# Doing the same things, that I have done for countries

# In[ ]:


usa = page_views_sample_df.loc[page_views_sample_df.geo_location.str[:2] == 'US', :]
usa.columns = ['uuid', 'State']

usa.State = usa.State.str[3:5]

# Drop Data with missing state info
usa = usa.loc[usa.State != '', :]

usa.loc[:, 'UserCount'] = usa.groupby('State')['State'].transform('count')
usa.loc[:, ['State', 'UserCount']] = usa.loc[:, ['State', 'UserCount']].drop_duplicates('State', keep='first')
usa.sort_values('UserCount', ascending=False, inplace=True)


# In[ ]:


stateCode2Name = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AS': 'American Samoa', 'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'GU': 'Guam', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN':'Minnesota', 'MO': 'Missouri', 'MP': 'Northern Mariana Islands', 'MS': 'Mississippi', 'MT': 'Montana', 'NA': 'National', 'NC': 'North Carolina', 'ND': 'North Dakota', 'NE':'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'PR': 'Puerto Rico', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VI': 'Virgin Islands', 'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'}
usa['StateName'] = usa['State'].map(stateCode2Name)
# Drop NAs
usa.dropna(inplace=True)

usa['CumulativePercentage'] = 100 * usa.UserCount.cumsum()/usa.UserCount.sum()
usa.reset_index(drop=True, inplace=True)
usa[['StateName', 'UserCount', 'CumulativePercentage']].head(50)


# ## Show USA map

# In[ ]:


import plotly.offline as py
py.offline.init_notebook_mode()

data = [ dict(
        type = 'choropleth',
        locations = usa['State'],
        z = usa['UserCount'],
        locationmode = 'USA-states',
        text = usa['StateName'],
        colorscale = [[0,"rgb(153, 241, 243)"],[0.33,"rgb(16, 64, 143)"],[1,"rgb(0, 0, 0)"]],
        autocolorscale = False,
        marker = dict(
            line = dict(color = 'rgb(58,100,69)', width = 0.6)),
            colorbar = dict(autotick = True, tickprefix = '', title = '# of Users')
            )
       ]

layout = dict(
    title = 'Total number of users by state',
    geo = dict(
        scope='usa',
        projection=dict( type='albers usa' ),
        showlakes = True,
        lakecolor = 'rgb(255, 255, 255)'),
    )

fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='USmap')

