
# coding: utf-8

# # Kiva Crowdfunding Starter Kernel
# 
# ---
# 
# Here's a little starter kernel to get you going with the Kiva Crowdfunding challenge. I've taken the liberty to import Kiva's data and also some additional resources from the Oxford Poverty & Human Development Initiative (OPHI) that I think might be helpful. 

# In[ ]:


# Import libraries
import pandas as pd

# Load data from Kiva
loan_data = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')
theme_regions = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')
mpi_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')

# Load additional resources from OPHI data
mpi_national = pd.read_csv('../input/mpi/MPI_national.csv')
mpi_subnational = pd.read_csv('../input/mpi/MPI_subnational.csv')


# ### Convert column names to all lower case
# When I import data from multiple sources, I find it useful to convert column names to lower case:

# In[ ]:


# Convert all column names to lower case
loan_data.columns = [col.lower() for col in loan_data.columns]
theme_ids.columns = [col.lower() for col in theme_ids.columns]
theme_regions = [col.lower() for col in theme_regions.columns]
mpi_region.columns = [col.lower() for col in mpi_region.columns]

# Convert OPHI columns to lower case
mpi_national.columns = [col.lower() for col in mpi_national.columns]
mpi_subnational.columns = [col.lower() for col in mpi_subnational.columns]


# # Piechart: Top 10 Kiva loan uses in Uganda

# In[ ]:


# Importing plotly for offline use 
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()


# Subset the loan theme types in Uganada
loan_types_in_uganda = loan_data['use'][loan_data['country'] == 'Uganda']

# Calculate percentages 
percentages = round(loan_types_in_uganda.value_counts() / len(loan_types_in_uganda) * 100, 2)[:10]

# Plotly piechart
plotly_config = {
    'modeBarButtonsToRemove': ['sendDataToCloud']
}

trace = go.Pie(
    labels=percentages.keys(),
    values=percentages.values,
    hoverinfo='label+percent', textinfo='percent', 
               textfont=dict(size=15, color='#000000'),
               marker=dict(line=dict(color='#000000', width=1))
)

data = [trace]
layout = go.Layout(
    width=700,
    height=700,
    title='Top 10 loan uses in Uganda',
    titlefont= dict(size=20),
    showlegend=True,
    legend=dict(x=0.1,y=-5),
)

fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, show_link=False, config=plotly_config)


# ## Choropleth: Average Dollar Value of Kiva Loans

# In[ ]:


dv = round(loan_data.groupby('country').mean()['loan_amount'], 2)

data = [dict(
    type='choropleth',
    locationmode='country names',
    locations=dv.index,
    z=dv.values,
    colorscale='Blues',
    reversescale=True,
    colorbar=dict(title='Dollar value'), 
    zauto=True,
    
)]

layout = dict(
    title='Average Dollar Value of Kiva Loans',
    geo=dict(
        resolution=110,
        showframe=True,
        showcoastlines=True,
        showcountries=True,
        showland=True,
        landcolor='#F5F5F5'
    )
)
figure = dict(data=data, layout=layout)
offline.iplot(figure, show_link=False)

