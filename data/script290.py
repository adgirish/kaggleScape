
# coding: utf-8

# **Loading the Data Science for Good: Kiva Crowdfunding Information**

# In[ ]:


import pandas as pd

datafiles = ['kiva_loans','loan_theme_ids','loan_themes_by_region','kiva_mpi_region_locations']
datafiles = {file: pd.read_csv(''.join(['../input/',file,'.csv'])) for file in datafiles}


# **Exploring: Size & Data Stats**

# In[ ]:


from IPython.display import display
for key in datafiles:
    print(key, len(datafiles[key]))
    display(datafiles[key].describe(include='all'))


# As mentioned in the Data page this is only a small snapshot of what's available through the API.

# In[ ]:


from IPython.display import HTML
head = """
<html><head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {'packages':['treemap']});
      google.charts.setOnLoadCallback(drawChart);
      function drawChart() {
        var data = google.visualization.arrayToDataTable([
          ['Region', 'Parent', 'Loan Amount'],
          ['Global', null, 0],"""

df = datafiles['kiva_loans'][['country','region','funded_amount']].copy()
df['country'] = df['country'].map(lambda x: str(x).replace("'",""))
df['region'] = df['region'].map(lambda x: str(x).replace("'",""))
Country = df.groupby(['country'], as_index=False)[['funded_amount']].sum()
Region = df.groupby(['region','country'], as_index=False)[['funded_amount']].sum()
Region = Region.groupby(['region'], as_index=False).first() #Remove similar Region names
Region = Region[~(Region['region'].isin(Country['country'].values))] #Remove similar Region names to Country
data = ''.join(["['" + str(c) + "','Global'," + str(int(a)) + "]," for c, a in Country.values])
data += ''.join(["['" + str(r) + "','" + str(c) + "'," + str(int(a)) + "]," for r, c, a in Region.values])
body = """
        ]);
        tree = new google.visualization.TreeMap(document.getElementById('chart_div'));
        tree.draw(data, {});
      }
    </script>
  </head>
  <body>
    <div id="chart_div" style="width: 800px; height: 300px;"></div>
  </body>
</html>
"""
HTML(head+data+body)

