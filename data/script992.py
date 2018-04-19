
# coding: utf-8

# # What to expect after graduation?
# ***

# ***Piotr Skalski - 05.11.2017***

# ## Table of Contents
# ***
# * [1. Introduction](#introduction) <br>
# * [2. Importing dataset and data preprocessing](#importing_dataset_and_data_preprocessing) <br>
#    * [2.1. Importing essential libraries](#importing_essential_libraries) <br>
#    * [2.2. Importing datasets](#importing_datasets) <br>
#    * [2.3. Let's summarize the datasets](#lets_summarize_the_dataset) <br>
#    * [2.4. Data preprocessing](#data_preprocessing) <br>
# * [3. Data Visualization](#data_visualization) <br>
#    * [3.1. Starting median salary distribution by major](#starting_median_salary_distribution) <br>
#    * [3.2. So You Want To Be A Lawyer?](#want_to_be_lawyer) <br>
#    * [3.3. Time makes all the difference](#time_makes_all_the_difference) <br>
#    * [3.4. Long term investment](#long_term_investment) <br>   
#    * [3.5. Biggest money makers](#biggest_money_makers) <br>   
# ***

# ## 1. Introduction
# <a id="introduction"></a>

# The <a href="https://www.kaggle.com/wsj/college-salaries">Where it Pays to Attend College</a> dataset is not new - it was published more than half a year ago. Unfortunately I was not a member of Kaggle back then. As I am a student myself, this problem is quite interesting to me, so I decided to check which majors offer the greatest prospects.

# ## 2. Importing dataset and data preprocessing
# <a id="importing_dataset_and_data_preprocessing"></a>

# ### 2.1. Importing essential libraries
# <a id="importing_essential_libraries"></a>

# In[ ]:


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as colors
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# ### 2.2. Importing datasets
# <a id="importing_datasets"></a>

# In[ ]:


college = pd.read_csv('../input/salaries-by-college-type.csv')
region = pd.read_csv('../input/salaries-by-region.csv')
majors = pd.read_csv('../input/degrees-that-pay-back.csv')
# List of all datasets
datasets_list = [college, region, majors]


# ### 2.3. Let's summarize the datasets
# <a id="lets_summarize_the_dataset"></a>

# In[ ]:


college.head()


# In[ ]:


college.info()


# In[ ]:


region.head()


# In[ ]:


region.info()


# In[ ]:


majors.head()


# In[ ]:


majors.info()


# <b>NOTE:</b> Right away we see two things: First, the data contained in the set are incomplete. Secondly, most of the data was loaded in type that is useless to us. Let's conduct simple data preprocessing to fix second problem.

# ### 2.4. Data preprocessing
# <a id="data_preprocessing"></a>

# <b>NOTE:</b> Let's start by renaming the columns. This makes it much easier to work with a datasets.

# In[ ]:


college_columns = {
    "School Name" : "name",
    "School Type" : "type",
    "Starting Median Salary" : "start_p50",
    "Mid-Career Median Salary" : "mid_p50",
    "Mid-Career 10th Percentile Salary" : "mid_p10",
    "Mid-Career 25th Percentile Salary" : "mid_p25",
    "Mid-Career 75th Percentile Salary" : "mid_p75",
    "Mid-Career 90th Percentile Salary" : "mid_p90"
}

college.rename(columns=college_columns, inplace=True)

region_columns = {
    "School Name" : "name",
    "Region" : "region",
    "Starting Median Salary" : "start_p50",
    "Mid-Career Median Salary" : "mid_p50",
    "Mid-Career 10th Percentile Salary" : "mid_p10",
    "Mid-Career 25th Percentile Salary" : "mid_p25",
    "Mid-Career 75th Percentile Salary" : "mid_p75",
    "Mid-Career 90th Percentile Salary" : "mid_p90"
}

region.rename(columns=region_columns, inplace=True)

majors_columns = {
    "Undergraduate Major" : "name",
    "Starting Median Salary" : "start_p50",
    "Mid-Career Median Salary" : "mid_p50",
    "Percent change from Starting to Mid-Career Salary" : "increase",
    "Mid-Career 10th Percentile Salary" : "mid_p10",
    "Mid-Career 25th Percentile Salary" : "mid_p25",
    "Mid-Career 75th Percentile Salary" : "mid_p75",
    "Mid-Career 90th Percentile Salary" : "mid_p90"
}

majors.rename(columns=majors_columns, inplace=True)


# <b>NOTE:</b> Let's use a dedicated function to convert the selected columns to numeric values.

# In[ ]:


selected_columns = ["start_p50", "mid_p50", "mid_p10", "mid_p25", "mid_p75", "mid_p90"]

for dataset in datasets_list:
    for column in selected_columns:
        dataset[column] = dataset[column].str.replace("$","")
        dataset[column] = dataset[column].str.replace(",","")
        dataset[column] = pd.to_numeric(dataset[column])


# ## 3. Data Visualization
# <a id="data_visualization"></a>

# ### 3.1. Starting median salary distribution by university
# <a id="starting_median_salary_distribution"></a>

# In[ ]:


short_term = college.sort_values("start_p50", ascending=False)
values = short_term["start_p50"].tolist()
ind = np.arange(len(values))


# Creating new plot
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax.yaxis.grid()
ax.xaxis.grid()
bars = ax.bar(ind, values)

for i, b in enumerate(bars):
    b.set_color(plt.cm.summer(1. * i / (len(values) - 1)))
    
plt.ylabel('Starting Median Salary [$]', fontsize=20)
plt.xlabel('Distribution of mean starting salary by university', fontsize=20)
plt.title('Instant income? Not for everyone.', fontsize=35, fontweight='bold')
plt.xticks(np.arange(0, len(ind), (len(ind)-1)/5), [0, 20, 40, 60, 80, 100])

plt.show()


# ### 3.2. So You Want To Be A Lawyer?
# <a id="want_to_be_lawyer"></a>

# In[ ]:


group_by_type = college.groupby("type")

x_data = []
y_data = []

colors = ['rgba(93, 164, 214, 0.5)', 
          'rgba(255, 144, 14, 0.5)', 
          'rgba(44, 160, 101, 0.5)', 
          'rgba(255, 65, 54, 0.5)', 
          'rgba(207, 114, 255, 0.5)']

for uni_type, uni_group in group_by_type:
    x_data.append(uni_type)
    y_data.append(uni_group["mid_p50"])

traces = []

for xd, yd, cls in zip(x_data, y_data, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=cls,
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
        ))

layout = go.Layout(
    title='So You Want To Be A Lawyer?',
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)


# ### 3.3. Time makes all the difference
# <a id="time_makes_all_the_difference"></a>

# In[ ]:


hist_data = [college["start_p50"].values, college["mid_p50"].values]

group_labels = ['Starting Median Salary', 'Mid-Career Median Salary']
colors = ['#A6ACEC', '#63F5EF']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                         bin_size=2000, show_rug=False)

# Add title
fig['layout'].update(title='Humble beginnings', legend=dict(x=0.65, y=0.8))

# Plot!
py.iplot(fig, filename='Hist and Curve')


# ### 3.4. Long term investment
# <a id="long_term_investment"></a>

# In[ ]:


majors_sort = majors.sort_values("mid_p50", ascending=False).head(20)

def cut_name(x):
    if len(x) <= 18:
        return x
    else:
        return x[0:15] + "..."

trace1 = go.Bar(
    x = majors_sort["name"].apply(cut_name).tolist(),
    y = majors_sort["start_p50"].tolist(),
    name='Starting',
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    )
)
trace2 = go.Bar(
    x = majors_sort["name"].apply(cut_name).tolist(),
    y = majors_sort["mid_p50"].tolist(),
    name='Mid-Career',
    marker=dict(
        color='rgba(219, 64, 82, 0.7)',
        line=dict(
            color='rgba(219, 64, 82, 1.0)',
            width=2,
        )
    )
)

trace3 = go.Scatter(
    x = majors_sort["name"].apply(cut_name).tolist(),
    y = majors_sort["increase"].tolist(),
    name='Percent change',
    mode = 'markers',
    marker=dict(
        symbol="hexagon-dot",
        size=15
    ),
    yaxis='y2'
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    barmode='group',
    title = 'Sometimes you have to wait for fruits to work',
    width=850,
    height=500,
    margin=go.Margin(
        l=75,
        r=75,
        b=120,
        t=80,
        pad=10
    ),
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    yaxis = dict(
        title= 'Median Salary [$]',
        anchor = 'x',
        rangemode='tozero'
    ),   
    yaxis2=dict(
        title='Change [%]',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right',
        anchor = 'x',
        rangemode = 'tozero',
        dtick = 19.95
    ),
    #legend=dict(x=-.1, y=1.2)
    legend=dict(x=0.1, y=0.05)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 3.5. Biggest money makers
# <a id="biggest_money_makers"></a>

# In[ ]:


# Sorted dataset
majors_sort_mid90 = majors.sort_values("mid_p90", ascending=True)

# Method that shortens long texts
def cut_name(x):
    if len(x) <= 25:
        return x
    else:
        return x[0:22] + "..."

# Prepared information
traces = {
    "mid_p10" : {
        "name" : "Mid-Career 10th Percentile",
        "color" : "rgba(255, 114, 114, 0.7)",
        "line_color" : "rgba(255, 114, 114, 1.0)"
    },
    "mid_p25" : {
        "name" : "Mid-Career 25th Percentile",
        "color" : "rgba(255, 202, 120, 0.7)",
        "line_color" : "rgba(255, 202, 120, 1.0)"
    },
    "mid_p50" : {
        "name" : "Mid-Career 50th Percentile",
        "color" : "rgba(253, 255, 88, 0.7)",
        "line_color" : "rgba(253, 255, 88, 1.0)"
    },
    "mid_p75" : {
        "name" : "Mid-Career 75th Percentile",
        "color" : "rgba(153, 255, 45, 0.7)",
        "line_color" : "rgba(153, 255, 45, 1.0)"
    },
    "mid_p90" : {
        "name" : "Mid-Career 90th Percentile",
        "color" : "rgba(49, 255, 220, 0.7)",
        "line_color" : "rgba(49, 255, 220, 1.0)"
    }
}

# List that stores information about data traces
data = []

# Single trace 
for key, value in traces.items():
    
    trace = go.Scatter(
        x = majors_sort_mid90[key].tolist(),
        y = majors_sort_mid90["name"].apply(cut_name).tolist(),
        name = value["name"],
        mode = 'markers',
        marker=dict(
            color = value["color"],
            line=dict(
                color = value["line_color"],
                width=2,
            ),
            symbol="hexagon-dot",
            size=10
        ),
    )
    data.append(trace)

# Chart layout
layout = go.Layout(
    title = 'Biggest money makers',
    width=850,
    height=1200,
    margin=go.Margin(
        l=180,
        r=50,
        b=80,
        t=80,
        pad=10
    ),
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    yaxis = dict(
        anchor = 'x',
        rangemode='tozero',
        tickfont=dict(
            size=10
        ),
        ticklen=1
    ),   
    legend=dict(x=0.6, y=0.07)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

