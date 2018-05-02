
# coding: utf-8

# # INTRODUCTION
# * In this kernel, we will learn how to use plotly library.
#     * Plotly library: Plotly's Python graphing library makes interactive, publication-quality graphs online. Examples of how to make line plots, scatter plots, area charts, bar charts, error bars, box plots, histograms, heatmaps, subplots, multiple-axes, polar charts, and bubble charts.
# 
# <br>Content:
# 1. [Loading Data and Explanation of Features](#1)
# 1. [Line Charts](#2)
# 1. [Scatter Charts](#3)
# 1. [Bar Charts](#4)
# 1. [Pie Charts](#5)
# 1. [Bubble Charts](#6)
# 1. [Histogram](#7)
# 1. [Word Cloud](#8)
# 1. [Box Plot](#9)
# 1. [Scatter Plot Matrix](#10)
# 1. Map Plots: https://www.kaggle.com/kanncaa1/time-series-prediction-with-eda-of-world-war-2
# 
# 
# 

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import plotly.graph_objs as go
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> <br>
# # Loading Data and Explanation of Features
# <font color='red'>
# * timesData includes 14 features that are:
#     <font color='black'>
#     * world_rank             
#     * university_name       
#     * country               
#     * teaching                
#     * international            
#     * research                 
#     * citations                
#     * income                   
#     * total_score              
#     * num_students             
#     * student_staff_ratio      
#     * international_students   
#     * female_male_ratio        
#     * year 

# In[ ]:


# Load data that we will use.
timesData = pd.read_csv("../input/timesData.csv")


# In[ ]:


# information about timesData
timesData.info()


# In[ ]:


timesData.head(10)


# <a id="2"></a> <br>
# # Line Charts
# <font color='red'>
# Line Charts Example: Citation and Teaching vs World Rank of Top 100 Universities
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * title = title of layout
#     * x axis = it is dictionary
#         * title = label of x axis
#         * ticklen = length of x axis ticks
#         * zeroline = showing zero line or not
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frame
df = timesData.iloc[:100,:]
# import graph objects as "go"
import plotly.graph_objs as go
# Creating trace1
trace1 = go.Scatter(
                    x = df.world_rank,
                    y = df.citations,
                    mode = "lines",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= df.university_name)
# Creating trace2
trace2 = go.Scatter(
                    x = df.world_rank,
                    y = df.teaching,
                    mode = "lines+markers",
                    name = "teaching",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.university_name)
data = [trace1, trace2]
layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id="3"></a> <br>
# # Scatter
# <font color='red'>
# Scatter Example: Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * title = title of layout
#     * x axis = it is dictionary
#         * title = label of x axis
#         * ticklen = length of x axis ticks
#         * zeroline = showing zero line or not
#     * y axis = it is dictionary and same with x axis
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frames
df2014 = timesData[timesData.year == 2014].iloc[:100,:]
df2015 = timesData[timesData.year == 2015].iloc[:100,:]
df2016 = timesData[timesData.year == 2016].iloc[:100,:]
# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2014.world_rank,
                    y = df2014.citations,
                    mode = "markers",
                    name = "2014",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2014.university_name)
# creating trace2
trace2 =go.Scatter(
                    x = df2015.world_rank,
                    y = df2015.citations,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2015.university_name)
# creating trace3
trace3 =go.Scatter(
                    x = df2016.world_rank,
                    y = df2016.citations,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2016.university_name)
data = [trace1, trace2, trace3]
layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id="4"></a> <br>
# # Bar Charts
# <font color='red'>
# First Bar Charts Example: citations and teaching of top 3 universities in 2014 (style1)
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * mode = type of plot like marker, line or line + markers
#     * name = name of the plots
#     * marker = marker is used with dictionary. 
#         * color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
#         * line = It is dictionary. line between bars
#             * color = line color around bars
#     * text = The hover text (hover is curser)
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * barmode = bar mode of bars like grouped
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frames
df2014 = timesData[timesData.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = df2014.university_name,
                y = df2014.citations,
                name = "citations",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
# create trace2 
trace2 = go.Bar(
                x = df2014.university_name,
                y = df2014.teaching,
                name = "teaching",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2014.country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <font color='red'>
# Second Bar Charts Example: citations and teaching of top 3 universities in 2014 (style2)
# <br> Actually, if you change only the barmode from *group* to *relative* in previous example, you achieve what we did here. However, for diversity I use different syntaxes. 
# <font color='black'>
# * Import graph_objs as *go*
# * Creating traces
#     * x = x axis
#     * y = y axis
#     * name = name of the plots
#     * type = type of plot like bar plot
# * data = is a list that we add traces into it
# * layout = it is dictionary.
#     * xaxis = label of x axis
#     * barmode = bar mode of bars like grouped( previous example) or relative
#     * title = title of layout
# * fig = it includes data and layout
# * iplot() = plots the figure(fig) that is created by data and layout

# In[ ]:


# prepare data frames
df2014 = timesData[timesData.year == 2014].iloc[:3,:]
# import graph objects as "go"
import plotly.graph_objs as go
x = df2014.university_name

trace1 = {
  'x': x,
  'y': df2014.citations,
  'name': 'citation',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': df2014.teaching,
  'name': 'teaching',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 3 universities'},
  'barmode': 'relative',
  'title': 'citations and teaching of top 3 universities in 2014'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# <font color='red'>
# Third Bar Charts Example: Horizontal bar charts.  (style3) Citation vs income for universities
# <font color='black'>
# * Import graph_objs as *go* and importing tools
#     *  Tools: used for subplots
# * Creating trace1 
#     * bar: bar plot
#         * x = x axis 
#         * y = y axis
#         * marker
#             * color: color of bars
#             * line: bar line color and width
#         * name: name of bar
#         * orientation: orientation like horizontal 
#    * creating trace2
#        * scatter: scatter plot
#            * x = x axis 
#             * y = y axis
#             * mode: scatter type line line + markers or only markers
#             * line: properties of line
#                 * color: color of line
#             * name: name of scatter plot
#     * layout: axis, legend, margin, paper and plot properties
#         * 

# In[ ]:


# import graph objects as "go" and import tools
import plotly.graph_objs as go
from plotly import tools

# prepare data frames
df2016 = timesData[timesData.year == 2016].iloc[:7,:]

y_saving = [each for each in df2016.research]
y_net_worth  = [float(each) for each in df2016.income]
x_saving = [each for each in df2016.university_name]
x_net_worth  = [each for each in df2016.university_name]
trace0 = go.Bar(
                x=y_saving,
                y=x_saving,
                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),
                name='research',
                orientation='h',
)
trace1 = go.Scatter(
                x=y_net_worth,
                y=x_net_worth,
                mode='lines+markers',
                line=dict(color='rgb(63, 72, 204)'),
                name='income',
)
layout = dict(
                title='Citations and income',
                yaxis1=dict(showticklabels=True,domain=[0, 0.85]),
                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),
                xaxis1=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),
                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),
                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),
                margin=dict(l=200, r=20,t=70,b=70),
                paper_bgcolor='rgb(248, 248, 255)',
                plot_bgcolor='rgb(248, 248, 255)',
)
annotations = []
y_s = np.round(y_saving, decimals=2)
y_nw = np.rint(y_net_worth)
# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, x_saving):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))

layout['annotations'] = annotations

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(layout)
iplot(fig)


# <a id="5"></a> <br>
# # Pie Charts
# <font color='red'>
# Pie Charts Example: Horizontal bar charts.  (style3) Citation vs income for universities
# <font color='black'>
# * fig: create figures
#     * data: plot type
#         * values: values of plot
#         * labels: labels of plot
#         * name: name of plots
#         * hoverinfo: information in hover
#         * hole: hole width
#         * type: plot type like pie
#     * layout: layout of plot
#         * title: title of layout
#         * annotations: font, showarrow, text, x, y

# In[ ]:


# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:7,:]
pie1 = df2016.num_students
pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]
labels = df2016.university_name
# figure
fig = {
  "data": [
    {
      "values": pie1_list,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number Of Students Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Universities Number of Students rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Students",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)


# <a id="6"></a> <br>
# # Bubble Charts
# <font color='red'>
# Bubble Charts Example: University world rank vs teaching score with number of students(size) and international score (color)
# <font color='black'>
# * x = x axis
# * y = y axis
# * mode = markers(scatter)
# *  marker = marker properties
#     * color = third dimension of plot. Internaltional score
#     * size = fourth dimension of plot. Number of students
# * text: university names

# In[ ]:


# data preparation
df2016 = timesData[timesData.year == 2016].iloc[:20,:]
num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]
international_color = df2016.international
data = [
    {
        'y':  df2016.teaching,
        'x': df2016.world_rank,
        'mode': 'markers',
        'marker': {
            'color': international_color,
            'size': num_students_size,
            'showscale': True
        },
        "text" :  df2016.university_name    
    }
]
iplot(data)


# <a id="7"></a> <br>
# # Histogram
# <font color='red'>
# Lets look at histogram of students-staff ratio in 2011 and 2012 years. 
#     <font color='black'>
# * trace1 = first histogram
#     * x = x axis
#     * y = y axis
#     * opacity = opacity of histogram
#     * name = name of legend
#     * marker = color of histogram
# * trace2 = second histogram
# * layout = layout 
#     * barmode = mode of histogram like overlay. Also you can change it with *stack*

# In[ ]:


# prepare data
x2011 = timesData.student_staff_ratio[timesData.year == 2011]
x2012 = timesData.student_staff_ratio[timesData.year == 2012]

trace1 = go.Histogram(
    x=x2011,
    opacity=0.75,
    name = "2011",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=x2012,
    opacity=0.75,
    name = "2012",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title=' students-staff ratio in 2011 and 2012',
                   xaxis=dict(title='students-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)

iplot(fig)


# <a id="8"></a> <br>
# # Word Cloud
# Not a pyplot but learning it is good for visualization. Lets look at which country is mentioned most in 2011.
# * WordCloud = word cloud library that I import at the beginning of kernel
#     * background_color = color of back ground
#     * generate = generates the country name list(x2011) a word cloud

# In[ ]:


# data prepararion
x2011 = timesData.country[timesData.year == 2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(x2011))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# <a id="9"></a> <br>
# # Box Plots
# <font color='red'>
# * Box Plots
#     * Median (50th percentile) = middle value of the data set. Sort and take the data in the middle. It is also called 50% percentile that is 50% of data are less that median(50th quartile)(quartile)
#         * 25th percentile = quartile 1 (Q1) that is lower quartile
#         * 75th percentile = quartile 3 (Q3) that is higer quartile
#         * height of box = IQR = interquartile range = Q3-Q1
#         * Whiskers = 2 * IQR from the median
#         * Outliers = being more than 2*IQR away from median commonly.
#         
#     <font color='black'>
#     * trace = box
#         * y = data we want to visualize with box plot 
#         * marker = color

# In[ ]:


# data preparation
x2015 = timesData[timesData.year == 2015]

trace0 = go.Box(
    y=x2015.total_score,
    name = 'total score of universities in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=x2015.research,
    name = 'research of universities in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)


# <a id="10"></a> <br>
# # Scatter Matrix Plots
# <font color='red'>
# Scatter Matrix = it helps us to see covariance and relation between more than 2 features
# <font color='black'>
# * import figure factory as ff
# * create_scatterplotmatrix = creates scatter plot
#     * data2015 = prepared data. It includes research, international and total scores with index from 1 to 401
#     * colormap = color map of scatter plot
#     * colormap_type = color type of scatter plot
#     * height and weight

# In[ ]:


# import figure factory
import plotly.figure_factory as ff
# prepare data
dataframe = timesData[timesData.year == 2015]
data2015 = dataframe.loc[:,["research","international", "total_score"]]
data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)


# # Conclusion
# * If you like it, thank you for you upvotes.
# * If you have any question, I will happy to hear it
# ## To be continued
