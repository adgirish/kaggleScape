
# coding: utf-8

# # INTRODUCTION
# In this tutorial, we are going to learn basics of bokeh library. Bokeh is interactive visualization library.
# <br> I divide bokeh tutorial into 2 parts. Because kaggle has problem while running bokeh that cause crash in browser.
# 1. PART 1:
#     1. Basic Data Exploration with Pandas
#     1. Explanation of Bokeh Packages
#     1. Plotting with Glyphs
#     1. Additional Glyps
#     1. Data Formats
#     1. Customizing Glyphs
#     1. Layouts
#     1. Linking Plots
# 1. PART 2: https://www.kaggle.com/kanncaa1/interactive-bokeh-tutorial-part-2/editnb
#     1. Callbacks 
#         * Slider
#         * dropdowns
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ##  Basic Data Exploration with Pandas
# <br>Before everyting, we need to import data with read_csv().
# <br>In order to make visualization we need to understand data and look whether there are nan values in data or not.
# <br>We will use pandas library:
#     * read_csv()
#     * info()
#     * head()
# 

# In[ ]:


# As you can see from info method. There are 16598.
# However, Year has 16327 entries. That means Year has NAN value.
# Also Year should be integer but it is given as float. Therefore we will convert it.
# In addition, publisher has NAN values.
data = pd.read_csv("../input/vgsales.csv")
data.info()


# In[ ]:


# Lets start with dropping nan values
data.dropna(how="any",inplace = True)
data.info()


# In[ ]:


# Then convert data from float to int
data.Year = data.Year.astype(int)
data.head()     # head method always gives you overview of data.


# ## Explanation of Bokeh Packages
# For bokeh library, we will use some packages:
# * output_file: that save our figure with .html extension
# * show: show the figure
# * figure: creates empty figure
# * ColumnarDataSource: Data source of bokeh
# * HoverTool: like cursor
# * CategoricalColorMapper: Like a hue in seaborn. If you do not know it look at my seaborn tutorial
#     * https://www.kaggle.com/kanncaa1/seaborn-for-beginners
# * Row and column: puts plots in row order or column order in figure
# * gridplot
# * Tabs and Panel: Panel is figure for each plot and tab is like button
#     
# 

# In[ ]:


# bokeh packages
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel
output_notebook()


# ## Plotting with Glyphs
# * Glyphs: visual shapes like circle, square, rectangle or diamond
# * figure: creates figure
#     * x_axis_label: label of x axis
#     * y_axis_label: label of y axis
#     * tools: tools to move or zoom plot
#         * pan: slides the plot
#         * box_zoom: zoom in 
# * circle: like scatter in matplotlib
#     * size: size of circles
#     * color: color
#     * alpha: opacity
# * output_file:  that save our figure with .html extension
# * show: show the figure
# 

# In[ ]:


plot = figure(x_axis_label = "x",y_axis_label = "y",tools = "pan,box_zoom")
plot.circle(x=[5,4,3,2,1],y=[1,2,3,4,5],size = 10,color = "black",alpha = 0.7)
output_file("my_first_bokeh_plot.html")
show(plot)


# In[ ]:


# There are other types of glyphs
plot = figure()
plot.diamond(x=[5,4,3,2,1],y=[1,2,3,4,5],size = 10,color = "black",alpha = 0.7)
plot.cross(x=[1,2,3,4,5],y=[1,2,3,4,5],size = 10,color = "red",alpha = 0.7)
show(plot)


# ##  Additional Glyps
# * line: line plot
#     * line_width: width of line
#     * fill_color: filling inside of circle with color
# * patches: multiple polynomial shapes at once on a plot
#     * fill_color: filling inside of patches
#     * line_color: color of line  around patches
#  

# In[ ]:


# line
plot = figure()
plot.line(x=[1,2,3,4,5,6,7],y = [1,2,3,4,5,5,5],line_width = 2)
plot.circle(x=[1,2,3,4,5,6,7],y = [1,2,3,4,5,5,5],fill_color = "white",size = 10)
show(plot)


# In[ ]:


# patches
plot = figure()
plot.patches(xs = [[1,1,2,2],[2,2,3,3]],ys = [[1,2,1,2],[1,2,1,2]],fill_color = ["purple","red"],line_color = ["black","black"])
#show(plot)


# ## Data Formats
# Bokeh can use list, numpy arrays and pandas as a data source. We have pandas data frame in this tutorial.

# In[ ]:


# Lets use source in a basic example
# As you know from info() method we have Year and  Global_Sales columns
# Lets plot it to learn how to use ColumnDataSource
source = ColumnDataSource(data)
plot = figure()
plot.circle(x="Year",y="Global_Sales",source = source)
show(plot)
# If you remember our column names are "Year" and "Global_Sales" in pandas data frame.
# Nothing change when we convert pandas data frame to source.
# You can think source is like pandas data frame at this point. Only for now :)


# ## Customizing Glyphs
# * Selection appearance: when you select some point on data, that points shine and others burn out
#     * tools:
#         * box_select and lasso_select: selection tools
#     * selection_color: When you select point, it becomes selected color
#     * nonselection_fill_alpha: Other non selected points become non selected alpha
#     * nonselection_fill_color: Other non selected points become non selected color
# * HoverTool: cursor
#     * Crosshair: line cursor
#     * hover_color: Color of hover
# * Color mapping: color map of chose field. (like hue in seaborn)
#     * factors: names of variable to color map
#     * palette: color of chose factors
# 
# 

# In[ ]:


# Selection appearance
plot = figure(tools="box_select,lasso_select")
plot.circle(x= "Year",y = "Global_Sales",source=source,color = "black",
            selection_color = "orange",
            nonselection_fill_alpha = 0.2,
           nonselection_fill_color = "blue")
show(plot)


# In[ ]:


# Hover appearance
hover = HoverTool(tooltips = [("Genre of game","@Genre"),("Publisher of game","@Publisher")], mode="hline")
plot = figure(tools=[hover,"crosshair"])
plot.circle(x= "Year",y = "Global_Sales",source=source,color ="black",hover_color ="red")
show(plot)


# In[ ]:


# Color mapping
factors = list(data.Genre.unique()) # what we want to color map. I choose genre of games
colors = ["red","green","blue","black","orange","brown","grey","purple","yellow","white","pink","peru"]
mapper = CategoricalColorMapper(factors = factors,palette = colors)
plot =figure()
plot.circle(x= "Year",y = "Global_Sales",source=source,color = {"field":"Genre","transform":mapper})
show(plot)
# plot looks like confusing but I think you got the idea of mapping 


# ## Layouts
# Arranging multiple plots like subplot in matplot library.
# * Row and columns: puts plots in row order or column order in figure
# * Grid arrangement: list of order for layout
#     * toolbar_location: location of tool bar that can be below above left right or none
# * Tabbed layout
#     * Panel: like a figure
#     * Tabs: like a button 
# 

# In[ ]:


# Row and column
p1 = figure()
p1.circle(x = "Year",y= "Global_Sales",source = source,color="red")
p2 = figure()
p2.circle(x = "Year",y= "EU_Sales",source = source,color="black")
p3 = figure()
p3.circle(x = "Year",y= "NA_Sales",source = source,color="blue")
p4 = figure()
p4.circle(x = "Year",y= "JP_Sales",source = source,color="orange")
layout1 = row(p1,p2)
layout2 = row(p3,p4)
layout3= column(layout1,layout2)
show(layout3)


# In[ ]:


#nested
# I use p1, p2 and p3 that are created at above
layout = row(column(p1,p2),p3)
show(layout)


# In[ ]:


# Grid plot 
layout = gridplot([[p1,p2],[p3,None]],toolbar_location="above")
show(layout)


# In[ ]:


#Tabbed layout
#I use p1 and p2 that are created at above
tab1 = Panel(child = p1,title = "Global_Sales")
tab2 = Panel(child = p2,title = "EU_Sales")
tabs = Tabs(tabs=[tab1,tab2])
show(tabs)


# ## Linking Plots
# Links plot together. For example, there are two plots and we zoom in one of them. Other one is zoomed automatically. 

# In[ ]:


# linking axis
# We will use p1 and p2 that are created at above
p2.x_range = p1.x_range
p2.y_range = p1.y_range
layout4=column(p1,p2)
show(layout4)


# # CONCLUSION
# If you like the bokeh library, I am going to dive deep into bokeh.
# <br> I divide bokeh tutorial into 2 parts. Because kaggle has problem while running bokeh that cause crash in browser.
# <br> Also look at Part 2: https://www.kaggle.com/kanncaa1/interactive-bokeh-tutorial-part-2/editnb
# ### If you have any question, I am happy to hear it. I thank c who is developer of Bokeh for this useful visualization library.
# 
