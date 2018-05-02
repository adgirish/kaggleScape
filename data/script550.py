
# coding: utf-8

# Quick Introduction to this Dataset
# ==================================
# 
# **Simple but informative** -- that's the goal of this dataset.  
# 
# This kernel will demonstrate the following:
# 
#  - Loading the data
#  - Google maps
#  - Pivot table
#  - Simple graphs
#  - Percent change
#  - Seaborn heatmap

# 
# 
# Loading Data
# ------------
# 

# In[ ]:


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

# Read data 
d=pd.read_csv("../input/911.csv",
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)


# Set index
d.index = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]


# In[ ]:


d.head()


# In[ ]:


# Title is the category of the call
d["title"].value_counts()


# ## Maps ##
# 
# If you're interested in making a Google map take a look at this  [kernel][1]
# 
# Here's the [code][2] to create this map. 
# 
# 
# ![Google Maps on Kaggle][3]
# 
# 
#   [1]: https://www.kaggle.com/mchirico/d/mchirico/montcoalert/map-of-helicopter-landings
#   [2]: https://www.kaggle.com/mchirico/d/mchirico/montcoalert/map-of-helicopter-landings/code
#   [3]: https://raw.githubusercontent.com/mchirico/mchirico.github.io/master/p/images/kaggleGoogleMap.png

# ## Working with the Data ##

# In[ ]:


# There are 3 groups -- EMS, Fire, Traffic
# We'll call these type.  This type is split on ':'
d['type'] = d["title"].apply(lambda x: x.split(':')[0])


# In[ ]:


d["type"].value_counts()


# 
# 
# Pivot Table
# -----------
# 

# In[ ]:


# Let's create a pivot table with just EMS
# It will be stored in a variable 'pp'
g=d[d['type'] == 'EMS' ]
p=pd.pivot_table(g, values='e', index=['timeStamp'], columns=['title'], aggfunc=np.sum)

# Resampling every week 'W'.  This is very powerful
pp=p.resample('W', how=[np.sum]).reset_index()
pp.head()


# In[ ]:


# That "sum" column is a pain...remove it

# Let's flatten the columns 
pp.columns = pp.columns.get_level_values(0)

pp.head()


# 
# 
# Graphs/Plots
# ------
# 

# In[ ]:


# Red dot with Line
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



ax.plot_date(pp['timeStamp'], pp['EMS: ASSAULT VICTIM'],'k')
ax.plot_date(pp['timeStamp'], pp['EMS: ASSAULT VICTIM'],'ro')


ax.set_title("EMS: ASSAULT VICTIM")
fig.autofmt_xdate()
plt.show()

# Note, you'll get a drop at the ends...not a complete week


# In[ ]:


# Remove the first and last row
pp = pp[pp['timeStamp'] < pp['timeStamp'].max()]
pp = pp[pp['timeStamp'] > pp['timeStamp'].min()]


# In[ ]:


# Get the best fitting line

# Need to import for legend
import matplotlib.lines as mlines

# For best fit line
from sklearn import linear_model

# Red dot with Line
fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  



ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 



# Build Linear Fit
Y = pp['EMS: ASSAULT VICTIM'].values.reshape(-1,1)
X=np.arange(Y.shape[0]).reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(X,Y)
m = model.coef_[0][0]
c = model.intercept_[0]
ax.plot(pp['timeStamp'],model.predict(X), color='blue',
         linewidth=2)
blue_line = mlines.Line2D([], [], color='blue', label='Linear Fit: y = %2.2fx + %2.2f' % (m,c))
ax.legend(handles=[blue_line], loc='best')


ax.plot_date(pp['timeStamp'], pp['EMS: ASSAULT VICTIM'],'k')
ax.plot_date(pp['timeStamp'], pp['EMS: ASSAULT VICTIM'],'ro')


ax.set_title("EMS: ASSAULT VICTIM")
fig.autofmt_xdate()
plt.show()


# In[ ]:



# Need to import for legend
import matplotlib.lines as mlines

fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)  


ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left() 
plt.xticks(fontsize=12) 


ax.plot_date(pp['timeStamp'], pp['EMS: ASSAULT VICTIM'],'k')
ax.plot_date(pp['timeStamp'], pp['EMS: ASSAULT VICTIM'],'ro')


ax.plot_date(pp['timeStamp'], pp['EMS: VEHICLE ACCIDENT'],'g')
ax.plot_date(pp['timeStamp'], pp['EMS: VEHICLE ACCIDENT'],'bo')


ax.set_title("EMS: ASSAULT VICTIM vs  EMS: VEHICLE ACCIDENT")


# Legend Stuff
green_line = mlines.Line2D([], [], color='green', marker='o',markerfacecolor='blue',
                          markersize=7, label='EMS: VEHICLE ACCIDENT')
black_line = mlines.Line2D([], [], color='black', marker='o',markerfacecolor='darkred',
                          markersize=7, label='EMS: ASSAULT VICTIM')

ax.legend(handles=[green_line,black_line], loc='best')


fig.autofmt_xdate()
plt.show()

# Note scale hides the assault increase 


# ## Functions -- Probably more useful ##

# In[ ]:


from sklearn import linear_model
import matplotlib.lines as mlines

def plotWLine(category='EMS: ASSAULT VICTIM'):

    
    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)  



    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    plt.xticks(fontsize=12) 



    # Build Linear Fit
    Y = pp[category].values.reshape(-1,1)
    X=np.arange(Y.shape[0]).reshape(-1,1)
    model = linear_model.LinearRegression()
    model.fit(X,Y)
    m = model.coef_[0][0]
    c = model.intercept_[0]
    ax.plot(pp['timeStamp'],model.predict(X), color='blue',
             linewidth=2)
    blue_line = mlines.Line2D([], [], color='blue', label='Linear Fit: y = %2.2fx + %2.2f' % (m,c))
    

    
    # Robustly fit linear model with RANSAC algorithm
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),random_state=23)
    model_ransac.fit(X, Y)
    mr = model_ransac.estimator_.coef_[0][0]
    cr = model_ransac.estimator_.intercept_[0]
    ax.plot(pp['timeStamp'],model_ransac.predict(X), color='green',
             linewidth=2)
    green_line = mlines.Line2D([], [], color='green', label='RANSAC Fit: y = %2.2fx + %2.2f' % (mr,cr))


    
    ax.legend(handles=[blue_line,green_line], loc='best')
    

    ax.plot_date(pp['timeStamp'], pp[category],'k')
    ax.plot_date(pp['timeStamp'], pp[category],'ro')


    ax.set_title(category)
    fig.autofmt_xdate()
    plt.show()
    print('\n')


    
def plot2WLine(cat1='EMS: ASSAULT VICTIM',cat2='EMS: VEHICLE ACCIDENT'):
    
    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)  



    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left() 
    plt.xticks(fontsize=12) 

    

    ax.plot_date(pp['timeStamp'], pp[cat1],'k')
    ax.plot_date(pp['timeStamp'], pp[cat1],'ro')


    ax.plot_date(pp['timeStamp'], pp[cat2],'g')
    ax.plot_date(pp['timeStamp'], pp[cat2],'bo')


    
    
# Build Linear Fit
    
    # cat 1
    Y = pp[cat1].values.reshape(-1,1)
    X=np.arange(Y.shape[0]).reshape(-1,1)
    model = linear_model.LinearRegression()
    model.fit(X,Y)
    m = model.coef_[0][0]
    c = model.intercept_[0]
    ax.plot(pp['timeStamp'],model.predict(X), color='black',
             linewidth=2)
    
    black_line = mlines.Line2D([], [], color='black', marker='o',markerfacecolor='darkred',
                               markersize=7,
                               label='%s, y = %2.2fx + %2.2f' % (cat1,m,c))
  
    # cat 2
    Y = pp[cat2].values.reshape(-1,1)
    X=np.arange(Y.shape[0]).reshape(-1,1)
    model = linear_model.LinearRegression()
    model.fit(X,Y)
    m = model.coef_[0][0]
    c = model.intercept_[0]
    ax.plot(pp['timeStamp'],model.predict(X), color='green',
             linewidth=2)
    
    green_line = mlines.Line2D([], [], color='green',marker='o',markerfacecolor='blue',
                          markersize=7, label='%s, y = %2.2fx + %2.2f' % (cat2,m,c))
  
 
    
    ax.set_title(cat1 + ' vs ' + cat2)
    ax.legend(handles=[green_line,black_line], loc='best')

    fig.autofmt_xdate()
    plt.show()
    print('\n')
       
    
# Create some plots
plotWLine('EMS: RESPIRATORY EMERGENCY')
plotWLine('EMS: NAUSEA/VOMITING')
plotWLine('EMS: CARDIAC EMERGENCY')
plotWLine('EMS: FALL VICTIM')
plotWLine('EMS: HEMORRHAGING')
plotWLine('EMS: ALLERGIC REACTION')






plot2WLine(cat1='EMS: ASSAULT VICTIM',cat2='EMS: VEHICLE ACCIDENT')


# 
# 
# Percent Change
# --------------
# 

# In[ ]:


# Get percent change
pp['EMS: ASSAULT VICTIM pc']=pp[('EMS: ASSAULT VICTIM')].pct_change(periods=1)

pp[['timeStamp','EMS: ASSAULT VICTIM pc','EMS: ASSAULT VICTIM']].head(6)


# ## Seaborn Heatmap ##

# In[ ]:


# Vehicle Accident -- yes, there is FIRE; maybe we should have include?
# Put this in a variable 'g'
g = d[(d.title.str.match(r'EMS:.*VEHICLE ACCIDENT.*') | d.title.str.match(r'Traffic:.*VEHICLE ACCIDENT.*'))]
g['Month'] = g['timeStamp'].apply(lambda x: x.strftime('%m %B'))
g['Hour'] = g['timeStamp'].apply(lambda x: x.strftime('%H'))
p=pd.pivot_table(g, values='e', index=['Month'] , columns=['Hour'], aggfunc=np.sum)
p.head()


# In[ ]:


cmap = sns.cubehelix_palette(light=2, as_cmap=True)
ax = sns.heatmap(p,cmap = cmap)
ax.set_title('Vehicle  Accidents - All Townships ');

