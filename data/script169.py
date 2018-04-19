
# coding: utf-8

# # Visualizing California Pollution [2000-2016] with Seaborn and Matplotlib
# 
# _By Nick Brooks, Date: December 2017_
# 
# Exploring the readings of four pollutants in California through their Air Quality Index for an "Apples to Apples" analysis. Air Quality Index scales the various pollutants to a normalized value which represents magnitude of harm to human inhalers.
# 
# ***
# 
# # Table of Content:
# 
# 1. **[Measurement Count and Macro Trend](#p1)**
# 2. **[Timeseries: Trend of Average Pollutants](p5)**
#     - 2.1 Trend over All Time [2000 - 2016] 
#     - 2.2 Trend over Day of Year
#     - 2.3 Trend over Weekday
# 3. **[Understanding Pollution Distribution and Interactions](#p5)**
#     - 3.1 One Dimensional Histograms
#     - 3.2 Correlation Heatmap
#     - 3.3 Two Dimensional Histograms
# 4. **[Heatmap for All AQI over time](#p6)**

# In[45]:


# I/O and Computation
import numpy as np
import pandas as pd

# Viz
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8, 6)

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Read
# df = pd.read_csv("Pollution/pollution_us_2000_2016.csv")
df = pd.read_csv("../input/pollution_us_2000_2016.csv")
print("Data Loaded")
# Time Formatting
df = df.drop(['Unnamed: 0'], axis=1)
df['Date Local'] = pd.to_datetime(df['Date Local'],format='%Y-%m-%d') # date parse
df['Year'] = df['Date Local'].dt.year # year


# **Dataset Characteristics:** <br>

# In[46]:


def custom_describe(df):
    """
    I am a non-comformist :)
    """
    unique_count = []
    for x in df.columns:
        mode = df[x].mode().iloc[0]
        unique_count.append([x,
                             len(df[x].unique()),
                             df[x].isnull().sum(),
                             mode,
                             df[x][df[x]==mode].count(),
                             df[x].dtypes])
    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
    return pd.DataFrame(unique_count, columns=["Column","Unique","Missing","Mode","Mode Occurence","dtype"]).set_index("Column").T
describe = custom_describe(df)
display(describe.iloc[:,0:15])
display(describe.iloc[:,15:])


# # 1. Measurement Count and Macro Trend
# <a id="p1"></a>

# In[47]:


f , ax = plt.subplots(1,2, figsize=[15,5])
df[['SO2 AQI','State']].groupby(["State"]).count().sort_values(by='SO2 AQI',ascending=False).plot.bar(ax=ax[0])
ax[0].set_title("Number of Measurements, by State")

var = "SO2 AQI"

# Df
temp_df = df[[var,'Year','State']].groupby(["Year"]).count().reset_index().sort_values(by='Year',ascending=False)
topstate = df[[var,'State']].groupby(["State"]).count().sort_values(by='SO2 AQI',ascending=False).index [:5]
state_col = ["green","red","yellow","orange","purple"]

# Plot
ax[1].set_title('Number of Observations for {} by Year'.format(var))
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Observation Count')
plt.plot(temp_df.Year,temp_df["SO2 AQI"],marker='o', linestyle='--', color='black', label='Square')
for (i,col) in zip(topstate, state_col):
    state_df= df[df.State==i][[var,'Year','State']].groupby(["Year"])    .count().reset_index().sort_values(by='Year',ascending=False)
    ax[1].plot(state_df.Year,state_df[var],marker='o', linestyle='--', color=col, label='Square')
ax[1].legend(topstate.insert(0, "All") , loc=2,fontsize='large')
plt.show()


# From these plots, I have concluded that I shall restrict my exploration to California, since it has disproportionately more data than the rest of the states.
# 
# However, before I go, I will take a quick glance at the general trend in pollution

# ***
# **Macro Trend:**

# In[48]:


# Plot the aggregate decrease of all pollutants
f, ax = plt.subplots(figsize=[10,4])
df.groupby(['Year']).agg({'SO2 AQI': 'mean',
                          'CO AQI': 'mean',
                          'NO2 AQI': 'mean',
                          'O3 AQI': 'mean'})\
.plot(lw=2,colormap='jet',marker='.',markersize=10, ax =ax,linewidth=2.5)
ax.set_title('Mean Pollutant AQI Over Time')
ax.set(xlabel="Average AQI", ylabel="Year")
plt.show()


# Trend of all pollutants, except for Ozone, is downward, signifying a general betterment of air quality in the United States.

# ***
# # 2. Timeseries Analysis on California Data:
# <a id="p4"></a>
# [Pandas Timeseries Link](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)
# 
# I have decided to narrow down the analysis to concern only California. The alternative would simply be too hard to make visual sense of.
# 
# ## 2.1 Trend over All Time [2000 - 2016]: 4x1 Format Line Plots
# 
# Pretty Obvious Trend. Could be further investigated with time-series modeling. Next, I want to investigate the trend at different levels of time.

# In[49]:


# Only California Dataset
cal = df[df.State=="California"]

# Temporary NA fix
cal= cal.dropna(axis='rows')

# Missing Values
miss = cal.isnull().sum().reset_index()
miss.columns = ['Column','Missing Count']

# Time Frames of Interest
cal["Date of Year"] = cal['Date Local'].dt.dayofyear # Day of Year
cal["Weekday"] = cal['Date Local'].dt.weekday 

# Input
cols = ["black","darkgreen","blue","red"]
polldata= ['NO2 AQI','O3 AQI','SO2 AQI',"CO AQI"]
# Plotter
def row_plots(data, time, rol_window):
    f, axarr = plt.subplots(len(data), sharex=True, squeeze=True)
    for index, x in enumerate(data):
        plot1 = cal[[x,time]].groupby([time]).mean()
        plot1[x] = plot1[x].rolling(window = rol_window).mean()
        axarr[index].set_ylabel("{}".format(x))
        axarr[index].plot(plot1, color=cols[index],label=x,linewidth=2)
        axarr[index].legend(fontsize='large', loc='center left',
                            bbox_to_anchor=(1, 0.5))
    plt.tight_layout(pad=0)
    plt.subplots_adjust(top=0.90)
    plt.suptitle("Trend of Average Pollutants by {}".format(time),fontsize=17)
    plt.show()

    # 
    site_poll = cal[["Site Num",time,"NO2 AQI","O3 AQI","CO AQI","SO2 AQI"]]    .groupby(['Site Num',time]).mean().groupby(level="Site Num")
    
    f, axarr = plt.subplots(len(data), sharex=True, squeeze=True)    
    for index, x in enumerate(data):
        pollutant_plot = site_poll[x]
        pollutant_plotTop = pollutant_plot.mean().nlargest(4).index
        for i in pollutant_plotTop:
            lineplot= pollutant_plot.get_group(i).groupby(pd.Grouper(level=time)).            mean().rolling(window = rol_window).mean()
            axarr[index].plot(lineplot)
        axarr[index].legend(pollutant_plotTop,fontsize='large', loc='center left',
                            bbox_to_anchor=(1, 0.5))
        axarr[index].set_ylabel("{}".format(x))  
    plt.tight_layout(pad=0)
    plt.subplots_adjust(top=0.90)
    plt.suptitle("Trend of Average Pollutants of Top 4 Sites by {}".format(time),fontsize=17)
    plt.show()
    plt.show()
    
    # City plots:
    city_poll = cal[["City",time,"NO2 AQI","O3 AQI","CO AQI","SO2 AQI"]]    .groupby(['City',time]).mean().groupby(level="City")

    f, axarr = plt.subplots(len(data), sharex=True, squeeze=True)    
    for index, x in enumerate(data):
        pollutant_plot = city_poll[x]
        pollutant_plotTop = pollutant_plot.mean().nlargest(4).index
        for i in pollutant_plotTop:
            lineplot= pollutant_plot.get_group(i).groupby(pd.Grouper(level=time)).mean().rolling(window = rol_window).mean()
            axarr[index].plot(lineplot)
        axarr[index].legend(pollutant_plotTop,fontsize='large', loc='center left',
                            bbox_to_anchor=(1, 0.5))
        axarr[index].set_ylabel("{}".format(x))
        
    plt.tight_layout(pad=0)
    plt.subplots_adjust(top=0.90)
    plt.suptitle("Trend of Average Pollutants of Top 4 Cities by {}".format(time),fontsize=17)
    plt.show()
    plt.show()
    
row_plots(data=polldata, time= "Date Local", rol_window=80)


# ## 2.2 Trend by Date of Year: 2x2 Format Line Plot
# Very Clear Seasonal Trends! There is defintely a weather related reason for this. Perhaps there is also a seasonality in the activity of the emittors of pollution.

# In[50]:


# Plot Mega-Helper
def years_site_city_plot(time, rol_window):
    plt.figure(figsize=(12,8))    
    for var,plot in [('NO2 AQI',221), ('O3 AQI',222),('SO2 AQI',223),("CO AQI",224)]:
        plt.subplot(plot)
        plot1 = cal[[var,time]].groupby([time]).mean()
        plot1[var] = plot1[var].rolling(window = rol_window).mean()
        plt.plot(plot1, color='purple', label=var)
        plt.title(var)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=4)
        plt.title(var)
        plt.xlabel(time)
        plt.ylabel("Air Quality Index")
    plt.tight_layout(pad=0, w_pad=0.5, h_pad=2.5)
    plt.subplots_adjust(top=0.89)
    plt.suptitle("Trend of Average Pollutants by {}".format(time),fontsize=17)
    plt.show()
    
    # Site
    site_poll = cal[["Site Num",time,"NO2 AQI","O3 AQI","CO AQI","SO2 AQI"]]    .groupby(['Site Num',time]).mean().groupby(level="Site Num")

    plt.figure(figsize=(12,8))
    for var,plot in [('NO2 AQI',221), ('O3 AQI',222),('SO2 AQI',223),("CO AQI",224)]:
        plt.subplot(plot)
        pollutant_plot = site_poll[var]
        pollutant_plotTop = pollutant_plot.mean().nlargest(4).index
        for i in pollutant_plotTop:
            plot1= pollutant_plot.get_group(i).groupby(pd.Grouper(level=time))            .mean().rolling(window = rol_window).mean().plot()
        plt.legend(pollutant_plotTop, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=4)
        plt.title(var)
        plt.xlabel(time)
        plt.ylabel("Air Quality Index")

    plt.tight_layout(pad=0, w_pad=0.5, h_pad=2.5)
    plt.subplots_adjust(top=0.89)
    plt.suptitle("Trend of Average Pollutant by Top 4 Site by {}".format(time),fontsize=17)
    plt.show()

    # City
    city_poll = cal[["City",time,"NO2 AQI","O3 AQI","CO AQI","SO2 AQI"]]    .groupby(['City',time]).mean().groupby(level="City")

    plt.figure(figsize=(12,8))
    for var,plot in [('NO2 AQI',221), ('O3 AQI',222),('SO2 AQI',223),("CO AQI",224)]:
        plt.subplot(plot)
        pollutant_plot = city_poll[var]
        pollutant_plotTop = pollutant_plot.mean().nlargest(4).index
        for i in pollutant_plotTop:
            plot1= pollutant_plot.get_group(i).groupby(pd.Grouper(level=time))            .mean().rolling(window = rol_window).mean().plot()
        plt.title(var)
        plt.xlabel(time)
        plt.ylabel("Air Quality Index")
        plt.legend(pollutant_plotTop, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=4)

    plt.tight_layout(pad=0, w_pad=0.5, h_pad=2.5)
    plt.subplots_adjust(top=0.89)
    plt.suptitle("Trend of Average Pollutant by Top 4 City by {}".format(time),fontsize=17)
    plt.show()


years_site_city_plot(time="Date of Year", rol_window=5)


# ## 2.3 Trend by Weekday: 2x2 Format Bar plot
# 
# This one, on other hand, points to a industrial production scheduling: either weekd-day based products, or week-day based machine maintenance.
# 
# [Grouped Bar Plots Link](https://chrisalbon.com/python/matplotlib_grouped_bar_plot.html)

# In[51]:


a = 0.80
def pol_bar_plot(time, rol_window):
    plt.figure(figsize=(12,8))
    width = .90 
    plot1 = cal.groupby([time]).mean()
    plot1 = plot1.rolling(window = rol_window).mean()
    X= list(range(len(set(plot1.index))))
    labels = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    
    for index, (var,plot) in enumerate([('NO2 AQI',221), ('O3 AQI',222),('SO2 AQI',223),("CO AQI",224)]):
        plt.subplot(plot)
        X= list(range(len(set(plot1.index))))
        plt.bar(left=[p + width for p in X], height=plot1[var],
                width=width,label=var,alpha=a)
        plt.title(var)
        # plt.xlabel(time)
        plt.ylabel("Air Quality Index")
        plt.xticks([p + width for p in X], labels)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=4)
    plt.tight_layout(pad=0, w_pad=0.5, h_pad=2.5)
    plt.subplots_adjust(top=0.89)
    plt.suptitle("Trend of Average Pollutant by {}".format(time),fontsize=17)
    plt.show()
    
    # Site
    site_poll = cal[["Site Num",time,"NO2 AQI","O3 AQI","CO AQI","SO2 AQI"]]    .groupby(['Site Num',time]).mean().groupby(level="Site Num")
    width = .22
    plt.figure(figsize=(12,8))
    for index, (var,plot) in enumerate([('NO2 AQI',221), ('O3 AQI',222),('SO2 AQI',223),("CO AQI",224)]):
        plt.subplot(plot)
        pollutant_plot = site_poll[var]
        pollutant_plotTop = pollutant_plot.mean().nlargest(4).index
        for index, i in enumerate(pollutant_plotTop):
            plot1= pollutant_plot.get_group(i).groupby(pd.Grouper(level=time))            .mean().rolling(window = rol_window).mean()
            plt.bar(left=[p + width*index for p in X], height=plot1,width=width, label=i,alpha=a)   
        plt.title(var)
        # plt.xlabel(time)
        plt.ylabel("Air Quality Index")
        plt.xticks([p + (width*len(pollutant_plotTop))/2 for p in X], labels)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)
    plt.tight_layout(pad=0, w_pad=0.5, h_pad=2.5)
    plt.subplots_adjust(top=0.89)
    plt.suptitle("Trend of Average Pollutant by Top 4 Site by {}".format(time),fontsize=17)
    plt.show()
    
   # City
    city_poll = cal[["City",time,"NO2 AQI","O3 AQI","CO AQI","SO2 AQI"]]    .groupby(['City',time]).mean().groupby(level="City")

    width = .22
    
    plt.figure(figsize=(12,8))
    for index, (var,plot) in enumerate([('NO2 AQI',221), ('O3 AQI',222),('SO2 AQI',223),("CO AQI",224)]):
        plt.subplot(plot)
        pollutant_plot = city_poll[var]
        pollutant_plotTop = pollutant_plot.mean().nlargest(4).index
        for index, i in enumerate(pollutant_plotTop):
            plot1= pollutant_plot.get_group(i).groupby(pd.Grouper(level=time))            .mean().rolling(window = rol_window).mean()
            plt.bar(left=[p + width*index for p in X], height=plot1,width=width, label=i,alpha=a)   
        plt.title(var)
        # plt.xlabel(time)
        plt.ylabel("Air Quality Index")
        plt.xticks([p + (width*len(pollutant_plotTop))/2 for p in X], labels)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=4)

    plt.tight_layout(pad=0, w_pad=0.5, h_pad=2.5)
    plt.subplots_adjust(top=0.89)
    plt.suptitle("Trend of Average Pollutant by Top 4 City by {}".format(time),fontsize=17)
    plt.show()

# New Color Theme
sns.set_palette([
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
])    
# Plot Data
pol_bar_plot(time="Weekday", rol_window=1)


# # 3. Understanding Pollution Distribution and Interactions:
# <a id="p5"></a>
# 
# _Univariate and Bivariate Analysis_

# In[52]:


Pollutants = cal[["NO2 AQI","O3 AQI","SO2 AQI", "CO AQI"]]
Pollutants.index = cal["Date Local"]

# Exclude Outliers
Pollutants.hist(bins=20)
plt.show()

print("Unique Sites:",len(cal["Site Num"].unique()))
print("Unique Cities:",len(cal["City"].unique()))
print("Unique Counties:",len(cal["County"].unique()))


# **Interpretation:** <br>
# CO and SO2 seem to have more extreme outliers. A common feature of pollution measurement whose consequence on human health is hardly known.
# 
# ***

# In[57]:


sns.heatmap(Pollutants.corr(), annot=True, fmt=".2f", cmap="viridis",cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Plot")
plt.show()


# **Interpretation:** <br>
# CO AQI and NO2 AQI stand out with the highest positiive correlation out of the bunch.
# 
# ***
# **2-D Distribution:** <br>

# In[ ]:


# Examine Strongest Correlation
sns.jointplot(x=cal["NO2 AQI"], y=cal["CO AQI"], kind='kde', xlim=(0,50),ylim=(0,15),color='g')
plt.show()


# **Interpretation:** <br>
# As Carbon Dioxide increases, so does Nitrogen Oxide. The shade/cloud is a two dimensional distribution.
# 
# ***
# # 4. Heatmap for All AQI over time
# <a id="p6"></a>
# 
# This plot is kinda crazy.. Bear with me. Althought it is incredibly difficult to read, it actually provides us with annual average AQI of all pollutants of all the states. Almost no other plot can present so much information so easily.

# In[58]:


polldata= ['NO2 AQI','O3 AQI','SO2 AQI',"CO AQI"]
stack_df = df[['State','Year','NO2 AQI','O3 AQI','SO2 AQI',"CO AQI"]]

for col in polldata:
    stack_df[col] =(stack_df[col] - stack_df[col].mean())/stack_df[col].std(ddof=0)
    
stack_df = stack_df.melt(["State", "Year"])#.stack()
stack_df.head()

# Create Heatmap Pivot with State as Row, Year as Col, So2 as Value
polldata= ['NO2 AQI','O3 AQI','SO2 AQI',"CO AQI"]

f, ax = plt.subplots(figsize=(16,40))
ax.set_title('All AQI by State and Year')
sns.heatmap(stack_df.pivot_table(values="value", index=["State", "variable"], columns=["Year"], aggfunc='mean',margins=True),
                annot=False, linewidths=.5, ax=ax,cbar_kws={'label': 'Annual Average'}, cmap="viridis")
plt.show()


# **Interpretation:** <br>
# Lots of information here. Since so much data is being averaged, the overall trend suggests that pollution is getting better on a macro scale. However, many communities in the US still suffer from high exposure to pollution and dangerous industral toxins. Indeed, pollution spikes still exist and their health impacts are not well understood.
