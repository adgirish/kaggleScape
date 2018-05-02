
# coding: utf-8

# # **Recruit Restaurant EDA**
# *Fabien Daniel (December 2017)*
# ___
# This notebook aims at providing a first glance at the data provided for the *"Recruit Restaurant Visitor Forecasting"* challenge. The purpose of this competition is to predict the number of people that will make reservations during 5 weeks, from April 2017 to May 2017. Historical data for a period $\sim$1 year prior to April 2017 is provided in order get some insights on customers' habits.
# ___
# **1. Load the data** <br>
# **2. Restaurant locations** <br>
# **3. Reservations against visits** <br>
# - 3.1 A global view of all *air* restaurants
# - 3.2 A spot check
#     * 3.2.1 case 1: the ideal case
#     * 3.2.2 case 2
# - 3.3 Test set reservations
# ___
# 
# ## 1. Load the data
# 
# First, I load all the packages that will be used throughout this notebook and set a few parameters related to display:

# In[ ]:


import numpy as np
import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, warnings
from mpl_toolkits.basemap import Basemap
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, HTML
InteractiveShell.ast_node_interactivity = "last_expr"
pd.options.display.max_columns = 50
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# The various files which are provided for this challenge are the following:

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
file_list = check_output(["ls", "../input"]).decode("utf8")
file_list = file_list.strip().split('\n')


# As a first step, I examine the content of these files by looking e.g at the variable types and the number of entries and null values. Some variables correspond to dates and times and for further use, I convert the corresponding variables to the `datetime`
#  format:

# In[ ]:


def get_info(df):
    print('Shape:',df.shape)
    print('Size: {:5.2f} MB'.format(df.memory_usage().sum()/1024**2))
    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values'}))
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
    display(tab_info)
    display(df[:5])


# In[ ]:


#_____________________________________________________________
# Read all the .csv files and show some info on their contents
for index, file in enumerate(file_list):
    var_name = file.rstrip('.csv')
    print(file)
    locals()[var_name] = pd.read_csv('../input/'+file)
    #____________________
    # convert to datetime
    for col in locals()[var_name].columns:
        if col.endswith('datetime') or col.endswith('date'):
            locals()[var_name][col] = pd.to_datetime(locals()[var_name][col])
    #__________________
    get_info(locals()[var_name])


# For these various dataframes, the meaning of the variables is given on the [competition home page](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data). As a reminder, I recall the content of each dataframe:
# - **air_reserve.csv, hpg_reserve.csv**: reservations made in the air (*AirREGI*) or hpg (*Hot Pepper Gourmet*) systems
# - **air_store_info.csv, hpg_store_info.csv**: information about *air* and *hpg* restaurants
# - **store_id_relation.csv**: link the restaurants ids of the air and hpg systems
# - **air_visit_data.csv**: historical visit data for the air restaurants
# - **date_info.csv**: information about the calendar dates in the dataset
# 
# It is interesting to note that there are no missing values in this dataset. Moreover, as stated in [this forum discussion](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/44502), there is not hpg couterpart to the **air_visit_data.csv** file. In other words, *we are not provided with the visit history for the restaurants which are only defined in the hpg system*.
# 
# ___
# ## 2. Restaurant locations
# 
# At first, in order to have a global view of the geo-localisation of the restaurants in Japan, I put locate them on a map, for both the hpg and air restaurants. At that point, we have to keep in mind that these localisations are only approximations of the true locations in order to keep the data anonymous:

# In[ ]:


def draw_map(df, title):
    plt.figure(figsize=(11,6))
    map = Basemap(resolution='i',llcrnrlon=127, urcrnrlon=147,
                  llcrnrlat=29, urcrnrlat=47, lat_0=0, lon_0=0,)
    map.shadedrelief()
    map.drawcoastlines()
    map.drawcountries(linewidth = 3)
    map.drawstates(color='0.3')
    parallels = np.arange(0.,360,10.,)
    map.drawparallels(parallels, labels = [True for s in range(len(parallels))])
    meridians = np.arange(0.,360,10.,)
    map.drawmeridians(meridians, labels = [True for s in range(len(meridians))])
    #______________________
    # put restaurants on map
    for index, (y,x) in df[['latitude','longitude']].iterrows():
        x, y = map(x, y)
        map.plot(x, y, marker='o', markersize = 5, markeredgewidth = 1, color = 'red',
                 markeredgecolor='k')
    plt.title(title, y = 1.05)


# In[ ]:


#draw_map(hpg_store_info, 'hpg store restaurant locations')
draw_map(air_store_info, 'air store restaurant locations')


# ___
# ## 3. Reservations against visists
# 
# First, I define a class that I will subsequently use to make the figures. I previously defined this class in [another notebook](https://www.kaggle.com/fabiendaniel/predicting-flight-delays-tutorial) that dealt with time series analysis and where the aim was to predict flight delays.
# 

# In[ ]:


class Figure_style():
    #_________________________________________________________________
    def __init__(self, size_x = 11, size_y = 5, nrows = 1, ncols = 1):
        sns.set_style("white")
        sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
        self.fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(size_x,size_y,))
        #________________________________
        # convert self.axs to 2D array
        if nrows == 1 and ncols == 1:
            self.axs = np.reshape(axs, (1, -1))
        elif nrows == 1:
            self.axs = np.reshape(axs, (1, -1))
        elif ncols == 1:
            self.axs = np.reshape(axs, (-1, 1))
    #_____________________________
    def pos_update(self, ix, iy):
        self.ix, self.iy = ix, iy
    #_______________
    def style(self):
        self.axs[self.ix, self.iy].spines['right'].set_visible(False)
        self.axs[self.ix, self.iy].spines['top'].set_visible(False)
        self.axs[self.ix, self.iy].yaxis.grid(color='lightgray', linestyle=':')
        self.axs[self.ix, self.iy].xaxis.grid(color='lightgray', linestyle=':')
        self.axs[self.ix, self.iy].tick_params(axis='both', which='major',
                                               labelsize=10, size = 5)
    #________________________________________
    def draw_legend(self, location='upper right'):
        legend = self.axs[self.ix, self.iy].legend(loc = location, shadow=True,
                                        facecolor = 'g', frameon = True)
        legend.get_frame().set_facecolor('whitesmoke')
    #_________________________________________________________________________________
    def cust_plot(self, x, y, color='b', linestyle='-', linewidth=1, marker=None, label=''):
        if marker:
            markerfacecolor, marker, markersize = marker[:]
            self.axs[self.ix, self.iy].plot(x, y, color = color, linestyle = linestyle,
                                linewidth = linewidth, marker = marker, label = label,
                                markerfacecolor = markerfacecolor, markersize = markersize)
        else:
            self.axs[self.ix, self.iy].plot(x, y, color = color, linestyle = linestyle,
                                        linewidth = linewidth, label=label)
        self.fig.autofmt_xdate()
    #________________________________________________________________________
    def cust_plot_date(self, x, y, color='lightblue', linestyle='-',
                       linewidth=1, markeredge=False, label=''):
        markeredgewidth = 1 if markeredge else 0
        self.axs[self.ix, self.iy].plot_date(x, y, color='lightblue', markeredgecolor='grey',
                                  markeredgewidth = markeredgewidth, label=label)
    #________________________________________________________________________
    def cust_scatter(self, x, y, color = 'lightblue', markeredge = False, label=''):
        markeredgewidth = 1 if markeredge else 0
        self.axs[self.ix, self.iy].scatter(x, y, color=color,  edgecolor='grey',
                                  linewidths = markeredgewidth, label=label)    
    #___________________________________________
    def set_xlabel(self, label, fontsize = 14):
        self.axs[self.ix, self.iy].set_xlabel(label, fontsize = fontsize)
    #___________________________________________
    def set_ylabel(self, label, fontsize = 14):
        self.axs[self.ix, self.iy].set_ylabel(label, fontsize = fontsize)
    #____________________________________
    def set_xlim(self, lim_inf, lim_sup):
        self.axs[self.ix, self.iy].set_xlim([lim_inf, lim_sup])
    #____________________________________
    def set_ylim(self, lim_inf, lim_sup):
        self.axs[self.ix, self.iy].set_ylim([lim_inf, lim_sup])  


# First, I make a census of the restaurant ids which are common to both the *hpg* and *air* systems and only keep in the **hpg_reserve** dataframe the restaurants which are common to both reservations systems:

# In[ ]:


convert_hpg = {k:v for k,v in list(zip(store_id_relation['hpg_store_id'].values,
                                       store_id_relation['air_store_id'].values))}
hpg_reserve["hpg_store_id"].replace(convert_hpg, inplace = True)
hpg_reserve = hpg_reserve[hpg_reserve['hpg_store_id'].str.startswith('air')]


# I create new variables that indicate the delay between the reservation and the visit dates and merge the **air_reserve** and **hpg_reserve** dataframes in the **total_reserve** dataframe:

# In[ ]:


def delta_reservation(df):
    df['delta_reservation'] = df['visit_datetime'] - df['reserve_datetime']
    df['delta_2days'] = df['delta_reservation'].apply(lambda x: int(x.days < 2))
    df['delta_7days'] = df['delta_reservation'].apply(lambda x: int(2 <= x.days < 7))
    df['delta_long'] = df['delta_reservation'].apply(lambda x: int(x.days >= 7))
    return df
#______________
air_reserve = delta_reservation(air_reserve)
hpg_reserve = delta_reservation(hpg_reserve)
#__________________________________________________________________________
air_reserve.rename(columns = {'air_store_id':'store_id'}, inplace = True)
hpg_reserve.rename(columns = {'hpg_store_id':'store_id'}, inplace = True)
total_reserve = pd.concat([air_reserve, hpg_reserve])
total_reserve['date'] = total_reserve['visit_datetime'].apply(lambda x:x.date())


# Considering the *air* restaurants, the **total_reserve** and **air_visit_data** dataframes respectively give informations on the number of visits and the reservations that were previously made. At first, I look at the number of unique restaurants in each dataframe:

# In[ ]:


list_visit_ids   = air_visit_data['air_store_id'].unique()
list_reserve_ids = total_reserve['store_id'].unique()
print("nb. of restaurants visited: {}".format(len(list_visit_ids)))
print("nb. of restaurants with reservations: {}".format(len(list_reserve_ids)))
print("intersections of ids: {}".format(len(set(list_visit_ids).intersection(set(list_reserve_ids)))))


# Hence, for all the restaurants where there is a record of the reservations, we also have the effective number of visitors. However, there is ~ 500 restaurants for which we don't have any informations on reservations. 
# 
# ### 3.1 A global view of all *air* restaurants
# 
# First, I consider the overall visits and reservations made in the *air* restaurants.

# In[ ]:


df1 = total_reserve[['date', 'reserve_visitors']].groupby('date').sum().reset_index()
df2 = air_visit_data.groupby('visit_date').sum().reset_index()

fig1 = Figure_style(11, 5, 1, 1)
fig1.pos_update(0, 0)
fig1.cust_plot(df2['visit_date'], df2['visitors'], linestyle='-', label = 'nb. of visits')
fig1.cust_plot(df1['date'], df1['reserve_visitors'], color = 'r', linestyle='-', label = 'nb. of reservations')
fig1.style() 
fig1.draw_legend(location = 'upper left')
fig1.set_ylabel('Visitors', fontsize = 14)
fig1.set_xlabel('Date', fontsize = 14)
#________
# limits
date_1 = datetime.datetime(2015,12,1)
date_2 = datetime.datetime(2017,6,1)
fig1.set_xlim(date_1, date_2)
fig1.set_ylim(-50, 25000)


# Here we can note a few things:
# - the number of reservations only account for a small fraction of the total number of visits (typically a factor 10). This partly comes from the fact that only ~1/3 of the total number of restaurants visited are present in the **total_reserve** dataframe. Additionally, many clients will probably go directly to the restaurants without reserving.
# - another interesting thing to note is that there is a clear high frequency peridocity and without looking at the data in details, we can infer that this periodicity arise from the day of the week, since it seems quite logical that the restaurant frequentation rises during the week-end.
# - finally, **_we see that for the dates that correspond to the test set_**, the number of visits is quite low: this point will be adressed later but we can infer that for the 5 weeks covered by the test set, **_we don't have the data that deal with the 'last minute' reservations._**
# 
# ___
# ### 3.2 A spot check
# 
# Now, we can examine the same thing but for a few selected restaurants.
# 
# #### 3.2.1 Case 1: the ideal case
# 
# We select a first restaurant in the **air_reserve** dataframe and compare the number of visits and number of reservations:

# In[ ]:


restaurant_id = air_reserve['store_id'][0]


# In[ ]:


df2 = air_visit_data[air_visit_data['air_store_id'] == restaurant_id]
df0 = total_reserve[total_reserve['store_id'] == restaurant_id]
df1 = df0[['date', 'reserve_visitors']].groupby('date').sum().reset_index()


# In[ ]:


fig1 = Figure_style(11, 5, 1, 1)
fig1.pos_update(0, 0)
fig1.cust_plot(df2['visit_date'], df2['visitors'], linestyle='-', label = 'nb. of visits')
fig1.cust_plot(df1['date'], df1['reserve_visitors'], color = 'r', linestyle='-', label = 'nb. of reservations')
fig1.style() 
fig1.draw_legend(location = 'upper left')
fig1.set_ylabel('Visitors', fontsize = 14)
fig1.set_xlabel('Date', fontsize = 14)
#________
# limits
date_1 = datetime.datetime(2015,12,21)
date_2 = datetime.datetime(2017,6,1)
fig1.set_xlim(date_1, date_2)
fig1.set_ylim(-3, 39)


# Here, we see that contrary to the case where we considered all the restaurants, the number of reservations closely follows the numbers of visits (when the reservations are available).
# 
# #### 3.2.2 Case 2
# 
# We select a second restaurant and proceed as before (except that I zoom on the data from Nov. 2016 to May 2017):

# In[ ]:


restaurant_id = air_reserve['store_id'][2]


# In[ ]:


df2 = air_visit_data[air_visit_data['air_store_id'] == restaurant_id]
df0 = total_reserve[total_reserve['store_id'] == restaurant_id]
df1 = df0[['date', 'reserve_visitors']].groupby('date').sum().reset_index()


# In[ ]:


fig1 = Figure_style(11, 5, 1, 1)
fig1.pos_update(0, 0)
fig1.cust_plot(df2['visit_date'], df2['visitors'], linestyle='-', label = 'nb. of visits')
fig1.cust_plot(df1['date'], df1['reserve_visitors'], color = 'r', linestyle='-',
               marker = ['r', 'o', 5], label = 'nb. of reservations')
fig1.style() 
fig1.draw_legend(location = 'upper left')
fig1.set_ylabel('Visitors', fontsize = 14)
fig1.set_xlabel('Date', fontsize = 14)
#________
# limits
date_1 = datetime.datetime(2016,11,1)
date_2 = datetime.datetime(2017,5,1)
fig1.set_xlim(date_1, date_2)
fig1.set_ylim(-3, 45)


# Here, we see that number of visits *usually*  exceeds the number of reservations.
# 
# Hence, we will be faced with two categories of restaurants: for some restaurants, it will be easy to predict the number of visitors since they will only attend people that reserved beforehand. Other restaurants will on the contrary attend everybody.
# 
# ### 3.3 Test set reservations
# 
# As commented earlier, there's a lack of information for the test set reservations. This is can be clearly seen in the graph below:

# In[ ]:


fig1 = Figure_style(11, 5, 1, 1)
fig1.pos_update(0, 0)

color = ['r', 'b', 'g']
label = ['delay < 2 days', '2 days < delay < 7 days', 'delay > 7 days']
for j, colonne in enumerate(['delta_2days', 'delta_7days', 'delta_long']):
    df0 = total_reserve[total_reserve[colonne] == 1]
    df1 = df0[['date', 'reserve_visitors']].groupby('date').sum().reset_index()
    fig1.cust_plot(df1['date'], df1['reserve_visitors'], linestyle='-', label = label[j], color = color[j])

fig1.style() 
fig1.draw_legend(location = 'upper left')
fig1.set_ylabel('Visitors', fontsize = 14)
fig1.set_xlabel('Date', fontsize = 14)
#________
# limits
date_1 = datetime.datetime(2017,2,1)
date_2 = datetime.datetime(2017,5,31)
fig1.set_xlim(date_1, date_2)
fig1.set_ylim(-3, 3000)
plt.show()


# On this graph, we see that after April 22, the number of reservations starts to decrease with time. Naturally, after a few days, we don't have any information concerning the resevations made 2 days before the visit. After April 29, we don't have any information on the reservations made 7 days before the visit. We see that long term reservations are avaliable for all the test set. However, their number tends to decrease with time and would thus correspond to a lower limit. 
# 
# As [outlined by BreakfastPirate](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/45120), this point is crucial in the prediction since most presumably, the public LB would correspond to the April 23 to April 28 period, for which we still have some information concerning the number of reservations. If the number of reservations is taken into account in the modeling, the models will probably perform reasonably well on the public LB. However, after April 28, there's a lack of information concerning the number of reservations. Models should take this into account and otherwise, we may expect that models that give good public LB scores would perform poorly on the private dataset.
