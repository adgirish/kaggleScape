
# coding: utf-8

# # Contents
# 1. Introduction
#     * Loading the Data
#     * Cleaning the Data
#     * Determining Usable Years
#     * Creating the Data Set to Analyze
# 2. Analyzing the Data Set
# 3. Conclusion
# 
# # 1.0 Introduction
# The question I am wanting to explore in this kernel is if the popularity of video game genre has changed over the years.  To do this I will use a simple linear regression to see if the median sales number has increased or decreased over time. 
# 
# ## 1.1 Loading the Data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

data = pd.read_csv('../input/Video_Game_Sales_as_of_Jan_2017.csv')
data.head()


# ## 1.2 Cleaning the Data
# As the data set stands, it needs a little bit of cleaning before it can be analyzed.  First, video games with unknown release years should be omitted.  Furthermore, the video games released in 2017 will also be omitted since the data was gathered in January and the sales of those video games have not yet reached full potential.  Finally, pandas is reading the years as floating points, to create nice plots the years should be changed to integers.

# In[ ]:


# Remove NaN data from Year of Release
data = data[data.Year_of_Release.notnull()]

# Omitting video games released in 2017.
data = data.loc[data.Year_of_Release < 2017]

# Converting year of release to integers
data.Year_of_Release = data['Year_of_Release'].astype(int)


# ## 1.3 Determining Usable Years
# Considering the video game sales before a certain point in time would not only be irrelevant for predicting sales in the close future but could skew the predictions.  To determine the starting year two things should be considered: the yearly global sales and at what point in time were all genres being continuously released.  Fortunately, a starting year can be chosen by looking at the total yearly global sales, the cumulative proportion of yearly global sales and the heat map of global sales of games release each year by genre.  Using these three plots below it can be seen that all genres are not fully represented until 1991 and more than 95% of the global sales occur after 1991.  Hence we will start by only considering the video game sales from 1991 to 2016.

# In[ ]:


# Creating a table of the total global sales for each genre and year
Sales_by_Gen_and_Yr = pd.pivot_table(data,index=['Year_of_Release'],
                     columns=['Genre'],values=['Global_Sales'],aggfunc=np.sum)
Sales_by_Gen_and_Yr.columns = Sales_by_Gen_and_Yr.columns.get_level_values(1)

# Finding the yearly totals and cumulative proportion of yearly global sales
Yearly_Tots = Sales_by_Gen_and_Yr.sum(axis=1)
Yearly_Tots = Yearly_Tots.sort_index()
YT1_cumsum = Yearly_Tots.cumsum()/Yearly_Tots.sum()

# Plotting the yearly totals and cumulative proportions
fig = plt.figure(figsize=(12,5))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
sns.barplot(y = Yearly_Tots.values, x = Yearly_Tots.index,ax=ax1)
ax1.set_title('Total Yearly Global Sales')
plt.setp(ax1.get_xticklabels(),rotation=90)
ax1.set_xlabel('Years')
ax1.set_ylabel('Number of games sold (in millions)')

sns.barplot(y = YT1_cumsum.values, x = YT1_cumsum.index, ax=ax2)
ax2.set_title('Cumulative Proportion of Yearly Global Sales')
plt.setp(ax2.get_xticklabels(),rotation=90)
ax2.set_xlabel('Years')
ax2.set_ylabel('Cummulative Proportion')
ax2.yaxis.set_ticks(np.arange(0,1,0.05))
fig.tight_layout()

# Plotting the heat map of global sales for games released each year by genre
plt.figure(figsize=(10,10))
sns.heatmap(Sales_by_Gen_and_Yr,annot = True, fmt = '.2f', cmap = 'Blues')
plt.tight_layout()
plt.ylabel('Year of Release')
plt.xlabel('Genre')
plt.title('Global Sales (in millions) of Games Released Each Year by Genre')
plt.show()


# ## 1.4 Creating the Data Set to Analyze
# With the selection of years now comes the choice of how to represent the yearly global sales for each genre.  Since the total number of games released each year varies, the total global sales for each genre would not be a consistent measure over time.  Furthermore, by plotting a histogram of the global sales it is easy to see the data is highly skewed to the right, meaning the yearly average of the global sales for each genre will not be an acceptable measure.  For highly skewed data, the median is an acceptable measure, so we will consider the median number of games sold per genre each year.

# In[ ]:


# Histogram of global sales
plt.figure(figsize=(9,5))
data.Global_Sales.hist(bins=50)
plt.show()


# In[ ]:


# Pulling only the data from 1991 to 2016
data = data.loc[data.Year_of_Release >= 1991]

# Finding the median sales value by genre and year
Med_Sales_by_Gen_and_Yr = pd.pivot_table(data,index=['Year_of_Release'],
                     columns=['Genre'],values=['Global_Sales'],aggfunc=np.median)
Med_Sales_by_Gen_and_Yr.columns = Med_Sales_by_Gen_and_Yr.columns.get_level_values(1)

Med_Sales_by_Gen_and_Yr.head()


# # 2.0 Analyzing the Data Set
# Simple linear regression can be used to explore basic increasing/decreasing trends.  

# In[ ]:


def Linear_Regression_Plot(Data):
    Regr_Coeff = []
    Regr_MSE = []
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10,12))

    x_data = np.transpose(np.matrix(Data.index))

    count = 0
    
    for genre in Data.columns:
        axs = axes[count//3,count%3]
        y_data = Data[genre].to_frame()
    
        # Linear regression
        regr = linear_model.LinearRegression()
        regr.fit(x_data,y_data)
        
        # Mean Squared Error
        MSE = np.mean((regr.predict(x_data)-y_data)**2)
        
        Regr_Coeff.append(regr.coef_[0][0])
        Regr_MSE.append(MSE[0])

        Data[genre].plot(ax=axs)
        axs.plot(x_data,regr.predict(x_data), color='black')

        y_lims = axs.get_ylim()
        
        
        txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,MSE)
        y_loc = 0.85*(y_lims[1]-y_lims[0])+y_lims[0]
        axs.text(2007,y_loc,txt)

        axs.set_title(genre)
        axs.set_xlabel('Year')
        axs.set_ylabel('Median')
        count+=1
    fig.tight_layout()
    
    return [Regr_Coeff,Regr_MSE]
    
[Regr_Coeff,Regr_MSE] = Linear_Regression_Plot(Med_Sales_by_Gen_and_Yr)


# Notice all but Shooter games have a negative trend; however, for many of the genres there is a large spike (an outlier) before 1995.  This initial spike in the data is affecting the slope of the trends and is not as relevant to recent sales. As a precaution, the same analysis can be performed starting after 1995 and will still include over 90% of the total global sales.

# In[ ]:


Med_Sales_by_Gen_and_Yr = Med_Sales_by_Gen_and_Yr.loc[Med_Sales_by_Gen_and_Yr.index >= 1995]

[Regr_Coeff_After_95,Regr_MSE_After_95] = Linear_Regression_Plot(Med_Sales_by_Gen_and_Yr)


# Even after excluding the years before 1995 all but Shooter games have shown a decline in sales. However, after excluding the years before 1995 the severity of the decline in all but the sports genre lessened.  Of the declining genres Simulation, Role-Playing, Racing, and Action games have incurred the steepest decline in sales.

# In[ ]:


Linear_Regression_Results = pd.DataFrame({'Regression Coeff After 1991':Regr_Coeff,
                                         'MSE After 1991':Regr_MSE,
                                         'Regression Coeff After 1995':Regr_Coeff_After_95,
                                         'MSE After 1995':Regr_MSE_After_95},
                                        index = list(Med_Sales_by_Gen_and_Yr.columns))
Column_Order = ['Regression Coeff After 1991','MSE After 1991','Regression Coeff After 1995',
                'MSE After 1995']

# Printing the linear regression results
Linear_Regression_Results[Column_Order].head(n=len(list(Med_Sales_by_Gen_and_Yr.columns)))


# With the decline in sales for all but one genre of video games, it would be normal to want to see if the same decline can be seen in the global sales for all genres.  By plotting the yearly median of all global sales, it is apparent there is an overall a decline in global sales.

# In[ ]:


Med_Sales_by_Yr = pd.pivot_table(data,index=['Year_of_Release'],
                     values=['Global_Sales'],aggfunc=np.median)


fig = plt.figure(figsize=(13,5))
Med_Sales_by_Yr.plot()

x_data = np.transpose(np.matrix(Med_Sales_by_Yr.index))
y_data = Med_Sales_by_Yr
regr = linear_model.LinearRegression()
regr.fit(x_data,y_data)

plt.plot(x_data,regr.predict(x_data), color='black')

txt = 'Coeff: %.3f \nMSE: %.3f' % (regr.coef_,np.mean((regr.predict(x_data)-y_data)**2))

plt.text(2011,0.8*Med_Sales_by_Yr.max(),txt)

plt.title('Median Global Sales')
plt.xlabel('Year')
plt.ylabel('Median Sales (in millions)')


# # 3.0 Conclusion
# With the exception of Shooter games the global sales of all other video game genres have been declining.  This could be due to the increase in games being played on tablets and smart phones.  First person shooter games are not easily played with touchscreen controls causing them to be unaffected by the increase in popularity of using tablets and smart phones as a gaming platform.  It would be interesting to find a data set that had the global downloads of games for tablets and smart phones and compare it with this data set.  If anyone has any interesting theories, please let me know.  Also I am still new to programming in Python and am open to any suggestions about creating a cleaner program.
