
# coding: utf-8

# #Looking at Global Inequality. 
# Investigating global inequality based on GNI per capita and wealth distributions over time.
# 
# 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns



pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

document = pd.read_csv('../input/Indicators.csv')

#want to see all the countries listed in the document  
document['CountryName'].unique()

#get rid of indicators that aren't countries 
list = ['Arab World', 'Caribbean small states', 'Central Europe and the Baltics',
 'East Asia & Pacific (all income levels)',
 'East Asia & Pacific (developing only)', 'Euro area',
 'Europe & Central Asia (all income levels)',
 'Europe & Central Asia (developing only)', 'European Union',
 'Fragile and conflict affected situations',
 'Heavily indebted poor countries (HIPC)', 'High income',
 'High income: nonOECD', 'High income: OECD',
 'Latin America & Caribbean (all income levels)',
 'Latin America & Caribbean (developing only)',
 'Least developed countries: UN classification', 'Low & middle income',
 'Low income', 'Lower middle income',
 'Middle East & North Africa (all income levels)',
 'Middle East & North Africa (developing only)', 'Middle income',
 'North America' 'OECD members' ,'Other small states',
 'Pacific island small states', 'Small states', 'South Asia',
 'Sub-Saharan Africa (all income levels)',
 'Sub-Saharan Africa (developing only)' ,'Upper middle income' ,'World', 'North America', 'OECD members']






# ##Identifying the "poor" countries
# What are the 15 countries that had the lowest average incomes from 1960-2014?

# In[8]:


document.head(10)


# In[22]:



years = document.loc[document['IndicatorCode'] == 'NY.GNP.PCAP.CD',['Year']].Year.unique()



# In[ ]:


lowestGNI_2014 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 2014").sort_values(by = 'Value', ascending = True)[:15]
lowestGNI_1960 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 1962").sort_values(by = 'Value', ascending = True)[:15]
    

fig = plt.subplots()

graph1 = sns.barplot(x = "Value", y = "CountryName", palette = "PuBu", data = lowestGNI_1960)
plt.xlabel('Average Income ($)', fontsize = 14)
plt.ylabel('Country',  fontsize=14)
plt.title('The 15 Countries with Lowest Average Income in 1962', fontsize = 14)






# In[ ]:


fig2 = plt.subplots()

graph2 = sns.barplot(x = "Value", y = "CountryName", palette = "PuBu", data = lowestGNI_2014)
plt.xlabel('Average Income($)', fontsize = 14)
plt.ylabel('Country', fontsize = 14)
plt.title('The 15 Countries with Lowest Average Income in 2014', fontsize = 14)



# ###Which countries have consistently been 'poor'?

# In[ ]:


for key,group in lowestGNI_1960.groupby(['CountryName']):
    for key2, group2 in lowestGNI_2014.groupby(['CountryName']):
        if key == key2:
            print (key)


# It is interesting to note the geographic differences of low income countries in 1962 and 2014. In 1962, 5 of the 15 countries (China, India, Korea, Pakistan, Nepal) with lowest average income in the world were located in Asia. Switch to 2014 and it is interesting to note that every single one of the countries with lowest income in the world are all African with the exception of Afghanistan. 
# 
# By running a simple for loop after visualizing the data, it can be seen that 4 countries- Burundi, Central African Republic, Malawi, and Togo- have been in the poorest 15 in both the past (1960s) and the present.

# ##Identifying the "rich" countries
# What countries had the highest average incomes in both 1960 and 2014?

# In[ ]:


rich_1960 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 1962").sort_values(by = 'Value')[-15:]
rich_2014 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 2014").sort_values(by= 'Value')[-15:]


# In[ ]:


fig = plt.subplots()

graph_rich = sns.barplot(x = "Value", y = "CountryName", palette = "BuGn", data = rich_1960)
plt.xlabel('Average Income ($)', fontsize = 14)
plt.ylabel('Country',  fontsize=14)
plt.title('The 15 Countries with Highest Average Income in 1960', fontsize = 14)




# In[ ]:


fig = plt.subplots()

graph_rich2 = sns.barplot(x = "Value", y = "CountryName", palette = "BuGn", data = rich_2014)
plt.xlabel('Average Income ($)', fontsize = 14)
plt.ylabel('Country',  fontsize=14)
plt.title('The 15 Countries with Highest Average Income in 2014', fontsize = 14)



# ###Which countries have consistently been 'rich'?

# In[ ]:


for key, group in rich_1960.groupby(['CountryName']):
    for key2, group2 in rich_2014.groupby(['CountryName']):
        if key == key2:
            print (key)


# There are a lot of unique attributes to note about the 'rich' countries. For they most part, they are located in the Western world- particularly in W. Europe, Scandinavia, and N.America. 9 of the 15 countries were wealthy in both the past (1960s) and the present- suggesting that having wealth in the past is a big indicator of having wealth in the present. 
# Another interesting thing to note is the appearance of newer 'rich' countries from the Middle East (Qatar, Kuwait) and East Asia (Macao, Singapore). 
# Furthermore, it can also be seen that transitioning from 1960s to the present, the average incomes increased significantly in the developed world- suggesting that wealth has been accumulating much quicker in only certain pockets of the planet.

# ##Comparing 'rich', 'emerging', and 'poor' countries

# ###Tracking Average Income from 1960-2014 

# In[ ]:


fig8, ax8 = plt.subplots(figsize = [15,8], ncols = 2)
ax6, ax7 = ax8

labels = []
GNP_revised = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['Australia','Austria','Canada', 'Luxembourg', 'Netherlands','Norway','United States']").groupby(['CountryName'])
for key, group in GNP_revised:
    ax6 = group.plot(ax = ax6, kind = "line", x = "Year", y = "Value", title = "Average Income from 1960-2014 in 'Rich' Countries")
    labels.append(key)

lines, _ = ax6.get_legend_handles_labels()
ax6.legend(lines, labels, loc='best')

labels2 = []
GNP_revised = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['Burundi', 'Togo', 'Malawi', 'Central African Republic']").groupby(['CountryName'])
for key, group in GNP_revised:
    ax7 = group.plot(ax = ax7, kind = "line", x = "Year", y = "Value", title = "Average Income from 1960-2014 in 'Poor' Countries")
    labels2.append(key)

lines, _ = ax7.get_legend_handles_labels()
ax7.legend(lines, labels2, loc='best')


# There are some noticeable observations here:
# 
# 1) Growth in income for "rich" countries is much greater than the growth in income for the "poorer" countries.
# 
# 2) In "poor" countries, we see that growth in income is not as steady as in "rich" countries. Since change in income is so much smaller overtime, even slight income changes appear very dramatic for "poor" countries.
# 

# ###Wealth is acculumating faster in certain countries than others

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [15,8], sharex = True)
income_query = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & Year == 1962 & CountryName == ['Malawi', 'China', 'Luxembourg', 'United States']")
income_query_graph = sns.barplot(x = 'CountryName', y = 'Value', order = ['Malawi', 'China', 'Luxembourg', 'United States'], ax = ax1, data = income_query)
ax1.set_title("Average Income in 1962", fontsize = 14)
ax1.set_xlabel('Country', fontsize = 14)
ax1.set_ylabel('Average Income ($)', fontsize = 14)

for p in income_query_graph.patches:
    height = p.get_height()
    income_query_graph.text(p.get_x() + p.get_width()/2., 1.05*height,
                '%d' % int(height), ha='center', va='bottom')
    
income_query_now=document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & Year == 2014 & CountryName == ['Malawi', 'China', 'Luxembourg', 'United States']")
income_query_now_graph = sns.barplot(x = 'CountryName', y = 'Value', order = ['Malawi', 'China', 'Luxembourg', 'United States'], ax = ax2, data = income_query_now)
ax2.set_title("Average Income in 2014", fontsize = 14)
ax2.set_xlabel('Country', fontsize = 14)
ax2.set_ylabel('Average Income ($)', fontsize = 14)
plt.ylim([0,90000])

for p in income_query_now_graph.patches:
    height = p.get_height()
    income_query_now_graph.text(p.get_x() + p.get_width()/2., 1.05*height,
                '%d' % int(height), ha='center', va='bottom')


# It turns out that compared to Luxembourg and United States, citizens of both Malawi and China had substantially lower incomes in 1962. The average citizen in Malawi only made $50/yr in 1962! The average American citizen made almost 7x's the average Malawian citizen!
# 
# So yes, the "rich" countries had a higher leg up compared to the "poor" countries!
# 
# It is also interesting to note that at $250 per year, the average income of a Malawian in 2014 is still substantially less than the average income of $3280 per yer of an American citizen in 1962. 
# 
# In fact, this is an interesting trend to note for several of the current 'poor' countries. The average incomes of these 'poor' countries is still considerably less compared to the incomes of 'rich' countries 40 years ago. This suggests that not only did 'poor' countries start off on an uneven foot, economic growth is affecting 'rich' countries significantly more than they are the 'poor' countries. This phenomena is occurring to the point that the poorest countries could not even achieve the incomes of the 'rich' countries 40 years ago before the 'rich' countries enormous growth in income.  

# In[ ]:


a = pd.Series(income_query_now['Value'].reset_index(drop = True))
b = pd.Series(income_query['Value'].reset_index(drop = True))
ratio = a/b

income_ratio = sns.barplot(x = ['China', 'Luxembourg', 'Malawi', 'United States'], y = ratio, order = ['China', 'Luxembourg', 'United States', 'Malawi'])
plt.title('Measuring Income Growth- Which countries have seen the most change in incomes?', fontsize = 11)
plt.xlabel('Country', fontsize = 10)
plt.ylabel('Income Ratio (2014 Income/1962 Income)', fontsize = 10)

for p in income_ratio.patches:
    height = p.get_height()
    income_ratio.text(p.get_x() + p.get_width()/2., 1.05*height,
                '%d' % int(height), ha='center', va='bottom')


# Change in income has been the most radical for China. The average income of a Chinese citizen has increased by over 100x's. China's insane growth makes Luxembourg and US considerable growth look miniscule. While in absolute numbers, China still lags behind 'rich' coutries in absolute income numbers, China's actual growth measured in terms of ratios is incredible.
# 
# Indeed, China has done a good itself lifting itself out of poverty going from being one of the poorest countries in the world in the 60s to garnering itself in the ranks of being a middle income economy. 
# 
# That is the good news. 
# 
# The bad news is that 'poor' countries such as Malawi have not encountered much income growth. Not only did these countries make less compared to other countries in the 60s, they have also not been able to keep up with other countries in terms of growth. 

# ###China vs Malawi-- how do their average incomes compare overtime?

# In[ ]:


fig11, ax21 = plt.subplots(figsize = [15,8])
labels_cGNP = []
for key, group in document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['China', 'Malawi']").groupby(['CountryName']):
    ax21 = group.plot(ax=ax21, kind = "line", x = "Year", y = "Value", title = "Comparing average incomes- China vs Malawi")
    labels_cGNP.append(key)

lines, _ = ax21.get_legend_handles_labels()
ax21.legend(lines, labels_cGNP, loc = 'best')


# China is truly the miracle economy of the world. It's growth is literally exponential. 

# ###China vs Malawi vs Luxembourg vs United States-- how their average incomes compare overtime

# In[ ]:


fig12, axs12 = plt.subplots(figsize = [15,8])
labels_cross3pt2 = []
for key, group in document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['China', 'Malawi', 'Luxembourg', 'United States']").groupby(['CountryName']):
    axs12 = group.plot(ax = axs12, kind = "line", x = "Year", y = "Value", title = "Comparing average income- China vs Malawi vs Luxembourg vs US")
    labels_cross3pt2.append(key)

lines,_ = axs12.get_legend_handles_labels()
axs12.legend(lines, labels_cross3pt2, loc = 'best')


# Even though China has showed extraordinary growth, it by no mean has an average income that compares to those of the 'rich' countries. China's growth is most definitely explosive, but it still has a long way to go to achieve the average income of a 'rich' country. While it's growth curve is by far the most impressive out of the 4 countries, there is still an obvious divide in income with China and Malawi belonging on one side and Luxembourg and the US on another.

# ##Wealth Distributions - Malawi vs China vs United States vs Luxembourg

# In[ ]:


income_share = document.query("IndicatorCode == ['SI.DST.FRST.20','SI.DST.02ND.20','SI.DST.03RD.20','SI.DST.04TH.20','SI.DST.05TH.20'] & CountryName == ['Malawi', 'China', 'Luxembourg', 'United States'] & Year == 2010 ").groupby("IndicatorCode")
N = 4
i1 = income_share.get_group('SI.DST.FRST.20')['Value']
i2 = income_share.get_group('SI.DST.02ND.20')['Value']
i3 = income_share.get_group('SI.DST.03RD.20')['Value']
i4 = income_share.get_group('SI.DST.04TH.20')['Value']
i5 = income_share.get_group('SI.DST.05TH.20')['Value']

f, ax_1 = plt.subplots(1, figsize = (15,8))
ind = np.arange(N)
width = 0.35
p1 = ax_1.bar(ind, i1, width, color = '#404040')
p2 = ax_1.bar(ind, i2, width, color = '#bababa', bottom = i1)
p3 = ax_1.bar(ind, i3, width, color = '#ffffff', bottom = [i+j for i,j in zip(i1,i2)])
p4 = ax_1.bar(ind, i4, width, color = '#f4a582', bottom = [i+j+k for i,j,k in zip(i1,i2,i3)])
p5 = ax_1.bar(ind, i5, width, color = '#ca0020', bottom = [i+j+k+l for i,j,k,l in zip(i1,i2,i3,i4)])
plt.ylabel('Percent', fontsize = 14)
plt.xlabel('Country Name', fontsize = 14)
plt.xticks(ind + (width/2), ('China', 'Luxembourg', 'Malawi', 'United States'))
plt.title('Examining wealth distributions- China, Luxembourg, Malawi, and US', fontsize = 14)
plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0]),('Income Share Lowest 20%', 'Income Share Second 20%', 'Income Share Third 20%', 'Income Share Fourth 20%', 'Income Share Highest 20%'), loc = 'upper right', bbox_to_anchor=(1.3, 0.9))
axes = plt.gca()
axes.set_ylim([0,100])


# China and Malawi have more skewed wealth distributions compared to Luxembourg and United States. Luxembourg has the most even wealth distribution while Malawi has the most skewed wealth distribution with the most of its income share beloging to the highest 20%.
# 
# However, wealth distributions do not change radically alter based on average income of the country. Regardless of the average income for a country, it can be clearly seen that income share is mostly allocated to the highest 20%!
