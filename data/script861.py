
# coding: utf-8

# **This is part two for my Exploratory Data Analysis for the gaming dataset. My part 1 can be found at:<br>**
# https://www.kaggle.com/etakla/d/rush4ratio/video-game-sales-with-ratings/exploring-the-dataset-univariate-analysis

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

from IPython.display import display, HTML
# Any results you write to the current directory are saved as output.

#For plotting
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


vg_df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
vg_df.User_Score = vg_df.User_Score.convert_objects(convert_numeric=True)


# # Bivariate General Overview

# ## Correlation

# In[ ]:


plt.figure(figsize=(12, 8))

vg_corr = vg_df.corr()
sns.heatmap(vg_corr, 
            xticklabels = vg_corr.columns.values,
            yticklabels = vg_corr.columns.values,
            annot = True);


# ## Crossplots

# In[ ]:


plt.figure(figsize=(14, 14))

sns.pairplot(vg_df, diag_kind='kde');


# # Evolution Over Time

# ## Sales vs Number of Releases

# In[ ]:


#Group the entries by year, then get how many entries are there; i.e. the number of releases
temp1 = vg_df.groupby(['Year_of_Release']).count()
temp1 = temp1.reset_index()

#Do the same, but sum the values to get the total values of everything by year.
temp2 = vg_df.groupby(['Year_of_Release']).sum()
temp2 = temp2.reset_index()

#Normalize the data, i.e. zero mean and unit std. I did this to be able to compare the shapes of both graphs, since 
#they have different ranges
normalised_df = pd.DataFrame()

normalised_df['release_count'] = temp1['Name']
normalised_df['global_sales'] = temp2['Global_Sales']
normalised_df = (normalised_df - normalised_df.mean()) / normalised_df.std()#(normalised_df.max() - normalised_df.min()) 
normalised_df['year'] = temp1['Year_of_Release']


#Plot
plt.figure(figsize=(15, 9))
ax = sns.pointplot(x = normalised_df.year, y = normalised_df.release_count, color = 'blue', label='Release Count')
ax = sns.pointplot(x = normalised_df.year, y = normalised_df.global_sales, color = 'red', label='Global Sales')

blue_patch = mpatches.Patch(color='blue', label='NUMBER OF RELEASES')
red_patch = mpatches.Patch(color='red', label='GLOBAL SALES')
plt.legend(handles=[blue_patch, red_patch], loc='upper left', fontsize = 16)

plt.xticks(rotation=45);


# The shapes follow each other well, there wasn't any weird too much sales with little releases or vice versa. The other thing to note is that the number of releases is smoother than the sales, which seem to harder to accurately predict.

# ## Genre Sales Evolution

# These are two plots, the first is an area plot to see the precentage of sales of each genre over the years. The second is the sales by year (Although still divided by genre). This last one is just for convenience, to remember how the games sales were.

# In[ ]:


fig = plt.figure(figsize=(10, 8))

genre_sales_percentages_by_year = (vg_df.groupby(['Year_of_Release', 'Genre']).Global_Sales.sum())*(100)/vg_df.groupby(['Year_of_Release']).Global_Sales.sum()
genre_sales_percentages_by_year.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', grid=False, figsize=(13, 4))

yearlySales = vg_df.groupby(['Year_of_Release','Genre']).Global_Sales.sum()
yearlySales.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(13, 4) ) ;


# The "action" genre is clearly what gamers are inclined to play the most. The interesting trend about it is that when the games sales started their decline around 2009, the "action" level remained **almost the same**. So this genre may seem to be resistent to hard-hit markets?  

# ## Genre Total Sales

# In[ ]:


x = vg_df.groupby(['Genre']).sum().copy()
ax = x.Global_Sales.sort_values(ascending=False).plot(kind='bar', figsize=(13, 5));

for p in ax.patches:
    ax.annotate(str( round( p.get_height() ) ) + "\n" + str(round( p.get_height() /89.170) )+ "%", 
                (p.get_x() * 1.007, p.get_height() * 0.75),
                color='black')


# Although there are changes in ranking, but there are no big surprises (By the number of releases, the rank was Action, Sports, Misc (Rank changed in sales), Role Playing, Shooter (Changed, sells more than it is released)..etc. For more information, refer to the first part, the univariate analysis)

# But these changes in ranking are interesting to explore more. I want to see how different genres perform according to different metrics. The ones I chose are:<br>
# 1) Number of Releases<br>
# 2) Total Sales<br>
# 3) Average Sales per Game

# In[ ]:


#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre
genre_difference_metric = [vg_df.Genre.value_counts().index, vg_df.groupby(['Genre']).sum().Global_Sales.sort_values(ascending=False).index, vg_df.groupby(['Genre']).mean().Global_Sales.sort_values(ascending=False).index]

#Dataframe to be used for plotting.
genre_evolution_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])

#Populate the dataframe
for metric in range(3):
    for genre in range(len(genre_difference_metric[metric])):
        genre_evolution_df = genre_evolution_df.append({'genre':genre_difference_metric[metric][genre], 'rank_type': metric, 'rank':genre},
                                   ignore_index=True)

        
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

sns.pointplot(x=genre_evolution_df.rank_type,
              y=12-genre_evolution_df['rank'], 
              hue=genre_evolution_df.genre)

for i in range(len(genre_difference_metric[0])):
    ax.text(-0.75, 12-i, genre_difference_metric[0][i], fontsize=11)
    ax.text(2.1, 12-i, genre_difference_metric[2][i], fontsize=11)
    
ax.set_xlim([-2,4])

xs = [0.0, 1.0, 2.0]
x_labels = ['total releases', 'total sales', 'average sales']
plt.xticks(xs, x_labels, rotation='vertical')

ax.set_xlabel('Sales Metric')

ys = range(1,13)
y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']
plt.yticks(ys, y_labels)
ax.set_ylabel('Genre Rank')

plt.show();


# I think that the graph is pretty interesting! Lots of interpretations can be made here.

# ## Rating Sales Evolution

# In[ ]:


rating_sales_percentages_by_year = (vg_df.groupby(['Year_of_Release', 'Rating']).Global_Sales.sum())*(100)/vg_df.groupby(['Year_of_Release']).Global_Sales.sum()
rating_sales_percentages_by_year.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(13, 4));


# This is a percentage area graph. So, the empty (grey) parts mean that a rating was missing.

# # Scores

# ## Critic vs User Scores

# In[ ]:


g = sns.jointplot(x = 'Critic_Score', 
              y = 'User_Score',
              data = vg_df, 
              kind = 'hex', 
              cmap= 'hot', 
              size=6)

#http://stackoverflow.com/questions/33288830/how-to-plot-regression-line-on-hexbins-with-seaborn
sns.regplot(vg_df.Critic_Score, vg_df.User_Score, ax=g.ax_joint, scatter=False, color='grey');


# The user scores appear to be more generous than the critic one. But which one has a better correlation with the sales? From the first graph, we can see that, clearly, the critic score correlates better with the sales. In fact, the user score does **not correlate at all** with the sales, it floats around the zero! (Except for the sales in Japan, but still a lower correlation than the critic's one)

# # Regional Sales

# ## Scatterplot between Global Sales and Regional Sales

# In[ ]:


sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
sales_normalised_df = vg_df[sales_cols].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.NA_Sales,    marker="+")
sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.EU_Sales,    marker=".")
sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.JP_Sales,    marker="x")
sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.Other_Sales, marker="o")

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
fig.tight_layout();


# ## Regional Sales of Genres

# In[ ]:


genre_geo_rankings = [vg_df.groupby('Genre').sum().unstack().NA_Sales.sort_values(ascending=False).index, 
                      vg_df.groupby('Genre').sum().unstack().EU_Sales.sort_values(ascending=False).index,
                      vg_df.groupby('Genre').sum().unstack().Other_Sales.sort_values(ascending=False).index,
                      vg_df.groupby('Genre').sum().unstack().JP_Sales.sort_values(ascending=False).index
                      ]

#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre
genre_geo_rank_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])

#for metric in genre_difference_metric:
for region in range(4):
    for genre in range(len(genre_geo_rankings[region])):
        genre_geo_rank_df = genre_geo_rank_df.append({'genre':genre_geo_rankings[region][genre], 'rank_type': region, 'rank':genre},
                                   ignore_index=True)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

sns.pointplot(x=genre_geo_rank_df.rank_type,
              y=12-genre_geo_rank_df['rank'], 
              hue=genre_geo_rank_df.genre)

for i in range(len(genre_geo_rankings[0])):
    ax.text(-0.9, 12-i, genre_geo_rankings[0][i], fontsize=11)
    ax.text(3.2, 12-i, genre_geo_rankings[3][i], fontsize=11)
    
ax.set_xlim([-2,5])

xs = [0.0, 1.0, 2.0, 3.0]
x_labels = ['North America', 'E.U.', 'Rest of the World', 'Japan']
plt.xticks(xs, x_labels, rotation='vertical')
ax.set_xlabel('Region')

ys = range(1,13)
y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']
plt.yticks(ys, y_labels)
ax.set_ylabel('Genre Rank')

plt.show();


# I think there are a lot of insights from this graph. Mainly, the taste of gamers in Japan is really different than that of the three other regions. The remaining three regions are, more or less, similar; with maybe only platform and racing genres moving more than one position between the three regions.

# # Most Selling Platform Each Year

# I will plot the top selling platform timeline. In the beginning the plot had all the platforms sales, but the result had too much information to digest, it was too clutered. The graph is followed by a table that describes the same thing, just for convenience

# In[ ]:


#temp is the sum of all variables for each platform by year
temp = vg_df.groupby(['Year_of_Release', 'Platform']).sum().reset_index().groupby('Year_of_Release')

platform_yearly_winner_df = pd.DataFrame()

for year, group in temp:
    current_year = temp.get_group(year)
    this_year_max_sales = 0.0
    current_year_winner = ""
    row = {'year':"", 'winner':"", 'sales':""}
    for index, platform_data in current_year.iterrows():
        if platform_data.Global_Sales > this_year_max_sales:
            this_year_max_sales = platform_data.Global_Sales
            current_year_winner = platform_data.Platform
    
    row['year'] = year
    row['winner'] = current_year_winner
    row['sales'] = this_year_max_sales
    platform_yearly_winner_df = platform_yearly_winner_df.append(row, ignore_index=True)

fig = plt.figure(figsize=(13, 4))

g = sns.pointplot(x = platform_yearly_winner_df.year ,
              y = platform_yearly_winner_df.sales , 
              hue = platform_yearly_winner_df.winner);

#http://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot
g.set_xticklabels(g.get_xticklabels(), rotation=90);


# In[ ]:


platform_yearly_winner_df.set_index('year', inplace=True)
HTML(platform_yearly_winner_df.to_html())


# # Publishers

# ## Top Publishers of all Time

# When thinking about what the term "top publisher" means, I think it has no simple answer. So I have decided to explore the matter from different points of view, and see if there are certain names that consistantly appear in all the points of view.

# ### By Sales

# In[ ]:


x = vg_df.groupby(['Publisher']).sum().Global_Sales.copy()
x.sort_values(ascending=False, inplace=True)
x.head(10)


# ### By Number of Releases

# In[ ]:


x = vg_df.groupby(['Publisher']).count().Name.copy()
x.sort_values(ascending=False, inplace=True)
x.head(10)


# The interesting part is that the publishers who made it to the top 10 in both lists are the same, with just a different ordering.

# ### By Average Yearly Earning

# In[ ]:


#http://stackoverflow.com/questions/30328646/python-pandas-group-by-in-group-by-and-average
vg_df.groupby(['Publisher', 'Year_of_Release'], as_index=False).mean().groupby('Publisher').mean().Global_Sales.sort_values(ascending=False).head(10)


# There was something that intuitively I didn't like about this last list. After some investigation, I found that there were some publishers who did not stay in the market for long, but had made some strong sales. I am not sure if this makes a publisher make it towards the top list of all time, but I have decided to add the condition that a top publisher must have been in the competition for more than 5 years. I am not sure if I am subconsciously pushing some names to reappear into this again, maybe because that makes it clearer who deserves to be on that list.

# In[ ]:


vg_df.groupby(['Publisher']).filter(lambda x: len(x) > 5).groupby(['Publisher', 'Year_of_Release'], as_index=False).mean().groupby('Publisher').mean().Global_Sales.sort_values(ascending=False).head(10)


# Still, some new names reappeared. But I am satisfied with the 5 years condition, so I am going to continue with this last list.

# ### By Average Earning per Game

# In[ ]:


x = vg_df.groupby(['Publisher']).mean().Global_Sales.copy()
x.sort_values(ascending=False, inplace=True)
x.head(10)


# Again, the same dilemma of publishers with little publishings making great sales and reappearing on the top list. This time, I am setting a condition of having released at least 10 games before competing for the first places. For example, the first place: Palcom. It is very interesting how, by a good margin, they are well ahead of the rest. 

# In[ ]:


vg_df[vg_df.Publisher == 'Palcom']


# So it released one game (And this makes sense to have this company on the top of the list). It was a big hit, making over 4 million 1989 dollars! Certainly impressive, but I am not sure if this is enough to make it one of the top publishers of all time. million 1989 dollars! Personally I don't like the fact that the average is topped by a publisher that made a single release, no matter how successful this release was. I will filter out publishers who published less than 10 games and then recompute the average

# In[ ]:


vg_df.groupby(['Publisher']).filter(lambda x: len(x) > 10).groupby(['Publisher']).Global_Sales.mean().sort_values(ascending=False).head(10)


# ## Top Publishers Final List

# So the list of top publisher is the union of the top 10 highest sales, top 10 highest release, top 10 highest average sales for publishers who have released more than 10 games and top 10 highest average sales for publishers who have been around for over 5 years. The "over 10 games" and "over 5 years" are just arbitrary values that seem reasonable to me, but feel free to experiment with these values (Change the value inside the lambda function within the filters)

# In[ ]:


top_publishers = ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment',                   'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive',
                  'Sony Computer Entertainment Europe', 'Microsoft Game Studios', 'Enix Corporation', 'Bethesda Softworks', 'SquareSoft'\
                  'Take-Two Interactive', 'LucasArts', '989 Studios', 'Hasbro Interactive', 'Universal Interactive']
#You can use set to create the list, I just have handtyped them just to be more attentive to the names.

top_publisher_df = vg_df[ vg_df['Publisher'].isin(top_publishers) ]


# ### How do these top publishers make-up of the gaming market?

# #### Number of releases

# In[ ]:


print("They make", 100*top_publisher_df.shape[0]/float(vg_df.shape[0]),"% of the number of releases in this dataset")


# #### Total Sales

# In[ ]:


total_games_sales = vg_df.Global_Sales.sum()
top_publisher_total_sales = top_publisher_df.Global_Sales.sum()

print("Total Video Games Sales:", total_games_sales, "Million US$")
print("Total Top Publishers Sales:", top_publisher_total_sales, "Million US$")
print("They make ", 100*top_publisher_total_sales/total_games_sales,"% of the total video games sales")


# ### Favorite Genre for Each Top Publisher

# #### By Number of Releases

# In[ ]:


x = top_publisher_df.groupby(['Publisher', 'Genre']).count().copy()
x.unstack().Name.idxmax(axis=1)


# #### By Sales

# In[ ]:


x = top_publisher_df.groupby(['Publisher', 'Genre']).sum().copy()
x.unstack().Global_Sales.idxmax(axis=1)


# ## Top Genre Producer

# #### By Number of Games Released

# In[ ]:


x = top_publisher_df.groupby(['Genre', 'Publisher']).count().copy()
x.unstack().Name.idxmax(axis=1)


# #### By Sales

# In[ ]:


x = top_publisher_df.groupby(['Genre', 'Publisher']).sum().copy()
x.unstack().Global_Sales.idxmax(axis=1)


# #### By Average Sales per game

# In[ ]:


x = top_publisher_df.groupby(['Genre', 'Publisher']).mean().copy()
x.unstack().Global_Sales.idxmax(axis=1)


# # Top Performers by Region

# ## Publishers

# #### North America

# Inside the bar, there will be the actual value of the total sales, followed by the publisher's share percentage of the sales in the region.

# In[ ]:


ax = vg_df.groupby('Publisher').sum().unstack().NA_Sales.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(13, 5));

for p in ax.patches:
    ax.annotate(str( round( p.get_height() ) ) + "\n" + str(round( 100.0* p.get_height() /vg_df.NA_Sales.sum()) )+ "%", 
                (p.get_x() + 0.13, p.get_height()-85),
                color='white', fontsize=12, fontweight='bold')


# #### European Union

# In[ ]:


vg_df.groupby('Publisher').sum().unstack().EU_Sales.sort_values(ascending=False).head(10).plot(kind='bar');


# #### Japan

# In[ ]:


vg_df.groupby('Publisher').sum().unstack().JP_Sales.sort_values(ascending=False).head(10).plot(kind='bar');


# #### Rest of the World

# In[ ]:


vg_df.groupby('Publisher').sum().unstack().Other_Sales.sort_values(ascending=False).head(10).plot(kind='bar');


# The pattern is similar in North America, E.U. and rest of the world. Again, Japan stands out as a region with its own peculiarities. 

# ## Genre

# #### North America

# In[ ]:


vg_df.groupby('Genre').sum().unstack().NA_Sales.sort_values(ascending=False).plot(kind='bar');


# #### European Union

# In[ ]:


vg_df.groupby('Genre').sum().unstack().EU_Sales.sort_values(ascending=False).head(10).plot(kind='bar');


# #### Japan

# In[ ]:


vg_df.groupby('Genre').sum().unstack().JP_Sales.sort_values(ascending=False).head(10).plot(kind='bar');


# We knew already that the Japanese taste for games was already different, but when we quantified the sales it showed an even more interesting insight: their games are dominated by the role-playing genre!

# #### Rest of the World

# In[ ]:


vg_df.groupby('Genre').sum().unstack().Other_Sales.sort_values(ascending=False).head(10).plot(kind='bar');


# # Honourable Mentions

# ## Most Profitable Games of all Time

# In[ ]:


vg_df.sort_values('Global_Sales', ascending=False).head(10).Name


# ## Most Profitable Games in each Genre

# In[ ]:


#There are games with duplicate names (For each platform for example), so let's deal with this
x = vg_df.groupby(['Genre', 'Name']).sum().reset_index().groupby('Genre')

#A dataframe that will hold rankings, for nice display
best_selling_titles_by_genre_df = pd.DataFrame()

for name, group in x:
    temp_col = group.sort_values('Global_Sales', ascending=False).head(10).Name.reset_index(drop=True)
    best_selling_titles_by_genre_df[name] = temp_col


# In[ ]:


best_selling_titles_by_genre_df


# My take over the best selling titles by genre:<br>
# 1- GTA dominates the action genre<br>
# 2- Wii sports stuff dominate the sports, followed by FIFAs<br>
# 3- Call of Duty dominates the shooting genre.<br>
# 4- The dataset need some cleaning, for example FIFA 2013 is listed as an action game while the rest of the FIFAs are listed as sports. Same for Assassin's Creed, some are listed as Action, others are adventure.<br>

# # Conclusion

# The sales and the number of releases go hand by hand, nothing special\suspicious.<br><br>
# 
# The "Action" genre seems to be the most resistant to sales decline.<br><br>
# 
# Genres seem to be reasonably persistent when we compare their number of releases and their revenues. However,  when we add to the picture the average revenue per game, the rankings change a lot:

# In[ ]:


#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre
genre_difference_metric = [vg_df.Genre.value_counts().index, vg_df.groupby(['Genre']).sum().Global_Sales.sort_values(ascending=False).index, vg_df.groupby(['Genre']).mean().Global_Sales.sort_values(ascending=False).index]

#Dataframe to be used for plotting.
genre_evolution_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])

#Populate the dataframe
for metric in range(3):
    for genre in range(len(genre_difference_metric[metric])):
        genre_evolution_df = genre_evolution_df.append({'genre':genre_difference_metric[metric][genre], 'rank_type': metric, 'rank':genre},
                                   ignore_index=True)

        
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

sns.pointplot(x=genre_evolution_df.rank_type,
              y=12-genre_evolution_df['rank'], 
              hue=genre_evolution_df.genre)

for i in range(len(genre_difference_metric[0])):
    ax.text(-0.75, 12-i, genre_difference_metric[0][i], fontsize=11)
    ax.text(2.1, 12-i, genre_difference_metric[2][i], fontsize=11)
    
ax.set_xlim([-2,4])

xs = [0.0, 1.0, 2.0]
x_labels = ['total releases', 'total sales', 'average sales']
plt.xticks(xs, x_labels, rotation='vertical')

ax.set_xlabel('Sales Metric')

ys = range(1,13)
y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']
plt.yticks(ys, y_labels)
ax.set_ylabel('Genre Rank')

plt.show();


# <br><br><br>The critic score is more conservative than the user score, but it correlates much better with the sales. In fact, user score does not correlate at all (coefficient almost equal zero) with the games sales:<br>

# In[ ]:


g = sns.jointplot(x = 'Critic_Score', 
              y = 'User_Score',
              data = vg_df, 
              kind = 'hex', 
              cmap= 'hot', 
              size=6)

#http://stackoverflow.com/questions/33288830/how-to-plot-regression-line-on-hexbins-with-seaborn
sns.regplot(vg_df.Critic_Score, vg_df.User_Score, ax=g.ax_joint, scatter=False, color='grey');


# <br><br>Japan is a unique region when it comes to games. The rest of the world is more or less consistent, but Japan is different. For example, this is how the different genres sales ranked within each of the four regions:

# In[ ]:


genre_geo_rankings = [vg_df.groupby('Genre').sum().unstack().NA_Sales.sort_values(ascending=False).index, 
                      vg_df.groupby('Genre').sum().unstack().EU_Sales.sort_values(ascending=False).index,
                      vg_df.groupby('Genre').sum().unstack().Other_Sales.sort_values(ascending=False).index,
                      vg_df.groupby('Genre').sum().unstack().JP_Sales.sort_values(ascending=False).index
                      ]

#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre
genre_geo_rank_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])

#for metric in genre_difference_metric:
for region in range(4):
    for genre in range(len(genre_geo_rankings[region])):
        genre_geo_rank_df = genre_geo_rank_df.append({'genre':genre_geo_rankings[region][genre], 'rank_type': region, 'rank':genre},
                                   ignore_index=True)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

sns.pointplot(x=genre_geo_rank_df.rank_type,
              y=12-genre_geo_rank_df['rank'], 
              hue=genre_geo_rank_df.genre)

for i in range(len(genre_geo_rankings[0])):
    ax.text(-0.9, 12-i, genre_geo_rankings[0][i], fontsize=11)
    ax.text(3.2, 12-i, genre_geo_rankings[3][i], fontsize=11)
    
ax.set_xlim([-2,5])

xs = [0.0, 1.0, 2.0, 3.0]
x_labels = ['North America', 'E.U.', 'Rest of the World', 'Japan']
plt.xticks(xs, x_labels, rotation='vertical')
ax.set_xlabel('Region')

ys = range(1,13)
y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']
plt.yticks(ys, y_labels)
ax.set_ylabel('Genre Rank')

plt.show();


# The Japanese also showed a strong inclination to Nintendo as a publisher, and for the "Role Playing" genre.

# The top publishers list is a list made of the top 20 publishers that consistently made it to the top 10 lists of different metrics (e.g. top selling games, top total sales..etc)

# <br>The top publishers are:<br>
# Electronic Arts<br>
# Activision<br>
# Namco Bandai Games<br>
# Ubisoft<br>
# Konami Digital Entertainment<br>
# THQ<br>
# Nintendo<br>
# Sony Computer Entertainment<br>
# Sega<br>
# Take-Two Interactive<br>
# Sony Computer Entertainment Europe<br>
# Microsoft Game Studios<br>
# Enix Corporation<br>
# Bethesda Softworks<br>
# SquareSoft<br>
# Take-Two Interactive<br>
# LucasArts<br>
# 989 Studios<br>
# Hasbro Interactive<br>
# Universal Interactive<br>

# These 20 companies make up a little over half (51.85%) of all the games produced, and a little over three quarters (76%) of all the global sales. This follows the general pattern found everywhere in this dataset: the exponential decay.
