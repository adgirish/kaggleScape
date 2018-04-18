
# coding: utf-8

# # About the dataset
# 
# Context
# Our world population is expected to grow from 7.3 billion today to 9.7 billion in the year 2050. Finding solutions for feeding the growing world population has become a hot topic for food and agriculture organizations, entrepreneurs and philanthropists. These solutions range from changing the way we grow our food to changing the way we eat. To make things harder, the world's climate is changing and it is both affecting and affected by the way we grow our food – agriculture. This dataset provides an insight on our worldwide food production - focusing on a comparison between food produced for human consumption and feed produced for animals.
# 
# Content
# The Food and Agriculture Organization of the United Nations provides free access to food and agriculture data for over 245 countries and territories, from the year 1961 to the most recent update (depends on the dataset). One dataset from the FAO's database is the Food Balance Sheets. It presents a comprehensive picture of the pattern of a country's food supply during a specified reference period, the last time an update was loaded to the FAO database was in 2013. The food balance sheet shows for each food item the sources of supply and its utilization. This chunk of the dataset is focused on two utilizations of each food item available:
# 
# Food - refers to the total amount of the food item available as human food during the reference period.
# Feed - refers to the quantity of the food item available for feeding to the livestock and poultry during the reference period.
# Dataset's attributes:
# 
# Area code - Country name abbreviation
# Area - County name
# Item - Food item
# Element - Food or Feed
# Latitude - geographic coordinate that specifies the north–south position of a point on the Earth's surface
# Longitude - geographic coordinate that specifies the east-west position of a point on the Earth's surface
# Production per year - Amount of food item produced in 1000 tonnes
# 
# This is a simple exploratory notebook that heavily expolits pandas and seaborn

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# importing data
df = pd.read_csv("../input/FAO.csv",  encoding = "ISO-8859-1")
pd.options.mode.chained_assignment = None


# Let's see what the data looks like...

# In[ ]:


df.head()


# # Plot for annual produce of different countries with quantity in y-axis and years in x-axis

# In[ ]:


area_list = list(df['Area'].unique())
year_list = list(df.iloc[:,10:].columns)

plt.figure(figsize=(24,12))
for ar in area_list:
    yearly_produce = []
    for yr in year_list:
        yearly_produce.append(df[yr][df['Area'] == ar].sum())
    plt.plot(yearly_produce, label=ar)
plt.xticks(np.arange(53), tuple(year_list), rotation=60)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode="expand", borderaxespad=0.)
plt.show()


# Clearly, India, USA and China stand out here. So, these are the countries with most food and feed production.
# 
# Now, let's have a close look at their food and feed data
# 
# # Food and feed plot for the whole dataset

# In[ ]:


sns.factorplot("Element", data=df, kind="count")


# So, there is a huge difference in food and feed production. Now, we have obvious assumptions about the following plots after looking at this huge difference.
# 
# # Food and feed plot for the largest producers(India, USA, China)

# In[ ]:


sns.factorplot("Area", data=df[(df['Area'] == "India") | (df['Area'] == "China, mainland") | (df['Area'] == "United States of America")], kind="count", hue="Element", size=8, aspect=.8)


# Though, there is a huge difference between feed and food production, these countries' total production and their ranks depend on feed production.

# 
# 
# Now, we create a dataframe with countries as index and their annual produce as columns from 1961 to 2013.

# In[ ]:


new_df_dict = {}
for ar in area_list:
    yearly_produce = []
    for yr in year_list:
        yearly_produce.append(df[yr][df['Area']==ar].sum())
    new_df_dict[ar] = yearly_produce
new_df = pd.DataFrame(new_df_dict)

new_df.head()


# Now, this is not perfect so we transpose this dataframe and add column names.

# In[ ]:


new_df = pd.DataFrame.transpose(new_df)
new_df.columns = year_list

new_df.head()


# Perfect! Now, we will do some feature engineering.
# 
# # First, a new column which indicates mean produce of each state over the given years. Second, a ranking column which ranks countries on the basis of mean produce.

# In[ ]:


mean_produce = []
for i in range(174):
    mean_produce.append(new_df.iloc[i,:].values.mean())
new_df['Mean_Produce'] = mean_produce

new_df['Rank'] = new_df['Mean_Produce'].rank(ascending=False)

new_df.head()


# Now, we create another dataframe with items and their total production each year from 1961 to 2013

# In[ ]:


item_list = list(df['Item'].unique())

item_df = pd.DataFrame()
item_df['Item_Name'] = item_list

for yr in year_list:
    item_produce = []
    for it in item_list:
        item_produce.append(df[yr][df['Item']==it].sum())
    item_df[yr] = item_produce


# In[ ]:


item_df.head()


# # Some more feature engineering
# 
# This time, we will use the new features to get some good conclusions.
# 
# # 1. Total amount of item produced from 1961 to 2013
# # 2. Providing a rank to the items to know the most produced item

# In[ ]:


sum_col = []
for i in range(115):
    sum_col.append(item_df.iloc[i,1:].values.sum())
item_df['Sum'] = sum_col
item_df['Production_Rank'] = item_df['Sum'].rank(ascending=False)

item_df.head()


# # Now, we find the most produced food items in the last half-century

# In[ ]:


item_df['Item_Name'][item_df['Production_Rank'] < 11.0].sort_values()


# So, cereals, fruits and maize are the most produced items in the last 50 years
# 
# # Food and feed plot for most produced items 

# In[ ]:


sns.factorplot("Item", data=df[(df['Item']=='Wheat and products') | (df['Item']=='Rice (Milled Equivalent)') | (df['Item']=='Maize and products') | (df['Item']=='Potatoes and products') | (df['Item']=='Vegetables, Other') | (df['Item']=='Milk - Excluding Butter') | (df['Item']=='Cereals - Excluding Beer') | (df['Item']=='Starchy Roots') | (df['Item']=='Vegetables') | (df['Item']=='Fruits - Excluding Wine')], kind="count", hue="Element", size=20, aspect=.8)


# # Now, we plot a heatmap of correlation of produce in difference years

# In[ ]:


year_df = df.iloc[:,10:]
fig, ax = plt.subplots(figsize=(16,10))
sns.heatmap(year_df.corr(), ax=ax)


# So, we gather that a given year's production is more similar to its immediate previous and immediate following years.

# In[ ]:


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10,10))
ax1.set(xlabel='Y1968', ylabel='Y1961')
ax2.set(xlabel='Y1968', ylabel='Y1963')
ax3.set(xlabel='Y1968', ylabel='Y1986')
ax4.set(xlabel='Y1968', ylabel='Y2013')
sns.jointplot(x="Y1968", y="Y1961", data=df, kind="reg", ax=ax1)
sns.jointplot(x="Y1968", y="Y1963", data=df, kind="reg", ax=ax2)
sns.jointplot(x="Y1968", y="Y1986", data=df, kind="reg", ax=ax3)
sns.jointplot(x="Y1968", y="Y2013", data=df, kind="reg", ax=ax4)
plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)
plt.savefig('joint.png')


# # Heatmap of production of food items over years
# 
# This will detect the items whose production has drastically increased over the years

# In[ ]:


new_item_df = item_df.drop(["Item_Name","Sum","Production_Rank"], axis = 1)
fig, ax = plt.subplots(figsize=(12,24))
sns.heatmap(new_item_df,ax=ax)
ax.set_yticklabels(item_df.Item_Name.values[::-1])
plt.show()


# There is considerable growth in production of Palmkernel oil, Meat/Aquatic animals, ricebran oil, cottonseed, seafood, offals, roots, poultry meat, mutton, bear, cocoa, coffee and soyabean oil.
# There has been exceptional growth in production of onions, cream, sugar crops, treenuts, butter/ghee and to some extent starchy roots.

# Now, we look at clustering.

# # What is clustering?
# Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data mining, and a common technique for statistical data analysis, used in many fields, including machine learning, pattern recognition, image analysis, information retrieval, bioinformatics, data compression, and computer graphics.

# # Today, we will form clusters to classify countries based on productivity scale

# For this, we will use k-means clustering algorithm.
# # K-means clustering
# (Source [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm) )
# ![http://gdurl.com/5BbP](http://gdurl.com/5BbP)

# This is the data we will use.

# In[ ]:


new_df.head()


# In[ ]:


X = new_df.iloc[:,:-2].values

X = pd.DataFrame(X)
X = X.convert_objects(convert_numeric=True)
X.columns = year_list


# # Elbow method to select number of clusters
# This method looks at the percentage of variance explained as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn't give much better modeling of the data. More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion". This "elbow" cannot always be unambiguously identified. Percentage of variance explained is the ratio of the between-group variance to the total variance, also known as an F-test. A slight variation of this method plots the curvature of the within group variance.
# # Basically, number of clusters = the x-axis value of the point that is the corner of the "elbow"(the plot looks often looks like an elbow)

# In[ ]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# As the elbow corner coincides with x=2, we will have to form **2 clusters**. Personally, I would have liked to select 3 to 4 clusters. But trust me, only selecting 2 clusters can lead to best results.
# Now, we apply k-means algorithm.

# In[ ]:


kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(X)

X = X.as_matrix(columns=None)


# Now, let's visualize the results.

# In[ ]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100,c='red',label='Others')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100,c='blue',label='China(mainland),USA,India')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of countries by Productivity')
plt.legend()
plt.show()


# So, the blue cluster represents China(Mainland), USA and India while the red cluster represents all the other countries.
# This result was highly probable. Just take a look at the plot of cell 3 above. See how China, USA and India stand out. That has been observed here in clustering too.
# 
# You should try this algorithm for 3 or 4 clusters. Looking at the distribution, you will realise why 2 clusters is the best choice for the given data

# This is not the end! More is yet to come.
