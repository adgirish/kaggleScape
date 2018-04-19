
# coding: utf-8

# <h1> Welcome to my Kernel </h1><br>
# 
# I will start this Kernel and will do some updates with new analysis !<br>
# 
# I hope you all like this exploration<br>
# 
# <h2>About this Dataset</h2><br>
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.
# <br>
# <i>It's a great dataset for evaluating simple regression models.,</i><br>
# <br>
# <i>* English is not my first language, so sorry about any error</i>
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import scipy.stats as st


# In[ ]:


df_usa = pd.read_csv("../input/kc_house_data.csv")


# In[ ]:


print(df_usa.shape)
print(df_usa.nunique())


# In[ ]:


print(df_usa.info())


# In[ ]:


df_usa.head()


# Knowning the Price variable

# In[ ]:


plt.figure(figsize = (8, 5))
plt.title('Price Distribuition')
sns.distplot(df_usa['price'])

plt.show()


# In[ ]:


print("Price Min")
print(df_usa['price'].min())
print("Price Mean")
print(df_usa['price'].mean())
print("Price Median")
print(df_usa['price'].median())
print("Price Max")
print(df_usa['price'].max())
print("Price Std")
print(df_usa['price'].std())


# In[ ]:


plt.figure(figsize = (8, 5))
sns.jointplot(df_usa.sqft_living, df_usa.price, 
              alpha = 0.5)
plt.xlabel('Sqft Living')
plt.ylabel('Sale Price')
plt.show()


# In[ ]:


condition = df_usa['condition'].value_counts()

print("Condition counting: ")
print(condition)

fig, ax = plt.subplots(ncols=2, figsize=(15,7))
sns.countplot(x='condition', data=df_usa, ax=ax[0])
sns.boxplot(x='condition', y= 'price',
            data=df_usa, ax=ax[1])
plt.show()


# In[ ]:


plt.figure(figsize = (12,8))
g = sns.FacetGrid(data=df_usa, hue='condition',size= 5, aspect=2)
g.map(plt.scatter, "sqft_living", "price")
plt.show()


# How can I plot this scatter together the graph above using seaborn ??? 

# <h1>Exploring bathrooms columns by price and conditions

# In[ ]:


df_usa["bathrooms"] = df_usa['bathrooms'].round(0).astype(int)

print("Freuency bathroom description:")
print(df_usa["bathrooms"].value_counts())

plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

ax1 = plt.subplot(221)
ax1 = sns.countplot(x="bathrooms", data=df_usa,
                    ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_title("Bathrooms counting", fontsize=15)
ax1.set_xlabel("Bathrooms number")
ax1.set_xlabel("count")

ax2 = plt.subplot(222)
ax2 = sns.boxplot(x="bathrooms", y='price',
                  data=df_usa, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.set_title("Bathrooms distribution price", fontsize=15)
ax2.set_xlabel("Bathrooms number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.stripplot(x="bathrooms", y="price",
                    data=df_usa, alpha=0.5,
                    jitter=True, hue="condition")
ax0.set_title("Better view distribuition through price", fontsize=15)
ax0.set_xlabel("Bathroom number")
ax0.set_ylabel("log Price(US)")
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)

plt.show()


# <h2>HOW CAN I SUBPLOTS ONE TYPE OF SCCATER THAT ACCEPTS HUE ??</h2>

# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

plt.figure(figsize = (12,6))
ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)
ax1.set_color_cycle(sns.color_palette('hls', 10))
for val in range(1,6,1):
    indeX = df_usa.condition == val
    ax1.scatter(df_usa.sqft_living.loc[indeX], df_usa.price.loc[indeX], label = val, alpha=0.5)
ax1.legend(bbox_to_anchor = [1.1, 1])
ax1.set_xlabel('sqfit living area')
ax1.set_ylabel('Price house')
ax1.set_title('Sqft Living - Price w.r.t Conditions')

ax2 = plt.subplot2grid((2,2), (1,0))
sns.boxplot(x = 'condition', y = 'price', data = df_usa, ax = ax2)
ax2.set_title('Box Plot Condition & Price', fontsize = 12)

ax3 = plt.subplot2grid((2,2), (1,1))
cubicQual = df_usa.groupby(['condition'])['price'].mean().round(0)
testTrain = df_usa.loc[:, ['condition', 'price']].copy()
testTrain['sqCond'] = np.power(testTrain['condition'],2)
mdl = linear_model.LinearRegression()
mdl.fit(testTrain[['condition', 'sqCond']], testTrain['price'])
y_pred = mdl.predict(testTrain[['condition', 'sqCond']])
print("Mean squared error: %.2f" % mean_squared_error(y_pred, testTrain.price))
# Plot outputs
ax3.scatter(testTrain['condition'], testTrain['price'],  color='black')
ax3.plot(testTrain['condition'], y_pred, color='blue', linewidth=3)
ax3.set_title('LinReg, price ~ condtion + sqft_cond', fontsize = 12)
ax3.set_xlabel('Condition Rate')
plt.subplots_adjust(hspace = 0.5, top = 0.9)
plt.suptitle('Condition Effect to Sale Price', fontsize = 14)
plt.show()


# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

plt.figure(figsize = (12,6))
ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)

for val in range(0,5,1):
    indeX = df_usa.view == val
    ax1.scatter(df_usa.sqft_living.loc[indeX], df_usa.price.loc[indeX], label = val, alpha=0.4)
ax1.legend(bbox_to_anchor = [1.1, 1])
ax1.set_xlabel('sqfit living area')
ax1.set_ylabel('Price house')
ax1.set_title('Sqft Living - Price w.r.t View')

ax2 = plt.subplot2grid((2,2), (1,0))
sns.boxplot(x = 'view', y = 'price', data = df_usa, ax = ax2)
ax2.set_title('Box Plot View & Price', fontsize = 12)

ax3 = plt.subplot2grid((2,2), (1,1))
cubicV = df_usa.groupby(['view'])['price'].mean().round(0)
testTrain = df_usa.loc[:, ['view', 'price']].copy()
testTrain['sqview'] = np.power(testTrain['view'],2)
mdl = linear_model.LinearRegression()
mdl.fit(testTrain[['view', 'sqview']], testTrain['price'])
y_pred = mdl.predict(testTrain[['view', 'sqview']])
print("Mean squared error: %.2f" % mean_squared_error(y_pred, testTrain.price))
# Plot outputs
ax3.scatter(testTrain['view'], testTrain['price'],  color='black')
ax3.plot(testTrain['view'], y_pred, color='blue', linewidth=3)
ax3.set_title('LinReg, price ~ condtion + sqft_cond', fontsize = 12)
ax3.set_xlabel('View rate')
plt.subplots_adjust(hspace = 0.5, top = 0.9)
plt.suptitle('"VIEW" Effect To SalePrice', fontsize = 14)
plt.show()


# In[ ]:


#How can I color the scatter plot by bedrooms? 


# In[ ]:


bedrooms = df_usa.bedrooms.value_counts()


plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)


ax1 = plt.subplot(221)
ax1 = sns.countplot(x="bedrooms", data=df_usa,
                    ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_title("bedrooms counting", fontsize=15)
ax1.set_xlabel("Bathrooms number")
ax1.set_ylabel("count")

ax2 = plt.subplot(222)
ax2 = sns.regplot(x="bedrooms", y='price', 
                  data=df_usa, ax=ax2, x_jitter=True)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.set_title("Bedrooms distribution price", fontsize=15)
ax2.set_xlabel("Bedrooms number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.lvplot(x="bedrooms", y="price",
                    data=df_usa)
ax0.set_title("Better understaning price", fontsize=15)
ax0.set_xlabel("Bedrooms")
ax0.set_ylabel("log Price(US)")
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)


plt.show()


# In[ ]:


print("Floors counting description")
print(df_usa['floors'].value_counts())


plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

ax1 = plt.subplot(221)
ax1 = sns.lvplot(x="floors", y='price', 
                    data=df_usa, ax=ax1, )
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_title("Floors counting", fontsize=15)
ax1.set_xlabel("Floors number")
ax1.set_ylabel("Count")

ax2 = plt.subplot(222)
ax2 = sns.countplot(x="floors",
                  data=df_usa, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.set_title("Floor distribution by price", fontsize=15)
ax2.set_xlabel("Floor number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.regplot(x="floors", y="price", #I need to change floors by sqft_living and hue bye floors
                    data=df_usa, x_jitter=True)
ax0.set_title("Better understaning price by floor", fontsize=15)
ax0.set_xlabel("Floor")
ax0.set_ylabel("log Price(US)")
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)

plt.show()


# In[ ]:


plt.figure(figsize = (12,8))
g=sns.lmplot(x="sqft_living", y="price",
                    data=df_usa, hue="floors")
g.set_titles("Floors by sqft_living and price", fontsize=15)
g.set_xlabels("Sqft Living")
g.set_ylabels("Price(US)")
plt.show()


# In[ ]:


print("Grade counting description")
print(df_usa['grade'].value_counts())


plt.figure(figsize = (12,8))
plt.subplots_adjust(hspace = 0.4, top = 0.8)

ax1 = plt.subplot(221)
ax1 = sns.lvplot(x="grade", y='price', 
                    data=df_usa, ax=ax1, )
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
ax1.set_title("grade counting", fontsize=15)
ax1.set_xlabel("Grade number")
ax1.set_ylabel("Count")

ax2 = plt.subplot(222)
ax2 = sns.countplot(x="grade",
                  data=df_usa, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
ax2.set_title("Grade distribution price", fontsize=15)
ax2.set_xlabel("Grade number")
ax2.set_ylabel("log Price(US)")

ax0 = plt.subplot(212)
ax0 = sns.regplot(x="grade", y="price",
                    data=df_usa, x_jitter=True)
ax0.set_title("Better understaning price by grade", fontsize=15)
ax0.set_xlabel("Grade")
ax0.set_ylabel("log Price(US)")
ax0.set_xticklabels(ax0.get_xticklabels(),rotation=90)

plt.show()


# In[ ]:


#Clearly view of bathrooms and bedrooms correlation

bath = ['bathrooms', 'bedrooms']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[bath[0]], df_usa[bath[1]]).style.background_gradient(cmap = cm)


# In[ ]:



bath_cond = ['bathrooms', 'condition']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[bath_cond[0]], df_usa[bath_cond[1]]).style.background_gradient(cmap = cm)


# In[ ]:


bed_cond = ['bedrooms', 'condition']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[bed_cond[0]], df_usa[bed_cond[1]]).style.background_gradient(cmap = cm)


# In[ ]:


cond_water = ['condition', 'waterfront']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[cond_water[0]], df_usa[cond_water[1]]).style.background_gradient(cmap = cm)


# In[ ]:


grade_cond = ['grade', 'condition']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[grade_cond[0]], df_usa[grade_cond[1]]).style.background_gradient(cmap = cm)


# In[ ]:


grade_bed = ['grade', 'bedrooms']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[grade_bed[0]], df_usa[grade_bed[1]]).style.background_gradient(cmap = cm)


# In[ ]:


grade_bath = ['grade', 'bathrooms']
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_usa[grade_bath[0]], df_usa[grade_bath[1]]).style.background_gradient(cmap = cm)


# In[ ]:


corr = df_usa[['bathrooms', 'bedrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'price']]

plt.figure(figsize=(10,8))
plt.title('Correlation of variables')
sns.heatmap(corr.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()


# In[ ]:


df_usa['yr_built'] = pd.to_datetime(df_usa['yr_built'])


# In[ ]:


g = sns.factorplot(x="yr_built", y = "price", data=df_usa[df_usa['price'] < 800000], 
                   size= 8, aspect = 2, kind="box" )
g.set_xticklabels(rotation=90)
plt.show()


# I am trying to incresse the visual of this time 
