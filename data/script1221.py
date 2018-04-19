
# coding: utf-8

# # Abstract
# The objective is to design system which will use existing yelp data to provide insightful analytics and help existing business owners, future business owners to make important decisions regarding new business or business expansion.40% of world population has internet connection today compared to 1% in 1995. Almost 3 exabytes of data is created per day using internet. Storing huge amount of data and retrieving knowledge out of it is challenging task these days. Yelp is a website which publishes crowd sourced reviews about local businesses (Restaurants, Department Stores, Bars, Home-Local Services, Cafes, Automotive). It provides opportunity to business owners to improve their services and users to choose best business amongst available.
# # Introduction
# Yelp is a local business directory service and review site with social networking features. It allows users to give ratings and review businesses. The review is usually short text consisting of few lines with about hundred words. Often, a review describes various dimensions about a business and the experience of user with respect to those dimensions.This dataset is a subset of Yelp's businesses, reviews, and user data. It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In the dataset you'll find information about businesses across 11 metropolitan areas in four countries.
# 
# # Glossary of terms
# 1. **Yelp**: A website which publishes crowd source reviews to help users and business owners . Business: A local body listed on yelp like Restaurants, Department Stores, Bars, Home-Local Services, Cafes, Automotive.
# 2. **Existing business owner**: A person who has listed his business on Yelp site and getting views from yelp users.
# 3. **Future business owner**: A person who wants to start new business in future time
# 4. **User**: A person who has registered on yelp who is writing reviews about different business after vising them or a person who is using yelp reviews to choose business.
# 5. **Analytics**: Extract knowledge out of data which can be used by system users to make important decisions which is very difficult just by looking at the data.
# 6. **Review**: It is text written by user after vising business about the over-all experience. I is also a numeric representation (out of 5) to compare it with other business.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
#import warnings
#warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 100)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


business = pd.read_csv('../input/yelp_business.csv')


# In[3]:


business.head(5)


# In[4]:


business_hours = pd.read_csv("../input/yelp_business_hours.csv")


# In[5]:


business_hours.head()


# In[6]:


business.columns


# In[7]:


business.shape


# In[8]:


#Null Values...
business.isnull().sum().sort_values(ascending=False)


# In[9]:


#are all business Id's unique?
business.business_id.is_unique #business_id is all unique


# In[10]:


business.city.value_counts()


#  # Top 50 most reviewed businesses

# In[11]:


business[['name', 'review_count', 'city', 'stars']].sort_values(ascending=False, by="review_count")[0:50]


# 
# # Number of businesses listed in different cities

# In[12]:


city_business_counts = business[['city', 'business_id']].groupby(['city'])['business_id'].agg('count').sort_values(ascending=False)


# In[13]:


city_business_counts = pd.DataFrame(data=city_business_counts)


# In[14]:


city_business_counts.rename(columns={'business_id' : 'number_of_businesses'}, inplace=True)


# In[15]:


city_business_counts[0:50].sort_values(ascending=False, by="number_of_businesses").plot(kind='barh', stacked=False, figsize=[10,10], colormap='winter')
plt.title('Top 50 cities by businesses listed')


# # Cities with most reviews and best ratings for their businesses

# In[16]:


city_business_reviews = business[['city', 'review_count', 'stars']].groupby(['city']).agg({'review_count': 'sum', 'stars': 'mean'}).sort_values(by='review_count', ascending=False)
city_business_reviews.head(10)


# In[17]:


city_business_reviews['review_count'][0:50].plot(kind='barh', stacked=False, figsize=[10,10],                                                  colormap='winter')
plt.title('Top 50 cities by reviews')


# In[18]:


city_business_reviews[city_business_reviews.review_count > 50000]['stars'].sort_values().plot(kind='barh', stacked=False, figsize=[10,10], colormap='winter')
plt.title('Cities with greater than 50k reviews ranked by average stars')


# # Distribution of stars

# In[19]:


business['stars'].value_counts()


# In[20]:


sns.distplot(business.stars, kde=False)


# # How many are open and how many closed?

# In[21]:


business['is_open'].value_counts()


# # Lets look into user tips on businesses before looking at reviews

# In[22]:


tip = pd.read_csv('../input/yelp_tip.csv')


# In[23]:


tip.head(10)


# In[24]:


tip.shape


# # How many of the selected words are used in the user tips?

# In[25]:


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 
                  'awful', 'wow', 'hate']
selected_words


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=selected_words, lowercase=False)
#corpus = ['This is the first document.','This is the second second document.']
#print corpus
selected_word_count = vectorizer.fit_transform(tip['text'].values.astype('U'))
vectorizer.get_feature_names()


# In[27]:


word_count_array = selected_word_count.toarray()
word_count_array.shape


# In[28]:


word_count_array.sum(axis=0)


# In[29]:


temp = pd.DataFrame(index=vectorizer.get_feature_names(),                     data=word_count_array.sum(axis=0)).rename(columns={0: 'Count'})


# In[30]:


temp.plot(kind='bar', stacked=False, figsize=[7,7], colormap='winter')


# We see that most of the tips are positive rather than negative!
# # Lets look at one restaurant with high star rating and one with low star rating and see what the user tips look like
# ## Lets look at "Earl of Sandwich" restaurant in Las Vegas which has 4.5 rating

# In[31]:


business[(business['city'] == 'Las Vegas') & (business['stars'] == 4.5)]


# In[32]:


business[business.name=='"Earl of Sandwich"']


# - Points to remember:
#     1. There are 4 branches
#     2. Two of them are on the strip
#     3. Since there are multiple, lets pick by index

# In[33]:


# This is where  have been to :)
business.loc[139699,:]


# In[34]:


earl_of_sandwich = tip[tip.business_id==business.loc[139699,:].business_id]


# In[35]:


earl_of_sandwich_selected_word_count = vectorizer.fit_transform(earl_of_sandwich['text'].values.astype('U'))


# In[36]:


word_count_array = earl_of_sandwich_selected_word_count.toarray()
temp = pd.DataFrame(index=vectorizer.get_feature_names(),                     data=word_count_array.sum(axis=0)).rename(columns={0: 'Count'})
temp


# In[37]:


temp.plot(kind='bar', stacked=False, figsize=[7,7], colormap='winter')


# We can see that the tips are mostly positive!

# In[38]:


business[['name', 'review_count', 'city', 'stars']][business.review_count>1000].sort_values(ascending=True, by="stars")[0:15]


# **Lets look into Luxor Hotel and Casino Las Vegas which has a 2.5 star**

# In[39]:


business[business['name'] == '"Luxor Hotel and Casino Las Vegas"']


# In[40]:


luxor_hotel = tip[tip.business_id==business.loc[6670,:].business_id]
luxor_hotel.info()


# In[41]:


luxor_hotel_selected_word_count = vectorizer.fit_transform(luxor_hotel['text'].values.astype('U'))


# In[42]:


word_count_array = luxor_hotel_selected_word_count.toarray()
temp = pd.DataFrame(index=vectorizer.get_feature_names(),                     data=word_count_array.sum(axis=0)).rename(columns={0: 'Count'})


# In[43]:


temp.plot(kind='bar', stacked=False, figsize=[10,5], colormap='winter')


# This has more positive words than negative, so the user tips for this restaurant are not very predictive of its star! This might make sense because while users write good and bad reviews, tips are naturally like to be what they liked and therefore positive!
# # Lets look into user reviews

# In[44]:


reviews = pd.read_csv('../input/yelp_review.csv')


# In[45]:


reviews.shape, tip.shape #there are 5.26 million reviews! 1 million tips


# In[46]:


reviews.head(5)


# In[47]:


tip.head()


# # How many of these restaurants serve Japanese food? Lets find out based on reviews!

# In[48]:


selected_words = ['sushi', 'miso', 'teriyaki', 'tempura', 'udon',                   'soba', 'ramen', 'yakitori', 'izakaya']


# 
# **Lets take subset of reviews since there are so many**

# **More data analysis and data science to follow in this and other notebooks!! :)**
