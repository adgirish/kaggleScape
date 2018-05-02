
# coding: utf-8

# # Introduction
# This Python notebook seeks to explore Mercari's data set, in order to extract "what to do" and "what not to do".
# 
# The main idea is to contribute to the community with quick, but deep, first view of the data.
# 
# It contains some visualizations and data manipulation, that can be further used to build your own Machine Learning Models.
# 
# Feel free to contribute to this Kernel with your thoughts.
# 
# Happy Kaggling!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import string

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Readindg data sets

# In[ ]:


df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')


# ### Training set first look

# In[ ]:


df_train.head()


# In[ ]:


print('Train shape:{}\nTest shape:{}'.format(df_train.shape, df_test.shape))


# ## Target distribution

# In[ ]:


plt.figure(figsize=(20, 15))
plt.hist(df_train['price'], bins=50, range=[0,250], label='price')
plt.title('Train "price" distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


# ### Price distribution

# In[ ]:


df_train['price'].describe()


# Looks like the target distribution is more concentrated between **0~100**, but there are still values until **2000**.

# In[ ]:


plt.figure(figsize=(20, 15))
bins=50
plt.hist(df_train[df_train['shipping']==1]['price'], bins, normed=True, range=[0,250],
         alpha=0.6, label='price when shipping==1')
plt.hist(df_train[df_train['shipping']==0]['price'], bins, normed=True, range=[0,250],
         alpha=0.6, label='price when shipping==0')
plt.title('Train price over shipping type distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Normalized Samples', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# The comparison of target class when shipping is 1 or 0 do not seems to be REALLY separated. But this does not means that this feature is useless, just that it can be further explored

# In[ ]:


df = df_train[df_train['price']<100]

my_plot = []
for i in df_train['item_condition_id'].unique():
    my_plot.append(df[df['item_condition_id']==i]['price'])

fig, axes = plt.subplots(figsize=(20, 15))
bp = axes.boxplot(my_plot,vert=True,patch_artist=True,labels=range(1,6)) 

colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes.yaxis.grid(True)

plt.title('BoxPlot price X item_condition_id', fontsize=15)
plt.xlabel('item_condition_id', fontsize=15)
plt.ylabel('price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

del df


# The training data was reduced to just have sample with **target < 100**.
# item_condition_id feature does not seems to vary to much in our data. Their medians do not change as the item_condition_id changes. Maybe it is just an ID and should be discarded or maybe it can help the learning algorithms or Feature Egeneering in some way.
# 
# Just to be fair, ID n°5 looks like a little bit different from others. It has a higher 3rd quartile and median.

# # Text exploration
# Lets take a look into our textual features. I bet we can have a lot of insights from it.
# ### Most commom letters in Items Descriptions

# In[ ]:


cloud = WordCloud(width=1440, height=1080).generate(" ".join(df_train['item_description']
.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')


# ### Verifying if products actually have description
# The words "description yet" took my attention as they are commom according to the Word Cloud. Lets investigate.

# In[ ]:


df_train['has_description'] = 1
df_train.loc[df_train['item_description']=='No description yet', 'has_description'] = 0


# In[ ]:


plt.figure(figsize=(20, 15))
bins=50
plt.hist(df_train[df_train['has_description']==1]['price'], bins, range=[0,250],
         alpha=0.6, label='price when has_description==1')
plt.hist(df_train[df_train['has_description']==0]['price'], bins, range=[0,250],
         alpha=0.6, label='price when has_description==0')
plt.title('Train price X has_description type distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


# Looking to this histograms, the distribution of prices when an item does not have a description yet is very similar to when they already have a description, considering that the amount of simples in each histogram is very different, of course. To be more clear, there are normed histograms in the next chart.
# #### Same as above, but normalized

# In[ ]:


plt.figure(figsize=(20, 15))
bins=50
plt.hist(df_train[df_train['has_description']==1]['price'], bins, normed=True,range=[0,250],
         alpha=0.6, label='price when has_description==1')
plt.hist(df_train[df_train['has_description']==0]['price'], bins, normed=True,range=[0,250],
         alpha=0.6, label='price when has_description==0')
plt.title('Train price X has_description type distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


# Explore products without description does not look to be the best initial approach in this competition.

# # TF-IDF
# Maybe Term Frequency – Inverse Document Frequency (TF-IDF) can be a good approach to deal with items descriptions.
# 
# I'll give it a chance here, even as the descriptions are pretty short, as it had given me good results in other contexts.

# In[ ]:


def compute_tfidf(description):
    description = str(description)
    description.translate(string.punctuation)

    tfidf_sum=0
    words_count=0
    for w in description.lower().split():
        words_count += 1
        if w in tfidf_dict:
            tfidf_sum += tfidf_dict[w]
    
    if words_count > 0:
        return tfidf_sum/words_count
    else:
        return 0

tfidf = TfidfVectorizer(
    min_df=5, strip_accents='unicode', lowercase =True,
    analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
    smooth_idf=True, sublinear_tf=True, stop_words='english')


# In[ ]:


tfidf.fit_transform(df_train['item_description'].apply(str))
tfidf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
df_train['tfidf'] = df_train['item_description'].apply(compute_tfidf)


# In[ ]:


plt.figure(figsize=(20, 15))
plt.scatter(df_train['tfidf'], df_train['price'])
plt.title('Train price X item_description TF-IDF', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('TF-IDF', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


# Well, looking to this scatter plot is possible to notice that as higher the TF-IDF, lower the 'price' gets. I'll definitely try this feature in my models.

# ## Item Description Lengths
# ### Characters count

# In[ ]:


train_ds = pd.Series(df_train['item_description'].tolist()).astype(str)
test_ds = pd.Series(df_test['item_description'].tolist()).astype(str)

bins=100
plt.figure(figsize=(20, 15))
plt.hist(train_ds.apply(len), bins, range=[0,600], label='train')
plt.hist(test_ds.apply(len), bins, alpha=0.6,range=[0,600], label='test')
plt.title('Histogram of character count', fontsize=15)
plt.xlabel('Characters Number', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


# The histograms seems to have a pretty similar distribution. There is a gap near **20 characters**, maybe because of "no description yet" descriptions.

# ### Word count

# In[ ]:


bins=100
plt.figure(figsize=(20, 15))
plt.hist(train_ds.apply(lambda x: len(x.split())), bins, range=[0,100], label='train')
plt.hist(test_ds.apply(lambda x: len(x.split())), bins, alpha=0.6,range=[0,100], label='test')
plt.title('Histogram of word count', fontsize=15)
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()


# As well as character count, word count histograms are very similar for train and test. Further exploration will be needed here!

# # Category Name
# In our data set, each item have fits in a specific category, or a group of categories (mainly 3 for item).
# 
# Basically, the categories are arranged as from top to bottom of comprehensiveness. This means that the first category is less specific and the next are more specifics.
# 
# Lets investigate the 'main' categories first.

# In[ ]:


def transform_category_name(category_name):
    try:
        main, sub1, sub2= category_name.split('/')
        return main, sub1, sub2
    except:
        return np.nan, np.nan, np.nan

df_train['category_main'], df_train['category_sub1'], df_train['category_sub2'] = zip(*df_train['category_name'].apply(transform_category_name))


# In[ ]:


main_categories = [c for c in df_train['category_main'].unique() if type(c)==str]
categories_sum=0
for c in main_categories:
    categories_sum+=100*len(df_train[df_train['category_main']==c])/len(df_train)
    print('{:25}{:3f}% of training data'.format(c, 100*len(df_train[df_train['category_main']==c])/len(df_train)))
print('nan\t\t\t {:3f}% of training data'.format(100-categories_sum))


# In[ ]:


df = df_train[df_train['price']<80]

my_plot = []
for i in main_categories:
    my_plot.append(df[df['category_main']==i]['price'])
    
fig, axes = plt.subplots(figsize=(20, 15))
bp = axes.boxplot(my_plot,vert=True,patch_artist=True,labels=main_categories) 

colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']*2
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes.yaxis.grid(True)

plt.title('BoxPlot price X Main product category', fontsize=15)
plt.xlabel('Main Category', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# Well, these boxplots are telling us more than the last ones. At least we have some variation when looking to their medians. But I feel like these results can be improved.
# 
# "Men" products are showing themselves more expensives than other in a general way.

# ### 3rd level categories
# As we have investigated main categories, let's take a look into the lowest level categories.

# In[ ]:


print('The data has {} unique 3rd level categories'.format(len(df_train['category_sub2'].unique())))

df = df_train.groupby(['category_sub2'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=False, inplace=True)
df = df[:20]
df.sort_values(by=['mean_price'], ascending=True, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5)
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('ASCENDING - 3rd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()
########################################################################

df = df_train.groupby(['category_sub2'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=True, inplace=True)
df = df[:50]
df.sort_values(by=['mean_price'], ascending=False, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5, color='r')
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('DESCENDING - 3rd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()


# ### 2nd level categories (middle level)
# To finish with categories for now, let's evaluate the middle level categories.

# In[ ]:


print('The data has {} unique 2nd level categories'.format(len(df_train['category_sub1'].unique())))

df = df_train.groupby(['category_sub1'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=False, inplace=True)
df = df[:20]
df.sort_values(by=['mean_price'], ascending=True, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', alpha=0.5, color='green')
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('ASCENDING - 2nd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()
########################################################################

df = df_train.groupby(['category_sub1'])['price'].agg(['size','sum'])
df['mean_price']=df['sum']/df['size']
df.sort_values(by=['mean_price'], ascending=True, inplace=True)
df = df[:50]
df.sort_values(by=['mean_price'], ascending=False, inplace=True)

plt.figure(figsize=(20, 15))
plt.barh(range(0,len(df)), df['mean_price'], align='center', color='pink')
plt.yticks(range(0,len(df)), df.index, fontsize=15)
plt.xticks(fontsize=15)
plt.title('DESCENDING - 2nd level categories sorted by its mean prices', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('3rd level categorie', fontsize=15)
plt.legend(fontsize=15)
plt.show()


# * It is interesting to see that we have a huge difference of price looking into items categories (as expected).
# * Splitting categories into "levels" in our data can make a big difference when training our models.

# # To be continued...
