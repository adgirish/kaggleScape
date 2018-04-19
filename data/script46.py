
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("../input/train.tsv", sep = "\t")
test = pd.read_csv("../input/test.tsv", sep = "\t")


# In[ ]:


train.info()


# In[ ]:


train.isnull().any()


# In[ ]:


def fill_missing_data(data):
    data.category_name.fillna(value = "Other/Other/Other", inplace = True)
    data.brand_name.fillna(value = "Unknown", inplace = True)
    data.item_description.fillna(value = "No description yet", inplace = True)
    return data

train = fill_missing_data(train)


# In[ ]:


train.describe()


# In[ ]:


train.head()


# # Visualize data

# ### Price

# In[ ]:


train.price.describe()


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize = (15, 10))
sns.boxplot(train.price, showfliers = False, ax = ax[0])
ax[1].hist(train.price, bins = 50, range = [0, 300], label = "price")
ax[1].set_title("Price Distribution (Training)", fontsize = 20)
ax[1].set_xlabel("Price", fontsize = 15)
ax[1].set_ylabel("Samples", fontsize = 15)
plt.show()


# In[ ]:


train["log_price"] = np.log(train["price"] + 1)
fig, ax = plt.subplots(2, 1, figsize = (15, 10))
sns.boxplot(train.log_price, ax = ax[0])
ax[1].hist(train.log_price, bins = 50)
ax[1].set_title("Log Price Distribution (Training)", fontsize = 20)
ax[1].set_xlabel("Log Price", fontsize = 15)
ax[1].set_ylabel("Samples", fontsize = 15)
plt.show()


# In[ ]:


print("There are", train[train["price"] == 0].price.size, "items with price 0.")


# ### Item Condition

# In[ ]:


fig, ax = plt.subplots(2, 1, figsize = (15, 12))
sns.countplot(train.item_condition_id, ax = ax[0])
rects = ax[0].patches
labels = train.item_condition_id.value_counts().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax[0].text(rect.get_x() + rect.get_width()/2, height + 5, label, ha = "center", va = "bottom")
sns.boxplot(x = train.item_condition_id, y = train.log_price, orient = "v", ax = ax[1])
plt.show()


# In[ ]:


fog, ax = plt.subplots(1, 5, figsize = (15, 8))
for i in range(1, 6):
    train[train.item_condition_id == i].log_price.plot.hist(ax = ax[i-1], bins = 50, alpha = 0.5)  
    ax[i-1].set_xlabel("Log Price")
    ax[i-1].set_title("Item Condition Id = " + " " + str(i))
plt.show()


# In[ ]:


train.groupby(["item_condition_id"]).price.std()


# In[ ]:


train.groupby(["item_condition_id"]).price.mean()


# In[ ]:


train.groupby(["item_condition_id"]).price.median()


# ### Shipping

# In[ ]:


train.shipping.value_counts()


# In[ ]:


sns.boxplot(x = train.shipping, y = train.log_price, orient = "v")


# In[ ]:


plt.figure(figsize = (15, 8))
plt.hist(train[train.shipping == 1].log_price, bins = 50, alpha = 0.5, label = "log price with free shipping")
plt.hist(train[train.shipping == 0].log_price, bins = 50, alpha = 0.5, label = "log price with shipping")
plt.legend(fontsize = 10)
plt.show()


# ### Brand

# In[ ]:


brands = train["brand_name"].value_counts()
print("There are", brands.size, "unique known brands.")


# In[ ]:


plt.figure(figsize = (10, 10))
sns.barplot(brands[1:11].values, brands[1:11].index)
plt.title("Top 10 known brand in store")
plt.show()


# In[ ]:


brand_std_price = train.groupby(["brand_name"], as_index = True).price.std().sort_values(ascending = False)
print("std price by brands", brand_std_price[:10])
brand_mean_price = train.groupby(["brand_name"], as_index = True).price.mean().sort_values(ascending = False)
print("mean price by brands", brand_mean_price[:10])
brand_median_price = train.groupby(["brand_name"]).price.median().sort_values(ascending = False)
print("median price by brands", brand_median_price[:10])


# In[ ]:


brands = ["PINK", "Nike", "Louis Vuitton", "Lululemon"]
nbrand = len(brands)

fig, ax = plt.subplots(2, 2, figsize = (15, 10))
for b in range(nbrand):
    brand = brands[b]
    for i in range(1, 6):
        sns.distplot(train[train.brand_name == brand][train["item_condition_id"] == i].log_price, hist = False,
                     label = "item_condition_id = " + " " + str(i), ax = ax[int(b/2)][b%2])
    ax[int(b/2)][b%2].set_xlabel("Log Price")
    ax[int(b/2)][b%2].set_title("Price of " + brand + " in each item condition")
plt.show()


# In[ ]:


train[train["brand_name"] == "Unknown"].price.describe()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize = (15, 8))
sns.countplot(train[train.brand_name == "Unknown"].item_condition_id, ax = ax[0])
ax[0].set_xlabel("Item Condition")
ax[0].set_title("Number of unknown brands in each item condition")
rects = ax[0].patches
labels = train[train.brand_name == "Unknown"].item_condition_id.value_counts().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax[0].text(rect.get_x() + rect.get_width()/2, height + 5, label, ha = "center", va = "bottom")
for i in range(1, 6):
    sns.distplot(train[train.brand_name == "Unknown"][train["item_condition_id"] == i].log_price, hist = False,
                 label = "item_condition_id = " + " " + str(i), ax = ax[1])
ax[1].set_xlabel("Log Price")
ax[1].set_title("Price of unknown brand in each item condition")
plt.show()


# In[ ]:


free_item = train[train.price == 0].brand_name
print(free_item.unique().size, "brands have free items")


# In[ ]:


known_free_item = free_item.value_counts()[1:21]
fig, ax = plt.subplots(1, 2, figsize = (15, 8))
sns.countplot(train[train.price == 0].shipping, ax = ax[0])
rects = ax[0].patches
labels = train[train.price == 0].shipping.value_counts().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax[0].text(rect.get_x() + rect.get_width()/2, height + 5, label, ha = "center", va = "bottom")
ax[0].set_title("Free items' ihipping")
sns.barplot(known_free_item.values, known_free_item.index, ax = ax[1])
ax[1].set_xlabel("count")
ax[1].set_title("Number of brands have free items")
plt.show()


# ## 315 items are totally free!!!

# ### Category

# In[ ]:


train["main_cat"] = train.category_name.str.extract("([^/]+)/[^/]+/[^/]+")
train["subcat1"] = train.category_name.str.extract("[^/]+/([^/]+)/[^/]+")
train["subcat2"] = train.category_name.str.extract("[^/]+/[^/]+/([^/]+)")


# In[ ]:


category = train.main_cat
order = sorted(category.unique())
fig, ax = plt.subplots(1, 2, figsize = (15, 10))
sns.boxplot(x = train.log_price, y = category, orient = "h", order = order, ax = ax[0])
ax[0].set_title("Log Price Base On Main Subcategories", fontsize = 20)
ax[0].set_ylabel("Categories", fontsize = 15)
sns.barplot(category.value_counts().values, category.value_counts().index, order = order, ax = ax[1])
ax[1].set_title("Number in eanch category", fontsize = 20)
plt.show()


# In[ ]:


import squarify

fig = plt.figure(figsize = (8, 8))
regions = train.main_cat.value_counts().to_frame()
ax = fig.add_subplot(111, aspect = "equal")
ax = squarify.plot(sizes = regions["main_cat"].values, label = regions.index,
              color = sns.color_palette("viridis", 10), alpha = 1)
ax.set_xticks([])
ax.set_yticks([])
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.title("Treemap of Main Category", fontsize = 18)
plt.show()


# In[ ]:


print("There are", len(train.subcat1.unique()), "second categories.")


# In[ ]:


mean_subcat1_price = pd.DataFrame(train.groupby(["subcat1"]).price.mean())
mean_subcat1_price = mean_subcat1_price.sort_values(by = "price", ascending = False)[:10]
mean_subcat1_price.reset_index(level = 0, inplace = True)

plt.figure(figsize = (10, 5))
sns.barplot(x = "price", y = "subcat1", data = mean_subcat1_price, orient = "h")
plt.title("Mean Price Base On First Subcategories (first 10)", fontsize = 20)
plt.ylabel("Subcategories", fontsize = 20)
plt.xlabel("Price", fontsize = 20)
plt.show()


# In[ ]:


sub_category1 = train.subcat1
order = sorted(sub_category1.unique())
fig, ax = plt.subplots(1, 2, figsize = (15, 30))
sns.boxplot(x = train.log_price, y = sub_category1, orient = "h", order = order, ax = ax[0])
ax[0].set_title("Log Price Base On Main Subcategories", fontsize = 20)
ax[0].set_ylabel("Categories", fontsize = 15)
sns.barplot(sub_category1.value_counts().values, sub_category1.value_counts().index, order = order, ax = ax[1])
ax[1].set_title("Number in eanch category", fontsize = 20)
plt.show()


# In[ ]:


fig = plt.figure(figsize = (15, 10))
regions = train.subcat1.value_counts().to_frame()
ax = fig.add_subplot(111, aspect = "equal")
ax = squarify.plot(sizes = regions["subcat1"].values, label = regions.index,
              color = sns.color_palette("viridis", 30), alpha = 1)
ax.set_xticks([])
ax.set_yticks([])
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.title("Treemap of First Category", fontsize = 18)
plt.show()


# In[ ]:


print("There are", len(train.subcat2.unique()), "second categories.")


# In[ ]:


mean_subcat2_price = pd.DataFrame(train.groupby(["subcat2"]).price.mean())
mean_subcat2_price = mean_subcat2_price.sort_values(by = "price", ascending = False)[:10]
mean_subcat2_price.reset_index(level = 0, inplace = True)

plt.figure(figsize = (10, 5))
sns.barplot(x = "price", y = "subcat2", data = mean_subcat2_price, orient = "h")
plt.title("Mean Price Base On Second Categories (first 10)", fontsize = 20)
plt.ylabel("Second Categories", fontsize = 20)
plt.xlabel("Price", fontsize = 20)
plt.show()


# In[ ]:


# https://www.kaggle.com/asindico/standard-prices-vs-outliers
# not outliers
def nol(data, m = 2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# outliers
def ol(data, m = 3):
    return data[(data - np.mean(data)) >= m * np.std(data)]


# In[ ]:


chist = train.groupby(["category_name"], as_index = False).count().sort_values(by = "train_id",
                                                                               ascending = False)[0:25]


# In[ ]:


k = 10
fig, ax = plt.subplots(5, 2, figsize = (15, 30))
for i in range(k):
    sns.distplot(nol(train[train["category_name"] == chist["category_name"].values[i]].price), ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_title(chist["category_name"].values[i])


# In[ ]:


fig,ax = plt.subplots(5, 2, figsize = (15, 30))
for i in range(k):
    sns.distplot(ol(train[train["category_name"] == chist["category_name"].values[i]].price), ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_title(chist["category_name"].values[i])


# Price outliers are generated by some specific brands

# In[ ]:


fig, ax = plt.subplots(5, 2, figsize = (15, 30))
for i in range(k):    
    ohist = train.iloc[(ol(train[
        train["category_name"] == chist["category_name"].values[i]
    ].price).index).values].groupby(["item_condition_id"], as_index = False).count()
    sns.barplot(x = ohist["item_condition_id"], y = ohist["train_id"], ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_title(chist["category_name"].values[i])
    ax[int(i/2)][i%2].set_xlabel("Item Condition")
    ax[int(i/2)][i%2].set_ylabel("Frequency")


# In[ ]:


fig, ax = plt.subplots(5, 2, figsize = (15, 30))
for i in range(k):    
    nohist = train.iloc[(nol(train[
        train["category_name"] == chist["category_name"].values[i]
    ].price).index).values].groupby(["item_condition_id"], as_index = False).count()
    sns.barplot(x = nohist["item_condition_id"], y = nohist["train_id"], ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_title(chist["category_name"].values[i])
    ax[int(i/2)][i%2].set_xlabel("Item Condition")
    ax[int(i/2)][i%2].set_ylabel("Frequency")


# Number of items' condition with excellent under outlier price  are about 125, but outliers' are varied.

# ### Item Description

# In[ ]:


from wordcloud import WordCloud

wordcloud = WordCloud(width = 1200, height = 1000).generate(" ".join(train.item_description.astype(str)))
plt.figure(figsize = (20, 15))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[ ]:


ncat = ["Women", "Beauty", "Kids", "Electronics", "Men", "Home", 
        "Other", "Vintage & Collectibles", "Handmade","Sports & Outdoors"]

fig, ax = plt.subplots(5, 2, figsize = (40, 40))
for i in range(len(ncat)):
    c = ncat[i]
    wordcloud = WordCloud(max_words = 200
                         ).generate(" ".join(train["item_description"][train["main_cat"] == c].astype(str)))
    ax[int(i/2)][i%2].axis("off")
    ax[int(i/2)][i%2].imshow(wordcloud)
    ax[int(i/2)][i%2].set_title(c, fontsize = 35)
plt.show()


# In[ ]:


# https://www.kaggle.com/huguera/mercari-data-analysis
from sklearn.feature_extraction.text import TfidfVectorizer
import string 

def compute_tfidf(description):
    description = str(description)
    description.translate(string.punctuation)

    tfidf_sum = 0
    words_count = 0
    for w in description.lower().split():
        words_count += 1
        if w in tfidf_dict:
            tfidf_sum += tfidf_dict[w]
    
    if words_count > 0:
        return tfidf_sum/words_count
    else:
        return 0
    
tfidf = TfidfVectorizer(
    min_df = 5, strip_accents = "unicode", lowercase = True,
    analyzer = "word", token_pattern = r"\w+", ngram_range =(1, 3), use_idf = True, 
    smooth_idf = True, sublinear_tf = True, stop_words = "english")

tfidf.fit_transform(train["item_description"].apply(str))
tfidf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
train["desc_tfidf"] = train["item_description"].apply(compute_tfidf)


# In[ ]:


plt.figure(figsize = (15, 10))
plt.scatter(x = train.desc_tfidf, y = train.price, alpha = 0.5)
plt.xlabel("TF-IDF")
plt.ylabel("Price")
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(np.hstack([train.name]))
train["name"] = le.transform(train.name)
le.fit(np.hstack([train.brand_name]))
train["brand_name"] = le.transform(train.brand_name)
le.fit(np.hstack([train.main_cat]))
train["main_cat"] = le.transform(train.main_cat)
le.fit(np.hstack([train.subcat1]))
train["subcat1"] = le.transform(train.subcat1)
le.fit(np.hstack([train.subcat2]))
train["subcat2"] = le.transform(train.subcat2)


# In[ ]:


columns = list(train.columns)
plt.figure(figsize = (10, 10))
sns.heatmap(train[columns].corr(), annot = True, linewidth = 0.5)
plt.show()


# In[ ]:


train.info()

