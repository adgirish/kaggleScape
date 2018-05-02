
# coding: utf-8

# # Description
# 
# Here's  simple content-based recommendation engine. An in-depth explanation is here (and the code is ported from the same place):
# 
# http://blog.untrod.com/2016/06/simple-similar-products-recommendation-engine-in-python.html

# ## Step 1 - Train the engine.
# Create a TF-IDF matrix of unigrams, bigrams, and trigrams for each product. The 'stop_words' param tells the TF-IDF module to ignore common english words like 'the', etc.
# 
# Then we compute similarity between all products using SciKit Leanr's linear_kernel (which in this case is equivalent to cosine similarity).
# 
# Iterate through each item's similar items and store the 100 most-similar. Stops at 100 because well...how many similar products do you really need to show?
# 
# Similarities and their scores are stored in a dictionary as a list of Tuples, indexed to their item id.
# 

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ds = pd.read_csv("../input/sample-data.csv")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    # First item is the item itself, so remove it.
    # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, item_id)
    results[row['id']] = similar_items[1:]
    
print('done!')


# ## Step 2: Predict!

# In[ ]:


# hacky little function to get a friendly item name from the description field, given an item ID
def item(id):
    return ds.loc[ds['id'] == id]['description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary. No real logic here.
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")

# Just plug in any item id here (1-500), and the number of recommendations you want (1-99)
# You can get a list of valid item IDs by evaluating the variable 'ds', or a few are listed below

recommend(item_id=11, num=5)


# # Try it yourself!
# 
# Here are some product IDs to try. Just call:
# 
#     recommend(<id>)
# 
# 1 - Active classic boxers
# 
# 2 - Active sport boxer briefs
# 
# 3 - Active sport briefs
# 
# 4 - Alpine guide pants
# 
# 5 - Alpine wind jkt
# 
# 6 - Ascensionist jkt
# 
# 8 - Print banded betina btm
# 
# 9 - Baby micro d-luxe cardigan
# 
# 10 - Baby sun bucket hat
# 
# 11 - Baby sunshade top
