
# coding: utf-8

# # Different point of view
# 
# You thought that this is an image classification competition. You thought you have to do CNN and stuff. But don't let organizers deceive you! There are texts and we are going to use them!
# 
# ## Idea
# 
# We have URLs for images in the data. And it makes sense to think that on different hostings there will be a different distribution of target classes. Or maybe we will be possible to find some meaningful words in the image filenames.
# 
# So let's train a text classification algorithm on the URLs and see what we can get.

# In[ ]:


import numpy as np
import pandas as pd
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


def read_json_to_dataframe(filepath, test_file=False):
    with open(filepath) as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df['url'] = df.images.map(lambda x: x['url'][0])
    if not test_file:
        df['image_id'] = df.annotations.map(lambda x: x['image_id'])
        df['label_id'] = df.annotations.map(lambda x: x['label_id'])
        df.drop(columns=['annotations', 'images'], inplace=True)
    return df


# In[ ]:


train_data = read_json_to_dataframe('../input/train.json', test_file=False)
validation_data = read_json_to_dataframe('../input/validation.json', test_file=False)
test_data = read_json_to_dataframe('../input/test.json', test_file=True)


# In[ ]:


print("Train size: ", train_data.shape)
print("Validation size: ", validation_data.shape)
print("Test size: ", test_data.shape)


# # Create text features: TF-IDF vectorizer
# 
# What features can we get from texts? Domain name, top-level domain name, some hints from file names, maybe something else. We can do all these automatically by counting char n-grams in each URL and use this numbers as features. And `sklearn` already got all the functions and methods we need.

# In[ ]:


tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=2500, lowercase=False)
tfidf.fit(train_data.url)


# If don't know what just happened, then long story short: we just found 2500 most common 1-, 2- and 3-grams form all URLs in the training data. [Here is a nice doc](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) if you want to know more. Now we can count occurrences of those n-grams in the URLs and apply some smoothing called [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

# In[ ]:


train_features = tfidf.transform(train_data.url)
validation_features = tfidf.transform(validation_data.url)
test_features = tfidf.transform(test_data.url)


# And now let's train Logistic Regression on those features. The thing is that we have a lot of features and a lot of points so LR will take forever (in this notebook environment). So we will take few thousand random points to train a model.

# In[ ]:


np.random.seed(0)
random_ids = np.random.choice(np.arange(len(train_data)), size=7500, replace=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lr = LogisticRegression(C=10.0)\nlr.fit(train_features[random_ids], train_data.label_id.values[random_ids])\nprint("Validation error: %.3f" % (1 - accuracy_score(validation_data.label_id, lr.predict(validation_features))))')


# In[ ]:


submission = pd.DataFrame({
    'id': 1 + np.arange(len(test_data)),
    'predicted': lr.predict(test_features),
})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# Here you go. You have a submission. You definitely can do better if you use more data points to train logistic regression and invest some time into a selection of better parameters of TfIdf Vectorizer. But I suggest you don't.
# 
# It's all (almost) a joke, it's really an image classification competition and better spend your precious time on real images :) In fact, as @fayzur noted, it's forbidden to use URLs by competition rules, so go with images and don't get banned.
# 
# Good luck to you!
