
# coding: utf-8

# ## TF-DNNRegressor - AllState Claims Severity (v0.2)
# 
# This script show a simple example of using [tf.contrib.learn][1] library to create our model.
# 
# The code is divided in following steps:
# 
#  - Load CSVs data
#  - Filtering Categorical and Continuous features
#  - Converting Data into Tensors
#  - Selecting and Engineering Features for the Model
#  - Defining The Regression Model
#  - Training and Evaluating Our Model
#  - Predicting output for test data
# 
# *v0.1: Added code for data loading, modeling and  prediction model.*
# 
# *v0.2: Removed unnecessary output logs.*
# 
# *PS: I was able to get a score of 1295.07972 using this script with 70% (of train.csv) data used for training and rest for evaluation. Script took 2hrs for training and 3000 steps were used.*
# 
# [1]: https://www.tensorflow.org/versions/r0.11/tutorials/tflearn/index.html#tf-contrib-learn-quickstart

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")


# ## Load CSVs data

# In[ ]:


df_train_ori = pd.read_csv('../input/train.csv')
df_test_ori = pd.read_csv('../input/test.csv')


# We only take first 1000 rows for training/testing and last 500 row for evaluation.
# 
# 
# This done so that this script does not consume a lot of kaggle system resources.

# In[ ]:


train_df = df_train_ori.head(1000)
evaluate_df = df_train_ori.tail(500)

test_df = df_test_ori.head(1000)

MODEL_DIR = "tf_model_full"

print("train_df.shape = ", train_df.shape)
print("evaluate_df.shape = ", evaluate_df.shape)
print("test_df.shape = ", test_df.shape)


# ## Filtering Categorical and Continuous features
# 
# We store Categorical, Continuous and Target features names in different variables. This will be helpful in later steps.

# In[ ]:


features = train_df.columns
categorical_features = [feature for feature in features if 'cat' in feature]
continuous_features = [feature for feature in features if 'cont' in feature]
LABEL_COLUMN = 'loss'


# ## Converting Data into Tensors
# 
# > When building a TF.Learn model, the input data is specified by means of an Input Builder function. This builder function will not be called until it is later passed to TF.Learn methods such as fit and evaluate. The purpose of this function is to construct the input data, which is represented in the form of Tensors or SparseTensors.
# 
# > Note that input_fn will be called while constructing the TensorFlow graph, not while running the graph. What it is returning is a representation of the input data as the fundamental unit of TensorFlow computations, a Tensor (or SparseTensor).
# 
# [More detail][2] on input_fn.
# 
# [2]: https://www.tensorflow.org/versions/r0.11/tutorials/input_fn/index.html#building-input-functions-with-tf-contrib-learn

# In[ ]:


# Converting Data into Tensors
def input_fn(df, training = True):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_features}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
        for k in categorical_features}

    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) +
                        list(categorical_cols.items()))

    if training:
        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABEL_COLUMN].values)

        # Returns the feature columns and the label.
        return feature_cols, label
    
    # Returns the feature columns    
    return feature_cols

def train_input_fn():
    return input_fn(train_df)

def eval_input_fn():
    return input_fn(evaluate_df)

def test_input_fn():
    return input_fn(test_df, False)


# ## Selecting and Engineering Features for the Model
# 
# We use tf.learn's concept of [FeatureColumn][FeatureColumn] which help in transforming raw data into suitable input features. 
# 
# These engineered features will be used when we construct our model.
# 
# [FeatureColumn]: https://www.tensorflow.org/versions/r0.11/tutorials/linear/overview.html#feature-columns-and-transformations

# In[ ]:


engineered_features = []

for continuous_feature in continuous_features:
    engineered_features.append(
        tf.contrib.layers.real_valued_column(continuous_feature))


for categorical_feature in categorical_features:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        categorical_feature, hash_bucket_size=1000)

    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,
                                                                  combiner="sum"))


# ## Defining The Regression Model
# 
# Following is the simple DNNRegressor model. More detail about hidden_units, etc can be found [here][123].
# 
# **model_dir** is used to save and restore our model. This is because once we have trained the model we don't want to train it again, if we only want to predict on new data-set.
# 
# [123]: https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html#DNNRegressor

# In[ ]:


regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=engineered_features, hidden_units=[10, 10], model_dir=MODEL_DIR)


# ## Training and Evaluating Our Model

# In[ ]:


# Training Our Model
wrap = regressor.fit(input_fn=train_input_fn, steps=500)


# In[ ]:


# Evaluating Our Model
print('Evaluating ...')
results = regressor.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))


# ## Predicting output for test data
# 
# Most of the time prediction script would be separate from training script (we need not to train on same data again) but I am providing both in same script here; as I am not sure if we can create multiple notebook and somehow share data between them in Kaggle.

# In[ ]:


predicted_output = regressor.predict(input_fn=test_input_fn)
print(predicted_output[:10])

