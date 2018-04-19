
# coding: utf-8

# # Local Validation for Instacart Market Basket Analysis
# 
# **Problems for creating local scorings**
# 
# This is a multiclass and multilabel prediction challenge.
# 
# The multiple labels that need to be predicted and which have a varying number of occurences for each row (the sample only shows the same ID per row but real predictions should have "None" or anywhere from 1 to about 80 IDs per row) make the typical scoring-implementations from SKlearn or similar packaged harder to use. 
# 
# These Packages don't (as far as I know) provide a way to score these multi-label predictions.
# 
# In order to calculate a F1 score (which is used by Kaggle to score your prediction) on a local test-set, we need to create our own way of scoring predictions locally to be able to do more testing.
# 
# **No Cross-Validation**
# 
# The competition asks you to predict the product orders for the next order of clients. The prior data is organized in a "continous" way and later orders depend on previous ones. Unlike other classifacation problems you shouldn't split the orders in a typical 5-fold-Cross-Validation because it doesn't helpt to predict earlier orders using data from later orders.
# 
# Fortunately, Instacart has already provided us with the last orders for some client in the "order_products_train.csv" file. The naming of the files could be a bit better in my opinion as you dont just train your model on this file but also/only on the priors file.
# 
# You can use the train-data as your local test set if you format it in a different way.

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 200) # use this to display more content for each column


# In[ ]:


train_df = pd.read_csv("../input/order_products__train.csv")


# ## Formatting the training-data
# 
# As noted above, you should use the later orders for validating your model. With the following code you can convert the training-order-data or any other subset into a submission-like format. 

# In[ ]:


# we will use 5 orders with different amounts of products bought for demonstration
train_5 = train_df.loc[train_df["order_id"].isin([199872, 427287, 569586, 894112, 1890016])]

train_5.head(10)


# In[ ]:


# concatenate all product-ids into a single string
# thanks to https://www.kaggle.com/eoakley/start-here-simple-submission

def products_concat(series):
    out = ''
    for product in series:
        if product > 0:
            out = out + str(int(product)) + ' '
    
    if out != '':
        return out.rstrip()
    else:
        return 'None'


# In[ ]:


# this creates a DataFrame in the same format as your (local) prediction. 
train_5 = pd.DataFrame(train_5.groupby('order_id')["product_id"].apply(products_concat)).reset_index()
train_5


# These product IDs now need to be compared to your predicted IDs for these order_ids

# ## F1-Score, Precision and Recall
# 
# After you have created a prediction on your test set (check other Kernels for script-ideas), we need to calculate the F1-score for each row and the total dataset. For more information on the F1-score, check [Wikipedia](https://en.wikipedia.org/wiki/F1_score) or [this article](http://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/).
# 
# For this we need to get the [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) for each row. These can be a bit confusing though at first.
# 
# - precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances
# - recall (also known as sensitivity) is the fraction of relevant instances that have been retrieved over total relevant instances
# 
# The F1-Score combines these two and gives a good score if you predict the right instances and the right amount of instances. 
# 
# **Example**
# 
# Let's say a customer actually bought 10 products (IDs are 0-9). Our model predicts 5 products (IDs 7, 8, 9, 10, 11) out of which 3 are correct (IDs 7, 8, 9) and 2 are incorrect.
# 
# - precision is 3/5
# - recall is 3/10
# 
# Instead of just rating your prediction on how many correct products you got, the F1-Score is penalizing predictions with too many false positives. Suppose we had 100 products in the whole inventory and we predicted that each customer bought all 100 of them, we would predict all the actually bought products but all the additional false positives would keep the score down.

# ### Comparing local test-data and prediction-data
# 
# For the ease of demonstration, lets create two DataFrames with easy to use product_ids. Incorrect predictions either have missing IDs or letters for products that were not bought.

# In[ ]:


df_real = pd.DataFrame({"order_id": [10, 11, 12, 13, 14, 15, 16],
                        "product_id": ["0", "0 1", "0 1 2 3",  "0 1 2 3", "0 1 2 3 4 5", 
                                       "0 1 2 3 4 5 6 7", "0 1 2 3 4 5 6 7 8 9"]},
                       index=np.arange(7))

df_pred = pd.DataFrame({"order_id": [10, 11, 12, 13, 14, 15, 16], 
                        "product_id": ["0", "0 X", "0 1 2 Y", "0 1 2 3 4 5", "0 1 2 3", 
                                       "0 1 2 3 4 5 6 7 8 9 X", "0 1 2 3 4 5 6 7 8"]},
                       index=np.arange(7))

df_real_preds = pd.merge(df_real, df_pred, on="order_id", suffixes=("_real", "_pred"))
df_real_preds


# In[ ]:


def score_order_predictions(df_real, df_pred, return_df=True, show_wrong_IDs=True):
    '''
    Print out the total weighted precision, recall and F1-Score for the given true and predicted orders.
    
    return_df:  if set to True, a new DataFrame with added columns for precision, recall and F1-Score will be returned.
    
    show_wrong_IDs: if set to True, two columns with the IDs that the prediction missed and incorrectly predicted will be added. Needs return_df to be True.
    '''
    df_combined = pd.merge(df_real, df_pred, on="order_id", suffixes=("_real", "_pred"))
    
    df_combined["real_array"] = df_combined["product_id_real"].apply(lambda x: x.split())
    df_combined["pred_array"] = df_combined["product_id_pred"].apply(lambda x: x.split())
    
    df_combined["num_real"] = df_combined["product_id_real"].apply(lambda x: len(x.split()))
    df_combined["num_pred"] = df_combined["product_id_pred"].apply(lambda x: len(x.split()))

    df_combined["num_pred_correct"] = np.nan
    for i in df_combined.index:
        df_combined.loc[i, "num_pred_correct"] = len([e for e in df_combined.loc[i,"real_array"]
                                                      if e    in df_combined.loc[i,"pred_array"]])
    if show_wrong_IDs==True:
        df_combined["IDs_missing"] = np.empty((len(df_combined), 0)).tolist()
        for i in df_combined.index:
            missing = np.in1d(df_combined.loc[i, "real_array"], df_combined.loc[i,"pred_array"], invert=True)
            missing_values = np.array(df_combined.loc[i, "real_array"])[missing]
            df_combined.set_value(i, "IDs_missing", missing_values)
 
        df_combined["IDs_not_ordered"] = np.empty((len(df_combined), 0)).tolist()
        for i in df_combined.index:
            not_ordered = np.in1d(df_combined.loc[i, "pred_array"], df_combined.loc[i,"real_array"], invert=True)
            not_ordered_values = np.array(df_combined.loc[i, "pred_array"])[not_ordered]
            df_combined.set_value(i, "IDs_not_ordered", not_ordered_values)

    df_combined["precision"] = np.round(df_combined["num_pred_correct"] / df_combined["num_pred"], 4)
    df_combined["recall"]    = np.round(df_combined["num_pred_correct"] / df_combined["num_real"], 4)
    df_combined["F1-Score"]  = np.round(2*( (df_combined["precision"]*df_combined["recall"]) / 
                                           (df_combined["precision"]+df_combined["recall"]) ), 4)
    
    recall_total =    df_combined["num_pred_correct"].sum() / df_combined["num_real"].sum()
    precision_total = df_combined["num_pred_correct"].sum() / df_combined["num_pred"].sum()
    F1_total =  2* ( (precision_total * recall_total) / (precision_total + recall_total) )      
    
    print("F1-Score: ", round(F1_total, 4))
    print("recall:   ", round(recall_total, 4))
    print("precision:", round(precision_total, 4))
    
    df_combined.drop(["real_array", "pred_array", "num_real", "num_pred", "num_pred_correct"], axis=1, inplace=True)
    
    # reorder columns so that the scoring-columns appear first and
    # all other on the right of them (bad readability with many IDs)
    df_combined = pd.concat([df_combined.loc[:, "order_id"], 
                             df_combined.iloc[:, -3:],
                             df_combined.iloc[:, 1:-3]], 
                            axis=1)
    if return_df==True:
        return df_combined
    else: 
        return None


# In[ ]:


df_scores = score_order_predictions(df_real, df_pred, return_df=True, show_wrong_IDs=True)
df_scores


# With this function you can split your orders-data and evaluate your predictions for a variety of different scenarios (predict only for weekends, for specific times of the day, for customers with less/more than 5 orders, for customers with rare or promotional products, etc).
# 
# Use the new columns in the resulting DataFrame to see if your predictions contain too many or too few IDs, if you have weak spots for specific cases (you can pass more DataFrames with more columns, e.g. the DayOfWeek, to the function) or if you keep missing/adding specific IDs. 
# 
# ---

# I have found a number of different ways to calculate the F1-Score. Scikit-Learn has a [few ways](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) to weight the classes. If you think another calculation would be more fitting to this challenge, please let me know.
# 
# **What other ideas do you have for validating predictions locally?**
