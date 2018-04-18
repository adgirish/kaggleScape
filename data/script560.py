
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_rows', 1000)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Note to other dear competitors
# 
# This is the beginning of a series of notebook that I am working on for educational purpose, to demonstrate some run-of-the mill techniques we use on Kaggle for a student audience.
# 
# So when most folks here are competiting to overfit the LB ^_^, we are doing some small effort to fill our students with practical machine learning knowledge and the virtue of solid CV. 
# 
# Since our students are also competing as teams in this competition, we decided to use the kernel facility as a way to share knowledge. Nevertheless, all comments are welcome, and let's all enjoy the last days of this competition!

# # 1. Porto Seguo - Data Overview
# 
# In this competition we are tasked with making predictive models that can predict if a given driver will make insurance claim. One of the key aspects of this dataset is that it has already been nicely cleaned, with many categorical information nicely labeled and name accordingly - some simply as unordered categorical features, other are with ordinal values that represent the underlying logic. 
# 
# As directly quoted from the data description about different postfix:
# * *bin* to indicate binary features
# * *cat* to indicate categorical features
# * Features without these designations are either continuous or ordinal
# 
# Additionally, value of -1 indicate missing values. 
# 
# let's do some simple data manipulation to generate an overview on all the features over the train and test dataset

# In[2]:


# load train and test data, and then join them together
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample_submission=pd.read_csv('../input/sample_submission.csv')


# In[3]:


traintest=train.drop(['id','target'], axis=1).append(test.drop(['id'], axis=1))
cols=traintest.columns

# defining column name for feature statistics
# as the name suggested, we are capturing the following statistic from the features:
# nunique: number of unique value
# freq1: most frequent value 
# freq1_val: number of occurance of the most frequent value
# freq2: second most frequent value 
# freq2_val: number of occurance of the second most frequent value
# freq3: 3rd most frequent value, if available
# freq3_val: number of occurance of the thrid most frequent value, if available
# describe stats: the following ones are the stat offer by our best friend .describe methods. 

stat_cols= ['nunique','freq1','freq1_val', 'freq2', 'req2_val',
             'freq3', 'freq3_val'] + traintest[cols[0]].describe().index.tolist()[1:]

stat_cols=['feature']+stat_cols

feature_stat=pd.DataFrame(columns=stat_cols)
i=0

for col in cols:
    stat_vals=[]
    
    # get stat value
    stat_vals.append(col)
    stat_vals.append(traintest[col].nunique())
    stat_vals.append(traintest[col].value_counts().index[0])
    stat_vals.append(traintest[col].value_counts().iloc[0])
    stat_vals.append(traintest[col].value_counts().index[1])
    stat_vals.append(traintest[col].value_counts().iloc[1])
    
    if len(traintest[col].value_counts())>2:
        stat_vals.append(traintest[col].value_counts().index[2])
        stat_vals.append(traintest[col].value_counts().iloc[2])
    else:
        stat_vals.append(np.nan)
        stat_vals.append(np.nan)
            
    stat_vals+=traintest[col].describe().tolist()[1:]

    feature_stat.loc[i]=stat_vals
    i+=1


# Let's sort this list by the number of unique value, and have a closer look on the categorical features, sine these are the one that requires "special treatment" later on

# In[4]:


feature_stat[feature_stat['feature'].str.contains("cat")].sort_values(by=['nunique'])


# Hum, there are quite a few of them, and especially the feature 'ps_car_11_cat' has many levels.
# 
# # 2. Categorical features encoding
# 
# Categorical features in this dataset are encoded in a numeric manner, i.e. we have '0', '1', '2' representing different classes of some car related concepts like 'FWD', 'RWD', '4WD' for drive types. It is a nice way to represent information that is not in numeric format - since most ML algorthims prefer number over string. 
# 
# This does present a problem, as you can imagine, there isn't necessary a logical order to rank different drive types from 'FF' to '4DW in an ordinal manner i.e. '0','1','2'.  Now that we encoding them in such a way, we have actually manually imposed such an ordinal logic which doesn't necessary exist within the observations.
# 
# Let's apply some tricks present them in a non-ordinal way, here we are going to use "frequency encoding" and "binary encoding"
# 
# ## 2.1 Frequency Encoding
# 
# The idea behind frequency encoding is to represent the levels within a categorical features with the frequency they are observe in a dataset. One of the reaons of doing this is more frequently observed level often behave in a similar way - but this is not always true. In the end we need our cross-validation effort to tell us if this is indeed helpful. 
# 
# Anyway, let's build a utility function to do this:

# In[5]:


# This function late in a list of features 'cols' from train and test dataset, 
# and performing frequency encoding. 
def freq_encoding(cols, train_df, test_df):
    # we are going to store our new dataset in these two resulting datasets
    result_train_df=pd.DataFrame()
    result_test_df=pd.DataFrame()
    
    # loop through each feature column to do this
    for col in cols:
        
        # capture the frequency of a feature in the training set in the form of a dataframe
        col_freq=col+'_freq'
        freq=train_df[col].value_counts()
        freq=pd.DataFrame(freq)
        freq.reset_index(inplace=True)
        freq.columns=[[col,col_freq]]

        # merge ths 'freq' datafarme with the train data
        temp_train_df=pd.merge(train_df[[col]], freq, how='left', on=col)
        temp_train_df.drop([col], axis=1, inplace=True)

        # merge this 'freq' dataframe with the test data
        temp_test_df=pd.merge(test_df[[col]], freq, how='left', on=col)
        temp_test_df.drop([col], axis=1, inplace=True)

        # if certain levels in the test dataset is not observed in the train dataset, 
        # we assign frequency of zero to them
        temp_test_df.fillna(0, inplace=True)
        temp_test_df[col_freq]=temp_test_df[col_freq].astype(np.int32)

        if result_train_df.shape[0]==0:
            result_train_df=temp_train_df
            result_test_df=temp_test_df
        else:
            result_train_df=pd.concat([result_train_df, temp_train_df],axis=1)
            result_test_df=pd.concat([result_test_df, temp_test_df],axis=1)
    
    return result_train_df, result_test_df


# let's apply this function on some of our categorical features:

# In[6]:


cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat',
          'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_11_cat']

# generate dataframe for frequency features for the train and test dataset
train_freq, test_freq=freq_encoding(cat_cols, train, test)

# merge them into the original train and test dataset
train=pd.concat([train, train_freq], axis=1)
test=pd.concat([test,test_freq], axis=1)


# Let's have a look at one of the frequency feature and compare to the original feature:

# In[7]:


train[train.columns[train.columns.str.contains('ps_ind_02_cat')]].head(5)


# As you can see they a frequency value is now assigned to different level for the feature 'ps_ind_02_cat' - we shall remove the original feature after this transformation, but we still want to apply binary encoding before doing so.
# 
# ## 2.2 Binary Encoding
# 
# One of the most demonstrated effort for categorical encoding is so care "one-hot-encoding", in which case you create a vector of 0s for each level, and simply assign a 1 to replace the nth 0 for the nth level. So for instance for the drive type example we used above: "FWD", "RWD and "4WD", we would represent "FWD" with 100, "RWD" 010, and "4WD" with 001.
# 
# It is good idea to represent information in this way, in that it remove the unwanted ordinal pattern in 0, 1 and 2. But now we introduce an issue of dimensitionaity - we are now using n "dummy columns" to represent an original columns with n levels.  For a feature with 3 levels we might be ok, but for a feature like "ps_car_11_cat" are are introducing 100+ column into a original data set with 50 columns. By doing this we more than triple the size of the dataset, and this spells trouble for machine learning due to [*"the curse of dimensionality"*](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
# 
# Binary Encoding, as describe by this cool article [*"Beyond One-Hot"*](https://www.kdnuggets.com/2015/12/beyond-one-hot-exploration-categorical-variables.html) from KDnuggets, is an approach that allow you to use less "dummy columns" and yet remove the ordinal pattern from numeric encoding. The idea is to represent already numeric encoded levels from categorical features using their binary representation. So for a feature with 4 levels 0, 1, 2, 3: 0 will be encoded as "00", 1 will be encoded as "01", 2 will be encoded as "10" and 3 will be encoded as "11". 
# 
# Let's put this into action, and create a utility function for 

# In[8]:


# perform binary encoding for categorical variable
# this function take in a pair of train and test data set, and the feature that need to be encode.
# it returns the two dataset with input feature encoded in binary representation
# this function assumpt that the feature to be encoded is already been encoded in a numeric manner 
# ranging from 0 to n-1 (n = number of levels in the feature). 

def binary_encoding(train_df, test_df, feat):
    # calculate the highest numerical value used for numeric encoding
    train_feat_max = train_df[feat].max()
    test_feat_max = test_df[feat].max()
    if train_feat_max > test_feat_max:
        feat_max = train_feat_max
    else:
        feat_max = test_feat_max
        
    # use the value of feat_max+1 to represent missing value
    train_df.loc[train_df[feat] == -1, feat] = feat_max + 1
    test_df.loc[test_df[feat] == -1, feat] = feat_max + 1
    
    # create a union set of all possible values of the feature
    union_val = np.union1d(train_df[feat].unique(), test_df[feat].unique())

    # extract the highest value from from the feature in decimal format.
    max_dec = union_val.max()
    
    # work out how the ammount of digtis required to be represent max_dev in binary representation
    max_bin_len = len("{0:b}".format(max_dec))
    index = np.arange(len(union_val))
    columns = list([feat])
    
    # create a binary encoding feature dataframe to capture all the levels for the feature
    bin_df = pd.DataFrame(index=index, columns=columns)
    bin_df[feat] = union_val
    
    # capture the binary representation for each level of the feature 
    feat_bin = bin_df[feat].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))
    
    # split the binary representation into different bit of digits 
    splitted = feat_bin.apply(lambda x: pd.Series(list(x)).astype(np.uint8))
    splitted.columns = [feat + '_bin_' + str(x) for x in splitted.columns]
    bin_df = bin_df.join(splitted)
    
    # merge the binary feature encoding dataframe with the train and test dataset - Done! 
    train_df = pd.merge(train_df, bin_df, how='left', on=[feat])
    test_df = pd.merge(test_df, bin_df, how='left', on=[feat])
    return train_df, test_df


# Let's try this out!

# In[9]:


cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat',
          'ps_ind_05_cat', 'ps_car_01_cat']

train, test=binary_encoding(train, test, 'ps_ind_02_cat')
train, test=binary_encoding(train, test, 'ps_car_04_cat')
train, test=binary_encoding(train, test, 'ps_car_09_cat')
train, test=binary_encoding(train, test, 'ps_ind_05_cat')
train, test=binary_encoding(train, test, 'ps_car_01_cat')


# let's have a look at now all the columns with 'ps_ind_02_cat', you can now see how the binary encodeing features are created now for 'ps_ind_02_cat'

# In[10]:


train[train.columns[train.columns.str.contains('ps_ind_02_cat')]].head(5)


# Those of you with a sharp eye will notice that we didn't perform binary encoding for the feature 'ps_car_11_cat' - a categorical feature with more than 100 levels. The reason we didn't do it here is that performing binary encoding on this feature will create 40+ encoded columns, it is a sizable reduction compare to one-hot-encoding, but neverthelss would still inflat this dataset by 80% from 50+ columns to 90+ columns. For a bigger dataset with more column this is still doable, however in this case, generating 40+ eencoded columns would still bring too much noise for the benfit of doing it.
# 
# What to do in this case? You can try more tricks, perhaps by referring to this wonder slide on [*feature engineering*](https://www.slideshare.net/HJvanVeen/feature-engineering-72376750) by [*Triskelion*](https://www.kaggle.com/triskelion). I personally would have try target encoding, but this need to be done in a k-fold manner, and I would skip it here to save some space for doing some acutal modelling. 
# 
# ## 3. Simple modelling pipeline
# 
# Ok, now let's try to do some actual modelling using the train and test dataset we have been working on.
# First of all, let's also trim the dataset a bit more. As it has already been shared in the kernel such as this wonderful [*interactive EDA*](https://www.kaggle.com/headsortails/steering-wheel-of-fortune-porto-seguro-eda) by [*Heads or Tails*](https://www.kaggle.com/headsortails) , that all the feature with the wording "cal" isn't useful. Here we are going to drop all these columns

# In[11]:


col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train.drop(col_to_drop, axis=1, inplace=True)  
test.drop(col_to_drop, axis=1, inplace=True)  


# Also, we now that we have processed some of the categorical features in the form of frequency encoding and binary encoding, lest's remove the original featrues. - We keep the feature 'ps_car_11_cat' for now as we haven't treated it with anything other than frequency encoding.

# In[12]:


cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat', 'ps_ind_05_cat', 'ps_car_01_cat']
train.drop(cat_cols, axis=1, inplace=True)  
test.drop(cat_cols, axis=1, inplace=True)  


# Next step, let's create the local train and validation data for this little exercise

# In[13]:


localtrain, localval=train_test_split(train, test_size=0.25, random_state=2017)

drop_cols=['id','target']
y_localtrain=localtrain['target']
x_localtrain=localtrain.drop(drop_cols, axis=1)

y_localval=localval['target']
x_localval=localval.drop(drop_cols, axis=1)


# In the next parts, we are going to build some simple ML model to make prediction on our dataset, with Random Forest and Logistic Regression 
# 
# ## 3.1 Random Forest model training and prediction
# Let's use sklearn's Random Forest API to fit our Random Forest model, and perform the prediction on local validation set

# In[14]:


print('Start training...')
start_time=time.time()

rf=RandomForestClassifier(n_estimators=250, n_jobs=6, min_samples_split=5, max_depth=7,
                          criterion='gini', random_state=0)

rf.fit(x_localtrain, y_localtrain)
rf_valprediction=rf.predict_proba(x_localval)[:,1]

end_time = time.time()
print("it takes %.3f seconds to train and predict" % (end_time - start_time))


# It has been shared in the [*competition forum*](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/40368) that there is an easy conversion from AUC to Gini normalised coefficient - this is further explained in the wiki page for [*Gini coeeficient*](https://en.wikipedia.org/wiki/Gini_coefficient#Relation_to_other_statistical_measures).

# In[15]:


rf_val_auc=roc_auc_score(y_localval, rf_valprediction)
rf_val_gininorm=2*rf_val_auc-1

print('Random Forest Validation AUC is {:.6f}'.format(rf_val_auc))
print('Random Forest Validation Normalised Gini Coefficient is {:.6f}'.format(rf_val_gininorm))


# Finally let's perform the prediction for the test data

# In[16]:


x_test=test.drop(['id'], axis=1)
y_testprediction=rf.predict_proba(x_test)[:,1]

rf_submission=sample_submission.copy()
rf_submission['target']=y_testprediction
rf_submission.to_csv('rf_submission.csv', compression='gzip', index=False)


# ##  3.2 Logistic Regression training and prediction
# 
# With sklearn, the training and prediction for Logistic Regression is quite similar to that of the Random Forest, the only major difference is that we have to remember for perform feature scaling first, and then we have to use the scaled features for training and prediction.

# In[17]:


scaler = StandardScaler().fit(x_localtrain.values)
x_localtrain_scaled = scaler.transform(x_localtrain)
x_localval_scaled = scaler.transform(x_localval)
x_test_scaled = scaler.transform(x_test)

print('Start training...')
start_time=time.time()

logit=LogisticRegression()
logit.fit(x_localtrain_scaled, y_localtrain)
logit_valprediction=logit.predict_proba(x_localval_scaled)[:,1]

end_time = time.time()
print("it takes %.3f seconds to train and predict" % (end_time - start_time))


# The prediction step is rather straightforward - exactly the same as the one for Random Forest

# In[18]:


logit_val_auc=roc_auc_score(y_localval, logit_valprediction)
logit_val_gininorm=2*logit_val_auc-1

print('Logistic Regression Validation AUC is {:.6f}'.format(logit_val_auc))
print('Logistic Regression Validation Normalised Gini Coefficient is {:.6f}'.format(logit_val_gininorm))


# Let's also generate the prediction for logistic regression

# In[19]:


y_testprediction=logit.predict_proba(x_test_scaled)[:,1]

logit_submission=sample_submission.copy()
logit_submission['target']=y_testprediction
logit_submission.to_csv('logit_submission.csv', compression='gzip', index=False)

