
# coding: utf-8

# # Predicting customers who will "charge-off"
# *produced by Vincenzo Pota in August 2017 *

# This notebook contains my attempt to predict customers who will charge-off in the future. I describe in detail the following steps:
# 1. Data Cleaning
# 2. Feature selection and transformation
# 3. Define the business case
# 4. Build the models
# 5. Test the models
# 
# Dataset is given in a flat file and in a database. Let's use the database for good practice and for performances. Once the dataset is better understood, we can perform data cleaning and aggregation in-database to avoid overloading computer memory. 

# Let's load the libraries, connect to the database, parse dates and load all data in-memory:

# In[ ]:


import sqlite3
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

conn = sqlite3.connect('../input/database.sqlite') # This might take a while to run...
to_parse = ['issue_d' , 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
df = pd.read_sql_query('select * from loan', con=conn, parse_dates = to_parse)


# The dataframe `df` has 887,384 rows and 75 columns. It occupies 0.5Gb, which might be problematic for later data modelling.

# In[ ]:


print('The shape is {}'.format(df.shape))
print('Memory : {} Mb'.format(int(df.memory_usage(deep=False).sum() / 1000000)))


# ## Data Cleaning: it is not as bad as it looks

# After a closer inspection in Excel, many of the columns seem to contain very little information. I will remove these columns to make the dataset more managable and to release some memory. In a real-case situation, I would not have adopted such a conservative approach.
# 
# ### Remove columns with more than 60% null values
# These are:

# In[ ]:


check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))
check_null[check_null>0.6]


# ...for a total of 21 columns. We can remove these columns with `inplace=True` to overwrite the current dataframe in memory. Remove also a line with all null values...

# In[ ]:


df.drop(check_null[check_null>0.5].index, axis=1, inplace=True) 
df.dropna(axis=0, thresh=30, inplace=True)


# ### Remove columns with little information
# Here are some columns we want to remove and why:
# 1. `index` is not needed because it's built-in the dataframe itself
# 2. `policy_code` is always `== 1`
# 3. `payment_plan` has only 10 `y` and 887372 `n`
# 4. `url` not needed, although it might be useful if it contains extra-data (e.g., payment history)
# 5. `id` and `member_id` are all unique, which is a bit misleading. I was expecting to find payment histories, but it seems that every record is a single customer.
# 6. `application_type` is 'INDIVIDUAL' for 99.94% of the records
# 7. `acc_now_delinq` is `0` for 99.5% of the records
# 8. `emp_title` not needed here, but it might be useful for the modelling (see below), 
# 9. `zip_code` not needed for this level of analysis,
# 10. `title` might be useful with NLP, but let's ignore it for now
# 
# Numbers above have been calculated by grouping by the metrics, counting the size of each group and sorting. For example:

# In[ ]:


df.groupby('application_type').size().sort_values()


# We can now delete the columns above:

# In[ ]:


delete_me = ['index', 'policy_code', 'pymnt_plan', 'url', 'id', 'member_id', 'application_type', 'acc_now_delinq','emp_title', 'zip_code','title']
df.drop(delete_me , axis=1, inplace=True) 


# ## Feature transformations

# The dataset has now 43 columns. We need to transform a few metrics which sound very important, but are formatted as strings. These transformations are performed with the __modelling__ in mind. Ultimatelly we want to produce a dataset almost ready to be fed to the model. Here is a summary of the operations performed:
# 1. Strip `months` from `term` and make it an integer
# 2. The Interest rate is a string. Remove `%` and make it a float
# 3. Extract numbers from `emp_length` and fill missing values with the median (see below). If `emp_length == 10+ years` then leave it as `10`
# 4. Transform `datetimes` to a Period 

# In[ ]:


# strip months from 'term' and make it an int
df['term'] = df['term'].str.split(' ').str[1]

#interest rate is a string. Remove % and make it a float
df['int_rate'] = df['int_rate'].str.split('%').str[0]
df['int_rate'] = df.int_rate.astype(float)/100.

# extract numbers from emp_length and fill missing values with the median
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
df['emp_length'] = df['emp_length'].fillna(df.emp_length.median())

col_dates = df.dtypes[df.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    df[d] = df[d].dt.to_period('M')


# In[ ]:


df.head()


# ## Data exploration
# We now have the data in a more suitable form for data exploration. I could plot different combinations of metrics on 2-dimensional plots and look for interesting trends. Instead, I want to touch briefly two techniques that can allow us to have an overview of the dataset without too much coding involved.
# 
# ### Use interactive pivot tables with javascript
# We can explore the dataset with one single javascript wrapper using the library `pivottablejs` which allows us to do aggregations and plotting using javascipt libraries. On this computer, this library cannot handle 800k rows and 43 columns in a reasonable amount of time, so I decided to input a __random__ selection of 10% of the dataframe. This should be ok for proportions and averaged, but not for absolute counts. This is when aggregating in-database would speed things up.

# In[ ]:


# pivot_ui(df.sample(frac=0.1))
# opens a new window


# A few things to notice:
# * A line plot of `issue_dt` vs. `grade` (counted as fraction of columns) reveals that the relative fraction of loan grade changes with time (especially after 2012-07). It would be interesting to understand if this change was due to business changes or to changes in customer behaviour. 
# * A stacked bar chart plot of `home_ownership` vs. `loan_status` (counted as fraction of columns) shows that a `loan status` of *Charged_off* is about 4% for customers who own, rent or with a mortgage. Even though the `loan_status` is 10% and 25% for customers with None or Other, the total counts for these categories are very small. 
# * A stacked bar chart plot of `grade` vs. `loan_status` (counted as fraction of columns) shows that, as expected, the *Charged_off* status becomes more and more relevant for higher interest rates (grades F and G)

# # Data Modelling

# __Let's build a model which predicts the status *charged_off*__. The fraction of this status in the whole dataset is low, only around 5%, but not as low as other status. 

# In[ ]:


loan_status_grouped = df.groupby('loan_status').size().sort_values(ascending=False)/len(df) * 100
loan_status_grouped


# This is indeed a bit problematic, but let's how the models perform first.
# 
# ## The business problem
# In developing the model we need to think about the business problem we are trying to solve. I have identified two different scenarios:
# 1. In the first scenario, the investor (assuming he/she has access to our same data) wants to predict the risk of *charged off* before lending the money to a borrower. The metrics associated to activity in the Loan Club are not known because the customer is still a prospect borrower.
# 2. In the second scenario, Loan Club wants to predict probability for a borrower to charge off while he/she is "Current", maybe to prevent the charge off from happening or try to minimise damage.
# 
# I do not fully understand the meaning of all metrics. Therefore I will adopt the first scenario because I believe is the one which makes more sense with my current understanding of the problem. 
# 
# The problem therefore becomes: __How well can we predict that a prospect customer will charge off at some point in the future?__ 

# ## More feature engeneering
# We can finally choose the metrics for the model remembering to check for missing values and transforming metrics in a way suitable for modelling. 
# 
# * Let's keep the `loan_amount`, but let's create a metric which indicates that the total amount committed by investors for that loan at that point in time (`funded_amnt_inv`) is less than what the borrower requested.

# In[ ]:


df['amt_difference'] = 'eq'
df.loc[(df['funded_amnt'] - df['funded_amnt_inv']) > 0,'amt_difference'] = 'less'


# * The interest rate is an important metrics, but it changes with time, whereas the interest grade does not. So, we will consider the interest `grade` only, exluding the `sub_grade` to keep it simple.
# 
# * the metrics `delinq_2yrs` is very skewed towards zero (80% are zeros). Let's make it categorical: `no` when `delinq_2yrs == 0` and `yes` when  `delinq_2yrs > 0`
# 
# * Same as above for `inq_last_6mths`: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
# 
# * Same as above for `pub_rec`: Number of derogatory public records
# 
# * I thought about computing difference between the date of the earliest credit line and the issue date `df['tmp'] = df.earliest_cr_line - df.issue_d`, but I do not understand the metrics well, so I will skip this
# 
# * Let's compute the ratio of the number of open credit lines in the borrower's credit file divided by the total number of credit lines currently in the borrower's credit file

# In[ ]:


# Make categorical

df['delinq_2yrs_cat'] = 'no'
df.loc[df['delinq_2yrs']> 0,'delinq_2yrs_cat'] = 'yes'

df['inq_last_6mths_cat'] = 'no'
df.loc[df['inq_last_6mths']> 0,'inq_last_6mths_cat'] = 'yes'

df['pub_rec_cat'] = 'no'
df.loc[df['pub_rec']> 0,'pub_rec_cat'] = 'yes'

# Create new metric
df['acc_ratio'] = df.open_acc / df.total_acc


# These are the features we want to model

# In[ ]:


features = ['loan_amnt', 'amt_difference', 'term', 
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',  
            'loan_status'
           ]


# Given the business problem stated above, we want to distinguish between a customer who will *charge off* and a customer who will pay in full. I will not model the cohort of *Current* customers because these are still "in progress" and belong to the second scenario. 

# In[ ]:


X_clean = df.loc[df.loan_status != 'Current', features]
X_clean.head()


# In[ ]:


mask = (X_clean.loan_status == 'Charged Off')
X_clean['target'] = 0
X_clean.loc[mask,'target'] = 1


# ## A few last touches
# We need to transform categorical variables in continuous variables using the One Hot Encoder. `pandas` has a built-in function for this.

# In[ ]:


cat_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'pub_rec_cat', 'initial_list_status']

# Drop any residual missing value (only 24)
X_clean.dropna(axis=0, how = 'any', inplace = True)

X = pd.get_dummies(X_clean[X_clean.columns[:-2]], columns=cat_features).astype(float)
y = X_clean['target']


# ## The models
# 
# Let's start modelling by importing a few modules. Features are all on different scale, so it is wise to rescale all features in the range -1, +1

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

X_scaled = preprocessing.scale(X)
print(X_scaled)
print('   ')
print(X_scaled.shape)


# Write a function that :
# 1. Takes train and test set under different assumptions
# 2. Runs a set of models. 3 in this case: Gradient Boosting, Logistic Regression and Random Forest
# 3. Makes prediction using the test set
# 4. Builds-up a table with evaluation metrics
# 5. Plots a roc curve of the estimators

# In[ ]:


def run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced'):
    
    clfs = {'GradientBoosting': GradientBoostingClassifier(max_depth= 6, n_estimators=100, max_features = 0.3),
            'LogisticRegression' : LogisticRegression(),
            #'GaussianNB': GaussianNB(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=10)
            }
    cols = ['model','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),
                         'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),
                         'precision_score': metrics.precision_score(y_test, y_pred),
                         'recall_score': metrics.recall_score(y_test, y_pred),
                         'f1_score': metrics.f1_score(y_test, y_pred)})

        models_report = models_report.append(tmp, ignore_index = True)
        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')
    
    return models_report, conf_matrix


# ### Model with unbalanced classes
# If we do not modify the class ratios our model has very poor predictive power. The area ander the curve (AUC) is about 0.6, suggesting that we perform better than random. However, the recall is zero: we cannot predict the target variable at all. This might be either because there is something wrong with the metrics or because the classes are too unbalanced. 

# In[ ]:


#mpl.rc("savefig", dpi=300)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.4, random_state=0)
models_report, conf_matrix = run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced')


# In[ ]:


models_report


# In[ ]:


conf_matrix['LogisticRegression']


# ### Model with synthetically balanced classes
# 
# We can artificially balance the classes using the algorithm SMOTE ( Synthetic Minority Over-sampling Technique). This uses a K-nearest neighbour approach to create feature vectors which resemble those of the target variable. The minority class is oversampled. With this trick, the performance of the model improves considerably.
# 
# We now have a recall of 70% using Logistic Regression. We get right 7 out of 10 customers who will "charge off". On the other hand we have a precision of 20%. 

# In[ ]:


index_split = int(len(X)/2)
X_train, y_train = SMOTE().fit_sample(X_scaled[0:index_split, :], y[0:index_split])
X_test, y_test = X_scaled[index_split:], y[index_split:]

#scores = cross_val_score(clf, X_scaled, y , cv=5, scoring='roc_auc')

models_report_bal, conf_matrix_bal = run_models(X_train, y_train, X_test, y_test, model_type = 'Balanced')


# In[ ]:


models_report_bal


# In[ ]:


conf_matrix_bal['LogisticRegression']

