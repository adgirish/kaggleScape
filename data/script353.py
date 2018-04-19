
# coding: utf-8

# # **WORK IN PROGRESS**
# 
# - To reduce the database in order to avoid any crash from the Kernel, we will select the 10 most frequent products in the train data base.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import seaborn as sns
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
from datetime import date, timedelta


# # **Preparation & Initial Study**
# 
# ## 0. Load the Data
# 
# Let's use a part of the code from [inversion's Kernel](https://www.kaggle.com/inversion/dataframe-with-all-date-store-item-combinations).

# In[ ]:


dtypes = {'store_nbr': np.dtype('int64'),
          'item_nbr': np.dtype('int64'),
          'unit_sales': np.dtype('float64'),
          'onpromotion': np.dtype('O')}

train = pd.read_csv('../input/train.csv', dtype=dtypes)
test = pd.read_csv('../input/test.csv', dtype=dtypes)
stores = pd.read_csv('../input/stores.csv')
items = pd.read_csv('../input/items.csv')
trans = pd.read_csv('../input/transactions.csv')
#oil = pd.read_csv('../input/oil.csv') #we upload this database later
holidays = pd.read_csv('../input/holidays_events.csv')


# In[ ]:


date_mask = (train['date'] >= '2017-07-15') & (train['date'] <= '2017-08-15')
pd_train = train[date_mask]

#Print the size
len(pd_train)


# ## 1. Feature engineering
# 
# ### **Oil.csv - Replace missing values**
# 
# - We can observe some missing values in the Oil prices database.
# 

# In[ ]:


#Load the data
oil = pd.read_csv('../input/oil.csv')

#add missing date
min_oil_date = min(pd_train.date)
max_oil_date = max(pd_train.date)

calendar = []

d1 = datetime.datetime.strptime(min_oil_date, '%Y-%m-%d')  # start date date(2008, 8, 15)
d2 = datetime.datetime.strptime(max_oil_date, '%Y-%m-%d')  # end date

delta = d2 - d1         # timedelta

for i in range(delta.days + 1):
    calendar.append(datetime.date.strftime(d1 + timedelta(days=i), '%Y-%m-%d'))

calendar = pd.DataFrame({'date':calendar})

oil = calendar.merge(oil, left_on='date', right_on='date', how='left')


# In[ ]:


#Check how many NA
print(oil.isnull().sum(), '\n')

#Type
print('Type : ', '\n', oil.dtypes)

#Print the 3 first line
oil.head(5)


# We will replace the missing value with the following formula : 
# 
# $$\frac{(dcoilwtico[t-1] + dcoilwtico[t+1])} {2}$$

# In[ ]:


#Check index to apply the formula
na_index_oil = oil[oil['dcoilwtico'].isnull() == True].index.values

#Define the index to use to apply the formala
na_index_oil_plus = na_index_oil.copy()
na_index_oil_minus = np.maximum(0, na_index_oil-1)

for i in range(len(na_index_oil)):
    k = 1
    while (na_index_oil[min(i+k,len(na_index_oil)-1)] == na_index_oil[i]+k):
        k += 1
    na_index_oil_plus[i] = min(len(oil)-1, na_index_oil_plus[i] + k )

#Apply the formula
for i in range(len(na_index_oil)):
    if (na_index_oil[i] == 0):
        oil.loc[na_index_oil[i], 'dcoilwtico'] = oil.loc[na_index_oil_plus[i], 'dcoilwtico']
    elif (na_index_oil[i] == len(oil)):
        oil.loc[na_index_oil[i], 'dcoilwtico'] = oil.loc[na_index_oil_minus[i], 'dcoilwtico']
    else:
        oil.loc[na_index_oil[i], 'dcoilwtico'] = (oil.loc[na_index_oil_plus[i], 'dcoilwtico'] + oil.loc[na_index_oil_minus[i], 'dcoilwtico'])/ 2    


# In[ ]:


#Plot the oil values
oil_plot = oil['dcoilwtico'].copy()
oil_plot.index = oil['date'].copy()
oil_plot.plot()
plt.show()


# ## 2. Merge all the database

# In[ ]:


#Merge train
pd_train = pd_train.drop('id', axis = 1)
pd_train = pd_train.merge(stores, left_on='store_nbr', right_on='store_nbr', how='left')
pd_train = pd_train.merge(items, left_on='item_nbr', right_on='item_nbr', how='left')
pd_train = pd_train.merge(holidays, left_on='date', right_on='date', how='left')
pd_train = pd_train.merge(oil, left_on='date', right_on='date', how='left')
pd_train = pd_train.drop(['description', 'state', 'locale_name', 'class'], axis = 1)


# #Merge test - here is the code
# test = test.drop('id', axis = 1)
# test = test.merge(stores, left_on='store_nbr', right_on='store_nbr', how='left')
# test = test.merge(items, left_on='item_nbr', right_on='item_nbr', how='left')
# test = test.merge(oil, left_on='date', right_on='date', how='left')
# test = test.merge(holidays, left_on='date', right_on='date', how='left')
# test = test.drop(['description', 'state', 'locale_name', 'class'], axis = 1)

# ## 3. Quick look and modification on the data
# 
# ### **- Newly created Train DataBase**

# In[ ]:


#Shape
print('Shape : ', pd_train.shape, '\n')

#Type
print('Type : ', '\n', pd_train.dtypes)

#Summary
pd_train.describe()


# In[ ]:


#5 random lines
pd_train.sample(10)


# In[ ]:


sns.countplot(x='store_nbr', data=pd_train);


# ### **- Let's extract only the 10 most purchased product**

# In[ ]:


#######Get the N most purchased products########
def N_most_labels(data, variable , N , all='TRUE'):
    labels_freq_pd = itemfreq(data[variable])
    labels_freq_pd = labels_freq_pd[labels_freq_pd[:, 1].argsort()[::-1]] #[::-1] ==> to sort in descending order
    
    if all == 'FALSE':
        main_labels = labels_freq_pd[:,0][0:N]
    else: 
        main_labels = labels_freq_pd[:,0][:]
        
    labels_raw_np = data[variable].as_matrix() #transform in numpy
    labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)

    labels_filtered_index = np.where(labels_raw_np == main_labels)
    
    return labels_freq_pd, labels_filtered_index

label_freq, labels_filtered_index = N_most_labels(data = pd_train, variable = "item_nbr", N = 10, all='FALSE')
print("labels_filtered_index[0].shape = ", labels_filtered_index[0].shape)

pd_train_filtered = pd_train.loc[labels_filtered_index[0],:]
print("pd_train_filtered.shape = ", pd_train_filtered.shape)


# In[ ]:


label_freq[0:10]


# In[ ]:


pd_train_filtered.sample(3)


# ### **- Replace NA for "holydays" variables**

# In[ ]:


#Fill in cells if there is no holyday by the value : "no_holyday"
na_index_pd_train = pd_train_filtered[pd_train_filtered['type_y'].isnull() == True].index.values
print("Size of na_index_pd_train : ", len(na_index_pd_train), '\n')

pd_train_filtered.loc[pd_train_filtered['type_y'].isnull(), 'type_y'] = "no_holyday"
pd_train_filtered.loc[pd_train_filtered['locale'].isnull(), 'locale'] = "no_holyday"
pd_train_filtered.loc[pd_train_filtered['transferred'].isnull(), 'transferred'] = "no_holyday"
    
#check is there is NA
pd_train_filtered.isnull().sum()


# ### **- Reformat the date**
# 
# We will extract the day, the month and the year from the 'date' variable.

# In[ ]:


def get_month_year(df):
    df['month'] = df.date.apply(lambda x: x.split('-')[1])
    df['year'] = df.date.apply(lambda x: x.split('-')[0])
    
    return df

get_month_year(pd_train_filtered);


# In[ ]:


pd_train_filtered['date'] = pd.to_datetime(pd_train_filtered['date'])
pd_train_filtered['day'] = pd_train_filtered['date'].dt.weekday_name
pd_train_filtered = pd_train_filtered.drop('date', axis=1)


# In[ ]:


pd_train_filtered.sample(10)


# ## 3. Dummy variables
# 
# We will create binary variables.

# In[ ]:


dummy_variables = ['onpromotion','city','type_x','cluster','store_nbr','item_nbr',
                'family','perishable','type_y', 'locale', 'transferred', 'month', 'day']

for var in dummy_variables:
    dummy = pd.get_dummies(pd_train_filtered[var], prefix = var, drop_first = False)
    pd_train_filtered = pd.concat([pd_train_filtered, dummy], axis = 1)

pd_train_filtered = pd_train_filtered.drop(dummy_variables, axis = 1)
pd_train_filtered = pd_train_filtered.drop(['year'], axis = 1)


# In[ ]:


print('Shape : ', pd_train_filtered.shape)
pd_train_filtered.sample(10)


# ## 4. Scale variables
# 
# We can scale the variables so they are normalized with 0 mean and with a standard deviation equal to 1.

# In[ ]:


#Re-scale
#We keep this value to re-scale the predicted unit_sales values in the following lines of code.
min_train, max_train = pd_train_filtered['unit_sales'].min(), pd_train_filtered['unit_sales'].max()


# In[ ]:


scalable_variables = ['unit_sales','dcoilwtico']

for var in scalable_variables:
    mini, maxi = pd_train_filtered[var].min(), pd_train_filtered[var].max()
    pd_train_filtered.loc[:,var] = (pd_train_filtered[var] - mini) / (maxi - mini)


# In[ ]:


print('Shape : ', pd_train_filtered.shape)
pd_train_filtered.sample(10)


# ## **Split the data into a train and a validation database**

# In[ ]:


#train database without uni_sales
pd_train_filtered = pd_train_filtered.reset_index(drop=True)  #we reset the index
y_labels = pd_train_filtered['unit_sales']
X_train_filtered = pd_train_filtered.drop(['unit_sales'], axis = 1)

print('Shape X :', X_train_filtered.shape)
print('Shape y :', y_labels.shape)


# We split the train database with the function train_test_split (very useful and easy to use function).

# In[ ]:


num_test = 0.20
X_train, X_validation, y_train, y_validation = train_test_split(X_train_filtered, y_labels, test_size=num_test, random_state=15)
print('X_train shape :', X_train.shape)
print('y_train shape :', y_train.shape)
print('X_validation shape :', X_validation.shape)
print('y_validation shape :', y_validation.shape)


# # **Random Forest**
# 
# Before trying to build a neural network, we will try to predict the sales with a random forest (for the fun). We already know that because we cannot use all the data in this kaggle kernel, the results might not be good. Nevertheless, it might be interesting to compare different method.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
RFR = RandomForestRegressor()

# Choose some parameter combinations to try
#YOU CAN TRY DIFFERENTS PARAMETERS TO FIND THE BEST MODEL
parameters = {'n_estimators': [5, 10, 100],
              #'criterion': ['mse'],
              #'max_depth': [5, 10, 15], 
              #'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1,5]
             }

# Type of scoring used to compare parameter combinations
#We have to use RandomForestRegressor's own scorer (which is R^2 score)

# Run the grid search
grid_obj = GridSearchCV(RFR, parameters,
                        cv=5, #Determines the cross-validation splitting strategy /to specify the number of folds in a (Stratified)KFold
                        n_jobs=-1, #Number of jobs to run in parallel
                        verbose=1)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
RFR = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
RFR.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

predictions = RFR.predict(X_validation)

#if we want to Re-scale, use this lines of code :
#predictions = predictions * (max_train - min_train) + min_train
#y_validation_RF = y_validation * (max_train - min_train) + min_train

#if not, keep this one:
y_validation_RF = y_validation

print('R2 score = ',r2_score(y_validation_RF, predictions), '/ 1.0')
print('MSE score = ',mean_squared_error(y_validation_RF, predictions), '/ 0.0')


# In[ ]:


#Check and plot the 50 first predictions
plt.plot(y_validation_RF.as_matrix()[0:50], '+', color ='blue', alpha=0.7)
plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)
plt.show()


# The results are nice ! We can do better by using more observations.

# # **Neural Network with Keras**
# 
# ## KERAS REGRESSION - NEURAL NETWORK

# In[ ]:


import keras

# Convert data as np.array
features = np.array(X_train)
targets = np.array(y_train.reshape(y_train.shape[0],1))
features_validation= np.array(X_validation)
targets_validation = np.array(y_validation.reshape(y_validation.shape[0],1))

print(features[:10])
print(targets[:10])


# In[ ]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# Building the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1))

# Compiling the model
model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) #mse: mean_square_error
model.summary()


# In[ ]:


# Training the model
epochs_tot = 1000
epochs_step = 250
epochs_ratio = int(epochs_tot / epochs_step)
hist =np.array([])

for i in range(epochs_ratio):
    history = model.fit(features, targets, epochs=epochs_step, batch_size=100, verbose=0)
    
    # Evaluating the model on the training and testing set
    print("Step : " , i * epochs_step, "/", epochs_tot)
    score = model.evaluate(features, targets)
    print("Training MSE:", score[1])
    score = model.evaluate(features_validation, targets_validation)
    print("Validation MSE:", score[1], "\n")
    hist = np.concatenate((hist, np.array(history.history['mean_squared_error'])), axis = 0)
    
# plot metrics
plt.plot(hist)
plt.show()


# In[ ]:


predictions = model.predict(features_validation, verbose=0)

print('R2 score = ',r2_score(y_validation, predictions), '/ 1.0')
print('MSE score = ',mean_squared_error(y_validation_RF, predictions), '/ 0.0')


# In[ ]:


#Check and plot the 50 first predictions
plt.plot(y_validation.as_matrix()[0:50], '+', color ='blue', alpha=0.7)
plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)
plt.show()


# I'm quite surprised the predictions are "okay" even if we did not take a huge number of data. By gathering all the data from the database, we might obtain much more accurate results.
