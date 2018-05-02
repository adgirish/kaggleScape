
# coding: utf-8

# Hi, Kagglers!
# 
# Hereafter I will try to publish **some basic approaches to climb up the Leaderboard**
# 
# **Competition goal**
# 
# In this competition, Daimler is challenging Kagglers to tackle the curse of dimensionality and reduce the time that cars spend on the test bench.
# <br>Competitors will work with a dataset representing different permutations of Mercedes-Benz car features to predict the time it takes to pass testing. <br>Winning algorithms will contribute to speedier testing, resulting in lower carbon dioxide emissions without reducing Daimler’s standards. 
# 
# **The Notebook offers basic Keras skeleton to start with**
# 
# ### Stay tuned, this notebook will be updated on a regular basis
# **P.s. Upvotes and comments would let me update it faster and in a more smart way :)**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# preprocessing/decomposition
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA

# keras 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# supportive models
from sklearn.ensemble import GradientBoostingRegressor
# feature selection (from supportive model)
from sklearn.feature_selection import SelectFromModel

# to make results reproducible
seed = 42 # was 42


# In[ ]:


# Read datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# save IDs for submission
id_test = test['ID'].copy()

# glue datasets together
total = pd.concat([train, test], axis=0)
print('initial shape: {}'.format(total.shape))

# binary indexes for train/test set split
is_train = ~total.y.isnull()

# find all categorical features
cf = total.select_dtypes(include=['object']).columns

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    total[cf],
    drop_first=False # you can set it = True to ommit multicollinearity (crucial for linear models)
)

print('oh-encoded shape: {}'.format(dummies.shape))

# get rid of old columns and append them encoded
total = pd.concat(
    [
        total.drop(cf, axis=1), # drop old
        dummies # append them one-hot-encoded
    ],
    axis=1 # column-wise
)

print('appended-encoded shape: {}'.format(total.shape))

# recreate train/test again, now with dropped ID column
train, test = total[is_train].drop(['ID'], axis=1), total[~is_train].drop(['ID', 'y'], axis=1)

# drop redundant objects
del total

# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))


# In[ ]:


# Calculate additional features: dimensionality reduction components
n_comp=10 # was 10

# uncomment to scale data before applying decompositions
# however, all features are binary (in [0,1] interval), i don't know if it's worth something
train_scaled = train.drop('y', axis=1).copy()
test_scaled = test.copy()
'''
ss = StandardScaler()
ss.fit(train.drop('y', axis=1))

train_scaled = ss.transform(train.drop('y', axis=1).copy())
test_scaled = ss.transform(test.copy())
'''

# PCA
pca = PCA(n_components=n_comp, random_state=seed)
pca2_results_train = pca.fit_transform(train_scaled)
pca2_results_test = pca.transform(test_scaled)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train_scaled)
ica2_results_test = ica.transform(test_scaled)

# Append it to dataframes
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
   


# In[ ]:


# create augmentation by feature importances as additional features
t = train['y']
tr = train.drop(['y'], axis=1)

# Tree-based estimators can be used to compute feature importances
clf = GradientBoostingRegressor(
                max_depth=4, 
                learning_rate=0.005, 
                random_state=seed, 
                subsample=0.95, 
                n_estimators=200
)

# fit regressor
clf.fit(tr, t)

# df to hold feature importances
features = pd.DataFrame()
features['feature'] = tr.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

# select best features
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(tr)


test_reduced = model.transform(test.copy())

# dataset augmentation
train = pd.concat([train, pd.DataFrame(train_reduced)], axis=1)
test = pd.concat([test, pd.DataFrame(test_reduced)], axis=1)

# check new shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))


# In[ ]:


# define custom R2 metrics for Keras backend
from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# In[ ]:


# base model architecture definition
def model():
    model = Sequential()
    #input layer
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    # hidden layers
    model.add(Dense(input_dims))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//2))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.3))
    
    model.add(Dense(input_dims//4, activation=act_func))
    
    # output layer (y_pred)
    model.add(Dense(1, activation='linear'))
    
    # compile this model
    model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                  metrics=[r2_keras] # you can add several if needed
                 )
    
    # Visualize NN architecture
    print(model.summary())
    return model


# In[ ]:


# initialize input dimension
input_dims = train.shape[1]-1

#activation functions for hidden layers
act_func = 'tanh' # could be 'relu', 'sigmoid', ...

# make np.seed fixed
np.random.seed(seed)

# initialize estimator, wrap model in KerasRegressor
estimator = KerasRegressor(
    build_fn=model, 
    nb_epoch=100, 
    batch_size=20,
    verbose=1
)


# In[ ]:


# X, y preparation
X, y = train.drop('y', axis=1).values, train.y.values
print(X.shape)

# X_test preparation
X_test = test
print(X_test.shape)

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=seed
)


# In[ ]:


# define path to save model
import os
model_path = 'keras_model.h5'

# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=10, # was 10
        verbose=1),
    
    ModelCheckpoint(
        model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=0)
]

# fit estimator
estimator.fit(
    X_tr, 
    y_tr, 
    epochs=10, # increase it to 20-100 to get better results
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=callbacks,
    shuffle=True
)


# In[ ]:


# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})

# check performance on train set
print('MSE train: {}'.format(mean_squared_error(y_tr, estimator.predict(X_tr))**0.5)) # mse train
print('R^2 train: {}'.format(r2_score(y_tr, estimator.predict(X_tr)))) # R^2 train

# check performance on validation set
print('MSE val: {}'.format(mean_squared_error(y_val, estimator.predict(X_val))**0.5)) # mse val
print('R^2 val: {}'.format(r2_score(y_val, estimator.predict(X_val)))) # R^2 val
pass


# In[ ]:


# predict results
res = estimator.predict(X_test.values).ravel()

# create df and convert it to csv
output = pd.DataFrame({'id': id_test, 'y': res})
output.to_csv('keras-baseline.csv', index=False)

