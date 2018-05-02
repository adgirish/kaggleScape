
# coding: utf-8

# The data section mentions that some training examples are missing the **"inc_angle"** field. Let's come up with a good way to fill in the missing values!

# First, let's see how many training examples we need to fix:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_json("../input/statoil-iceberg-classifier-challenge/train.json")

training_examples = train.shape[0]
missing_angles = len(train[train['inc_angle'] == 'na'])
percent_missing = (missing_angles/training_examples)*100

print("{0}/{1} ({2:.2f}%) of examples are missing inc_angle".format(
    missing_angles, training_examples, percent_missing))


# Not too many... but 8% is a sinificant portion of the training data. It's definetely more than we can fill in manually.

# Mean, Median and Mode
# ===========
# One quick and easy solution would be to fill in all the missing angles with the mean, median or mode of all known inc_angles. Let's try it: 

# In[ ]:


# Include the test data in our calculations: 
test = pd.read_json("../input/statoil-iceberg-classifier-challenge/test.json")
train_no_ib = train.drop(['is_iceberg'],axis=1)
examples = pd.concat([train_no_ib,test])

inc_angles = examples[examples['inc_angle'] != 'na']['inc_angle']

mean = inc_angles.mean()
median = inc_angles.median()
mode = inc_angles.astype(np.double).round(1).mode()[0] # round to the nearest tenth for mode
print("Mean: {0}\nMedian: {1}\nMode: {2}".format(mean,median,mode))


# **Which one should we use?  **
# 
# To get a better sense of the quality of our fill in values, let's treat this like a learning problem. We can create train & validation sets and evaluate performance with *mean absolute error*:

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

inc_angles_train, inc_angles_valid = train_test_split(inc_angles, random_state=1, train_size=0.8, test_size=0.2)

ones = np.ones(inc_angles_valid.shape[0])
mean_mae = mean_absolute_error(ones*inc_angles_train.mean(), inc_angles_valid)
median_mae = mean_absolute_error(ones*inc_angles_train.median(), inc_angles_valid)
mode_mae = mean_absolute_error(ones*inc_angles_train.astype(np.double).round(1).mode()[0], inc_angles_valid)

print("Mean Error: {0}\nMedian Error: {1}\nMode Error: {2}".format(mean_mae,median_mae,mode_mae))


# So on average, our examples' incidence angles are ~3.43 degrees away from the median and the mean. It looks like mode is a little worse.
# 
# If we wanted to stop here, we could fill in our missing inc_angles with the median (+ small random value) and write out a new training data file:

# In[ ]:


from random import uniform

train_out = train.copy()

min_var = median_mae*-0.5
max_var = median_mae*0.5

train_out['inc_angle'] = [(median + uniform(min_var,max_var)) if angle == 'na' 
                          else angle 
                          for angle in train_out['inc_angle']]

train_out.to_json('train_median_fill.json')


# *But we can do better!*

# Predicting Incidence Angles
# ===
# Since the incidence angle describes a relationship between our 2 radar bands we should be able to build a model that predicts the incidence angle given the two bands as inputs. Let's give it a try with Keras:

# In[ ]:


from keras.models import Input,Model
from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.regularizers import l2
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

def model(dropout=0.1, regularization=0.00005):

    x_input = Input(shape=(75,75,2,1,)) 

    # Layer 1
    x = Conv3D(96, kernel_size=(5, 5, 2),activation='relu',input_shape=(75, 75, 2,1), kernel_regularizer=l2(regularization))(x_input)
    x = BatchNormalization()(x)
    x = MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1))(x)
    x = Dropout(dropout)(x)

    x = Reshape((35,35,96))(x)

    # Layer 2
    x = Conv2D(128, kernel_size=(3, 3), activation='relu' , kernel_regularizer=l2(regularization))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)
    
    # Layer 3
    x = Conv2D(256, kernel_size=(3, 3), activation='relu' , kernel_regularizer=l2(regularization))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(dropout+0.1)(x)
    
    # Layer 4
    x = Conv2D(128, kernel_size=(3, 3), activation='relu' , kernel_regularizer=l2(regularization))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(dropout)(x)
    
    x = Flatten()(x)
    
    # Layer 5
    x = Dense(768, kernel_regularizer=l2(regularization))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout+0.1)(x)
    
    # Layer 6
    x = Dense(384, kernel_regularizer=l2(regularization))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout+0.1)(x)
    
    # Linear Output Layer
    y_ = Dense(1)(x)
    
    model = Model(inputs=x_input, outputs=y_)
    adam_otim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam_otim, metrics=['mae'])
    
    model.summary()
    return model


# This is the best of a few different model setups I tried. It's loosly based on DeveshMaheshwari's [kernal](https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d). There's probably room for improvement. Feel free to tweak the hyperparameters and let me know what works best for you.
# 
# Notice we're using the same evaluation metric, mean absolute error ('mae'), that we used with our mean/median/mode fills above.

# In[ ]:


def load_train_data():
    train = pd.read_json("../input/statoil-iceberg-classifier-challenge/train.json")
    test = pd.read_json("../input/statoil-iceberg-classifier-challenge/test.json")
    
    train = train.drop(['is_iceberg'],axis=1)
    train = pd.concat([train,test])
    train = train[train['inc_angle'] != 'na']
    
    band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    bands = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis]], axis=-1)
    bands = bands.reshape((-1, 75, 75, 2, 1))
    
    angles = train["inc_angle"]
    
    return train_test_split(bands, angles, random_state=1, train_size=0.8, test_size=0.2)


# It took my macbook a few hours to train the model. So to speed things up, I included my pretrained weights as an input to this kernal. If you want to tweak the model and train it yourself locally, I included code to do that too:

# In[ ]:


m = model()
x_train, x_valid, y_train, y_valid = load_train_data()
weights_file = '../input/pretrained-weights-for-inc-angle/inc_angle_weights_pretrained.hdf5'

TRAIN_FROM_SCRATCH = False

if TRAIN_FROM_SCRATCH:
    checkpoint = ModelCheckpoint(weights_file, save_best_only=True)
    model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=1,
              validation_data=(x_valid, y_valid),
              callbacks=[checkpoint])
else:
    m.load_weights(filepath=weights_file)


# Now that it's trained, let's see if our model does any better than the median at predicting inc_angles:

# In[ ]:


predicted_angles = m.predict(x_valid, verbose=1)
model_mae = mean_absolute_error(predicted_angles, y_valid)
print('Model Error: {0}'.format(model_mae))


# **Success!**
# 
# Our model is ~16% better than the median at predicting incidence angles. It's not a huge improvement. And it probably wasn't worth all the effort considering it only helps ~8% of the training data... but nevertheless! It's still pretty cool that we can get a rough angle estimate given the two radar images. I suspect it wouldn't be too much effort to improve our accuracy even further.
# 
# Using our awesome new angle-prediction-model, let's create a new training data file without any missing inc_angles:

# In[ ]:


def predict_inc_angle(ex, model):
    band_1 = np.array([np.array(ex["band_1"]).astype(np.float32).reshape(75, 75)])
    band_2 = np.array([np.array(ex["band_2"]).astype(np.float32).reshape(75, 75)])
    bands = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis]], axis=-1)
    bands = bands.reshape((1, 75, 75, 2, 1))
    inc_angle = model.predict(bands)
    return inc_angle.reshape(1)[0]
    
train_out_model = train.copy()

train_out_model['inc_angle'] = [predict_inc_angle(ex,m) if ex['inc_angle'] == 'na' 
                          else ex['inc_angle'] 
                          for _,ex in train_out_model.iterrows()]

train_out_model.to_json('train_model_fill.json')
#print(train_out_model)


# As a bonus, if you take a look at the output (1574->1603 were all previously 'na') you'll notice there's a lot more variation in our predicted angles compared to the "median + random value" method seen earlier.  The variations from our *model*'s output should be more meaningful too. It's also nice to see that our model wasn't just learning the median. 
# 
# Feel free to use either of the results in your own models. Let me know if you have any questions or suggestions below. Upvote if this was helpful and thanks for reading!
# 
# -Ryan
