
# coding: utf-8

# # Introduction
# In order to bring some data augmentation to my model I wanted to use keras's ImageDataGenerator and fit_generator functions. The only issue being that this will not work out of the box, as the generator will not work with multiple inputs. I found a solution that works for me, and I don't currently see any other keras/python implementations with this working so thought I'd share.
# 
# Significant snippets of code are "borrowed" from these kernels:
# - [Exploring the Icebergs with skimage and Keras](https://www.kaggle.com/kmader/exploring-the-icebergs-with-skimage-and-keras) - Kevin Mader  
# - [A keras prototype (0.21174 on PL)](https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl) - noobhound

# In[ ]:


# Do some imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


# Load data and process data
data_dir = "../input/"

def load_data(data_dir):
    train = pd.read_json(data_dir+"train.json")
    test = pd.read_json(data_dir+"test.json")
    # Fill 'na' angles with zero
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    test.inc_angle = test.inc_angle.replace('na', 0)
    test.inc_angle = test.inc_angle.astype(float).fillna(0.0)
    return train, test

train, test = load_data(data_dir)

# Process data into images
def process_images(df):
    X_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    X_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    # Merge bands and add another band as the mean of Band 1 and Band 2 (useful for the ImageDataGenerator later)
    imgs = np.concatenate([X_band1[:, :, :, np.newaxis]
                            , X_band2[:, :, :, np.newaxis]
                            ,((X_band1+X_band2)/2)[:, :, :, np.newaxis]], axis=-1)
    return imgs

X_train = process_images(train)
X_test = process_images(test)

X_angle_train = np.array(train.inc_angle)
X_angle_test = np.array(test.inc_angle)
y_train = np.array(train["is_iceberg"])


# In[ ]:


# Create a train and validation split, 75% of data used in training
from sklearn.model_selection import train_test_split

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train,
                                    X_angle_train, y_train, random_state=666, train_size=0.75)


# # Create a basic CNN
# Using keras functional API to concatenate the angle input and convolutional model of images

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Reshape, concatenate, Conv2D, Flatten, MaxPooling2D
from keras.layers import BatchNormalization, Dropout, GlobalMaxPooling2D

def simple_cnn():
    pic_input = Input(shape=(75, 75, 3))
    ang_input = Input(shape=(1,))

    cnn = BatchNormalization()(pic_input)
    for i in range(4):
        cnn = Conv2D(8*2**i, kernel_size = (3,3), activation='relu')(cnn)
        cnn = MaxPooling2D((2,2))(cnn)
    cnn = GlobalMaxPooling2D()(cnn)
    cnn = concatenate([cnn,ang_input])
    cnn = Dense(32,activation='relu')(cnn)
    cnn = Dense(1, activation = 'sigmoid')(cnn)

    simple_cnn = Model(inputs=[pic_input,ang_input],outputs=cnn)

    simple_cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return simple_cnn


# # Create ImageDataGenerator
# Create a standard keras ImageDataGenerator, and then use a helper function to return multiple inputs as a list along with the y values necessary to train using fit_generator

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
batch_size=64
# Define the image transformations here
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 40)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=666)
    genX2 = gen.flow(X1,X2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

# Finally create generator
gen_flow = gen_flow_for_two_inputs(X_train, X_angle_train, y_train)


# # Finally fit the model

# In[ ]:


# Create the model
model = simple_cnn()

# Fit the model using our generator defined above
model.fit_generator(gen_flow, validation_data=([X_valid, X_angle_valid], y_valid),
                    steps_per_epoch=len(X_train) / batch_size, epochs=20)


# Create predictions.csv

# In[ ]:


# Predict on test data
test_predictions = model.predict([X_test,X_angle_test])

# Create .csv
pred_df = test[['id']].copy()
pred_df['is_iceberg'] = test_predictions
pred_df.to_csv('predictions.csv', index = False)
pred_df.sample(3)

