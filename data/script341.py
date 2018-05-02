
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
base_path = os.path.join('..', 'input')


# # Concatenate and Reshape
# Here we load the data and then combine the two bands and recombine them into a single image/tensor for training

# In[ ]:


def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images
train_df, train_images = load_and_format(os.path.join(base_path, 'train.json'))
print('training', train_df.shape, 'loaded', train_images.shape)
test_df, test_images = load_and_format(os.path.join(base_path, 'test.json'))
print('testing', test_df.shape, 'loaded', test_images.shape)
train_df.sample(3)


# # Show single examples

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
ax1.matshow(train_images[0,:,:,0])
ax1.set_title('Band 1')
ax2.matshow(train_images[0,:,:,1])
ax2.set_title('Band 2')


# # Training Overview
# Here we use the montage functionality of skimage to make preview tiles of randomly selected icebergs and ships. This helps us get a better idea about the diversity in the data.

# In[ ]:


fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
obj_list = dict(ships = train_df.query('is_iceberg==0').sample(16).index,
     icebergs = train_df.query('is_iceberg==1').sample(16).index)
for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
    ax1.set_title('%s Band 1' % obj_type)
    ax1.axis('off')
    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
    ax2.set_title('%s Band 2' % obj_type)
    ax2.axis('off')


# # Testing Data Overview
# To see how different the test data looks from the training and it looks like the data is much messier (multiple objects, different skews and angles). Clearly we will require some augmentation to do well here

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,12))
idx_list = test_df.sample(49).index
obj_type = 'Test Data'
ax1.imshow(montage2d(test_images[idx_list,:,:,0]))
ax1.set_title('%s Band 1' % obj_type)
ax1.axis('off')
ax2.imshow(montage2d(test_images[idx_list,:,:,1]))
ax2.set_title('%s Band 2' % obj_type)
ax2.axis('off')


# # Simple CNN
# While the competition explicitly mentioned these are not like 'normal' images and the values have meaning, I am going to be lazy and treat them like 'normal' images and dump them into a magical CNN and hope good things pop out.

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
X_train, X_test, y_train, y_test = train_test_split(train_images,
                                                   to_categorical(train_df['is_iceberg']),
                                                    random_state = 2017,
                                                    test_size = 0.5
                                                   )
print('Train', X_train.shape, y_train.shape)
print('Validation', X_test.shape, y_test.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, GlobalMaxPooling2D, Dense
simple_cnn = Sequential()
simple_cnn.add(BatchNormalization(input_shape = (75, 75, 2)))
for i in range(4):
    simple_cnn.add(Conv2D(8*2**i, kernel_size = (3,3)))
    simple_cnn.add(MaxPooling2D((2,2)))
simple_cnn.add(GlobalMaxPooling2D())
simple_cnn.add(Dropout(0.5))
simple_cnn.add(Dense(8))
simple_cnn.add(Dense(2, activation = 'softmax'))
simple_cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
simple_cnn.summary()


# In[ ]:


simple_cnn.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, shuffle = True)


# # Make Predictions
# Here we make predictions on the output and export the CSV so we can submit

# In[ ]:


test_predictions = simple_cnn.predict(test_images)


# In[ ]:


pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
pred_df.to_csv('predictions.csv', index = False)
pred_df.sample(3)

