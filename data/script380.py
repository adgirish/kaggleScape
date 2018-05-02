
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")


# In[ ]:


# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train_df["is_iceberg"])
print("Xtrain:", X_train.shape)

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
print("Xtest:", X_test.shape)


# # Simple keras model

# In[ ]:


from keras.models import Sequential
from keras.layers import Convolution2D, GlobalAveragePooling2D, Dense, Dropout


# In[ ]:


model = Sequential()
model.add(Convolution2D(32, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(Convolution2D(64, 3, activation="relu", input_shape=(75, 75, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


model.fit(X_train, y_train, validation_split=0.2)


# In[ ]:


# Make predictions
prediction = model.predict(X_test, verbose=1)


# In[ ]:


submit_df = pd.DataFrame({'id': test_df["id"], 'is_iceberg': prediction.flatten()})
submit_df.to_csv("./naive_submission.csv", index=False)


# # Conclusion
# No free lunch! Happy hyperparameter tuning.
# 
