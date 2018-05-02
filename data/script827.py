
# coding: utf-8

# ### Building a Convolutional Network on only 2 features:
# Author - Alexandru Papiu
# 
# Convolutional Nets can be thought of as feature extractors - they take in an image that is very high dimensional and return a learned representation of this image that consists of higher level features that help with the learning task at hand. Unfortunately these representations are themselves high dimensional so it's hard to see exactly what is going on. To try to understand things better I'll build a CNN on the MNIST dataset that has a 2-unit layer right before the classification layer. This will force the model to capture as much information as possible in those two features. Let's see how well the 2-D model does!

# ###Loading modules and data:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Input
from keras.optimizers import adam
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


X_train = train.iloc[:,1:].values
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #reshape to rectangular
X_train = X_train/255 #pixel values are 0 - 255 - this makes puts them in the range 0 - 1

y_train = train["label"].values


# In[ ]:


y_ohe = to_categorical(y_train)


# ### Building the Model:

# Let's build the CNN. The architecture is pretty "classic" - a bunch of convolutional layers with relu activations followed by some fully connected layers.  The only thing that's different is the we make the output gradually smaller so that we can see how much signal the model can capture in only 2 dimensions.

# In[ ]:


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation="relu"))
model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 3, 3, activation="relu"))
#model.add(Convolution2D(32, 3, 3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = "relu"))
model.add(Dense(16, activation = "relu"))
model.add(Dense(2))
model.add(Dense(10, activation="softmax"))


# In[ ]:


model.compile(loss='categorical_crossentropy', 
              optimizer = adam(lr=0.001), metrics = ["accuracy"])


# In[ ]:


hist = model.fit(X_train, y_ohe,
          validation_split = 0.05, batch_size = 128, nb_epoch = 7)


# 97% accuracy! That's not bad at all for only 2 dimensions.  Let's all try to visualize projections of the digits onto the 2-d feature space:

# In[ ]:


#getting the 2D output:
output = model.get_layer("dense_3").output
extr = Model(model.input, output)


# ### The Learned 2-D representation:

# In[ ]:



X_proj = extr.predict(X_train[:10000])
X_proj.shape

proj = pd.DataFrame(X_proj[:,:2])
proj.columns = ["comp_1", "comp_2"]
proj["labels"] = y_train[:10000]


matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
sns.lmplot("comp_1", "comp_2",hue = "labels", data = proj, fit_reg=False, size = 8)


# That is really impressive. The CNN has managed to encode these high dimensional images into only 2 dimensions in such a way the classes are separated in a clean symmetric pattern. It would be interesting to dig a little deeper to see exactly what the two axes represent here - at the end of the day they're just mathematical functions from a 784 dimensional space to a 2 dimensional one. However that might not be too illuminating, the transformations are just messy compositions of convolutions, linear combinations and relu's.  What's really striking I think, is how the model starts with some raw inputs and by simply minimizing a loss function builds this beautiful symmetric structure. I guess there's beauty in math after all :)
