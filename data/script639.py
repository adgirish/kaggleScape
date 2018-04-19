
# coding: utf-8

# ## Introduction ##
# The Kaggle Leaderboard is a special kind of place. It does not resemble a real-life situation, but you are based on only one metric: the Log Loss. This means that where we would normally care about for example speed or the size of our models, we don't now. If you have the right models, the tricks in this notebook will help you improve your score a little bit more (hopefully).

# ## Log Loss ##
# There is a really good notebook going more into detail on log loss. You can find it [here][1]. The thing we have to take away from this notebook is that when we're more certain about a class and we're wrong, we're punished harder than linear. Just like the Dutch tax laws.
# 
# 
#   [1]: https://www.kaggle.com/grfiv4/log-loss-depicted-1

# ## Trick 1: Clipping ##
# Clipping is a simple operation on our predictions where we set a maximum and a minimum certainty. This avoids really hard punishment in case we're wrong. This means while your model gets better, the less clipping will help you improve your score. For example in the earlier stages of this competition, a clip of 0.90 improved our score from 0.94380 to 0.91815.

# In[ ]:


import pandas as pd
import os

clip = 0.90
classes = 8

def clip_csv(csv_file, clip, classes):
    # Read the submission file
    df = pd.read_csv(csv_file, index_col=0)

    # Clip the values
    df = df.clip(lower=(1.0 - clip)/float(classes - 1), upper=clip)
    
    # Normalize the values to 1
    df = df.div(df.sum(axis=1), axis=0)

    # Save the new clipped values
    df.to_csv('clip.csv')
    print(df.head(10))
    
# Of course you are going to use your own submission here
clip_csv('../input/sample_submission_stg1.csv', clip, classes)


# ## Trick 2: Blending ##
# When you have different models or different model configurations, then it could be that some models are experts at recognizing all kinds of tuna, while others are better at distinguishing fish vs no fish. Good specialist models are only very certain in their own area. In this case it helps to let them work together to a solution. A way of combining the outputs of multiple models or model settings is blending. It's a very simple procedure where all predictions are added to each other for each image, class pair and then divided by the number of models.
# 
# Blending can for example be used for test augmentation: all test image are augmented with several operations (flipping, rotation, zooming etc.). When you augment each image a couple of times, you can use them as separate submission files, which can be combined using blending afterwards. This approach improved our score from 0.89893 to 0.86401 for this competition.
# 
# An other simple alternative for blending is majority voting, where every model is allowed to make one prediction per image.

# In[ ]:


import numpy as np
import pandas as pd

def blend_csv(csv_paths):
    if len(csv_paths) < 2:
        print("Blending takes two or more csv files!")
        return
    
    # Read the first file
    df_blend = pd.read_csv(csv_paths[0], index_col=0)
    
    # Loop over all files and add them
    for csv_file in csv_paths[1:]:
        df = pd.read_csv(csv_file, index_col=0)
        df_blend = df_blend.add(df)
        
    # Divide by the number of files
    df_blend = df_blend.div(len(csv_paths))

    # Save the blend file
    df_blend.to_csv('blend.csv')
    print(df_blend.head(10))

# Obviously replace this with two or more of your files
blend_csv(['../input/sample_submission_stg1.csv', '../input/sample_submission_stg1.csv'])


# # Trick 3: Pseudo-Labeling
# This is an technique where test images are used during training. The labels are provide by the predictions of the model. One of the first thoughts that come to mind is: *"WhaAT the HECK!?? How can this even work!?"*.  There is a theorem why this could work. Until we know if it's true, applying Pseudo-Labeling is it's at least worth trying.
# 
# Quick summary of the explanation given in the paper below: In semi-supervised learning the goal is to make a clear separation between the classes. The decision boundary should be in low-density regions. This way the model generalizes better. The network should use similar activations for the same class.
# 
# When you want to read more, [here][1] is a paper on the subject.
# 
#   [1]: http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf

# ## Keras example
# Below is an example of Pseudo-Labling using Keras. The code is cherry-picked from the code from a lesson from fast.ai. The source files are [this][1] and [this][2] one.
# 
# You can play with the amount of test data you add to the batches. In the example below we choose to take about one third of non-training data, being 1/16th validation data and 1/4th test data. 
# 
#   [1]: https://github.com/fastai/courses/blob/master/deeplearning1/nbs/utils.py
#   [2]: https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson7.ipynb

# In[ ]:


# Create a MixIterator object
# This class is a simple method to create batches from several other batch generators
class MixIterator(object):
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list
        if self.multi:
            self.N = sum([it[0].N for it in self.iters])
        else:
            self.N = sum([it.N for it in self.iters])

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
        if self.multi:
            nexts = [[next(it) for it in o] for o in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
            return (n0, n1)
        else:
            nexts = [next(it) for it in self.iters]
            n0 = np.concatenate([n[0] for n in nexts])
            n1 = np.concatenate([n[1] for n in nexts])
        return (n0, n1)


# In[ ]:


# Example usage in Keras
# [replace by your own code]
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
validation_data=(x_test, y_test))

batch_size = 8
# [/replace by your own code]

predictions = model.predict(x_test, batch_size=batch_size)

gen = ImageDataGenerator()

train_batches = gen.flow(x_train, y_train, batch_size=44)
val_batches = gen.flow(x_val, y_val, batch_size=4)
test_batches = gen.flow(x_test, predictions, batch_size=16)

mi = MixIterator([train_batches, test_batches, val_batches])
model.fit_generator(mi, mi.N, nb_epoch=8, validation_data=(x_val, y_val))

