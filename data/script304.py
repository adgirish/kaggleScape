
# coding: utf-8

# # A Tensorflow based Convolutional Neural Network for Iceberg Classification
# 
# In an attempt to improve my understanding of how to best design and use neural networks, I set out to build a convolutional neural network using just tensorflow (as opposed to my usual workflow of using nifty wrappers like Keras to make life easier). I am sharing the code here in the hopes that other people can learn a bit and maybe be tempted into experiment with using tensorflow themselves!
# 
# NOTE: This solution abandons the non-image data in the train and test json files so it is definitely far from an optimal solution. This network is not designed to win the competition, but rather with the idea of me learning more about neural network design through the use of base tensorflow.
# 
# NOTE2: The model scores in the 0.27 range if the n_epochs=250 line is changed to n_epochs=2500. Expanding the number of neurons and adding connected layers also further improves the score.
# 
# NOTE3: The kernel ran over time... so I made the change in NOTE2 on the master so no one else has to worry about that!
# 
# ## Imports
# Relitavely short list of libraries and imports, making use of pandas, numpy, the train/test split function from SciKit-Learn and obviously tensorflow.
# 

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ### Citations
# 
# Here is a list of the resorces that went in to helping me design this convolutional neural network:
# 
# https://www.tensorflow.org/api_docs/python/tf/layers
# 
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
# 
# This example is using TensorFlow layers API
# TensorFlow’s high-level machine learning API (tf.estimator) makes it easy to configure, 
# train, and evaluate a variety of machine learning models:
# 
# https://www.tensorflow.org/get_started/estimator
# 
# I also read this book. It was neat.
# http://shop.oreilly.com/product/0636920052289.do
# 
# ## Load the data, split the training data into a training and validation set
# 
# Before the model is built there is the same old basic housekeeping of loading in the data and splitting off a validation set. As I mentioned above I am discarding everything except for the images and the labels... which is most definitely a loss of useful information. I am also not augmenting the data here (as I want to deal with one thing at a time. see this kernel for how I generated more image instances for training: https://www.kaggle.com/camnugent/expanded-training-set-keras-imagedatagenerator)
# Thank you to Kevin Mader, I have appropriated your input function below.

# In[ ]:


#####
# Load in the data
#####
print('loading data')
# load function from: https://www.kaggle.com/kmader/exploring-the-icebergs-with-skimage-and-keras
# b/c I didn't want to reinvent the wheel
def load_and_format(in_path):
    """ take the input data in .json format and return a df with the data and an np.array for the pictures """
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images


train_df, train_images = load_and_format('../input/train.json')

test_df, test_images = load_and_format('../input/test.json')

X_train, X_valid, y_train, y_valid = train_test_split(train_images,
                                                   train_df['is_iceberg'].as_matrix(),
                                                   test_size = 0.3
                                                   )
print('Train', X_train.shape, y_train.shape)
print('Validation', X_valid.shape, y_valid.shape)


# ## Convert data to float32 
# 
# Tensorflow likes its data to be in float32, if you skip this step and pass in float64 values... it will yell at you. I don't like being yelled at so I avoid this.

# In[ ]:


#convert to np.float32 for use in tensorflow
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
y_valid= y_valid.astype(np.float32)


# ## Define a reset function
# 
# This is here for iterative design purposes. If you define the neural network and don't do exactly how you want, then try to do it again without resetting the graph, then funny things can happen as tensorflow will try to patch the new onto the old. We must therefore always throw away the old!

# In[ ]:


#for stability
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()


# ## Set necessary paramaters/hyperparamaters
# 
# For this model I'm using a slow learning rate (0.005) and a high number of epochs (2500).
# 
# The input to the network is the length * width of the image (# of pixels).
# The dropout is used to prevent overfitting by randomly dropping components of neural network. 
# 

# In[ ]:


print('designing model')
# Training Parameters
learning_rate = 0.005
n_epochs = 2500 # changed to 2500 for a LB score of ~2.69


# Network Parameters
num_input = 75*75 #size of the images
num_classes = 2 # Binary
dropout = 0.4 # Dropout, probability to keep units


# ## Design the convolutional neural network
# 
# Here we get to the design of the network, first set is to design the graph in tensorflow. The variables X and y below are placeholders for the actual data we will pass in to the network. Note the shape of X is (None, 75, 75, 2). The None is so that the # of rows is flexiable, the 75,75 is the pixel dimensions of the image and the 2 is because there are two channels of image data being passed in. Note y has shape=(None) because it will be a 1-D vector with one input for each row. If we had multiple classes this could be changed to shape=(None, 5) (for 5 classes).
# 
# I use the tensorflow layers API because it is easier to understand the makeup of the network and also easier to design the network.
# 
# The network used here uses an initial set of convolutional layers, followed by a pooling step and several additional fully connected layers. The second to last layer applies dropout, which we defined as 0.3. Throughout the network the rectified linear unit (ReLU) activation function(https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is used, along with an He Kernel initializer (https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf).
# The Sigmoid activation function is important for the final layer as this lets us get meaningful probabilities returned from the network.

# In[ ]:


X = tf.placeholder(tf.float32, shape=(None, 75, 75, 2), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


with tf.variable_scope('ConvNet'):

    he_init = tf.contrib.layers.variance_scaling_initializer()

    # Convolution Layer with 32 filters and a kernel size of 5
    conv1 = tf.layers.conv2d(X, filters=32,  kernel_size=[5, 5], activation=tf.nn.relu)
    # Max Pooling 
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(pool1, filters=64,  kernel_size=[3,3], activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3,3], activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(pool3, filters=256, kernel_size=[3,3], activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2)
    
    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(pool4)

    # Fully connected layer 
    fc2 = tf.layers.dense(fc1, 32, 
                        kernel_initializer=he_init, activation=tf.nn.relu)

    # Apply Dropout 
    fc3 = tf.layers.dropout(fc2, rate=dropout)

    logits = tf.layers.dense(fc3, num_classes, activation=tf.nn.sigmoid)


# ## Define the loss function
# 
# With the network defined we next define the loss function which compares the predicted values to the actual values of the training set. The sparse_softmax_cross_entropy_with_logits function used here computes the sparse softmax cross entropy between logits and labels and the reduce_mean function is then used to compute the mean of the tensor.
# 

# In[ ]:


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


# ## Define the training method
# 
# Gradient descent is defined as the training method used to minimize the loss function

# In[ ]:


with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# ## Define the evalutation method
# 
# This explains the evaluation method better then I can, so have a look if you're curious about how tf.nn.in_top_k() works!
# 
# https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/evaluation

# In[ ]:


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# Next we initialize the network.
# I've commented out the saver portion, I use this when running the network locally to maintain a copy of the model after training (so we don't have to start from scratch each time). The saver, and corresponding use lines below are turned off because there is no need to save the model to memory when I run on Kaggle (you can turn them on though).

# In[ ]:


init = tf.global_variables_initializer()
#saver = tf.train.Saver()


# ## Train the model
# 
# Recall the number of epochs was defined above as 2500, so the model will be trained on the entire training set for 2500 iterations. Here I have it print the training and testing accuracy after each epoch.
# 
# Here we initiate the model and for each epoch we use sess.run() to pass the data into the model and train the network. Predictions for the train and validation data are then made and the accuracy is assessed and printed to the screen.

# In[ ]:


print('training model\n')
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})   
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        acc_test = accuracy.eval(feed_dict={X: X_valid,
                                            y: y_valid})
    
        print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_test)
    save_path = saver.save(sess, "./cam_iceberg_model_final.ckpt")


# ## Prepare the test data
# 
# As we did with the training and validation data, before making predictions I convert the type of the test data to float32.
# 

# In[ ]:


#convert the test images to float32
test_images =test_images.astype(np.float32) 
test_images.shape


# ## Make predictions
# 
# The last line y_pred = Z[:,1] selects the second column of the predictions because we want 'probability of iceberg' not 'probability of not iceberg' which would be column 0.

# In[ ]:



print('making predictions\n')
#make external predictions on the test_dat
with tf.Session() as sess:
    saver.restore(sess, "./cam_iceberg_model_final.ckpt") # or better, use save_path
    Z = logits.eval(feed_dict={X: test_images}) #outputs switched to logits
    y_pred = Z[:,1]



# ## Write output to file
# 
# Lastly we take the predictions and construct a dataframe which we output to a .csv and can then submit for evalutation!

# In[ ]:


output = pd.DataFrame(test_df['id'])
output['is_iceberg'] = y_pred

output.to_csv('cam_tf_cnn.csv', index=False)

