
# coding: utf-8

# This is inspired by [Ceshine Lee](https://www.kaggle.com/ceshine/lgbm-starter?scriptVersionId=1852107) and [LingZhi's](https://www.kaggle.com/vrtjso/lgbm-one-step-ahead?scriptVersionId=1965435) LGBM kernel. 
# 
# This kernel tackles the problem using a 2-layer dense neural network that looks something like this:
# ![](https://www.pyimagesearch.com/wp-content/uploads/2016/08/simple_neural_network_header.jpg)
# 
# Technically, Tensorflow is used to build this neural network. Before feeding the data into the second layer, batch normalization is used for faster learning(quicker convergent in gradient descent) and Dropout layer is used for regularisation to prevent overfitting.  Instead of a constant learning rate, I have used AdamOptimizer that decays the learning rate over time so that the whole training of network takes much lesser time in my experiment.
# 
# I'm sorry that the naming conventions is a little confusing but feel free to ask questions!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error
import gc
import tensorflow as tf


# In[ ]:


df_train_X = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/x_train.csv')
df_train_Y = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/y_train.csv')
df_test_X = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/x_test.csv')
df_test_Y = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/y_test.csv')


# In[ ]:


itemsDF = pd.read_csv('../input/fork-of-lgbm-one-step-ahead-xgb/items_reindex.csv')


# In[ ]:


def NWRMSLE(y, pred, w):
    return mean_squared_error(y, pred, sample_weight=w)**0.5


# In[ ]:


df_train_X.drop(['Unnamed: 0'], inplace=True,axis=1)
df_test_X.drop(['Unnamed: 0'], inplace=True,axis=1)
df_train_Y.drop(['Unnamed: 0'], inplace=True,axis=1)
df_test_Y.drop(['Unnamed: 0'], inplace=True,axis=1)


# This is the start of building the computation graph of TensorFlow NN model.
# 
# Let's declare some constant values for our TF NN model.

# In[ ]:


numFeatures = df_train_X.shape[1]
numLabels = 1
hiddenUnit = 20
learningRate = 0.01
numEpochs = 1000


# Declare the placeholders for the input(x) and output(y_) layer.

# In[ ]:


x = tf.placeholder(tf.float64, [None, numFeatures],name="X_placeholder")
y_ = tf.placeholder(tf.float64, [None, numLabels],name="Y_placeholder")


# Declare the first and second hidden layer by initializing the weights to a range of random normally distributed values.

# In[ ]:


weights = tf.Variable(tf.random_normal([numFeatures,hiddenUnit],stddev=0.1,name="weights", dtype=tf.float64))
weights2 = tf.Variable(tf.random_normal([hiddenUnit,1],name="weights2", dtype=tf.float64))


# Declare the bias that will be multiplied together with the weights later. Similarly,  we'll initializing the bias to a range of random normally distributed values.

# In[ ]:


bias = tf.Variable(tf.random_normal([1,hiddenUnit],stddev=0.1,name="bias", dtype=tf.float64))
bias2 = tf.Variable(tf.random_normal([1,1],stddev=0.1,name="bias2", dtype=tf.float64))


# We'll define a placeholder for inputting the "perishable" feature which is used to compute the weighted loss

# In[ ]:


weightsNWR = tf.placeholder(tf.float32, [None, 1],name="weightsNWR")


# Take this chance to populate the weight variables which will be used to pass to the placeholder during the training phase.

# In[ ]:


itemWeightsTrain = pd.concat([itemsDF["perishable"]] * 6) * 0.25 + 1
itemWeightsTrain = np.reshape(itemWeightsTrain,(itemWeightsTrain.shape[0], 1))


# In[ ]:


itemWeightsTest = itemsDF["perishable"]* 0.25 + 1
itemWeightsTest = np.reshape(itemWeightsTest,(itemWeightsTest.shape[0], 1))


# First hidden layer is composed of multiplication of input, weights and the bias we have declared above

# In[ ]:


y = tf.matmul(x,weights) + bias


# We'll pass the results of the first layer to a relu activation function to convert the linear values into a non-linear one.

# In[ ]:


y = tf.nn.relu(y)


# Next, we'll set up a batch normalization function that normalize the values that comes from out the relu function.

# Normalization can improve learning speed because the path to the global minimum is reduced:
# ![](http://cs231n.github.io/assets/nn2/prepro1.jpeg)

# Although many literatures say that batch norm is applied **before** activation function, I believe that it would be more beneficial if batch normalization is applied **after** the activation function so that the range of linear values will not be restricted to a down-sized range.

# In[ ]:


epsilon = 1e-3
batch_mean2, batch_var2 = tf.nn.moments(y,[0])
scale2 = tf.Variable(tf.ones([hiddenUnit],dtype=tf.float64),dtype=tf.float64)
beta2 = tf.Variable(tf.zeros([hiddenUnit],dtype=tf.float64),dtype=tf.float64)
y = tf.nn.batch_normalization(y,batch_mean2,batch_var2,beta2,scale2,epsilon)


# We set up a dropout layer to intentionally deactivate certain units. This will improve generalization and reduce overfitting(better validation set score) because it force your layer to learn with different neurons the same "concept".
# 
# ![](http://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_5/dropout.jpeg)
# 
# Note that during the prediction phase, the dropout is deactivated.

# In[ ]:


dropout_placeholder = tf.placeholder(tf.float64,name="dropout_placeholder")
y=tf.nn.dropout(y,dropout_placeholder)


# Next we'll build the second hidden layer. As usual, it's the multiplication of input, weights and the bias we have declared above

# In[ ]:


#create 1 more hidden layer
y = tf.matmul(y,weights2)+bias2


# Pass the results to another relu activation function

# In[ ]:


y = tf.nn.relu(y)


# The loss function that are trying to optimize, or the goal of training, is to minimize the weighted mean squared error. 
# 
# Perishable items are given a weight of 1.25 where all other items are given a weight of 1.00, as described in the competition details. 
# 

# In[ ]:


loss = tf.losses.mean_squared_error(predictions=y,labels=y_,weights=weightsNWR)
cost = tf.reduce_mean(loss)


# As stated above, I have found AdamOptimizer, which decays the learning rate over time to be better than the GradientOptimizer option in terms of training speed. Beside that, AdamOptimizer also dampens the oscillations in the direction that do not point to the minimal so that the back-and-forth between these walls will be reduced and at the same time, we'll build up momentum in the direction of the minimum.

# In[ ]:


optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)


# Finally, we'll create a TF session for training our model.

# In[ ]:


sess = tf.Session()


# Initilize the variables that we have been setting up

# In[ ]:


sess.run(tf.global_variables_initializer())


# Finally, it's time to train our NN model! We are actually training 16 NN model (1 for each column of y values). There are 16 columns in Y train and Y test, which represents prediction for 16 days.
# 
# We are training for 1000 epoch. At every 100 epoch, we do an output in terms of weighted mse to see if it overfits for test set. After 1000 epoch is done, we'll use the trained model for prediction by feeding the X submission data.
# 
# Note that Dropout rate is set at 0.6(deactivate 40% of units) during training and back at 1.0(no deactivation) during prediction. Check these values when you are building your own drop out layers to ensure that you are not throwing away results during prediction.
# 
# Also, training will be longer than Kaggle's allowable timeout limit so run this at your own local machine.

# In[ ]:


val_pred_nn = []
test_pred_nn = []
cate_vars_nn = []
submit_pred_nn=[]

trainingLoss=[]
validationLoss=[]


#step through all the dates(16)
for i in range(16):
    print("Step %d" % (i+1))
    
    trainY_NN = np.reshape(df_train_Y.iloc[:,i],(df_train_Y.shape[0], 1))
    testY_NN = np.reshape(df_test_Y.iloc[:,i],(df_test_Y.shape[0], 1))
    
    for epoch in range(numEpochs):
        _,loss = sess.run([optimizer,cost], feed_dict={x: df_train_X, y_: trainY_NN,weightsNWR:itemWeightsTrain,dropout_placeholder:0.6})

        if epoch%100 == 0:
            print('Epoch', epoch, 'completed out of',numEpochs,'loss:',loss)
            #trainingLoss.append(loss)
            #check against test dataset
            test_pred = sess.run(cost, feed_dict={x:df_test_X,y_: testY_NN,weightsNWR:itemWeightsTest,dropout_placeholder:1.0})
            print('Acc for test dataset ',test_pred)
            #validationLoss.append(test_pred)
    
    tf_pred = sess.run(y,feed_dict={x:df_test_X,weightsNWR:itemWeightsTest,dropout_placeholder:1.0})
    tf_predY = np.reshape(tf_pred,(tf_pred.shape[0],))
    test_pred_nn.append(tf_predY)
    print('score for step',(i+1))
    print("Validation mse:", mean_squared_error(df_test_Y.iloc[:,i], tf_predY))
    print('NWRMSLE:',NWRMSLE(df_test_Y.iloc[:,i], tf_predY,itemsDF["perishable"]*0.25+1))

    #predict for submission set
    nn_submit_predY = sess.run(y,feed_dict={x:df_Submission_X,dropout_placeholder:1.0})
    nn_submit_predY = np.reshape(nn_submit_predY,(nn_submit_predY.shape[0],))
    submit_pred_nn.append(nn_submit_predY)
    
    gc.collect()
    sess.run(tf.global_variables_initializer())


# In[ ]:


nnTrainY= np.array(test_pred_nn).transpose()
pd.DataFrame(nnTrainY).to_csv('nnTrainY.csv')
nnSubmitY= np.array(submit_pred_nn).transpose()
pd.DataFrame(nnSubmitY).to_csv('nnSubmitY.csv')


# You can use the below NWRMSLE to compare test set score with other benchmarks, or finding out the optimal weights for your ensemble.

# In[ ]:


print('NWRMSLE:',NWRMSLE(df_test_Y,nnTrainY,itemsDF["perishable"]* 0.25 + 1))


# With the prediction values from the NN model, prepare for submission. The following cells are pretty self-explantory.

# In[ ]:


#to reproduce the testing IDs
df_train = pd.read_csv(
    '../input/favorita-grocery-sales-forecasting/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
del df_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)


# In[ ]:


#submitDF = pd.read_csv('../input/testforsubmit/testForSubmit.csv',index_col=False)
df_test = pd.read_csv(
    "../input/favorita-grocery-sales-forecasting/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)


# In[ ]:


print("Making submission...")
df_preds = pd.DataFrame(
    combinedSubmitPredY, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)


# In[ ]:


submission = df_test[["id"]].join(df_preds, how="left").fillna(0)


# In[ ]:


submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)


# In[ ]:


submission[['id','unit_sales']].to_csv('submit_nn.csv',index=None)


# **TODO/Areas to improve/To be Updated:**
# * Normalize X data before inputting into input layer.
# * Use Tensorboard to graphically visualize the model to see if there's any bottlenecks or areas that could improve the robustness of the model.
