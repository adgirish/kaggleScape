
# coding: utf-8

# * * ****<font color='green'>
# **Welcome to Deep Learning Tutorial for Beginners** 
# * I am going to explain <u>every thing</u> one by one.
# * Instead of writing long and hard for reading paragraphs, I define and emphasize keywords line by line.
# * At the end of this tutorial, you will have enough information about deep learning to go deeper inside it.
# * Lets look at content.
# 
# <font color='red'>
# <br>Content:
# * [Introduction](#1)
# * [Overview the Data Set](#2)
# * [Logistic Regression](#3)
#     * [Computation Graph](#4)
#     * [Initializing parameters](#5)
#     * [Forward Propagation](#6)
#         * Sigmoid Function
#         * Loss(error) Function
#         * Cost Function
#     * [Optimization Algorithm with Gradient Descent](#7)
#         * Backward Propagation
#         * Updating parameters
#     * [Logistic Regression with Sklearn](#8)
#     * [Summary and Questions in Minds](#9)
#     
# * [Artificial Neural Network](#10)
#     * [2-Layer Neural Network](#11)
#         * [Size of layers and initializing parameters weights and bias](#12)
#         * [Forward propagation](#13)
#         * [Loss function and Cost function](#14)
#         * [Backward propagation](#15)
#         * [Update Parameters](#16)
#         * [Prediction with learnt parameters weight and bias](#17)
#         * [Create Model](#18)
#     * [L-Layer Neural Network](#19)
#         * [Implementing with keras library](#22)
# * Time Series Prediction: https://www.kaggle.com/kanncaa1/time-series-prediction-with-eda-of-world-war-2
# * [Conclusion](#20)
# 
# 

# <a id="1"></a> <br>
# # INTRODUCTION
# * **Deep learning:** One of the machine learning technique that learns features directly from data. 
# * **Why deep learning:** When the amounth of data is increased, machine learning techniques are insufficient in terms of performance and deep learning gives better performance like accuracy.
# <a href="http://ibb.co/m2bxcc"><img src="http://preview.ibb.co/d3CEOH/1.png" alt="1" border="0"></a>
# * **What is amounth of big:** It is hard to answer but intuitively 1 million sample is enough to say "big amounth of data"
# * **Usage fields of deep learning:** Speech recognition, image classification, natural language procession (nlp) or recommendation systems
# * **What is difference of deep learning from machine learning:** 
#     * Machine learning covers deep learning. 
#     * Features are given machine learning manually.
#     * On the other hand, deep learning learns features directly from data.
# <a href="http://ibb.co/f8Epqx"><img src="http://preview.ibb.co/hgpNAx/2.png" alt="2" border="0"></a>
# 
# <br>Lets look at our data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# <a id="Overview the Data Set"></a> <br>
# # Overview the Data Set
# * We will use "sign language digits data set" for this tutorial.
# * In this data there are 2062 sign language digits images.
# * As you know digits are from 0 to 9. Therefore there are 10 unique sign.
# * At the beginning of tutorial we will use only sign 0 and 1 for simplicity. 
# * In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.
# * Also sign one is between indexes 822 and 1027. Number of one sign is 206. Therefore, we will use 205 samples from each classes(labels).
# * Note: Actually 205 sample is very very very little for deep learning. But this is tutorial so it does not matter so much. 
# * Lets prepare our X and Y arrays. X is image array (zero and one signs) and Y is label array (0 and 1).

# In[ ]:


# load data set
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')


# * In order to create image array, I concatenate zero sign and one sign arrays
# * Then I create label array 0 for zero sign images and 1 for one sign images.

# In[ ]:


# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)


# * The shape of the X is (410, 64, 64)
#     * 410 means that we have 410 images (zero and one signs)
#     * 64 means that our image size is 64x64 (64x64 pixels)
# * The shape of the Y is (410,1)
#     *  410 means that we have 410 labels (0 and 1) 
# * Lets split X and Y into train and test sets.
#     * test_size = percentage of test size. test = 15% and train = 75%
#     * random_state = use same seed while randomizing. It means that if we call train_test_split repeatedly, it always creates same train and test distribution because we have same random_state.

# In[ ]:


# Then lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]


# * Now we have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use as input for our first deep learning model.
# * Our label array (Y) is already flatten(2D) so we leave it like that.
# * Lets flatten X array(images array).
# 

# In[ ]:


X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)


# * As you can see, we have 348 images and each image has 4096 pixels in image train array.
# * Also, we have 62 images and each image has 4096 pixels in image test array.
# * Then lets take transpose. You can say that WHYY, actually there is no technical answer. I just write the code(code that you will see oncoming parts) according to it :)

# In[ ]:


x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# <font color='purple'>
# What we did up to this point:
# * Choose our labels (classes) that are sign zero and sign one
# * Create and flatten train and test sets
# * Our final inputs(images) and outputs(labels or classes) looks like this:
# <a href="http://ibb.co/bWMK7c"><img src="http://image.ibb.co/fOqCSc/3.png" alt="3" border="0"></a>

# <a id="3"></a> <br>
# # Logistic Regression
# * When we talk about binary classification( 0 and 1 outputs) what comes to mind first is logistic regression.
# * However, in deep learning tutorial what to do with logistic regression there??
# * The answer is that  logistic regression is actually a very simple neural network. 
# * By the way neural network and deep learning are same thing. When we will come artificial neural network, I will explain detailed the terms like "deep".
# * In order to understand logistic regression (simple deep learning) lets first learn computation graph.

# <a id="4"></a> <br>
# ##  Computation Graph
# * Computation graphs are a nice way to think about mathematical expressions.
# * It is like visualization of  mathematical expressions.
# * For example we have $$c = \sqrt{a^2 + b^2}$$
# * It's computational graph is this. As you can see we express math with graph.
# <a href="http://imgbb.com/"><img src="http://image.ibb.co/hWn6Lx/d.jpg" alt="d" border="0"></a>

# * Now lets look at computation graph of logistic regression
# <a href="http://ibb.co/c574qx"><img src="http://preview.ibb.co/cxP63H/5.jpg" alt="5" border="0"></a>
#     * Parameters are weight and bias.
#     * Weights: coefficients of each pixels
#     * Bias: intercept
#     * z = (w.t)x + b  => z equals to (transpose of weights times input x) + bias 
#     * In an other saying => z = b + px1*w1 + px2*w2 + ... + px4096*w4096
#     * y_head = sigmoid(z)
#     * Sigmoid function makes z between zero and one so that is probability. You can see sigmoid function in computation graph.
# * Why we use sigmoid function?
#     * It gives probabilistic result
#     * It is derivative so we can use it in gradient descent algorithm (we will see as soon.)
# * Lets make example:
#     * Lets say we find z = 4 and put z into sigmoid function. The result(y_head) is almost 0.9. It means that our classification result is 1 with 90% probability.
# * Now lets start with from beginning and examine each component of computation graph more detailed.

# <a id="5"></a> <br>
# ## Initializing parameters
# * As you know input is our images that has 4096 pixels(each image in x_train).
# * Each pixels have own weights.
# * The first step is multiplying each pixels with their own weights.
# * The question is that what is the initial value of weights?
#     * There are some techniques that I will explain at artificial neural network but for this time initial weights are 0.01.
#     * Okey, weights are 0.01 but what is the weight array shape? As you understand from computation graph of logistic regression, it is (4096,1)
#     * Also initial bias is 0.
# * Lets write some code. In order to use at coming topics like artificial neural network (ANN), I make definition(method).

# In[ ]:


# short description and example of definition (def)
def dummy(parameter):
    dummy_parameter = parameter + 5
    return dummy_parameter
result = dummy(3)     # result = 8

# lets initialize parameters
# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b


# <a id="6"></a> <br>
# ## Forward Propagation
# * The all steps from pixels to cost is called forward propagation
#     * z = (w.T)x + b => in this equation we know x that is pixel array, we know w (weights) and b (bias) so the rest is calculation. (T is transpose)
#     * Then we put z into sigmoid function that returns y_head(probability). When your mind is confused go and look at computation graph. Also equation of sigmoid function is in computation graph.
#     * Then we calculate loss(error) function. 
#     * Cost function is summation of all loss(error).
#     * Lets start with z and the write sigmoid definition(method) that takes z as input parameter and returns y_head(probability)

# In[ ]:


# calculation of z
#z = np.dot(w.T,x_train)+b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
#y_head = sigmoid(z)


# * As we write sigmoid method and calculate y_head. Lets learn what is loss(error) function
# * Lets make example, I put one image as input then multiply it with their weights and add bias term so I find z. Then put z into sigmoid method so I find y_head. Up to this point we know what we did. Then e.g y_head became 0.9 that is bigger than 0.5 so our prediction is image is sign one image. Okey every thing looks like fine. But, is our prediction is correct and how do we check whether it is correct or not? The answer is with loss(error) function:
#     * Mathematical expression of log loss(error) function is that: 
#     <a href="http://imgbb.com/"><img src="http://image.ibb.co/dFCR3H/6.jpg" alt="6" border="0"></a>
#     * It says that if you make wrong prediction, loss(error) becomes big.
#         * Example: our real image is sign one and its label is 1 (y = 1), then we make prediction y_head = 1. When we put y and y_head into loss(error) equation the result is 0. We make correct prediction therefore our loss is 0. However, if we make wrong prediction like y_head = 0, loss(error) is infinity.
# * After that, the cost function is summation of loss function. Each image creates loss function. Cost function is summation of loss functions that is created by each input image.
# * Lets implement forward propagation.
# 

# In[ ]:


# Forward propagation steps:
# find z = w.T*x+b
# y_head = sigmoid(z)
# loss(error) = loss(y,y_head)
# cost = sum(loss)
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    return cost 


# <a id="7"></a> <br>
# ##  Optimization Algorithm with Gradient Descent
# * Well, now we know what is our cost that is error.
# * Therefore, we need to decrease cost because as we know if cost is high it means that we make wrong prediction.
# * Lets think first step, every thing starts with initializing weights and bias. Therefore cost is dependent with them.
# * In order to decrease cost, we need to update weights and bias.
# * In other words, our model needs to learn the parameters weights and bias that minimize cost function. This technique is called gradient descent.
# * Lets make an example:
#     * We have w = 5 and bias = 0 (so ignore bias for now). Then we make forward propagation and our cost function is 1.5.
#     * It looks like this. (red lines)
#     <a href="http://imgbb.com/"><img src="http://image.ibb.co/dAaYJH/7.jpg" alt="7" border="0"></a>
#     * As you can see from graph, we are not at minimum point of cost function. Therefore we need to go through minimum cost. Okey, lets update weight. ( the symbol := is updating)
#     * w := w - step. The question is what is this step? Step is slope1. Okey, it looks remarkable. In order to find minimum point, we can use slope1. Then lets say slope1 = 3 and update our weight. w := w - slope1 => w = 2.
#     * Now, our weight w is 2. As you remember, we need to find cost function with forward propagation again. 
#     * Lets say according to forward propagation with w = 2, cost function is 0.4. Hmm, we are at right way because our cost function is decrease. We have new value for cost function that is cost = 0.4. Is that enough? Actually I do not know lets try one more step.
#     * Slope2 = 0.7 and w = 2. Lets update weight w : = w - step(slope2) => w = 1.3 that is new weight. So lets find new cost.
#     * Make one more forward propagation with w = 1.3 and our cost = 0.3. Okey, our cost even decreased, it looks like fine but is it enough or do we need to make one more step? The answer is again I do not know, lets try.
#     * Slope3 = 0.01 and w = 1.3. Updating weight w := w - step(slope3) => w = 1.29 ~ 1.3. So weight does not change because we find minimum point of cost function. 
#     * Everything looks like good but how we find slope? If you remember from high school or university, in order to find slope of function(cost function) at given point(at given weight) we take derivative of function at given point. Also you can ask that okey well we find slope but how it knows where it go. You can say that it can go more higher cost values instead of going minimum point. The asnwer is that slope(derivative) gives both step and direction of step. Therefore do not worry :)
#     * Update equation is this. It says that there is a cost function(takes weight and bias). Take derivative of cost function according to weight and bias. Then multiply it with  α learning rate. Then update weight. (In order to explain I ignore bias but these all steps will be applied for bias)
#     <a href="http://imgbb.com/"><img src="http://image.ibb.co/hYTTJH/8.jpg" alt="8" border="0"></a>
#     * Now, I am sure you are asking what is learning rate that I mentioned never. It is very simple term that determines learning rate. Hovewer there is tradeoff between learning fast and never learning. For example you are at Paris(current cost) and want to go Madrid(minimum cost). If your speed(learning rate) is small, you can go Madrid very slowly and it takes too long time. On ther other hand, if your speed(learning rate) is big, you can go very fast but maybe you make crash and never go to Madrid. Therefore, we need to choose wisely our speed(learning rate).
#     * Learning rate is also called hyperparameter that need to be chosen and tuned. I will explain it more detailed in artificial neural network with other hyperparameters. For now just say learning rate is 1 for our previous example.
#   
# * I think now you understand the logic behind forward propagation(from weights and bias to cost) and backward propagation(from cost to weights and bias to update them). Also you learn gradient descent. Before implementing the code you need to learn one more thing that is how we take derivative of cost function according to weights and bias. It is not related with python or coding. It is pure mathematic. There are two option first one is to google how to take derivative of log loss function and second one is even to google what is derivative of log loss function :) I choose second one because I cannot explain math without talking :) 
# $$ \frac{\partial J}{\partial w} = \frac{1}{m}x(  y_head - y)^T$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (y_head-y)$$

# In[ ]:


# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients


# * Up to this point we learn 
#     * Initializing parameters (implemented)
#     * Finding cost with forward propagation and cost function (implemented)
#     * Updating(learning) parameters (weight and bias). Now lets implement it.

# In[ ]:


# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)


# * Woow, I get tired :) Up to this point we learn our parameters. It means we fit the data. 
# * In order to predict we have parameters. Therefore, lets predict.
# * In prediction step we have x_test as a input and while using it, we make forward prediction.

# In[ ]:


# prediction
def predict(w,b,x_test):
   # x_test is a input for forward propagation
   z = sigmoid(np.dot(w.T,x_test)+b)
   Y_prediction = np.zeros((1,x_test.shape[1]))
   # if z is bigger than 0.5, our prediction is sign one (y_head=1),
   # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
   for i in range(z.shape[1]):
       if z[0,i]<= 0.5:
           Y_prediction[0,i] = 0
       else:
           Y_prediction[0,i] = 1

   return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test)


# * We make prediction.
# * Now lets put them all together.

# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)


# * We learn logic behind simple neural network(logistic regression) and how to implement it.
# * Now that we have learned logic, we can use sklearn library which is easier than implementing all steps with hand for logistic regression.
# 
# 
# 

# <a id="8"></a> <br>
# ## Logistic Regression with Sklearn
# * In sklearn library, there is a logistic regression method that ease implementing logistic regression.
# * I am not going to explain each parameter of logistic regression in sklear but if you want you can read from there http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# * The accuracies are different from what we find. Because logistic regression method use a lot of different feature that we do not use like different optimization parameters or regularization.
# * Lets make conclusion for logistic regression and continue with artificial neural network.

# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))


# <a id="9"></a> <br>
# ## Summary and Questions in Minds
# <font color='purple'>
# What we did at this first part:
# * Initialize parameters weight and bias
# * Forward propagation
# * Loss function
# * Cost function
# * Backward propagation (gradient descent)
# * Prediction with learnt parameters weight and bias
# * Logistic regression with sklearn
# 
# <br> If you have any question up to this point ask me please bacause we will contruct artificial neural network on logistic regression.
# <br> HOMEWORK: Here is good place to stop and make practice. Your homework is that create your own logistic regression method and classify two different sign language digits. 

# <a id="10"></a> <br>
# # Artificial Neural Network (ANN)
# * It is also called deep neural network or deep learning.
# * **What is neural network:** It is basically taking logistic regression and repeating it at least 2 times.
# * In logistic regression, there are input and output layers. However, in neural network, there is at least one hidden layer between input and output layer.
# * **What is deep, in order to say "deep" how many layer do I need to have:** When I ask this question to my teacher, he said that ""Deep" is a relative term; it of course refers to the "depth" of a network, meaning how many hidden layers it has. "How deep is your swimming pool?" could be 12 feet or it might be two feet; nevertheless, it still has a depth--it has the quality of "deepness". 32 years ago, I used two or three hidden layers. That was the limit for the specialized hardware of the day. Just a few years ago, 20 layers was considered pretty deep. In October, Andrew Ng mentioned 152 layers was (one of?) the biggest commercial networks he knew of. Last week, I talked to someone at a big, famous company who said he was using "thousands". So I prefer to just stick with "How deep?""
# * **Why it is called hidden:** Because hidden layer does not see inputs(training set)
# * For example you have input, one hidden and output layers. When someone ask you "hey my friend how many layers do your neural network have?" The answer is "I have 2 layer neural network". Because while computing layer number input layer is ignored. 
# * Lets see 2 layer neural network: 
# <a href="http://ibb.co/eF315x"><img src="http://preview.ibb.co/dajVyH/9.jpg" alt="9" border="0"></a>
# * Step by step we will learn this image.
#     * As you can see there is one hidden layer between input and output layers. And this hidden layer has 3 nodes. If yoube curious why I choose number of node 3, the answer is there is no reason, I only choose :). Number of node is hyperparameter like learning rate. Therefore we will see hyperparameters at the end of artificial neural network.
#     * Input and output layers do not change. They are same like logistic regression.
#     * In image, there is a tanh function that is unknown for you. It is a activation function like sigmoid function. Tanh activation function is better than sigmoid for hidden units bacause mean of its output is closer to zero so it centers the data better for the next layer. Also tanh activation function increase non linearity that cause our model learning better.
#     * As you can see with purple color there are two parts. Both parts are like logistic regression. The only difference is activation function, inputs and outputs.
#         * In logistic regression: input => output
#         * In 2 layer neural network: input => hidden layer => output. You can think that hidden layer is output of part 1 and input of part 2.
# * Thats all. We will follow the same path like logistic regression for 2 layer neural network.
#    
#     
#     

# <a id="11"></a> <br>
# ## 2-Layer Neural Network
# * Size of layers and initializing parameters weights and bias
# * Forward propagation
# * Loss function and Cost function
# * Backward propagation
# * Update Parameters
# * Prediction with learnt parameters weight and bias
# * Create Model

# <a id="12"></a> <br>
# ## Size of layers and initializing parameters weights and bias
# * For x_train that has 348 sample $x^{(348)}$:
# $$z^{[1] (348)} =  W^{[1]} x^{(348)} + b^{[1] (348)}$$ 
# $$a^{[1] (348)} = \tanh(z^{[1] (348)})$$
# $$z^{[2] (348)} = W^{[2]} a^{[1] (348)} + b^{[2] (348)}$$
# $$\hat{y}^{(348)} = a^{[2] (348)} = \sigma(z^{ [2] (348)})$$
# 
# * At logistic regression, we initialize weights 0.01 and bias 0. At this time, we initialize weights randomly. Because if we initialize parameters zero each neuron in the first hidden layer will perform the same comptation. Therefore, even after multiple iterartion of gradiet descent each neuron in the layer will be computing same things as other neurons. Therefore we initialize randomly. Also initial weights will be small. If they are very large initially, this will cause the inputs of the tanh to be very large, thus causing gradients to be close to zero. The optimization algorithm will be slow.
# * Bias can be zero initially.

# In[ ]:


# intialize parameters and layer sizes
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters


# <a id="13"></a> <br>
# ## Forward propagation
# * Forward propagation is almost same with logistic regression.
# * The only difference is we use tanh function and we make all process twice.
# * Also numpy has tanh function. So we do not need to implement it.

# In[ ]:



def forward_propagation_NN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# <a id="14"></a> <br>
# ## Loss function and Cost function
# * Loss and cost functions are same with logistic regression
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$

# In[ ]:


# Compute cost
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost


# <a id="15"></a> <br>
# ## Backward propagation
# * As you know backward propagation means derivative.
# * If you want to learn (as I said I cannot explain without talking bc it is little confusing), please watch video in youtube.
# * However the logic is same, lets write code.

# In[ ]:


# Backward Propagation
def backward_propagation_NN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads


# <a id="16"></a> <br>
# ## Update Parameters 
# * Updating parameters also same with logistic regression.
# * We actually do alot of work with logistic regression

# In[ ]:


# update parameters
def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    
    return parameters


# <a id="17"></a> <br>
# ## Prediction with learnt parameters weight and bias
# * Lets write predict method that is like logistic regression.

# In[ ]:


# prediction
def predict_NN(parameters,x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# <a id="18"></a> <br>
# ## Create Model
# * Lets put them all together

# In[ ]:


# 2 - Layer neural network
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters
parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)


# <font color='purple'>
# Up to this point we create 2 layer neural network and learn how to implement
# * Size of layers and initializing parameters weights and bias
# * Forward propagation
# * Loss function and Cost function
# * Backward propagation
# * Update Parameters
# * Prediction with learnt parameters weight and bias
# * Create Model
# 
# <br> Now lets learn how to implement L layer neural network with keras.

# <a id="19"></a> <br>
# # L Layer Neural Network
# * **What happens if number of hidden layer increase:** Earlier layerls can detect simple features.
# * When model composing simple features together in later layers of neural network that it can learn more and more complex functions. For example, lets look at our sign one.
# <a href="http://ibb.co/dNgDJH"><img src="http://preview.ibb.co/mpD4Qx/10.jpg" alt="10" border="0"></a>
# * For example first hidden layer learns edges or basic shapes like line. When number of layer increase, layers start to learn more complex things like convex shapes or characteristic features like forefinger.
# * Lets create our model
#     * There are some hyperparameters we need to choose like learning rate, number of iterations, number of hidden layer, number of hidden units, type of activation functions. Woww it is too much :)
#     * These hyperparameters can be chosen intiutively if you spend a lot of time in deep learning world.
#     * However, if you do not spend too much time, the best way is to google it but it is not necessary. You need to try hyperparameters to find best one.
#     * In this tutorial our model will have 2 hidden layer with 8 and4 nodes, respectively. Because when number of hidden layer and node increase, it takes too much time. 
#     * As a activation function we will use relu(first hidden layer), relu(second hidden layer) and sigmoid(output layer) respectively.
#     * Number of iteration will be 100.
# * Our way is same with previous parts however as you learn the logic behind deep learning, we can ease our job and use keras library for deeper neural networks.
# * First lets reshape our x_train, x_test, y_train and y_test.
# 
#     
#     

# In[ ]:


# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T


# <a id="22"></a> <br>
# ## Implementing with keras library
# Lets look at some parameters of keras library:
# * units: output dimensions of node
# * kernel_initializer: to initialize weights
# * activation: activation function, we use relu
# * input_dim: input dimension that is number of pixels in our images (4096 px)
# * optimizer: we use adam optimizer
#     * Adam is one of the most effective optimization algorithms for training neural networks.
#     * Some advantages of Adam is that relatively low memory requirements and usually works well even with little tuning of hyperparameters
# * loss: Cost function is same. By the way the name of the cost function is cross-entropy cost function that we use previous parts.
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$
# * metrics: it is accuracy.
# * cross_val_score: use cross validation. If you do not know cross validation please chech it from my machine learning tutorial. https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
# * epochs: number of iteration

# In[ ]:


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# <a id="20"></a> <br>
# # Conclusion
# * First of all thanks for this data set.
# * If you see my wrong spelling (it can be so much) please ignore them :)
# * This tutorial is shallow, if you want more detail about some concepts you can comment.
# * If you think that I cannot explain some concepts please learn it from youtube(especially Andrew) and then continue.
# * If you do not understand some thing that is related with python or machine learning please check my other tutorials
#     * Data Science: https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
#     * Machine learning: https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
# 
# ### <br> If you like it, please upvote :)
# ## <br> If you have any question, I will be appreciate to hear it.
# 
