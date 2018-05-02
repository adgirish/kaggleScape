
# coding: utf-8

# # TensorFlow in vogue
# 
# The [researchers at Zalando](https://research.zalando.com/) recently released a dataset of images intended to serve as a drop-in replacement for the [MNIST hand-written digit database](http://yann.lecun.com/exdb/mnist/) that is commonly used to introduce people to different machine learning frameworks.
# 
# This new dataset is called [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist). The `fashion-mnist` [README](https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md) explains Zalando's motives for releasing the dataset.
# 
# The dataset consists of various images of clothing, footwear, and fashion accessories. The corresponding task is to match each image to the type of clothing, footwear, or accessory it represents.
# 
# Let us make sure that the data is available to us:

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


TRAIN_FILE = '../input/fashion-mnist_train.csv'
TEST_FILE = '../input/fashion-mnist_test.csv'


# We will use TensorFlow to build and train a very simple model which classifies `fashion-mnist` images.

# ## Setting expectations
# 
# TensorFlow is built to operate at scale. This can make it *seem* unnecessarily complex when it is *not* used with complex models or trained on large datasets. At various points in this tutorial, you may find yourself feeling this way. I certainly did as I was writing it.
# 
# This situation is made somewhat worse by the fact that, every time I had to make a choice of interfaces for certain functionality, I chose the one that would scale the best over perhaps the one that would be easiest to understand. Even though this may increase friction within this notebook, I hope that it will *decrease* your friction in using TensorFlow for more serious applications.
# 
# It may feel painful, and that you could much more easily do this particular task using other frameworks. Nevertheless, I urge you to stick with it. No other tool will reward this investment as much as TensorFlow.

# ## Defining the classifier
# 
# TensorFlow offers a high-level, scikit-learn-inspired API in its [Estimators](https://www.tensorflow.org/programmers_guide/estimators). This API provides a few pre-built estimators to solve regression and classification problems.
# 
# In this notebook, we will use the pre-built [linear classifier](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier).
# 
# Some amount of setup is required beforehand.

# In[ ]:


import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

print('TensorFlow version: ', tf.__version__)


# First, we specify a directory in which we will store checkpoints of the classifier parameters during training. This is also the directory from which we will load the parameters when we are evaluating the model or performing predictions using it.

# In[ ]:


MODEL_DIR = './fashion-model'


# Next, TensorFlow's [feature_column API](https://www.tensorflow.org/api_docs/python/tf/feature_column) allows us to specify to our classifier what kind of input it should expect.
# 
# Each of the `fashion-mnist` images is 28 pixels $\times$ 28 pixels, and is black-and-white.

# In[ ]:


feature_columns = [tf.feature_column.numeric_column('pixels', shape=[28,28])]


# It is now very easy to define our classifier:

# In[ ]:


classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=10,
    model_dir=MODEL_DIR
)


# The Estimator interface requires us to specify inputs to the classifier at training, evaluation, or prediction time by providing it with an input function. This input function is expected to be a function of no arguments which returns:
# 
# 1. A `features` dictionary describing the input features. The keys should be strings representing the names of the features. The values should be [tensors](https://www.tensorflow.org/api_docs/python/tf/Tensor) representing the features values. In our case, we will treat each image as a single feature represented by a $28 \times 28$ matrix, and we will call this feature `'pixels'`.
# 
# 2. A `labels` [tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor) containing the labelled classes for the training data.
# 
# The example below shows how to use TensorFlow queues to read data from CSV files, including a shuffle operation on the CSV rows. Although this may seem a little convoluted, and although TensorFlow does have a [simpler mechanism for loading data](https://www.tensorflow.org/api_guides/python/reading_data#Feeding), this is the preferred way to do so for reasons of performance. The intention here is that you be able to repurpose this code to many other situations with only slight modifications.
# 
# Note that the `generate_labelled_input_fn` is creating an `input_fn` which we will inject into our estimator when we use it for training, evaluation, or prediction. This allows us to define the sources of input - `csv_files` and the `batch_size` within the same scope as the `input_fn`, and is a common pattern in TensorFlow.

# In[ ]:


def generate_labelled_input_fn(csv_files, batch_size):
    def input_fn():
        file_queue = tf.train.string_input_producer(csv_files)
        reader = tf.TextLineReader(skip_header_lines=1)
        _, rows = reader.read_up_to(file_queue, num_records=100*batch_size)
        expanded_rows = tf.expand_dims(rows, axis=-1)
        
        shuffled_rows = tf.train.shuffle_batch(
            [expanded_rows],
            batch_size=batch_size,
            capacity=20*batch_size,
            min_after_dequeue=5*batch_size,
            enqueue_many=True
        )

        record_defaults = [[0] for _ in range(28*28+1)]

        columns = tf.decode_csv(shuffled_rows, record_defaults=record_defaults)

        labels = columns[0]

        pixels = tf.concat(columns[1:], axis=1)

        return {'pixels': pixels}, labels
    
    return input_fn


# You can use the parameters below to reconfigure the training and evaluation:

# In[ ]:


BATCH_SIZE = 40


# In[ ]:


TRAIN_STEPS = 2000


# We can now train the classifier.

# In[ ]:


classifier.train(
    input_fn=generate_labelled_input_fn([TRAIN_FILE], BATCH_SIZE),
    steps=TRAIN_STEPS
)


# Let's see how the trained classifiers fares on our test data.

# In[ ]:


classifier.evaluate(
    input_fn=generate_labelled_input_fn([TEST_FILE], BATCH_SIZE),
    steps=100
)


# Not too shabby for something so simple.

# ## Predictions
# 
# Alright, we have a trained model. Let's use it to make predictions!

# In[ ]:


import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


CLASSES = {
    '0': 'T-shirt/top',
    '1': 'Trouser',
    '2': 'Pullover',
    '3': 'Dress',
    '4': 'Coat',
    '5': 'Sandal',
    '6': 'Shirt',
    '7': 'Sneaker',
    '8': 'Bag',
    '9': 'Ankle boot'
}


# We'll make predictions for random samples from the test data.

# In[ ]:


test_data = pd.read_csv(TEST_FILE)


# In[ ]:


sample_row = test_data.sample()


# In[ ]:


sample = list(sample_row.iloc[0])
label = sample[0]
pixels = sample[1:]


# In[ ]:


image_array = np.asarray(pixels, dtype=np.float32).reshape((28, 28))


# This is the image that we're running the prediction on:

# In[ ]:



plt.imshow(image_array, cmap='gray')


# Once again, we need to provide the estimator's `predict` method with an input function that produces the data that we want to classify.
# 
# This input function should conform to the same interface as the one used for training and evaluation, with one concession - as we don't expect to have labelled data at prediction time, the input function that we pass the `predict` method of the estimator need not actually produce any labels. We can represent this by either returning `features, None` or by simply returning `features` by themselves.
# 
# We shall do the latter.

# In[ ]:


def generate_prediction_input_fn(image_arrays):
    def input_fn():
        queue = tf.train.input_producer(
            tf.constant(np.asarray(image_arrays)),
            num_epochs=1
        )
        
        image = queue.dequeue()
        return {'pixels': [image]}
    
    return input_fn


# Note that the estimator's `predict` method returns a generator.

# In[ ]:


predictions = classifier.predict(
    generate_prediction_input_fn([image_array]),
    predict_keys=['probabilities', 'classes']
)


# In[ ]:


prediction = next(predictions)


# In[ ]:


print('Prediction output: {}'.format(prediction))


# In[ ]:


print('Actual label: {} - {}'.format(label, CLASSES[str(label)]))
predicted_class = prediction['classes'][0].decode('utf-8')
probability = prediction['probabilities'][int(predicted_class)]
print('Predicted class: {} - {} with probability {}'.format(
    predicted_class,
    CLASSES[predicted_class],
    probability
))


# Not bad (I hope).

# ## What now?
# 
# The classifier you trained in this notebook should have achieved somewhere around $82\%$ accuracy on the evaluation data. That leaves us with a lot of room for improvement!
# 
# The natural next step would be to try and design your own models to beat the linear classifier.
# 
# If you would like to make use of the Estimator API to build your own models, you can follow the guide [here](https://github.com/GoogleCloudPlatform/ml-on-gcp/blob/master/tensorflow/tf-estimators.ipynb).
# 
# There are also many examples of TensorFlow models [here](https://github.com/tensorflow/models), which may provide you with inspiration.
# 
# TensorFlow has an [nn](https://www.tensorflow.org/api_docs/python/tf/nn) submodule that exposes building blocks you can use to define your own neural network models if you want even more control.
# 
# If there is enough interest, I could also publish a template notebook that makes it easier to start prototyping your custom models. Please let me know in the comments.
# 
# And finally, I would be very grateful for any feedback that you have.
# 
# Good luck!
