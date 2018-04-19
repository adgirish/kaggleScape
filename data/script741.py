
# coding: utf-8

# <H1>Overview</H1>
# 
# The Passenger Screening Algorithm Challenge asks the data science community to assist with improving threat detection at US airports while minimizing false positives to avoid long lines and delays. (Can I get an amen!).  This notebook is a follow up to my first effort for this contest called [Exploratory Data Analysis and Example Generation](https://www.kaggle.com/jbfarrar/exploratory-data-analysis-and-example-generation) (I'll call it EDA from now on).  As I mentioned in the EDA notebook, the HD-AIT system files supplied in this contest range from 10MB to approximately 2GB per subject.  In the instructions, the organizers suggest that one may even be able to win the contest with one of the smaller image suites. In that notebook, in addition to a review of the data and its vagueries, I supplied some basic building blocks for a preprocessing pipeline.
# 
# In this notebook, I continue the series with a full preprocessing pipeline using the building blocks from before as well as a first pass through a CNN based on the Alexnet using Tensorflow.  Clearly, no one is going to win the contest with this method, but I thought it would be helpful to everyone working on this to have an end to end working pipeline.  I hope you find it useful, and if you do, I hope you'll give me an up vote!
# 
# As previously noted, I'm not an expert on these systems or the related scans.  If you see something I've misunderstood or you think I've made an error, let me know and I'll correct it.  TSA has made it harder for people to get into this contest by disallowing even masked images to be protrayed on Kaggle, so you'll have to put these scripts in your own environment to take them around the track.  In any event, I am convinced that data science can improve the predictive veracity of these scans.  I'll get off the soap box now and move on.
# 
# To begin I collect all of the imports used in the notebook at the top.  It makes it easier when you're converting to a preprocessing script.  Make sure to take note of the last import, tsahelper. You will need to install tsahelper and uncomment this line in order for this pipeline to work. The tsahelper package is made from the EDA and is now available as a pip install (no warranties!). 
# 

# In[ ]:


# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer

#import tsahelper as tsa


# Here I collect the constants all in one place.  Once you are running training routinely, you'll want it to be easy to try different parameters. One tricky thing about the preprocessing approach employed here, is determining when a mini batch is complete.  There are 182 views of the threat zones per subject (more on the terminology and approach for this covered in detail in the [EDA](https://www.kaggle.com/jbfarrar/exploratory-data-analysis-and-example-generation)), so the batch counts depend on this fact to know when we have a complete minibatch.  Note also that FILE_LIST, TRAIN_FILE_LIST, and TEST_FILE_LIST are empty until after preprocessing.  More on that below.

# In[ ]:


#---------------------------------------------------------------------------------------
# Constants
#
# INPUT_FOLDER:                 The folder that contains the source data
#
# PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files 
# 
# STAGE1_LABELS:                The CSV file containing the labels by subject
#
# THREAT_ZONE:                  Threat Zone to train on (actual number not 0 based)
#
# BATCH_SIZE:                   Number of Subjects per batch
#
# EXAMPLES_PER_SUBJECT          Number of examples generated per subject
#
# FILE_LIST:                    A list of the preprocessed .npy files to batch
# 
# TRAIN_TEST_SPLIT_RATIO:       Ratio to split the FILE_LIST between train and test
#
# TRAIN_SET_FILE_LIST:          The list of .npy files to be used for training
#
# TEST_SET_FILE_LIST:           The list of .npy files to be used for testing
#
# IMAGE_DIM:                    The height and width of the images in pixels
#
# LEARNING_RATE                 Learning rate for the neural network
#
# N_TRAIN_STEPS                 The number of train steps (epochs) to run
#
# TRAIN_PATH                    Place to store the tensorboard logs
#
# MODEL_PATH                    Path where model files are stored
#
# MODEL_NAME                    Name of the model files
#
#----------------------------------------------------------------------------------------
INPUT_FOLDER = 'tsa_datasets/stage1/aps'
PREPROCESSED_DATA_FOLDER = 'tsa_datasets/preprocessed/'
STAGE1_LABELS = 'tsa_datasets/stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'tsa_logs/train/'
MODEL_PATH = 'tsa_logs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, 
                                                IMAGE_DIM, THREAT_ZONE )) 


# <H3>The Preprocessor</H3>
# 
# Throughout this notebook, passengers who are being scanned for contraband are referred to as "subjects".  The preprocessor begins with 3 different ways you can choose to read in a list of subjects.  If you are training on the full data set, OPTION 1 (see the comments) is your best bet.  If you want to preprocess all subjects for whom you have data, then OPTION 2 is your best choice, and if you are running in a sample or low volume notebook environment, then you can just give a short list of subject IDs for whom you have data loaded using OPTION 3.
# 
# My approach in this notebook is to isolate each individual threat zone from every visible angle and then make features out of each individual threat zone from each angle that a given threat zone is visible. This allows us to train on each threat zone individually from every view in a 2D format.  (This is covered in some detail in the [EDA](https://www.kaggle.com/jbfarrar/exploratory-data-analysis-and-example-generation)).
# 
# The preprocessor loops through the data one subject at a time, transforms the images, isolates threat zones, and uses a set of vertices to crop each image to 250x250.  Images are saved in minibatches by threat zone, so that they can be read into the trainer.
# 
# Note that the trainer depends upon the threat zone number being present in the minibatch file name.  (Not my favorite approach, but it was fastest and easiest given what I was doing).  If you have a better idea, pass it along!

# In[ ]:


#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
#
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------

def preprocess_tsa_data():
    
    # OPTION 1: get a list of all subjects for which there are labels
    #df = pd.read_csv(STAGE1_LABELS)
    #df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    #SUBJECT_LIST = df['Subject'].unique()

    # OPTION 2: get a list of all subjects for whom there is data
    #SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]
    
    # OPTION 3: get a list of subjects for small bore test purposes
    SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
                    '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
                    '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e']
    
    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()
    
    for subject in SUBJECT_LIST:

        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, 
                                                                     subject))
        print('--------------------------------------------------------------')
        images = tsa.read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list, 
                                                             tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num, 
                             tsa.get_subject_labels(STAGE1_LABELS, subject)))

            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                print('Threat Zone Label -> {}'.format(label))
                
                if threat_zone[img_num] is not None:

                    # correct the orientation of the image
                    print('-> reorienting base image') 
                    base_img = np.flipud(img)
                    print('-> shape {}|mean={}'.format(base_img.shape, 
                                                       base_img.mean()))

                    # convert to grayscale
                    print('-> converting to grayscale')
                    rescaled_img = tsa.convert_to_grayscale(base_img)
                    print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                       rescaled_img.mean()))

                    # spread the spectrum to improve contrast
                    print('-> spreading spectrum')
                    high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                       high_contrast_img.mean()))

                    # get the masked image
                    print('-> masking image')
                    masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                    print('-> shape {}|mean={}'.format(masked_img.shape, 
                                                       masked_img.mean()))

                    # crop the image
                    print('-> cropping image')
                    cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                    print('-> shape {}|mean={}'.format(cropped_img.shape, 
                                                       cropped_img.mean()))

                    # normalize the image
                    print('-> normalizing image')
                    normalized_img = tsa.normalize(cropped_img)
                    print('-> shape {}|mean={}'.format(normalized_img.shape, 
                                                       normalized_img.mean()))

                    # zero center the image
                    print('-> zero centering')
                    zero_centered_img = tsa.zero_center(normalized_img)
                    print('-> shape {}|mean={}'.format(zero_centered_img.shape, 
                                                       zero_centered_img.mean()))

                    # append the features and labels to this threat zone's example array
                    print ('-> appending example to threat zone {}'.format(tz_num))
                    threat_zone_examples.append([[tz_num], zero_centered_img, label])
                    print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                                                         len(threat_zone_examples),
                                                         len(threat_zone_examples[0]),
                                                         len(threat_zone_examples[0][0]),
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]),
                                                         len(threat_zone_examples[0][2])))
                else:
                    print('-> No view of tz:{} in img:{}. Skipping to next...'.format( 
                                tz_num, img_num))
                print('------------------------------------------------')

        # each subject gets EXAMPLES_PER_SUBJECT number of examples (182 to be exact, 
        # so this section just writes out the the data once there is a full minibatch 
        # complete.
        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + PREPROCESSED_DATA_FOLDER + 
                                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format( 
                                        tz_num+1,
                                        len(threat_zone_examples[0][1][0]),
                                        len(threat_zone_examples[0][1][1]), 
                                        batch_num))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples if example[0] == 
                               [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                            for features_label in tz_examples])

                # save batch.  Note that the trainer looks for tz{} where {} is a 
                # tz_num 1 based in the minibatch file to select which batches to 
                # use for training a given threat zone
                np.save(PREPROCESSED_DATA_FOLDER + 
                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]), 
                                                         batch_num), 
                                                         tz_examples_to_save)
                del tz_examples_to_save

            #reset for next batch 
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1
    
    # we may run out of subjects before we finish a batch, so we write out 
    # the last batch stub
    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER 
                    + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                      len(threat_zone_examples[0][1][0]),
                      len(threat_zone_examples[0][1][1]), 
                                                                                                                  batch_num))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] == 
                           [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                        for features_label in tz_examples])

            #save batch
            np.save(PREPROCESSED_DATA_FOLDER + 
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                     len(threat_zone_examples[0][1][0]),
                                                     len(threat_zone_examples[0][1][1]), 
                                                     batch_num), 
                                                     tz_examples_to_save)
# unit test ---------------------------------------
#preprocess_tsa_data()


# <H3>Train and Test Split</H3>
# 
# The next function takes the full minibatch list and splits it between train and test sets, using the TRAIN_TEST_SPLIT_RATIO.  Note that as mentioned above building FILE_LIST searches through the minibatch file name and looks for the string '-tz' + THREAT_ZONE + '-' in the file name.  If you used the preprocessor above, it creates the files in that form.

# In[ ]:


#---------------------------------------------------------------------------------------
# get_train_test_file_list(): gets the batch file list, splits between train and test
#
# parameters:      none
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def get_train_test_file_list():
    
    global FILE_LIST
    global TRAIN_SET_FILE_LIST
    global TEST_SET_FILE_LIST

    if os.listdir(PREPROCESSED_DATA_FOLDER) == []:
        print ('No preprocessed data available.  Skipping preprocessed data setup..')
    else:
        FILE_LIST = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER) 
                     if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        train_test_split = len(FILE_LIST) -                            max(int(len(FILE_LIST)*TRAIN_TEST_SPLIT_RATIO),1)
        TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
        TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
        print('Train/Test Split -> {} file(s) of {} used for testing'.format( 
              len(FILE_LIST) - train_test_split, len(FILE_LIST)))
        
# unit test ----------------------------
#get_train_test_file_list()
#print (


# <H3>Generating an Input Pipeline</H3>
# 
# The following function reads in a minibatch, extracts features and labels, and then returns the data in a form that can be easily streamed into a tensorfow feed dictionary, or as we will do below, as a feed dictionary to a TFLearn based CNN.

# In[ ]:


#---------------------------------------------------------------------------------------
# input_pipeline(filename, path): prepares a batch of features and labels for training
#
# parameters:      filename - the file to be batched into the model
#                  path - the folder where filename resides
#
# returns:         feature_batch - a batch of features to train or test on
#                  label_batch - a batch of labels related to the feature_batch
#
#---------------------------------------------------------------------------------------

def input_pipeline(filename, path):

    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []
    
    #Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))
        
    #Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)
    
    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])
    
    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)
    
    return feature_batch, label_batch
  
# unit test ------------------------------------------------------------------------
#print ('Train Set -----------------------------')
#for f_in in TRAIN_SET_FILE_LIST:
#    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
#    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
#                                                len(feature_batch[0]), 
#                                                len(feature_batch[0][0])))
#    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
    
#print ('Test Set -----------------------------')
#for f_in in TEST_SET_FILE_LIST:
#    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
#    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
#                                                len(feature_batch[0]), 
#                                                len(feature_batch[0][0])))
#    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))


# <H3>Shuffling the Training Set</H3>
# 
# Below we use TFLearn, an abstraction of Tensorflow, to build the convnet. Using TFLearn we can set the fit operation to shuffle rows within a mini batch.  This function shuffles the minibatch list, so that in addition to intra-minibatch shuffling, the there is also  shuffling of the order the mini batches are fed to the model.

# In[ ]:


#---------------------------------------------------------------------------------------
# shuffle_train_set(): shuffle the list of batch files so that each train step
#                      receives them in a different order since the TRAIN_SET_FILE_LIST
#                      is a global
#
# parameters:      train_set - the file listing to be shuffled
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list
    
# Unit test ---------------
#print ('Before Shuffling ->', TRAIN_SET_FILE_LIST)
#shuffle_train_set(TRAIN_SET_FILE_LIST)
#print ('After Shuffling ->', TRAIN_SET_FILE_LIST)


# <H3>Defining the Alexnet CNN</H3>
# 
# The Alexnet was first put to the real world test during the ImageNet Large Scale Visual Recognition Challenge in 2012. The performance of this network was a quantum shift for its time as the model achieved a top-5 error of 15.3%, more than 10.8 percentage points ahead of the runner up.  The solution is elaborated in  [this paper by the original author](https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf) if you are interested in learning more. 
# 
# But in short the network consists of 7 layers, 5 convolutions/maxpools, plus 2 regression layers at the end.  The structure of the model looks like this: 
# 
# <img src="https://kratzert.github.io/images/finetune_alexnet/alexnet.png" width="700" height="600">
# 
# Using TFLearn makes this definition is quite intuitive and simple.  

# In[ ]:


#---------------------------------------------------------------------------------------
# alexnet(width, height, lr): defines the alexnet
#
# parameters:      width - width of the input image
#                  height - height of the input image
#                  lr - learning rate
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model


# <H3>The Trainer</H3>
# 
# Finally, the trainer is straight forward.  Set up the network, loop to read in minibatches for test and train and run the fit method.  Note that TFLearn treats each "minibatch" as an epoch.  For the illustration purposes noted here, its not a big deal, but after may runs it may be quite annoying.  
# 
# Up until now, all the work I've personally done has been using the lower level interface.  This was my first time trying TFLearn.  I liked a lot about TFLearn.  Network construction is easy-peasy.  I haven't worked with Keras as of yet, but it looks like it may have a few advantages worth considering.

# In[ ]:


#---------------------------------------------------------------------------------------
# train_conv_net(): runs the train op
#
# parameters:      none
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def train_conv_net():
    
    val_features = []
    val_labels = []
    
    # get train and test batches
    get_train_test_file_list()
    
    # instantiate model
    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    
    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_features, val_labels = input_pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
        else:
            tmp_feature_batch, tmp_label_batch = input_pipeline(test_f_in, 
                                                                PREPROCESSED_DATA_FOLDER)
            val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

    
    
    # start training process
    for i in range(N_TRAIN_STEPS):

        # shuffle the train set files before each step
        shuffle_train_set(TRAIN_SET_FILE_LIST)
        
        # run through every batch in the training set
        for f_in in TRAIN_SET_FILE_LIST:
            
            # read in a batch of features and labels for training
            feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            #print ('Feature Batch Shape ->', feature_batch.shape)                
                
            # run the fit operation
            model.fit({'features': feature_batch}, {'labels': label_batch}, n_epoch=1, 
                      validation_set=({'features': val_features}, {'labels': val_labels}), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=MODEL_NAME)
            
# unit test -----------------------------------
#train_conv_net()


# <H3>Wrap Up!</H3>
# 
# Alright, now its time to cut it loose.  I convert this notebook into a script and let the magic begin.  So far I have run training against the first three threat zones (1-3),  I am currently seeing validation accuracy in the 92-96% range. If you checked out the [EDA](http://s://www.kaggle.com/jbfarrar/exploratory-data-analysis-and-example-generation) you'll recall that the probabilities for the first three zones are 11.6%, 11%, 9.1% respectively.  So a model that just predicts "no contraband", should perform at 88.4%, 89%, and 90.8%.  So while it appears we may be getting some predictive value, much work would be needed to drive those accuracy numbers higher.  With the pipeline working, I'm going to fire up a meaningful training run andI will update this section with a fullsome view by threat zone of the accuracy, once I've run enough epochs to have useful view.
# 
# If you've found this helpful, I hope you'll give me an up vote!
# 
# Good Luck!
# 
# 
