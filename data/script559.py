
# coding: utf-8

# **GitHub Link -- https://github.com/groverpr/deep-learning/blob/master/image_comp_1_2.ipynb **
# 
# ### Deep learning using **fastai** library
# (https://github.com/fastai) 
# 
# ***Statoil/C-CORE Iceberg Classifier Challenge***
# *In this competition, you’re challenged to build an algorithm that automatically identifies if a remotely sensed target is a ship or iceberg*
# 
# **Because, this code needs us to work on fastai environment, I could not upload the jupyter notebook on kernel. Here is the github link for the notebook. **
# 
# https://github.com/groverpr/deep-learning/blob/master/image_comp_1_2.ipynb
# 
# This kernel is specifically for beginners who want's to experiment building CNN and learn how to tune hyperparameters. By using this kernel, you can expect to get decent score and also learn deep learning using fastai. Fastai has made building deep neural networks very easy.
# 
# I have used pretrained **resnet18** model based on **Imagenet** data for this
# 
# Steps to use **fastai library** --
# 
# 1. git clone https://github.com/fastai/fastai  
# 2. cd fastai  
# 3. conda create -n fastai python=3.6 anaconda  
# 4. conda env update  
# 5. source activate fastai  

# ### Summary of steps --
# 
# 1. Convert HH and HV band to RGB color composite and save files as .png
# 2. Visualization of random ships and random icebergs 
# 3. Save all .png files in seperate train, test and valid directories (validation = 10% of train data)
# 4. Compute weights on the top layer of **resnet18** after finding appropriate learning rate. (Top-down transformation of images to increase sample size)
# 5. Unfreeze all layers to retrain weights using **stochastic gradient descent with restart and differential annealing**. (Retune learning rate)
# 6. Retrain until it starts overfitting on the train. (Retune hyperparameter [learning rate, cycle multiple and cycle length])
# 7. Predict using **Test Time Augmentation** , i.e. making prediction on randomly selected augmented images and taking average
# 8. Post analysis -- Visualizing which images were most correctly and incorrectly predicted
# 
# *(I am attending in-person DL fastai course and it will be released online by the end of year after it completes)*
# 
