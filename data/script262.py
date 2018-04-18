
# coding: utf-8

# # Open tensorflow kernel with CLI download, Multi-GPU support and much more
# 
# [Code @GitHub](https://github.com/Cognexa/cdiscount-kernel)
# 
# Just run a VGG-like convnet baseline while you analyze the data!
# 
# Works on Linux with Python 3.5+.
# 
# Features:
# - CLI data download
# - Data validation with SHA256 hash
# - Simple data visualization
# - Train-Valid splitting
# - Low memory footprint data streams
# - Base VGG-like convnet
# - Multi-GPU training with a single argument!
# - TensorBoard training tracking
# 
# ## Quick start
# Install tensorflow and 7z.
# 
# Clone repo and install the requirements
# ```
# git clone https://github.com/Cognexa/cdiscount-kernel && cd cdiscount-kernel
# pip3 install -r requirements.txt --user
# ```
# 
# Download dataset with kaggle-cli (this may take a while, 3 hours in my case)
# ```
# # requires >57Gb of free space
# KG_USER="<YOUR KAGGLE USERNAME" KG_PASS="<YOUR KAGGLE PASSWORD>" cxflow dataset download cdc
# ```
# 
# Or if you have downloaded the data earlier:
# ```
# mkdir data
# # mv/cp your etracted files to data directory
# ```
# 
# Validate your download and see the example data:
# ```
# # in the root directory (cdiscount-kernel)
# cxflow dataset validate cdc
# cxflow dataset show cdc
# # now see the newly created visual directory
# ```
# 
# Create a random validation split with 10% of the data and start training:
# ```
# cxflow dataset split cdc
# cxflow train cdc model.n_gpus=<NUMBER OF GPUS TO USE>
# ```
# 
# Observe the training with TensorBoard (note: a summary is written only after each epoch)
# ```
# tensorboard --logdir=log
# ```
# 
# ## UPDATE [LB 0.65]
# **important:** update **cxflow** and **cxflow-tensorflow** with `pip3 install cxflow cxflow-tensorflow --user --upgrade`
# 
# Main features:
# - XCeption net (https://arxiv.org/abs/1610.02357)
# - Fast random data access
# 
# Resize the data to `dataset.size` with (this may take a few hours)
# ```
# cxflow dataset resize cdc/xception.yaml
# cxflow dataset split cdc/xception.yaml
# ```
# 
# Run the training with
# ```
# cxflow train cdc/xception.yaml
# ```
# 
# Training procedure that reached 0.65:
# - Train with original size, LR 0.0001, 4 middle flow repeats until stalled
# - Fine-tune with 128x128, LR 0.0001, 0.5 dropout, 0.00001 weight decay until stalled
# - Fine-tune as above but with LR 0.00001 (10x smaller)
# 
# Tips:
# - Use small images right away
# - The final GlobalAveragePooling may be a bottleneck
# - Net does not overfit so far, no augmentations needed
# 
# ## Example output:
# ```
# 2017-09-16 00:22:14.000262: INFO    @common         : Creating dataset
# 2017-09-16 00:22:14.000776: INFO    @common         : 	CDCNaiveDataset created
# 2017-09-16 00:22:14.000777: INFO    @common         : Creating a model
# 2017-09-16 00:22:20.000724: INFO    @model          : 	Creating TF model on 2 GPU devices
# 2017-09-16 00:22:21.000362: INFO    @cdc_net        : Flatten shape `(?, 8192)`
# 2017-09-16 00:22:21.000387: INFO    @cdc_dataset    : Loading metadata
# 2017-09-16 00:22:26.000826: INFO    @cdc_net        : Output shape `(?, 5270)`
# 2017-09-16 00:22:26.000893: INFO    @cdc_net        : Flatten shape `(?, 8192)`
# 2017-09-16 00:22:26.000901: INFO    @cdc_net        : Output shape `(?, 5270)`
# 2017-09-16 00:22:29.000351: INFO    @common         : 	CDCNaiveNet created
# 2017-09-16 00:22:29.000354: INFO    @common         : Creating hooks
# 2017-09-16 00:22:29.000355: INFO    @common         : 	ShowProgress created
# 2017-09-16 00:22:29.000355: INFO    @common         : 	ComputeStats created
# 2017-09-16 00:22:29.000355: INFO    @common         : 	LogVariables created
# 2017-09-16 00:22:29.000356: INFO    @common         : 	LogProfile created
# 2017-09-16 00:22:29.000356: INFO    @common         : 	SaveEvery created
# 2017-09-16 00:22:29.000356: INFO    @common         : 	SaveBest created
# 2017-09-16 00:22:29.000357: INFO    @common         : 	CatchSigint created
# 2017-09-16 00:22:29.000357: INFO    @common         : 	StopAfter created
# 2017-09-16 00:22:30.000968: INFO    @common         : 	WriteTensorBoard created
# 2017-09-16 00:22:30.000968: INFO    @common         : Creating main loop
# 2017-09-16 00:22:30.000968: INFO    @common         : Running the main loop
# 2017-09-16 03:13:01.000457: INFO    @log_variables  : After epoch 1
# 2017-09-16 03:13:01.000457: INFO    @log_variables  : 	train loss mean: 4.243194
# 2017-09-16 03:13:01.000457: INFO    @log_variables  : 	train accuracy mean: 0.320435
# 2017-09-16 03:13:01.000457: INFO    @log_variables  : 	valid loss mean: 3.313541
# 2017-09-16 03:13:01.000457: INFO    @log_variables  : 	valid accuracy mean: 0.434122
# 2017-09-16 03:13:03.000486: INFO    @save           : Model saved to: ./log/CDCNaiveNet_2017-09-16-00-22-14_ngz6u4_b/model_1.ckpt
# 2017-09-16 03:13:05.000217: INFO    @save           : Model saved to: ./log/CDCNaiveNet_2017-09-16-00-22-14_ngz6u4_b/model_best.ckpt
# 2017-09-16 03:13:05.000219: INFO    @log_profile    : 	T read data:	1594.948686
# 2017-09-16 03:13:05.000219: INFO    @log_profile    : 	T train:	8347.117133
# 2017-09-16 03:13:05.000219: INFO    @log_profile    : 	T eval:	282.549242
# 2017-09-16 03:13:05.000219: INFO    @log_profile    : 	T hooks:	8.592250
# 2017-09-16 03:13:05.000219: INFO    @main_loop      : Epochs done: 1
# 2017-09-16 06:03:17.000103: INFO    @log_variables  : After epoch 2
# 2017-09-16 06:03:17.000103: INFO    @log_variables  : 	train loss mean: 2.952012
# 2017-09-16 06:03:17.000103: INFO    @log_variables  : 	train accuracy mean: 0.480100
# 2017-09-16 06:03:17.000103: INFO    @log_variables  : 	valid loss mean: 2.863293
# 2017-09-16 06:03:17.000104: INFO    @log_variables  : 	valid accuracy mean: 0.496674
# 2017-09-16 06:03:18.000840: INFO    @save           : Model saved to: ./log/CDCNaiveNet_2017-09-16-00-22-14_ngz6u4_b/model_2.ckpt
# 2017-09-16 06:03:20.000762: INFO    @save           : Model saved to: ./log/CDCNaiveNet_2017-09-16-00-22-14_ngz6u4_b/model_best.ckpt
# 2017-09-16 06:03:20.000764: INFO    @log_profile    : 	T read data:	1581.478134
# 2017-09-16 06:03:20.000764: INFO    @log_profile    : 	T train:	8342.576470
# 2017-09-16 06:03:20.000764: INFO    @log_profile    : 	T eval:	281.916230
# 2017-09-16 06:03:20.000764: INFO    @log_profile    : 	T hooks:	8.502520
# 2017-09-16 06:03:20.000764: INFO    @main_loop      : Epochs done: 2
# 
# ...```
# 
# 
# ## About
# This kernel is written in [cxflow-tensorflow](https://github.com/Cognexa/cxflow-tensorflow), a plugin for [cxflow](https://github.com/Cognexa/cxflow) framework. Make sure you check it out!
# 
# A simple submission script will be added soon, stay tuned!
# 
