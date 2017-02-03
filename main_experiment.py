#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import autoencoder
import numpy as np
import dataset_manupulation as dm
#import matplotlib.image as img

#config variable
number_of_kernel=np.array(  [16,    8,      8]);
kernel_shape=np.array([     [3,3],  [3,3],  [3,3]]);

                      
listTrainpath='/media/buckler/DataSSD/Phd/fall_detection/lists/novelty/skf4FoldDevTest/train/';
trainNameLists=['trainset.lst']

listPath='/media/buckler/DataSSD/Phd/fall_detection/lists/novelty/skf4FoldDevTest/dev+test/case1/'  
testNamesLists=['testset_1.lst','testset_2.lst','testset_3.lst','testset_4.lst']  
devNameLists=['devset_1.lst','devset_2.lst','devset_3.lst','devset_4.lst']             
           


#gestione dataset       
a3fall = dm.load_A3FALL('/media/buckler/DataSSD/Phd/fall_detection/dataset/spectrograms/') #load dataset

trainsets = dm.split_A3FALL_from_lists(a3fall,listTrainpath,trainNameLists)
devsets = dm.split_A3FALL_from_lists(a3fall,listPath,devNameLists)
testsets = dm.split_A3FALL_from_lists(a3fall,listPath,testNamesLists)


trainsets[0], mean, std =dm.normalize_data(trainsets[0]); #compute mean and std of the trainset and normalize the trainset     
testsets[1], _ , _ = dm.normalize_data(testsets[1],mean,std)  # normalize testset with the mean and std of the trainset
trainset_z = dm.awgn_padding_set(trainsets[0]);
testset_z = dm.awgn_padding_set(testsets[1]);
                                           
y_train , x_train = dm.reshape_set(trainset_z);
y_test , x_test = dm.reshape_set(testset_z);
       


#gestione esperimenti                                     

experiment_prova=autoencoder.autoencoder_fall_detection(kernel_shape,number_of_kernel);#init class


model=experiment_prova.network_architecture_autoencoder();#define net architecture
model.summary();     

experiment_prova.network_architecture_autoencoder_fit(x_train, y_train, x_dev, y_dev, x_test, y_test); 
                     
decoded_images = experiment_prova.reconstruct_images(x_test);  
                                                    
experiment_prova.compute_score(x_test,decoded_images,y_test);