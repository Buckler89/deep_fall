#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: buckler
"""


import autoencoder
import numpy as np
import dataset_manupulation as dm
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