#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: buckler
"""


import autoencoder_diego
import numpy as np

#config variable
number_of_kernel=np.array(  [16,    8,      8]);
kernel_shape=np.array([     [3,3],  [3,3],  [3,3]]);


                                 
experiment_prova=autoencoder_diego.autoencoder_fall_detection(kernel_shape,number_of_kernel);#init class

            
a3fall = experiment_prova.load_A3FALL('/media/buckler/DataSSD/Phd/fall_detection/dataset/spectrograms/') #load dataset

trainset=experiment_prova.select_list('/media/buckler/DataSSD/Phd/fall_detection/lists/novelty/skf4FoldDevTest/train/trainFilePaths.lst',a3fall)                                            

