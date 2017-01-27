#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import autoencoder_diego
import numpy as np
#import matplotlib.image as img
import matplotlib.pyplot as plt

#config variable
number_of_kernel=np.array(  [16,    8,      8]);
kernel_shape=np.array([     [3,3],  [3,3],  [3,3]]);

#define model and fit it                    
#experiment_mnist=autoencoder_diego.autoencoder_fall_detection(kernel_shape,number_of_kernel) ;
#                                                                
#X_train, y_train, x_test, y_test = experiment_mnist.pre_process_data_mnist();
#                                                                          
#experiment_mnist.network_architecture_autoencoder(X_train, y_train, x_test, y_test);
#
#decoded_images = experiment_mnist.reconstruct_images(x_test);  
#                                                    
#experiment_mnist.compute_score(x_test,decoded_images,y_test);

#image = img.imread("stupid_test_image/emoticon-5700-src-7edfa11b86dc141a-28x28.png")
#imageBN=image[:,:,0]
#imageBN2=np.zeros((1,1,28,28))
#imageBN2[0,0,:,:]=imageBN;
#experiment_mnist.reconstruct_image(imageBN2);               

#TEST load_dataset function
                                 
experiment_prova=autoencoder_diego.autoencoder_fall_detection(kernel_shape,number_of_kernel);
experiment_prova.load_dataset('/media/buckler/DataSSD/Phd/fall_detection/dataset/spectrogram_zero_pad/','_')
a=experiment_prova.allData;

plt.figure()
plt.matshow(a[1,0,:,:], fignum=100)
