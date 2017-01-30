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
                                 
experiment_prova=autoencoder_diego.autoencoder_fall_detection(kernel_shape,number_of_kernel);#init class

            
a3fall = experiment_prova.load_A3FALL('/media/buckler/DataSSD/Phd/fall_detection/dataset/spectrograms/') #load dataset
trainset, testset  = experiment_prova.split_A3FALL_simple();  #split the data
trainset, mean, std =experiment_prova.normalize_data(trainset); #compute mean and std of the trainset and normalize the trainset     
testset, _ , _ = experiment_prova.normalize_data(testset,mean,std)  # normalize testset with the mean and std of the trainset
#to do: reshape of the train and test for the network: partire dalle matrici già caricate e non utilizzare piu la vecchia funzione che leggeva dal dico gli spettri zeropaddati
trainset_z = experiment_prova.zeropadding_set(trainset);
testset_z = experiment_prova.zeropadding_set(testset);
                                            
y_train , x_train = experiment_prova.reshape_set(trainset_z);
y_test , x_test = experiment_prova.reshape_set(testset_z)
                                            



model=experiment_prova.network_architecture_autoencoder();#define net architecture
model.summary();     

experiment_prova.network_architecture_autoencoder_fit(x_train, y_train, x_test, y_test);                      
#experiment_prova.normalize_A3FALL(experiment_prova.allData)
#a=experiment_prova.data_std;
#
#plt.figure()
#plt.matshow(a[1,0,:,:], fignum=100)

decoded_images = experiment_prova.reconstruct_images(x_test);  
                                                    
experiment_prova.compute_score(x_test,decoded_images,y_test);