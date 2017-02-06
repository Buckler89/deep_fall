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
           


#GESTIONE DATASET       
a3fall = dm.load_A3FALL('/media/buckler/DataSSD/Phd/fall_detection/dataset/spectrograms/') #load dataset

                       #il trainset è 1 e sempre lo stesso per tutti gli esperimenti
trainset = dm.split_A3FALL_from_lists(a3fall,listTrainpath,trainNameLists)[0]; #creo i trainset per calcolare media e varianza per poter normalizzare 
trainset , mean, std =dm.normalize_data(trainset); #compute mean and std of the trainset and normalize the trainset  

a3fall_n , _, _= dm.normalize_data(a3fall, mean , std); #ormalize the dataset with the mean and std of the trainset
a3fall_n_z = dm.awgn_padding_set(a3fall_n);
                                
                                
#creo i set partendo dal dataset normalizzato e paddato
trainsets = dm.split_A3FALL_from_lists(a3fall_n_z,listTrainpath,trainNameLists)
devsets = dm.split_A3FALL_from_lists(a3fall_n_z,listPath,devNameLists);
testsets = dm.split_A3FALL_from_lists(a3fall_n_z,listPath,testNamesLists);

#reshape dataset per darli in ingresso alla rete

x_trains = list()
y_trains = list()
x_devs = list()
y_devs = list()
x_tests = list()
y_tests = list()

for s in trainsets:
    x, y = dm.reshape_set(s)
    x_trains.append(x)
    y_trains.append(y)
for s in devsets:
    x, y = dm.reshape_set(s)
    x_devs.append(x)
    y_devs.append(y)
for s in testsets:
    x, y = dm.reshape_set(s)
    x_tests.append(x)
    y_tests.append(y)


#cross validation
                                                                          
                        #pseudo codice
                        #for param in paramlist:
                        #    #carico modello 
                        #    #setto hyperparametri della funzione fit e compile
                        #    net.fit(trainset,validationset?)     #se il trainset è unico allora questa fit può stare fuori dal "for fold"!!!
                        #    for fold in folds:
                        #        net.predict(devset)
                        #        net.compute_score+=score #sommole score di tutte le fold
                        #        score=score/nfold
                        #    score.max()
                        #    save param di score.max()
                        
                        #una volta scoperto quali solo i parametri ottimi si fa il test vero e proprio:
                        #for fold in folds
                        #   net.load_model(parametri ottimi)
                        #   net.predict(testset)
                        #   net.compute_score+=score #sommole score di tutte le fold

params=[1]; #quesa variabile rappresenta tutti i parametri che dovranno essere variati, ovviamente poi andrà modifivata. Per ora è fittizia
for param in params:                        
    #carico modello con parametri di defautl
    net=autoencoder.autoencoder_fall_detection();
    net.define_arch();                                         
    
    #parametri di defautl anche per compile e fit
    net.model_compile()
    
    net.model_fit(x_trains[0], _ )
    for x_dev, y_dev in zip (x_devs, y_devs): #sarebbero le fold

        decoded_images = net.reconstruct_images(x_dev);  
        net.compute_score(x_dev, decoded_images, y_dev)
    
        #raggruppare ora i dati per fold: modifica la funzione compute score per fare questo
    
