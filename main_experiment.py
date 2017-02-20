#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
from py_files import autoencoder
from py_files import dataset_manupulation as dm
from os import path
import numpy as np
import argparse
#import matplotlib.image as img


parser = argparse.ArgumentParser(description="Novelty Deep Fall Detection")

# Global params
parser.add_argument("-c", "--config-file", dest = "config_filename", default = None)

#Esempi di utilizzo argparse. Documentazione completa https://docs.python.org/3/library/argparse.html
###############################################################################
# parser.add_argument("--batch-size", dest = "batch_size", default = 128, type=int)
# parser.add_argument("--no-shuffle", dest = "shuffle", action = 'store_false', default = True)
# parser.add_argument("--noise-std", dest = "noise_std", default = 0.0, type=float)
# parser.add_argument("--csv-file", dest ="csv_filename", default = None)
# parser.add_argument("--error-file", dest = "error_filename", default = None)
# parser.add_argument("--mode", dest="mode", default = "classic", choices = ["classic", "inverse", "goodfellow"])
# parser.add_argument("--discriminator-decides", dest = "discriminator_decides", default = False, action = 'store_true')
#
###############################################################################

np.random.seed(888)

args = parser.parse_args()

if (args.config_filename is not None):
    with open(args.config_filename, 'r') as f:
        lines = f.readlines()
        arguments = []
        for line in lines:
            arguments.extend(line.split())
        # First parse the arguments specified in the config file
        args = parser.parse_args(args=arguments)
        # Then append the command line arguments
        # Command line arguments have the priority: an argument is specified both
        # in the config file and in the command line, the latter is used
        args = parser.parse_args(namespace=args)





#config variable
number_of_kernel=np.array([16,    8,      8]);
kernel_shape=np.array([[3,3],  [3,3],  [3,3]]);

root_dir = path.realpath('.')

listTrainpath=path.join(root_dir,'lists','train');
trainNameLists=['trainset.lst']

listPath=path.join(root_dir,'lists','dev+test','case5');
testNamesLists=['testset_1.lst','testset_2.lst','testset_3.lst','testset_4.lst']  
devNameLists=['devset_1.lst','devset_2.lst','devset_3.lst','devset_4.lst']             

#GESTIONE DATASET       
a3fall = dm.load_A3FALL(path.join(root_dir,'dataset','spectrograms')) #load dataset

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
print("------------------------CROSS VALIDATION---------------")

params=[1]; #quesa variabile rappresenta tutti i set parametri che dovranno essere variati, ovviamente poi andrà modifivata. Per ora è fittizia
#init scoreAucMatrix
scoreAucMatrix=np.zeros((len(x_devs),len(params)))   #matrice che conterra tutte le auc ottenute per le diverse fold e diversi set di parametri                 
scoreThMatrix=np.zeros((len(x_devs),len(params)))   #matrice che conterra tutte le threshold ottime ottenute per le diverse fold e diversi set di parametri                 
f=p=0; #indici della scoreAucMatrix
for param in params: 
    f=0;    
    #carico modello con parametri di default
    net=autoencoder.autoencoder_fall_detection();
    net.define_arch();                                         
    
    #parametri di defautl anche per compile e fit
    net.model_compile()
    
    net.model_fit(x_trains[0], _ , nb_epoch=50)
    for x_dev, y_dev in zip (x_devs, y_devs): #sarebbero le fold

        decoded_images = net.reconstruct_spectrogram(x_dev);  
        auc, optimal_th, _, _, _ = net.compute_score(x_dev, decoded_images, y_dev);
        scoreAucMatrix[f,p]=auc;
        #scoreThMatrix[f,p]=th
        f+=1;
    p+=1;

#score=np.amax(scoreAucMatrix,axis=1);
#
idxBestParamPerFolds=scoreAucMatrix.argmax(axis=1);

             
#test-finale-------------------------------
print("------------------------TEST---------------")
idx=0;
my_cm=np.zeros((2,2));
sk_cm=np.zeros((2,2));
tot_y_pred=[];
tot_y_true=[];
for x_test, y_test in zip (x_tests, y_tests):
    
    param=params[idxBestParamPerFolds[idx]];#carico i parametri ottimi per una data fold
    #poi questi parametri verrano utilizzani nella creazione del modello/nel compile/e nel fit
    net.model_compile();
    net.model_fit(x_trains[0], _ );
                 
    decoded_images = net.reconstruct_spectrogram(x_test);  
    auc,_ , my_cm, y_true , y_pred = net.compute_score(x_test, decoded_images, y_test);
    #raccolto tutti i risultati delle fold, per poter fare un report generale
    for x in y_pred:
        tot_y_pred.append(x);
    for x in y_true:
        tot_y_true.append(x);                                             
    my_cm+=my_cm;  
    idx+=1;
    
    
#report finale
print('\n\n\n')
print("------------------------FINAL REPORT---------------")

net.print_score(my_cm,tot_y_pred,tot_y_true);
               
        
    
