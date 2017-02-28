#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import numpy as np

np.random.seed(888)#for experiment repetibility: this goes here, befor importing keras (inside autoencoder modele)
#from py_files import autoencoder
#from py_files import dataset_manupulation as dm
import autoencoder
import dataset_manupulation as dm
from os import path
import argparse
import os
import errno
import json
from keras.models import Model,load_model
#import matplotlib.image as img






###################THIS GOES IN THE CONFIGURATION EXPERIMENT SCRIPT. NON IN THIS!!!!###################################
#inizializzo i file dove memorizzare gli score più altri. 
#Inolte inizializzo i file dove memorizzare anche i modelli gli args e la threshold che hanno dato tali risultati


###################END THIS GOES IN THE CONFIGURATION EXPERIMENT SCRIPT. NON IN THIS!!!!###################################

parser = argparse.ArgumentParser(description="Novelty Deep Fall Detection")

# Global params
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-sp", "--score-path", dest="scorePath", default=os.path.join("score"))
parser.add_argument("-tl", "--trainset-list", dest="trainNameLists", nargs='+', default=['trainset.lst'])
parser.add_argument("-c", "--case", dest="case", default='case6')
parser.add_argument("-tln", "--test-list-names", dest="testNamesLists", nargs='+', default=['testset_1.lst','testset_2.lst','testset_3.lst','testset_4.lst'])
parser.add_argument("-dln", "--dev-list-names", dest="devNamesLists", nargs='+', default=['devset_1.lst','devset_2.lst','devset_3.lst','devset_4.lst'])
parser.add_argument("-it", "--input-type", dest="input_type", default='spectrograms')

# CNN params 
#parser.add_argument('-kn','--kernels-number', dest="number_of_kernel", nargs='+', default=[16, 8, 8], type=int)
#parser.add_argument('-ks','--kernel-shape', dest="kernel_shape", nargs='+', action='append', type=int) # default after parser.parse_args()
#parser.add_argument('-is','--cnn-input-shape', dest="cnn_input_shape", nargs='+', default=[1, 129, 197], type=int)

# fit params
parser.add_argument("-e", "--epoch", dest = "epoch", default=50, type=int)
parser.add_argument("-ns", "--no-shuffle", dest = "shuffle", default = True, action = 'store_false')
parser.add_argument("-bs", "--batch-size", dest = "batch_size", default=128, type=int)
parser.add_argument("-f", "--fit-net", dest = "fit_net", default = False, action = 'store_true')
parser.add_argument("-o", "--optimizer", dest = "optimizer", default="adadelta", choices = ["adadelta","adam", "sgd"])
parser.add_argument("-l", "--loss", dest = "loss", default="mse", choices = ["mse"])


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



args = parser.parse_args()

if (args.config_filename is not None):
    with open(args.config_filename, 'r') as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        if '#' not in line:
            arguments.extend(line.split())
    # First parse the arguments specified in the config file
    args = parser.parse_args(args=arguments)
    # Then append the command line arguments
    # Command line arguments have the priority: an argument is specified both
    # in the config file and in the command line, the latter is used
    args = parser.parse_args(namespace=args)
    # default values
    
    
    
#    if not args.kernel_shape:
#        args.kernel_shape = [[3, 3], [3, 3], [3, 3]]



root_dir = path.realpath('.')

listTrainpath=path.join(root_dir,'lists','train');
listPath=path.join(root_dir,'lists','dev+test', args.case);
          
#GESTIONE DATASET       
a3fall = dm.load_A3FALL(path.join(root_dir,'dataset',args.input_type)) #load dataset

#il trainset è 1 e sempre lo stesso per tutti gli esperimenti
trainset = dm.split_A3FALL_from_lists(a3fall,listTrainpath,args.trainNameLists)[0]; #creo i trainset per calcolare media e varianza per poter normalizzare 
trainset , mean, std =dm.normalize_data(trainset); #compute mean and std of the trainset and normalize the trainset  

a3fall_n , _, _= dm.normalize_data(a3fall, mean , std); #ormalize the dataset with the mean and std of the trainset
a3fall_n_z = dm.awgn_padding_set(a3fall_n);
                                
                                
#creo i set partendo dal dataset normalizzato e paddato
trainsets = dm.split_A3FALL_from_lists(a3fall_n_z,listTrainpath,args.trainNameLists)
devsets = dm.split_A3FALL_from_lists(a3fall_n_z,listPath,args.devNamesLists);
testsets = dm.split_A3FALL_from_lists(a3fall_n_z,listPath,args.testNamesLists);

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
scoreAucNew=np.zeros(len(x_devs))   #matrice che conterra tutte le auc ottenute per le diverse fold e diversi set di parametri                 
scoreThsNew=np.zeros(len(x_devs))   #matrice che conterra tutte le threshold ottime ottenute per le diverse fold e diversi set di parametri                 
f=0;
    
net=autoencoder.autoencoder_fall_detection( [3,3], [16, 8, 8], args.fit_net);
net.define_arch();                                         
#parametri di defautl anche per compile e fit
net.model_compile(optimizer=args.optimizer, loss=args.loss)
model=net.model_fit(x_trains[0], _ , nb_epoch=args.epoch, batch_size=args.batch_size, shuffle=args.shuffle) 

for x_dev, y_dev in zip (x_devs, y_devs): #sarebbero le fold

    decoded_images = net.reconstruct_spectrogram(x_dev);  
    auc, optimal_th, _, _, _ = net.compute_score(x_dev, decoded_images, y_dev);
    scoreAucNew[f]=auc;
    scoreThsNew[f]=optimal_th
    f+=1;
                                          
                                          
                                          
                                          
#check score and save data

# in questi 2 file ogni riga corrisponde ad una fold
scoreAucFileName='score_auc.txt';
thFileName='thresholds.txt';
scoreCasePath=os.path.join(args.scorePath,args.case);
if not os.path.exists(args.scorePath):#se non esisrte significa che è il primo esperimento
    try:    #quindi creo le cartelle necessarie e salvo i parametri e il modello
            #del primo esperimento ( che sarà sicuramentoe il migliore fino ad ora: è il primo!!)
            
        os.makedirs(os.path.join(scoreCasePath))
        os.makedirs(os.path.join(scoreCasePath,'args'))
        os.makedirs(os.path.join(scoreCasePath,'models'))
        for fold in np.arange(1,len(args.devNamesLists)+1):
            jsonargs=json.dumps(args.__dict__)
            with open(os.path.join(scoreCasePath,'args','argsFold'+str(fold)+'.txt'), 'w') as file:
                file.write(jsonargs);
            net.save_model(model,os.path.join(scoreCasePath,'models'),'modelFold'+str(fold));

        print("make dir")
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

if os.path.exists(os.path.join(scoreCasePath,scoreAucFileName)):
    print("loadtxt")
    scoreAuc=np.loadtxt(os.path.join(scoreCasePath,scoreAucFileName))
    scoreThs=np.loadtxt(os.path.join(scoreCasePath,thFileName))
  
    for auc, oldAuc, idx in zip(scoreAucNew, scoreAuc, enumerate(scoreAuc)):
        if auc > oldAuc:#se in una fold ho ottenuto una auc migliore rispetto ad un esperimento precedente
                        #allora sostituisco i valori di quella fold (ovvero una riga) con i nuovi: lo faccio sia per le auc
                        #che per la threshold ottime, i parametri usati e il modello adattato.
            #per le auc e le th uso dei file singoli (ogni riga una fold) per comodità
            scoreAucNew[idx[0]]=auc;
            scoreThsNew[idx[0]]=scoreThs[idx[0]];
            #per args e model uso file separati per ogni fold
            #salvo parametri
            jsonargs=json.dumps(args.__dict__)
            with open(os.path.join(scoreCasePath,'args','argsFold'+str(idx[0]+1)+'.txt'), 'w') as file:
                file.write(jsonargs);
            #salvo modello e pesi
                    
print("savetxt")
np.savetxt(os.path.join(scoreCasePath, scoreAucFileName),scoreAucNew)
np.savetxt(os.path.join(scoreCasePath, thFileName),scoreThsNew)

                                          
                                          
                                          
                                          
                                          
    
#TODO Spostare test nel file apposito che verra eseguito dopo la cross validation
#test-finale-------------------------------VA SPOSTATO NEL FILE APPOSITO
print("------------------------TEST---------------")
idx=0;
my_cm=np.zeros((2,2));
old_my_cm=np.zeros((2,2));#matrice d'appoggio
sk_cm=np.zeros((2,2));
tot_y_pred=[];
tot_y_true=[];
for x_test, y_test in zip (x_tests, y_tests):
    
    #in realtà questo fit non serve più: va caricato il modello fittato nella validation!!!
    net.model_compile();
    net.model_fit(x_trains[0], _ );
                 
    decoded_images = net.reconstruct_spectrogram(x_test);  
    auc,_ , my_cm, y_true , y_pred = net.compute_score(x_test, decoded_images, y_test);
    #raccolto tutti i risultati delle fold, per poter fare un report generale
    for x in y_pred:
        tot_y_pred.append(x);
    for x in y_true:
        tot_y_true.append(x);                                             
    my_cm = np.add(old_my_cm,my_cm); 
    old_my_cm=my_cm;
    idx+=1;
    
    
#report finale
print('\n\n\n')
print("------------------------FINAL REPORT---------------")

net.print_score(my_cm,tot_y_pred,tot_y_true);


               
        
    
