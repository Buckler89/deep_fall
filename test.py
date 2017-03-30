#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:13:28 2017

@author: buckler
"""

import numpy as np
from os import path
import dataset_manupulation as dm
from keras.models import Model, load_model
import autoencoder
import utility as u
import sys
import logging


trainNameLists = ['trainset.lst']
input_type = 'spectrograms'
case = 'case6'
log=False

# devNamesLists=['devset_1.lst','devset_2.lst','devset_3.lst','devset_4.lst']
testNamesLists = ['testset_1.lst', 'testset_2.lst', 'testset_3.lst', 'testset_4.lst']
root_dir = path.realpath('.')

listTrainpath = path.join(root_dir, 'lists', 'train')
listPath = path.join(root_dir, 'lists', 'dev+test', case)

# scoreAucFileName = 'score_auc.txt'
# thFileName = 'thresholds.txt'
argsFolder = 'args'
modelFolder = 'models'
resultsPath=path.join(root_dir, 'results', case)
BestScorePath = path.join(resultsPath,'bestResults')
scoreCasePath = path.join(BestScorePath)
scoreModelPath = path.join(BestScorePath, modelFolder)
scoreArgsPath = path.join(BestScorePath,  argsFolder)
nameFileLog = path.join(scoreCasePath, 'test_'+case+'.log')

#LOGGER
if log is True:
    stdout_logger = logging.getLogger(case)
    sl = u.StreamToLogger(stdout_logger, nameFileLog, logging.INFO)
    sys.stdout = sl  # ovverride funcion

    stderr_logger = logging.getLogger(case)
    sl = u.StreamToLogger(stderr_logger, nameFileLog, logging.ERROR)
    sys.stderr = sl  # ovverride funcion
#----------
# GESTIONE DATASET
a3fall = dm.load_A3FALL(path.join(root_dir, 'dataset', input_type))  # load dataset

# il trainset è 1 e sempre lo stesso per tutti gli esperimenti
trainset = dm.split_A3FALL_from_lists(a3fall, listTrainpath, trainNameLists)[0]  # creo i trainset per calcolare media
                                                                                # e varianza per poter normalizzare
trainset, mean, std = dm.normalize_data(trainset)  # compute mean and std of the trainset and normalize the trainset

a3fall_n, _, _ = dm.normalize_data(a3fall, mean, std)  # ormalize the dataset with the mean and std of the trainset
a3fall_n_z = dm.awgn_padding_set(a3fall_n)

# creo i set partendo dal dataset normalizzato e paddato
trainsets = dm.split_A3FALL_from_lists(a3fall_n_z, listTrainpath, trainNameLists)
# devsets = dm.split_A3FALL_from_lists(a3fall_n_z,listPath,devNamesLists)
testsets = dm.split_A3FALL_from_lists(a3fall_n_z, listPath, testNamesLists)

# reshape dataset per darli in ingresso alla rete

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
# for s in devsets:
#    x, y = dm.reshape_set(s)
#    x_devs.append(x)
#    y_devs.append(y)
for s in testsets:
    x, y = dm.reshape_set(s)
    x_tests.append(x)
    y_tests.append(y)

print("------------------------TEST---------------")
fold = 1
my_cm = np.zeros((2, 2))
old_my_cm = np.zeros((2, 2))  # matrice d'appoggio
sk_cm = np.zeros((2, 2))
tot_y_pred = []
tot_y_true = []
for x_test, y_test in zip(x_tests, y_tests):
    print('FOLD'+str(fold)+'-----------------------------------')
    # caricare modello
    net = autoencoder.autoencoder_fall_detection(_,case,fold);
    modelPath = path.join(scoreModelPath, 'modelfold' + str(fold) + '.h5')
    net.load_model(modelPath)


    decoded_images = net.reconstruct_spectrogram(x_test)
    auc, _, my_cm, y_true, y_pred = autoencoder.compute_score(x_test, decoded_images, y_test)
    # raccolto tutti i risultati delle fold, per poter fare un report generale
    for x in y_pred:
        tot_y_pred.append(x)
    for x in y_true:
        tot_y_true.append(x)
    my_cm = np.add(old_my_cm, my_cm)
    old_my_cm = my_cm
    fold += 1

# report finale
print('\n\n\n')
print("------------------------FINAL REPORT---------------")

autoencoder.print_score(my_cm, tot_y_pred, tot_y_true)
