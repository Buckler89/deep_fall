#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import numpy as np

np.random.seed(888)  # for experiment repetibility: this goes here, before importing keras (inside autoencoder modele)
# from py_files import autoencoder
# from py_files import dataset_manupulation as dm
import autoencoder
import dataset_manupulation as dm

from os import path
import argparse
import os
import errno
import json
import fcntl
import time

import utility as u


###################################################PARSER ARGUMENT SECTION########################################
parser = argparse.ArgumentParser(description="Novelty Deep Fall Detection")

class eval_action(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(eval_action, self).__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        values = eval(values)
        setattr(namespace, self.dest, values)

# Global params
parser.add_argument("-id", "--exp-index", dest="id", default=0, type=int)
parser.add_argument("-log", "--logging", dest="log", default=False, action="store_true")

parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
parser.add_argument("-sp", "--score-path", dest="scorePath", default="score")
parser.add_argument("-tl", "--trainset-list", dest="trainNameLists", action=eval_action, default=["trainset.lst"])
parser.add_argument("-c", "--case", dest="case", default="case6")
parser.add_argument("-tln", "--test-list-names", dest="testNamesLists", action=eval_action,
                    default=["testset_1.lst", "testset_2.lst", "testset_3.lst", "testset_4.lst"])
parser.add_argument("-dl", "--dev-list-names", dest="devNamesLists", action=eval_action,
                    default=["devset_1.lst", "devset_2.lst", "devset_3.lst", "devset_4.lst"])
parser.add_argument("-it", "--input-type", dest="input_type", default="spectrograms")

# CNN params
parser.add_argument("-cln", "--conv-layers-numb", dest="conv_layer_numb", default=3, type=int)
parser.add_argument("-is", "--cnn-input-shape", dest="cnn_input_shape", action=eval_action, default=[1, 129, 197])
parser.add_argument("-kn", "--kernels-number", dest="kernel_number", action=eval_action, default=[16, 8, 8])
parser.add_argument("-ks", "--kernel-shape", dest="kernel_shape", action=eval_action, default=[[3, 3],[3, 3],[3, 3]])
parser.add_argument("-mp", "--max-pool-shape", dest="m_pool", action=eval_action, default=[[2, 2],[2, 2],[2, 2]])
parser.add_argument("-s", "--strides", dest="strides", action=eval_action, default=[[1, 1],[1, 1],[1, 1]])

parser.add_argument("-dln", "--dense-layers-numb", dest="dense_layer_numb", default=1, type=int)
parser.add_argument("-ds", "--dense-shapes", dest="dense_layers_inputs", action=eval_action, default=[64])
parser.add_argument("-i", "--cnn-init", dest="cnn_init", default="glorot_uniform", choices=["glorot_uniform"])
parser.add_argument("-ac", "--cnn-conv-activation", dest="cnn_conv_activation", default="tanh", choices=["tanh"])
parser.add_argument("-ad", "--cnn-dense-activation", dest="cnn_dense_activation", default="tanh", choices=["tanh"])
parser.add_argument("-bm", "--border-mode", dest="border_mode", default="same", choices=["valid", "same"])
parser.add_argument("-wr", "--w-reg", dest="w_reg", default=None) # in autoencoder va usato con eval("funz(parametri)")
parser.add_argument("-br", "--b-reg", dest="b_reg", default=None)
parser.add_argument("-ar", "--act-reg", dest="a_reg", default=None)
parser.add_argument("-wc", "--w-constr", dest="w_constr", default=None)
parser.add_argument("-bc", "--b-constr", dest="b_constr", default=None)
parser.add_argument("-nb", "--no-bias", dest="bias", default=True, action="store_false")
parser.add_argument("-p", "--pool-type", dest="pool_type", default="all", choices=["all", "only_end"])

# fit params
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--no-shuffle", dest="shuffle", default=True, action="store_false")
parser.add_argument("-bs", "--batch-size", dest="batch_size", default=128, type=int)
parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-o", "--optimizer", dest="optimizer", default="adadelta", choices=["adadelta", "adam", "sgd"])
parser.add_argument("-l", "--loss", dest="loss", default="mse", choices=["mse"])

args = parser.parse_args()

if args.config_filename is not None:
    with open(args.config_filename, "r") as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        arguments.extend(line.split("#")[0].split())
    # First parse the arguments specified in the config file
    args, unknown = parser.parse_known_args(args=arguments)
    # Then append the command line arguments
    # Command line arguments have the priority: an argument is specified both
    # in the config file and in the command line, the latter is used
    args = parser.parse_args(namespace=args)

###################################################END PARSER ARGUMENT SECTION########################################



###################################################INIT LOG########################################
#redirect all the stream of both standar.out, standard.err to the same logger
strID = str(args.id)

if args.log:
    import logging
    import sys

    logFolder = 'logs'
    nameFileLog = os.path.join(logFolder, 'process_' + strID + '.log')
    u.makedir(logFolder)  # crea la fold solo se non esiste
    if os.path.isfile(nameFileLog):  # if there is a old log, save it with another name
        fileInFolder = [x for x in os.listdir(logFolder) if x.startswith('process_')]
        os.rename(nameFileLog, nameFileLog + '_' + str(len(fileInFolder) + 1))  # so the name is different

    stdout_logger = logging.getLogger(strID)
    sl = u.StreamToLogger(stdout_logger, nameFileLog, logging.INFO)
    sys.stdout = sl  #ovverride funcion

    stderr_logger = logging.getLogger(strID)
    sl = u.StreamToLogger(stderr_logger, nameFileLog, logging.ERROR)
    sys.stderr = sl #ovverride funcion
print("LOG OF PROCESS ID = "+strID)

###################################################END INIT LOG########################################


######################################CHECK SCORE FOLDER STRUCTURE############################################
# check the score folder structure #TODO PORTARE IN N FILE ESTERNO CHE PREPARE TUTTO? ALTRIMENTI SE LO FACCIAMO QUI, SI
# POTREBBERO CREARE PROBLEMI DI ACCESSO TRA I VARI PROCESSI

# in questi 2 file ogni riga corrisponde ad una fold
scoreAucsFileName = 'score_auc.txt'
thFileName = 'thresholds.txt'

scoreCasePath = os.path.join(args.scorePath, args.case)
scoreAucsFilePath = os.path.join(scoreCasePath, scoreAucsFileName)
scoreThsFilePath = os.path.join(scoreCasePath, thFileName)
argsFolder = 'args'
modelFolder = 'models'
argsPath = os.path.join(scoreCasePath, argsFolder)
modelPath = os.path.join(scoreCasePath, modelFolder)
jsonargs = json.dumps(args.__dict__)

if not os.path.exists(scoreCasePath):
    u.makedir(scoreCasePath)
    u.makedir(argsPath)
    u.makedir(modelPath)
    np.savetxt(scoreAucsFilePath, np.zeros(len(args.testNamesLists)))
    np.savetxt(scoreThsFilePath, np.zeros(len(args.testNamesLists)))
elif os.listdir(scoreCasePath) == []:  # se è vuota significa che è il primo esperimento
    # quindi creo le cartelle necessarie e salvo un file delle auc e th inizializzato a 0
    print("make arg and model dir and init scoreFile")
    u.makedir(argsPath)
    u.makedir(modelPath)
    np.savetxt(scoreAucsFilePath, np.zeros(len(args.testNamesLists)))
    np.savetxt(scoreThsFilePath, np.zeros(len(args.testNamesLists)))

# TODO in realtà questo controllo non scansiona se mancano i modelli o/e i parametri
# se la cartella già esiste devo verificare la consistenza dei file all'interno
elif not set([scoreAucsFileName, thFileName, argsFolder, modelFolder]).issubset(set(os.listdir(scoreCasePath))):
    message='Score fold inconsistency detected. Check if all the file are present in ' + scoreCasePath + '. Process aborted'
    #print(message)
    stderr_logger.error(message)

    raise Exception(message)

######################################END CHECK SCORE FOLDER STRUCTURE############################################


root_dir = path.realpath('.')

listTrainpath = path.join(root_dir, 'lists', 'train')
listPath = path.join(root_dir, 'lists', 'dev+test', args.case)

# GESTIONE DATASET
a3fall = dm.load_A3FALL(path.join(root_dir, 'dataset', args.input_type))  # load dataset

# il trainset è 1 e sempre lo stesso per tutti gli esperimenti
trainset = dm.split_A3FALL_from_lists(a3fall, listTrainpath, args.trainNameLists)[0]  # creo i trainset per calcolare
# media e varianza per poter normalizzare
trainset, mean, std = dm.normalize_data(trainset)  # compute mean and std of the trainset and normalize the trainset

a3fall_n, _, _ = dm.normalize_data(a3fall, mean, std)  # ormalize the dataset with the mean and std of the trainset
a3fall_n_z = dm.awgn_padding_set(a3fall_n)

# creo i set partendo dal dataset normalizzato e paddato
trainsets = dm.split_A3FALL_from_lists(a3fall_n_z, listTrainpath, args.trainNameLists)
devsets = dm.split_A3FALL_from_lists(a3fall_n_z, listPath, args.devNamesLists)
testsets = dm.split_A3FALL_from_lists(a3fall_n_z, listPath, args.testNamesLists)

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
for s in devsets:
    x, y = dm.reshape_set(s)
    x_devs.append(x)
    y_devs.append(y)
for s in testsets:
    x, y = dm.reshape_set(s)
    x_tests.append(x)
    y_tests.append(y)

# CROSS VALIDATION
print("------------------------CROSS VALIDATION---------------")

# init score matrix
#TODO sistemare nomi
scoreAucNew = np.zeros(len(
    args.testNamesLists))  # matrice che conterra tutte le auc ottenute per le diverse fold e diversi set di parametri
scoreThsNew = np.zeros(len(
    args.testNamesLists))  # matrice che conterra tutte le threshold ottime ottenute per le diverse fold e diversi set di parametri
f = 0
net = autoencoder.autoencoder_fall_detection()
# net.define_static_arch()
net.define_cnn_arch(args)
# parametri di defautl anche per compile e fit
models = list()

for x_dev, y_dev in zip(x_devs, y_devs):  # sarebbero le fold
    print('\n\n\n----------------------------------FOLD {}-----------------------------------'.format(f))
    net.model_compile(optimizer=args.optimizer, loss=args.loss)
    #L'eralysstopping viene fatto in automatico se vengono passati anche x_dev e y_dev

    m = net.model_fit(x_trains[0], _, x_dev=x_dev, y_dev=y_dev, nb_epoch=args.epoch, batch_size=args.batch_size, shuffle=args.shuffle,
                      fit_net=args.fit_net)
    models.append(m)
    decoded_images = net.reconstruct_spectrogram(x_dev, m)
    auc, optimal_th, _, _, _ = autoencoder.compute_score(x_dev, decoded_images, y_dev)
    scoreAucNew[f] = auc
    scoreThsNew[f] = optimal_th
    f += 1

print("------------------------SCORE SELECTION---------------")

# check score and save data
if os.path.exists(scoreAucsFilePath):  # sarà presumibilmente sempre vero perche viene creata precedentemente
    try:
        print("open File to lock")
        fileToLock = open(scoreAucsFilePath, 'a+')  # se metto w+ mi cancella il vecchio!!!
    except OSError as exception:
        stderr_logger.error(exception)
        raise
    # prova a bloccare il file: se non riesce ritenta dopo un po. Non va avanti finche non riesce a bloccare il file
    try:
        while True:
            try:
                print("file Lock")
                fcntl.flock(fileToLock,
                            fcntl.LOCK_EX | fcntl.LOCK_NB)  # NOTA BENE: file locks on Unix are advisory only:ecco perche
                                                            #serve tutto questo giro
                break
            except IOError as e:
                # raise on unrelated IOErrors
                if e.errno != errno.EAGAIN:
                    #print('ERROR occured trying acquuire file')
                    stderr_logger.error('ERROR occured trying acquuire file')
                    stderr_logger.error(e)
                    raise
                else:
                    print("wait fo file to Lock")
                    time.sleep(0.1)
        print("loadtxt")
        scoreAuc = np.loadtxt(scoreAucsFilePath)
        scoreThs = np.loadtxt(scoreThsFilePath)
        print('check if new best score is achieved')
        for auc, oldAuc, foldsIdx in zip(scoreAucNew, scoreAuc, enumerate(scoreAuc)):
            if auc > oldAuc:  # se in una fold ho ottenuto una auc migliore rispetto ad un esperimento precedente
                # allora sostituisco i valori di quella fold (ovvero una riga) con i nuovi: lo faccio sia per le auc
                # che per la threshold ottime, i parametri usati e il modello adattato.
                # per le auc e le th uso dei file singoli (ogni riga una fold) per comodità
                scoreAucNew[foldsIdx[0]] = auc
                scoreThs[foldsIdx[0]] = scoreThsNew[foldsIdx[0]]
                # per args e model uso file separati per ogni fold
                # salvo i parametri
                with open(os.path.join(argsPath, 'argsfold' + str(foldsIdx[0] + 1) + '.txt'), 'w') as file:
                    file.write(json.dumps(jsonargs, indent=4, sort_keys=True))
                # salvo modello e pesi
                net.save_model(models[foldsIdx[0]], modelPath, 'modelfold' + str(foldsIdx[0] + 1))

        print("savetxt")
        np.savetxt(scoreAucsFilePath, scoreAucNew)
        np.savetxt(scoreThsFilePath, scoreThs)
    finally:
        print("file UnLock")
        fcntl.flock(fileToLock, fcntl.LOCK_UN)
print("------------------------FINE CROSS VALIDATION---------------")

# # test-finale-------------------------------
# print("------------------------TEST---------------")
# idx = 0
# my_cm = np.zeros((2, 2))
# old_my_cm = np.zeros((2, 2))  # matrice d'appoggio
# sk_cm = np.zeros((2, 2))
# tot_y_pred = []
# tot_y_true = []
# for x_test, y_test in zip(x_tests, y_tests):
#
#     # in realtà questo fit non serve più: va caricato il modello fittato nella validation!!!
#     net.model_compile()
#     net.model_fit(x_trains[0], _)
#
#     decoded_images = net.reconstruct_spectrogram(x_test)
#     auc, _, my_cm, y_true, y_pred = net.compute_score(x_test, decoded_images, y_test)
#     # raccolto tutti i risultati delle fold, per poter fare un report generale
#     for x in y_pred:
#         tot_y_pred.append(x)
#     for x in y_true:
#         tot_y_true.append(x)
#     my_cm = np.add(old_my_cm, my_cm)
#     old_my_cm = my_cm
#     idx += 1
#
# # report finale
# print('\n\n\n')
# print("------------------------FINAL REPORT---------------")
#
# net.print_score(my_cm, tot_y_pred, tot_y_true)

