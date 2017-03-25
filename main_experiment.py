#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:11:09 2017

@author: buckler
"""
import numpy as np

np.random.seed(888)  # for experiment repetibility: this goes here, before importing keras (inside autoencoder modele) It works?
import autoencoder
import dataset_manupulation as dm

from os import path
import argparse
import os
import errno
import json
import fcntl
import time
import datetime
from copy import deepcopy
import utility as u


###################################################PARSER ARGUMENT SECTION########################################
print('parser')
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
parser.add_argument("-rp", "--root-path", dest="root_path", default=None)
parser.add_argument("-id", "--exp-index", dest="id", default=0, type=int)
parser.add_argument("-log", "--logging", dest="log", default=False, action="store_true")
parser.add_argument("-sifg", "--save-Img-For-Gif", dest="saveImgForGif", default=False, action="store_true")

parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)
#parser.add_argument("-sp", "--score-path", dest="scorePath", default="score") #non serve più
parser.add_argument("-tl", "--trainset-list", dest="trainNameLists", action=eval_action, default=["trainset.lst"])
parser.add_argument("-c", "--case", dest="case", default="case6")
parser.add_argument("-tln", "--test-list-names", dest="testNamesLists", action=eval_action,
                    default=["testset_1.lst", "testset_2.lst", "testset_3.lst", "testset_4.lst"])
parser.add_argument("-dl", "--dev-list-names", dest="devNamesLists", action=eval_action,
                    default=["devset_1.lst", "devset_2.lst", "devset_3.lst", "devset_4.lst"])
parser.add_argument("-it", "--input-type", dest="input_type", default="spectrograms")

# CNN params
parser.add_argument("-is", "--cnn-input-shape", dest="cnn_input_shape", action=eval_action, default=[1, 129, 197])
parser.add_argument("-cln", "--conv-layers-numb", dest="conv_layer_numb", default=3, type=int)
parser.add_argument("-kn", "--kernels-number", dest="kernel_number", action=eval_action, default=[16, 8, 8])
parser.add_argument("-ks", "--kernel-shape", dest="kernel_shape", action=eval_action, default=[[3, 3], [3, 3], [3, 3]])
parser.add_argument("-mp", "--max-pool-shape", dest="m_pool", action=eval_action, default=[[2, 2], [2, 2], [2, 2]])
parser.add_argument("-s", "--strides", dest="strides", action=eval_action, default=[[1, 1], [1, 1], [1, 1]])
parser.add_argument("-cwr", "--cnn-w-reg", dest="cnn_w_reg",
                    default="None")  # in autoencoder va usato con eval("funz(parametri)")
parser.add_argument("-cbr", "--cnn-b-reg", dest="cnn_b_reg", default="None")
parser.add_argument("-car", "--cnn-act-reg", dest="cnn_a_reg", default="None")
parser.add_argument("-cwc", "--cnn-w-constr", dest="cnn_w_constr", default="None")
parser.add_argument("-cbc", "--cnn-b-constr", dest="cnn_b_constr", default="None")
parser.add_argument("-ac", "--cnn-conv-activation", dest="cnn_conv_activation", default="tanh", choices=["tanh"])

#Dense params
parser.add_argument("-dln", "--dense-layers-numb", dest="dense_layer_numb", default=1, type=int)
parser.add_argument("-ds", "--dense-shapes", dest="dense_shapes", action=eval_action, default=[64])
parser.add_argument("-i", "--cnn-init", dest="cnn_init", default="glorot_uniform", choices=["glorot_uniform"])
parser.add_argument("-ad", "--cnn-dense-activation", dest="cnn_dense_activation", default="tanh", choices=["tanh"])
parser.add_argument("-bm", "--border-mode", dest="border_mode", default="same", choices=["valid", "same"])
parser.add_argument("-dwr", "--d-w-reg", dest="d_w_reg",
                    default="None")  # in autoencoder va usato con eval("funz(parametri)")
parser.add_argument("-dbr", "--d-b-reg", dest="d_b_reg", default="None")
parser.add_argument("-dar", "--d-act-reg", dest="d_a_reg", default="None")
parser.add_argument("-dwc", "--d-w-constr", dest="d_w_constr", default="None")
parser.add_argument("-dbc", "--d-b-constr", dest="d_b_constr", default="None")
parser.add_argument("-drp", "--dropout", dest="dropout", default=False, action="store_true")
parser.add_argument("-drpr", "--drop-rate", dest="drop_rate", default=0.5, type=float)

parser.add_argument("-nb", "--no-bias", dest="bias", default=True, action="store_false")
parser.add_argument("-p", "--pool-type", dest="pool_type", default="all", choices=["all", "only_end"])

# fit params
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--no-shuffle", dest="shuffle", default=True, action="store_false")
parser.add_argument("-bs", "--batch-size-fract", dest="batch_size_fract", default=0.1, type=float)
parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-o", "--optimizer", dest="optimizer", default="adadelta", choices=["adadelta", "adam", "sgd"])
parser.add_argument("-l", "--loss", dest="loss", default="mse", choices=["mse", "msle"])
parser.add_argument("-pt", "--patiance", dest="patiance", default=20, type=int)
parser.add_argument("-ami", "--aucMinImp", dest="aucMinImprovment", default=0.01, type=float)
parser.add_argument("-lr", "--learning-rate", dest="learning_rate", default=1.0, type=float)

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

try:


    ###################################################INIT LOG########################################
    # redirect all the stream of both standar.out, standard.err to the same logger
    strID = str(args.id)
    if args.root_path is None:
        args.root_path = os.path.realpath(".")
    print("init log")

    allResultBasePath = os.path.join(args.root_path,'resluts',args.case)
    nameFileLogCsv = None  # init the name
    logFolder = os.path.join(allResultBasePath, 'logs')  # need also for saving csv file!
    u.makedir(logFolder)
    nameFileLog = os.path.join(logFolder, 'process_' + strID + '.log')
    if args.log:
        import logging
        import sys
        if os.path.isfile(nameFileLog):  # if there is a old log, save it with another name
            fileInFolder = [x for x in os.listdir(logFolder) if x.startswith('process_')]
            os.rename(nameFileLog, nameFileLog + '_old_' + str(len(fileInFolder) + 1))  # so the name is different
            # rename also the csv log for the losses
            # if os.path.isfile(nameFileLogCsv):  # if there is a old log, save it with another name
            #     os.rename(nameFileLogCsv, nameFileLogCsv + '_' + str(len(fileInFolder) + 1))  # so the name is different

        stdout_logger = logging.getLogger(strID)
        sl = u.StreamToLogger(stdout_logger, nameFileLog, logging.INFO)
        sys.stdout = sl  # ovverride funcion

        stderr_logger = logging.getLogger(strID)
        sl = u.StreamToLogger(stderr_logger, nameFileLog, logging.ERROR)
        sys.stderr = sl  # ovverride funcion
    ###################################################END INIT LOG########################################

    print("LOG OF PROCESS ID = " + strID)
    ts0 = time.time()
    st0 = datetime.datetime.fromtimestamp(ts0).strftime('%Y-%m-%d %H:%M:%S')
    print("experiment start in date: " + st0)

    ######################################CHECK SCORE FOLDER STRUCTURE############################################
    # check the score folder structure #TODO PORTARE IN UN FILE ESTERNO CHE PREPARE TUTTO? ALTRIMENTI SE LO FACCIAMO QUI, SI
    # POTREBBERO CREARE PROBLEMI DI ACCESSO TRA I VARI PROCESSI

    # in questi 2 file ogni riga corrisponde ad una fold
    scoreAucsFileName = 'score_auc.txt'
    thFileName = 'thresholds.txt'

    BestScorePath = os.path.join(allResultBasePath, 'bestResults')
    scoreAucsFilePath = os.path.join(BestScorePath, scoreAucsFileName)
    scoreThsFilePath = os.path.join(BestScorePath, thFileName)
    argsFolder = 'args'
    modelFolder = 'models'
    argsPath = os.path.join(BestScorePath, argsFolder)
    modelPath = os.path.join(BestScorePath, modelFolder)
    jsonargs = json.dumps(args.__dict__)

    if args.id == 0: #if is the first process init the best score folder and file
        u.makedir(argsPath)
        u.makedir(modelPath)
        u.makedir(BestScorePath)

        print("init scoreFile")
        np.savetxt(scoreAucsFilePath, np.zeros(len(args.testNamesLists)))
        np.savetxt(scoreThsFilePath, np.zeros(len(args.testNamesLists)))

        # # TODO in realtà questo controllo non scansiona se mancano i modelli o/e i parametri
        # # se la cartella già esiste devo verificare la consistenza dei file all'interno
        # elif not set([scoreAucsFileName, thFileName, argsFolder, modelFolder]).issubset(set(os.listdir(BestScorePath))):
        #     message = 'Score fold inconsistency detected. Check if all the file are present in ' + BestScorePath + '. Process aborted'
        #     print(message)
        #
        #     raise Exception(message)

    ######################################END CHECK SCORE FOLDER STRUCTURE############################################



    listTrainpath = path.join(args.root_path, 'lists', 'train')
    listPath = path.join(args.root_path, 'lists', 'dev+test', args.case)

    # Manage DATASET
    a3fall = dm.load_A3FALL(path.join(args.root_path, 'dataset', args.input_type))  # load dataset

    # il trainset è 1 e sempre lo stesso per tutti gli esperimenti
    trainset = dm.split_A3FALL_from_lists(a3fall, listTrainpath, args.trainNameLists)[0]  # need a traiset in order to compute the mean and variance.
    # Then use this mea and variance for normalize all the dataset

    trainset, mean, std = dm.normalize_data(trainset)  # compute mean and std of the trainset and normalize the trainset
    # calcolo il batch size
    batch_size = int(len(trainset)*args.batch_size_fract)

    a3fall_n, _, _ = dm.normalize_data(a3fall, mean, std)  # normalize the dataset with the mean and std of the trainset
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
    # TODO sistemare nomi
    scoreAucNew = np.zeros(len(
        args.testNamesLists))  # matrice che conterra tutte le auc ottenute per le diverse fold e diversi set di parametri
    scoreThsNew = np.zeros(len(
        args.testNamesLists))  # matrice che conterra tutte le threshold ottime ottenute per le diverse fold e diversi set di parametri


    models = list()
    f = 0
    for x_dev, y_dev in zip(x_devs, y_devs):  # sarebbero le fold
        print('\n\n\n----------------------------------FOLD {}-----------------------------------'.format(f + 1))
        logCsvFolder = os.path.join(allResultBasePath, 'logscsv', 'fold_'+str(f+1))  # need also for saving csv file!
        u.makedir(logCsvFolder)
        if args.saveImgForGif is True:
            imgForGifPath = os.path.join(allResultBasePath, 'ImgForGif', 'fold_'+str(f+1),'process_'+strID)
            u.makedir(imgForGifPath)
        else:
            imgForGifPath = None
        # Need to redefine the same architecture and compile it for each fold.
        # If you do the net does't start fit from the beginnig at the second fold
        net = autoencoder.autoencoder_fall_detection(str(args.id), args.case, str(f + 1))
        # net.define_static_arch()
        net.define_cnn_arch(args)
        prefitted_model = net.model_compile(optimizer=args.optimizer, loss=args.loss, learning_rate=args.learning_rate)
        # print('\n\n\n----------------------------------PRefitted-----------------------------------')
        #
        # decoded_images = net.reconstruct_spectrogram(x_dev, prefitted_model)
        # auc, optimal_th, _, _, _ = autoencoder.compute_score(x_dev, decoded_images, y_dev)
        # print('auc prefitted: ' + str(auc))
        # pathSaveFigName = os.path.join('imgForGif', 'prefitted-img-1'+str(f+1)+'.png')
        # autoencoder.plot_decoded_img(y_dev[11], x_dev[11], decoded_images[11], pathSaveFigName)
        # print('\n\n\n----------------------------------end PRefitted-----------------------------------')

        # L'eralystopping viene fatto in automatico se vengono passati anche x_dev e y_dev
        m = net.model_fit(x_trains[0], y_trains[0], x_dev=x_dev, y_dev=y_dev, nb_epoch=args.epoch,
                          batch_size=batch_size, shuffle=args.shuffle, model=prefitted_model,
                          fit_net=args.fit_net, patiance=args.patiance, aucMinImprovment=args.aucMinImprovment,
                          pathFileLogCsv=logCsvFolder, imgForGifPath=imgForGifPath)
        models.append(m)
        decoded_images = net.reconstruct_spectrogram(x_dev)
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
                    # serve tutto questo giro
                    break
                except IOError as e:
                    # raise on unrelated IOErrors
                    if e.errno != errno.EAGAIN:
                        # print('ERROR occured trying acquuire file')
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
                    with open(os.path.join(argsPath, 'argsfold' + str(foldsIdx[0] + 1) + '.json'), 'w') as file:
                        file.write(json.dumps(jsonargs, indent=4))
                    # salvo modello e pesi
                    net.save_model(models[foldsIdx[0]], modelPath, 'modelfold' + str(foldsIdx[0] + 1))

            print("savetxt")
            np.savetxt(scoreAucsFilePath, scoreAucNew)
            np.savetxt(scoreThsFilePath, scoreThs)
        finally:
            print("file UnLock")
            fcntl.flock(fileToLock, fcntl.LOCK_UN)
    print("------------------------END CROSS VALIDATION---------------")
    print("------------------------Cross Validation Summary---------------")
    f = 0
    for auc in scoreAucNew:
        print('Fold_' + str(f + 1) + ' auc :' + str(auc))
        f += 1

    ts1 = time.time()
    st1 = datetime.datetime.fromtimestamp(ts0).strftime('%Y-%m-%d %H:%M:%S')
    print("experiment start in date: " + st1)
    print("Experiment time (DAYS:HOURS:MIN:SEC):" + u.GetTime(ts1 - ts0))

    if args.log is True:
        u.logcleaner(nameFileLog)  # remove garbage character from log file

    print('DONE')

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

except:
    with open('Status_Pocesses_Report.txt', 'a') as statusFile:
        statusFile.write('process_'+str(args.id)+' ERROR')


