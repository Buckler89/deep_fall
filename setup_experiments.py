#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 11:19:08 2017

@author: daniele
"""

import argparse
from scipy.stats import uniform, norm
import numpy as np
import math


parser = argparse.ArgumentParser(description="Novelty Deep Fall Detection")

# Global params
parser.add_argument("-log", '--logging', dest='log', default=False, action='store_true')
parser.add_argument('-ss', '--search-strategy', dest="search_strategy", default="grid", choices=["grid", "random"])
parser.add_argument("-rnd", "--rnd-exp-number", dest="N", default=0, type=int)
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)

parser.add_argument("-sp", "--score-path", dest="scorePath", default="score")
parser.add_argument("-tl", "--trainset-list", dest="trainNameLists", nargs='+', default=['trainset.lst'])
parser.add_argument("-c", "--case", dest="case", default='case6')
parser.add_argument("-tn", "--test-list-names", dest="testNamesLists", nargs='+',
                    default=['testset_1.lst', 'testset_2.lst', 'testset_3.lst', 'testset_4.lst'])
parser.add_argument("-dn", "--dev-list-names", dest="devNamesLists", nargs='+',
                    default=['devset_1.lst', 'devset_2.lst', 'devset_3.lst', 'devset_4.lst'])
parser.add_argument("-it", "--input-type", dest="input_type", default='spectrograms',
                    choices=["spectrograms", "mel_coefficients"])

# CNN params
parser.add_argument('-is', '--cnn-input-shape', dest="cnn_input_shape", nargs='+', default=[1, 129, 197], type=int)
parser.add_argument('-cln', '--conv-layers-numb', dest="conv_layer_numb", nargs='+', default=[3], type=int)
parser.add_argument('-kn', '--kernels-number', dest="kernel_number", nargs='+', default=[16, 8, 8], type=int)
parser.add_argument('-kst', '--kernel-number-type', dest="kernel_number_type", default="any",
                    choices=["decrease", "encrease", "equal", "any"])
parser.add_argument('-ks', '--kernel-shape', dest="kernel_shape", nargs='+', action='append', type=int)
parser.add_argument('-kt', '--kernel-type', dest="kernel_type", default="square", choices=["square", "+cols", "+rows", "any"])
parser.add_argument('-mp', '--max-pool-shape', dest="m_pool", nargs='+', action='append', type=int)
parser.add_argument('-mpt', '--max-pool-type', dest="m_pool_type", default="square", choices=["square", "+cols", "+rows", "any"])
parser.add_argument("-p", "--pool-type", dest="pool_type", nargs='+', default=['all'], choices=["all", "only_end"])

parser.add_argument('-i', '--cnn-init', dest="cnn_init", nargs='+', default=["glorot_uniform"], choices=["glorot_uniform"])
parser.add_argument('-ac', '--cnn-conv-activation', dest="cnn_conv_activation", nargs='+', default=["tanh"], choices=["tanh"])
parser.add_argument('-ad', '--cnn-dense-activation', dest="cnn_dense_activation", nargs='+', default=["tanh"], choices=["tanh"])
parser.add_argument('-bm', '--border-mode', dest="border_mode", default="same", choices=["valid", "same"])
parser.add_argument('-st', '--strides-type', dest="strides_type", default="square", choices=["square", "+cols", "+rows", "any"])
parser.add_argument('-s', '--strides', dest="strides", nargs='+', action='append', type=int)
parser.add_argument('-wr', '--w-reg', dest="w_reg",
                    default=None)  # in autoencoder va usato con eval('funzione(parametri)')
parser.add_argument('-br', '--b-reg', dest="b_reg", default=None)
parser.add_argument('-ar', '--act-reg', dest="a_reg", default=None)
parser.add_argument('-wc', '--w-constr', dest="w_constr", default=None)
parser.add_argument('-bc', '--b-constr', dest="b_constr", default=None)
parser.add_argument("-nb", "--no-bias", dest="bias", default=True, action='store_false')

# dense params
parser.add_argument('-dln', '--dense-layers-numb', dest="dense_layer_numb", default=1, type=int)
parser.add_argument('-ds', '--dense-shape', dest="dense_lys_neurons", nargs='+', default=[64], type=int)
parser.add_argument('-dst', '--dense-shape-type', dest="dense_shape_type", default="any",
                    choices=["decrease", "encrease", "equal", "any"])

# fit params
parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action='store_true')
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--no-shuffle", dest="shuffle", default=True, action='store_false')
parser.add_argument("-bs", "--batch-size", dest="batch_size", nargs='+', default=[128], type=int)
parser.add_argument("-o", "--optimizer", dest="optimizer", nargs='+', default=["adadelta"],
                    choices=["adadelta", "adam", "sgd"])
parser.add_argument("-l", "--loss", dest="loss", nargs='+', default=["mse"], choices=["mse"])

args = parser.parse_args()

if args.config_filename is not None:
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
# special.default values
if args.search_strategy == "grid":
    if args.kernel_shape is None:
        args.kernel_shape = [[3, 3], [3, 3], [3, 3]]
    if args.m_pool is None:
        args.m_pool = [[2, 2], [2, 2], [2, 2]]
    if args.strides is None:
        args.strides = [[1, 1]]
elif args.search_strategy == "random":
    if args.kernel_shape is None:
        args.kernel_shape = [3, 3, 3, 3]
    if args.m_pool is None:
        args.m_pool = [2, 2, 2, 2]
    if args.strides is None:
        args.strides = [1, 1, 1, 1]



############################################ definizione classe contenitore esperimento
class experiment:
    pass


def check_dimension(e): #--------------------------------------------------------------- da verificare utilità con Diego
    if not hasattr(e, 'conv_layer_numb'):
        return False

    h = e.cnn_input_shape[1]
    w = e.cnn_input_shape[2]
    for i in range(e.cnn_layer_numb):
        if e.border_mode == 'same':
            ph = e.kernel_shape[i][0] - 1
            pw = e.kernel_shape[i][1] - 1
        else:
            ph = pw = 0
        h = int((h - e.kernel_shape[i][0] + ph) / e.strides[0]) + 1
        w = int((w - e.kernel_shape[i][1] + pw) / e.strides[1]) + 1

        if e.pool_type[0] == "all":
            # if border=='valid' h=int(h/params.params.m_pool[i][0])
            h = math.ceil(h / e.m_pool[i][0])
            w = math.ceil(w / e.m_pool[i][1])
    if e.pool_type[0] == "only_end":
        # if border=='valid' h=int(h/params.params.m_pool[i][0])
        h = math.ceil(h / e.m_pool[-1][0])
        w = math.ceil(w / e.m_pool[-1][1])

    if h*w <= 9:  #------------------------------------------------------------------------------------- cosa ci va qui?
        return False

    return True


############################################ def grid_search:
def grid_search(args):
    exp_list = []









    return exp_list



def check_shape_tie(rows, cols, tie_type):
    if rows == cols and tie_type == "square" or \
        rows < cols and tie_type == "+cols" or \
        rows > cols and tie_type == "+rows" or \
        tie_type == "any":
        return True
    return False


def check_num_tie(x, tie, tie_type):
    if x == tie and tie_type == "equal" or \
        x > tie and tie_type == "decrease" or \
        x < tie and tie_type == "encrease" or \
        tie_type == "any":
        return True
    return False


def random_search(args):
    exp_list = []
    for n in range(args.N):
        e = experiment()
        e.id = n
        e.cnn_input_shape = args.cnn_input_shape

        while check_dimension(e):

            ################################################################################### Convolutional layers
            cnn_layer_numb = np.random.choice(range(args.conv_layer_numb)) + 1
            e.conv_layer_numb = cnn_layer_numb

            rows = int(np.round(uniform.rvs(args.kernel_shape[0], args.kernel_shape[1]-args.kernel_shape[0])))
            cols = int(np.round(uniform.rvs(args.kernel_shape[2], args.kernel_shape[3]-args.kernel_shape[2])))
            while check_shape_tie(rows, cols, args.kernel_type):
                rows = int(np.round(uniform.rvs(args.kernel_shape[0], args.kernel_shape[1]-args.kernel_shape[0])))
                cols = int(np.round(uniform.rvs(args.kernel_shape[2], args.kernel_shape[3]-args.kernel_shape[2])))
            # for now all kernel of the same shape
            e.kernel_shape = [[rows, cols]]*3

            e.kernel_number = [int(np.round(uniform.rvs(args.kernel_number[0], args.kernel_number[1] - args.kernel_number[0])))]
            for i in range(cnn_layer_numb-1):
                c = int(np.round(uniform.rvs(args.kernel_number[0], args.kernel_number[1] - args.kernel_number[0])))
                while check_num_tie(e.kernel_number[i-1], c, args.kernel_number_type):
                    c = int(np.round(uniform.rvs(args.kernel_number[0], args.kernel_number[1] - args.kernel_number[0])))
                e.kernel_number.append(c)

            e.cnn_init = np.random.choice(args.cnn_init)
            e.cnn_conv_activation = np.random.choice(args.cnn_conv_activation)
            e.cnn_dense_activation = np.random.choice(args.cnn_dense_activation)
            e.border_mode = np.random.choice(args.border_mode)

            # strides
            rows = int(np.round(uniform.rvs(args.strides[0], args.strides[1]-args.strides[0])))
            cols = int(np.round(uniform.rvs(args.strides[2], args.strides[3]-args.strides[2])))
            while check_shape_tie(rows, cols, args.strides_type):
                rows = int(np.round(uniform.rvs(args.strides[0], args.strides[1]-args.strides[0])))
                cols = int(np.round(uniform.rvs(args.strides[2], args.strides[3]-args.strides[2])))
            # for now all cnn layer have the same strides shape
            e.strides = [[rows, cols]]*3

            # max pool
            rows = int(np.round(uniform.rvs(args.m_pool[0], args.m_pool[1]-args.m_pool[0])))
            cols = int(np.round(uniform.rvs(args.m_pool[2], args.m_pool[3]-args.m_pool[2])))
            while check_shape_tie(rows, cols, args.m_pool_type):
                rows = int(np.round(uniform.rvs(args.m_pool[0], args.m_pool[1]-args.m_pool[0])))
                cols = int(np.round(uniform.rvs(args.m_pool[2], args.m_pool[3]-args.m_pool[2])))
            # for now all max pool layer have the same shape
            e.m_pool = [[rows, cols]]*3

            e.optimizer = np.random.choice(args.optimizer)
            e.pool_type = np.random.choice(args.pool_type)

            ################################################################################### Danse layers
            dense_layer_numb = np.random.choice(range(args.dense_layer_numb)) + 1
            e.dense_layer_numb = dense_layer_numb

            e.dense_lys_neurons = [int(np.round(uniform.rvs(args.dense_lys_neurons[0],
                                                            args.dense_lys_neurons[1] - args.dense_lys_neurons[0])))]
            for i in range(dense_layer_numb-1):
                c = int(np.round(uniform.rvs(args.dense_lys_neurons[0], args.dense_lys_neurons[1] - args.dense_lys_neurons[0])))
                while check_num_tie(e.dense_lys_neurons[i-1], c, args.dense_shape_type):
                    c = int(np.round(uniform.rvs(args.dense_lys_neurons[0], args.dense_lys_neurons[1] - args.dense_lys_neurons[0])))
                    e.dense_lys_neurons.append(c)

            ################################################################################### Learning params
            e.shuffle = np.random.choice([True, False])
            e.optimizer = np.random.choice(args.optimizer)
            e.loss = np.random.choice(args.loss)
            e.batch_size = int(np.round(uniform.rvs(args.batch_size[0], args.batch_size[1] - args.batch_size[0])))

            # lv = random.choice(range(args.maxLayer)) + 1
            # dropout = norm.rvs(loc=(args.minDrop + args.maxDrop) / 2, scale=(args.maxDrop - args.minDrop) / 4)  # normale
            # maxNorm = uniform.rvs(args.minMaxNorm, args.maxMaxNorm - args.minMaxNorm)
            # x = int(2 ** uniform.rvs(args.minExpNeu, args.maxExpNeu - args.minExpNeu))
            # learningRate = 10 ** uniform.rvs(args.minExpLr, args.maxExpLr - args.minExpLr)

        exp_list.append(e)
    return exp_list

############################################ inizializzalizzazioni
experiments = []

############################################ creazione della lista dei parametri secondo la strategia scelta
if args.search_strategy == "grid":
    experiments = grid_search(args)
elif args.search_strategy == "random":
    experiments = random_search(args)

############################################ creazione dei file per lo scheduler
for e in experiments:

    # apri template in lettura
    # apri un file di testo (nome?, path?)  ---> configuration_name = str(i).zfill(5) + '.cfg'
    # copia le righe
    # crea la stringa con i parametri
    # scrivi la stringa
    # chiudi il file
    pass