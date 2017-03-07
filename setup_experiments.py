#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 11:19:08 2017

@author: daniele
"""

import argparse

############################################ Parsing argomenti (intervalli, nomi file ecc...)
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
parser.add_argument('-ks', '--kernel-shape', dest="kernel_shape", nargs='+', action='append', type=int)
parser.add_argument('-kt', '--kernel-type', dest="kernel_type", default="square", choices=["square", "rect", "any"])
parser.add_argument('-mp', '--max-pool-shape', dest="m_pool", nargs='+', action='append', type=int)
parser.add_argument('-mpt', '--max-pool-type', dest="m_pool_type", default="square", choices=["square", "rect", "any"])

parser.add_argument('-dln', '--dense-layers-numb', dest="dense_layer_numb", nargs='+', default=[1], type=int)
parser.add_argument('-ds', '--dense-shape', dest="dense_layers_inputs", nargs='+', default=[64], type=int)
parser.add_argument('-dst', '--dense-shape-type', dest="dense_shape_type", default="any",
                    choices=["decrease", "decrease", "equal", "any"])

parser.add_argument('-i', '--cnn-init', dest="cnn_init", nargs='+', default=["glorot_uniform"], choices=["glorot_uniform"])
parser.add_argument('-ac', '--cnn-conv-activation', dest="cnn_conv_activation", nargs='+', default=["tanh"], choices=["tanh"])
parser.add_argument('-ad', '--cnn-dense-activation', dest="cnn_dense_activation", nargs='+', default=["tanh"], choices=["tanh"])
parser.add_argument('-bm', '--border-mode', dest="border_mode", default="same", choices=["valid", "same"])
parser.add_argument('-st', '--strides-type', dest="strides_type", default="square", choices=["square", "rect", "any"])
parser.add_argument('-s', '--strides', dest="strides", nargs='+', action='append', type=int)
parser.add_argument('-wr', '--w-reg', dest="w_reg",
                    default=None)  # in autoencoder va usato con eval('funzione(parametri)')
parser.add_argument('-br', '--b-reg', dest="b_reg", default=None)
parser.add_argument('-ar', '--act-reg', dest="a_reg", default=None)
parser.add_argument('-wc', '--w-constr', dest="w_constr", default=None)
parser.add_argument('-bc', '--b-constr', dest="b_constr", default=None)
parser.add_argument("-nb", "--no-bias", dest="bias", default=True, action='store_false')
parser.add_argument("-p", "--pool-type", dest="pool_type", nargs='+', default=['all'], choices=["all", "only_end"])

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

############################################ def grid_search:
def grid_search(args):
    exp_list = []


    # do something



    return exp_list

############################################ def random_search:
def random_search(args):
    exp_list = []
    for n in range(args.N):
        e = experiment()


        # do something

        #check if dimension is ok

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



    # apri un file di testo (nome?, path?)
    configuration_name = str(i).zfill(4) + '.cfg'

    # scrivi intestazione
    # idea: apro un templete e ci appendo la chiamata del processo




    # crea la stringa con i parametri
    # scrivi la stringa
    # chiudi il file
    pass


