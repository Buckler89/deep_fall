#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 3 11:19:08 2017

@author: daniele
"""

import argparse
from scipy.stats import uniform
import numpy as np
import math
import os
from shutil import copyfile
import json


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
parser.add_argument("-log", "--logging", dest="log", default=False, action="store_true")
parser.add_argument("-ss", "--search-strategy", dest="search_strategy", default="grid", choices=["grid", "random"])
parser.add_argument("-rnd", "--rnd-exp-number", dest="N", default=0, type=int)
parser.add_argument("-cf", "--config-file", dest="config_filename", default=None)

parser.add_argument("-sp", "--score-path", dest="scorePath", default="score")
parser.add_argument("-shp", "--script-path", dest="scriptPath", default="scripts")
parser.add_argument("-tl", "--trainset-list", dest="trainNameLists", action=eval_action, default=["trainset.lst"])
parser.add_argument("-c", "--case", dest="case", default="case6")
parser.add_argument("-tn", "--test-list-names", dest="testNamesLists", action=eval_action,
                    default=["testset_1.lst", "testset_2.lst", "testset_3.lst", "testset_4.lst"])
parser.add_argument("-dn", "--dev-list-names", dest="devNamesLists", action=eval_action,
                    default=["devset_1.lst", "devset_2.lst", "devset_3.lst", "devset_4.lst"])
parser.add_argument("-it", "--input-type", dest="input_type", default="spectrograms",
                    choices=["spectrograms", "mel_coefficients"])

# CNN params
parser.add_argument("-is", "--cnn-input-shape", dest="cnn_input_shape", action=eval_action, default=[1, 129, 197])
parser.add_argument("-cln", "--conv-layers-numb", dest="conv_layer_numb", action=eval_action, default=[3])
parser.add_argument("-kn", "--kernels-number", dest="kernel_number", action=eval_action, default=[16, 8, 8])
parser.add_argument("-kst", "--kernel-number-type", dest="kernel_number_type", default="any")
parser.add_argument("-ks", "--kernel-shape", dest="kernel_shape", action=eval_action, default=[[3,3],[3,3],[3,3]])
parser.add_argument("-kt", "--kernel-type", dest="kernel_type", default="square")
parser.add_argument("-mp", "--max-pool-shape", dest="m_pool", action=eval_action, default=[[2,2],[2,2],[2,2]])
parser.add_argument("-mpt", "--max-pool-type", dest="m_pool_type", default="square")

parser.add_argument("-p", "--pool-type", dest="pool_type", action=eval_action, default=["all"]) # choices=["all", "only_end"]
parser.add_argument("-i", "--cnn-init", dest="cnn_init", action=eval_action, default=["glorot_uniform"])
parser.add_argument("-ac", "--cnn-conv-activation", dest="cnn_conv_activation", action=eval_action, default=["tanh"])
parser.add_argument("-ad", "--cnn-dense-activation", dest="cnn_dense_activation", action=eval_action, default=["tanh"])
parser.add_argument("-bm", "--border-mode", dest="border_mode", default=["same"], action=eval_action)
parser.add_argument("-st", "--strides-type", dest="strides_type", default="square")

parser.add_argument("-s", "--strides", dest="strides", action=eval_action, default=[[1,1],[1,1],[1,1]])
parser.add_argument("-wr", "--w-reg", dest="w_reg", action=eval_action, default=None) # in autoencoder va usato con eval("funz(parametri)")
parser.add_argument("-br", "--b-reg", dest="b_reg", action=eval_action, default=None)
parser.add_argument("-ar", "--act-reg", dest="a_reg", action=eval_action, default=None)
parser.add_argument("-wc", "--w-constr", dest="w_constr", action=eval_action, default=None)
parser.add_argument("-bc", "--b-constr", dest="b_constr", action=eval_action, default=None)
parser.add_argument("-nb", "--bias", dest="bias", default=[True], action=eval_action)

# dense params
parser.add_argument("-dln", "--dense-layers-numb", dest="dense_layer_numb", action=eval_action, default=[1])
parser.add_argument("-ds", "--dense-shapes", dest="dense_shapes", action=eval_action, default=[64])
parser.add_argument("-dst", "--dense-shape-type", dest="dense_shape_type", default="any")

# fit params
parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--shuffle", dest="shuffle", default=[True], action=eval_action)
parser.add_argument("-bs", "--batch-size", dest="batch_size", action=eval_action, default=[128])
parser.add_argument("-o", "--optimizer", dest="optimizer", action=eval_action, default=["adadelta"])
parser.add_argument("-l", "--loss", dest="loss", action=eval_action, default=["mse"])

args = parser.parse_args()

if args.config_filename is not None:
    with open(args.config_filename, "r") as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        arguments.extend(line.split("#")[0].split())
    # First parse the arguments specified in the config file
    args = parser.parse_args(args=arguments)
    # Then append the command line arguments
    # Command line arguments have the priority: an argument is specified both
    # in the config file and in the command line, the latter is used
    args = parser.parse_args(namespace=args)


# container class for single experiments
class experiment:
    pass

                        #--------------------------------------------------------------- verifica formule e poi
def check_dimension(e): #--------------------------------------------------------------- da verificare utilità con Diego
    if not hasattr(e, "conv_layer_numb"):
        # for emulate do while struct
        return True

    h = e.cnn_input_shape[1]
    w = e.cnn_input_shape[2]
    for i in range(e.conv_layer_numb):
        if e.border_mode == "same":
            ph = e.kernel_shape[i][0] - 1
            pw = e.kernel_shape[i][1] - 1
        else:
            ph = pw = 0
        h = int((h - e.kernel_shape[i][0] + ph) / e.strides[i][0]) + 1
        w = int((w - e.kernel_shape[i][1] + pw) / e.strides[i][1]) + 1

        if e.pool_type == "all":
            # if border=="valid" h=int(h/params.params.m_pool[i][0])
            h = math.ceil(h / e.m_pool[i][0])
            w = math.ceil(w / e.m_pool[i][1])
    if e.pool_type == "only_end":
        # if border=="valid" h=int(h/params.params.m_pool[i][0])
        h = math.ceil(h / e.m_pool[-1][0])
        w = math.ceil(w / e.m_pool[-1][1])

    if h*w <= 9:  #------------------------------------------------------------------------------------- cosa ci va qui?
        return False

    return True


############################################ def grid_search:
def grid_search(args):
    exp_list = []
    n=0

    for cln in args.conv_layer_numb:
        for kn in args.kernel_number:
            for ks in args.kernel_shape:
                for mp in args.m_pool:
                    for p in args.pool_type:
                        for s in args.strides:
                            for dln in args.dense_layer_numb:
                                for ds in args.dense_shapes:
                                    for ci in args.cnn_init:
                                        for ac in args.cnn_conv_activation:
                                            for ad in args.cnn_dense_activation:
                                                for bm in args.border_mode:
                                                    for bs in args.batch_size:
                                                        for o in args.optimizer:
                                                            for l in args.loss:
                                                                for nb in args.bias:
                                                                    for ns in args.shuffle:
                                                                        # only sane combination
                                                                        if len(kn) == len(ks) == len(s) == len(mp) == cln:
                                                                            if len(ds) == dln:
                                                                                e = experiment()
                                                                                e.id = n
                                                                                n += 1
                                                                                e.cnn_input_shape = args.cnn_input_shape
                                                                                e.conv_layer_numb = cln
                                                                                e.kernel_number = kn
                                                                                e.kernel_shape = ks
                                                                                e.strides = s
                                                                                e.m_pool = mp
                                                                                e.pool_type = p
                                                                                e.cnn_init = ci
                                                                                e.cnn_conv_activation = ac
                                                                                e.cnn_dense_activation = ad
                                                                                e.border_mode = bm
                                                                                e.dense_layer_numb = dln
                                                                                e.dense_shapes = ds
                                                                                e.batch_size = bs
                                                                                e.optimizer = o
                                                                                e.loss = l
                                                                                e.shuffle = ns
                                                                                e.bias = nb

                                                                                exp_list.append(e)

    return exp_list


def gen_with_shape_tie(rng, tie_type):
    ack=False
    while not ack:
        rows = np.random.choice(range(rng[0], rng[1] + 1))
        cols = np.random.choice(range(rng[2], rng[3] + 1))
        if rows == cols and "square" in tie_type or \
            rows <= cols and "+cols" in tie_type or \
            rows >= cols and "+rows" in tie_type or \
            tie_type == "any":
            ack = True
    return [rows, cols]


def gen_with_ties(dim, num, bounds, tie_type):
    ties=tie_type.split(",")
    if dim == 1:
        if "equal" in ties:
            v = [np.random.choice(range(bounds[0], bounds[1] + 1))] * num
        else:
            v = [np.random.choice(range(bounds[0], bounds[1] + 1))]
            for j in range(1, num):
                ack = False
                c = [np.random.choice(range(bounds[0], bounds[1] + 1))]
                while not ack:
                    if c <= v[j-1] and "decrease" in ties or \
                       c >= v[j-1] and "encrease" in ties or \
                       "any" in ties:
                        ack = True
                    else:
                        c = [np.random.choice(range(bounds[0], bounds[1] + 1))]
                v.extend(c)
    elif dim == 2:
        if ties[1] == ties[2] =="equal":
            v = [gen_with_shape_tie(bounds, ties[0])] * num
        else:
            v = [gen_with_shape_tie(bounds, ties[0])]
            for j in range(1, num):
                ack = False
                while not ack:
                    c = gen_with_shape_tie(bounds, ties[0])
                    ack = True
                    if c[0] > v[j-1][0] and ties[1] == "decrease" or \
                       c[0] < v[j-1][0] and ties[1] == "encrease":
                        ack = False
                    if c[1] > v[j-1][1] and "decrease" in tie_type or \
                       c[1] < v[j-1][1] and "encrease" in tie_type:
                        ack = False
                v.append(c)
    return v


def random_search(args):
    exp_list = []
    for n in range(args.N):
        e = experiment()
        e.id = n
        e.cnn_input_shape = args.cnn_input_shape

        while check_dimension(e):

            ################################################################################### Convolutional layers
            conv_layer_numb = np.random.choice(args.conv_layer_numb)
            e.conv_layer_numb = conv_layer_numb
            e.kernel_shape = gen_with_ties(2, conv_layer_numb, args.kernel_shape, args.kernel_type)
            e.kernel_number = gen_with_ties(1, conv_layer_numb, args.kernel_number, args.kernel_number_type)
            e.border_mode = np.random.choice(args.border_mode)
            # strides
            e.strides = gen_with_ties(2, conv_layer_numb, args.strides, args.strides_type)
            # max pool
            e.m_pool = gen_with_ties(2, conv_layer_numb, args.m_pool, args.m_pool_type)
            e.pool_type = np.random.choice(args.pool_type)
            e.cnn_init = np.random.choice(args.cnn_init)
            e.cnn_conv_activation = np.random.choice(args.cnn_conv_activation)
            ################################################################################### Danse layers
            #dense_layer_numb = np.random.choice(args.dense_layer_numb)
            dense_layer_numb = np.random.choice(range(args.dense_layer_numb[0], args.dense_layer_numb[1] + 1))
            e.dense_layer_numb = dense_layer_numb
            e.dense_shapes = gen_with_ties(1, dense_layer_numb, args.dense_shapes, args.dense_shape_type)
            e.cnn_dense_activation = np.random.choice(args.cnn_dense_activation)
            ################################################################################### Learning params
            e.shuffle = np.random.choice(args.shuffle)
            e.optimizer = np.random.choice(args.optimizer)
            e.loss = np.random.choice(args.loss)
            e.batch_size = int(np.round(uniform.rvs(args.batch_size[0], args.batch_size[1] - args.batch_size[0])))
            e.shuffle = np.random.choice(args.shuffle)
            e.bias = np.random.choice(args.bias)

        exp_list.append(e)
    return exp_list

############################################ inizializzalizzazioni
experiments = []
root_dir = os.path.realpath(".")

############################################ creazione della lista dei parametri secondo la strategia scelta
if args.search_strategy == "grid":
    print("define point of grid search")
    experiments = grid_search(args)
elif args.search_strategy == "random":
    print("define point of random search")
    experiments = random_search(args)

############################################ creazione dei file per lo scheduler

i=0
for e in experiments:
    script_name = os.path.join(args.scriptPath, str(i).zfill(5) + "_fall.pbs")  #--------------------------- dove vanno gli script?
    copyfile(os.path.join(root_dir, "template_script.txt"), script_name)  #-------------------------------------- da settare virtual env su template
    command = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python " + \
              os.path.join(root_dir, "main_experiment.py") + \
              " --exp-index " + str(e.id) + \
              " --score-path " + str(args.scorePath) + \
              " --trainset-list " + '"' + str(args.trainNameLists).replace(" ", "") + '"' + \
              " --case " + str(args.case) + \
              " --test-list-names " + '"' + str(args.testNamesLists).replace(" ", "") + '"' + \
              " --dev-list-name " + '"' + str(args.devNamesLists).replace(" ", "") + '"' + \
              " --input-type " + str(args.input_type) + \
              " --cnn-input-shape " + str(args.cnn_input_shape).replace(" ", "") + \
              " --conv-layers-numb " + str(e.conv_layer_numb) + \
              " --kernels-number " + str(e.kernel_number).replace(" ", "") + \
              " --pool-type " + str(e.pool_type) + \
              " --kernel-shape " + str(e.kernel_shape).replace(" ", "") + \
              " --strides " + str(e.strides).replace(" ", "") + \
              " --max-pool-shape " + str(e.m_pool).replace(" ", "") + \
              " --cnn-init " + str(e.cnn_init) + \
              " --cnn-conv-activation " + str(e.cnn_conv_activation) + \
              " --cnn-dense-activation " + str(e.cnn_dense_activation) + \
              " --border-mode " + str(e.border_mode) + \
              " --w-reg " + str(args.w_reg) + \
              " --b-reg " + str(args.b_reg) + \
              " --act-reg " + str(args.a_reg) + \
              " --w-constr " + str(args.w_constr) + \
              " --b-constr " + str(args.b_constr) + \
              " --dense-layers-numb " + str(e.dense_layer_numb) + \
              " --dense-shape " + str(e.dense_shapes).replace(" ", "") + \
              " --epoch " + str(args.epoch) + \
              " --batch-size " + str(e.batch_size) + \
              " --optimizer " + str(e.optimizer) + \
              " --loss " + str(e.loss)

    if not e.shuffle:
        command += " --no-shuffle"
    if not e.bias:
        command += " --no-bias"
    if args.log:
        command += " --logging"
    if args.fit_net:
        command += " --fit-net"

    with open(script_name, "a") as f:
        f.write(command)

    i+=1

np.save(os.path.join(root_dir, "scripts", "experiments.npy"), experiments)
