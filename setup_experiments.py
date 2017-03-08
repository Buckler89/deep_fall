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
parser.add_argument("-kst", "--kernel-number-type", dest="kernel_number_type", default="any",
                    choices=["decrease", "encrease", "equal", "any"])
parser.add_argument("-ks", "--kernel-shape", dest="kernel_shape", action=eval_action, default=[[3,3],[3,3],[3,3]])
parser.add_argument("-kt", "--kernel-type", dest="kernel_type", default="square", choices=["square", "+cols", "+rows", "any"])
parser.add_argument("-mp", "--max-pool-shape", dest="m_pool", action=eval_action, default=[[2,2],[2,2],[2,2]])
parser.add_argument("-mpt", "--max-pool-type", dest="m_pool_type", default="square", choices=["square", "+cols", "+rows", "any"])

parser.add_argument("-p", "--pool-type", dest="pool_type", action=eval_action, default=["all"]) # choices=["all", "only_end"]
parser.add_argument("-i", "--cnn-init", dest="cnn_init", action=eval_action, default=["glorot_uniform"])
parser.add_argument("-ac", "--cnn-conv-activation", dest="cnn_conv_activation", action=eval_action, default=["tanh"])
parser.add_argument("-ad", "--cnn-dense-activation", dest="cnn_dense_activation", action=eval_action, default=["tanh"])
parser.add_argument("-bm", "--border-mode", dest="border_mode", default="same", choices=["valid", "same"])
parser.add_argument("-st", "--strides-type", dest="strides_type", default="square", choices=["square", "+cols", "+rows", "any"])

parser.add_argument("-s", "--strides", dest="strides", action=eval_action, default=[[1,1],[1,1],[1,1]])
parser.add_argument("-wr", "--w-reg", dest="w_reg", default=None) # in autoencoder va usato con eval("funz(parametri)")
parser.add_argument("-br", "--b-reg", dest="b_reg", default=None)
parser.add_argument("-ar", "--act-reg", dest="a_reg", default=None)
parser.add_argument("-wc", "--w-constr", dest="w_constr", default=None)
parser.add_argument("-bc", "--b-constr", dest="b_constr", default=None)
parser.add_argument("-nb", "--no-bias", dest="bias", default=[True], action=eval_action)

# dense params
parser.add_argument("-dln", "--dense-layers-numb", dest="dense_layer_numb", action=eval_action, default=[1])
parser.add_argument("-ds", "--dense-shapes", dest="dense_shapes", action=eval_action, default=[64])
parser.add_argument("-dst", "--dense-shape-type", dest="dense_shape_type", default="any",
                    choices=["decrease", "encrease", "equal", "any"])

# fit params
parser.add_argument("-f", "--fit-net", dest="fit_net", default=False, action="store_true")
parser.add_argument("-e", "--epoch", dest="epoch", default=50, type=int)
parser.add_argument("-ns", "--no-shuffle", dest="shuffle", default=[True], action=eval_action)
parser.add_argument("-bs", "--batch-size", dest="batch_size", action=eval_action, default=[128])
parser.add_argument("-o", "--optimizer", dest="optimizer", action=eval_action, default=["adadelta"])
parser.add_argument("-l", "--loss", dest="loss", action=eval_action, default=["mse"])

args = parser.parse_args()

if args.config_filename is not None:
    with open(args.config_filename, "r") as f:
        lines = f.readlines()
    arguments = []
    for line in lines:
        if "#" not in line:
            arguments.extend(line.split())
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
        return False

    h = e.cnn_input_shape[1]
    w = e.cnn_input_shape[2]
    for i in range(e.cnn_layer_numb):
        if e.border_mode == "same":
            ph = e.kernel_shape[i][0] - 1
            pw = e.kernel_shape[i][1] - 1
        else:
            ph = pw = 0
        h = int((h - e.kernel_shape[i][0] + ph) / e.strides[0]) + 1
        w = int((w - e.kernel_shape[i][1] + pw) / e.strides[1]) + 1

        if e.pool_type[0] == "all":
            # if border=="valid" h=int(h/params.params.m_pool[i][0])
            h = math.ceil(h / e.m_pool[i][0])
            w = math.ceil(w / e.m_pool[i][1])
    if e.pool_type[0] == "only_end":
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
                                                                        e = experiment()
                                                                        e.id = n
                                                                        n += 1
                                                                        e.cnn_input_shape = args.cnn_input_shape
                                                                        e.conv_layer_num = cln
                                                                        e.kernel_number = kn
                                                                        e.pool_type = p
                                                                        e.kernel_shape = ks
                                                                        e.strides = s
                                                                        e.m_pool = mp
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
            for j in range(cnn_layer_numb-1):
                c = int(np.round(uniform.rvs(args.kernel_number[0], args.kernel_number[1] - args.kernel_number[0])))
                while check_num_tie(e.kernel_number[j-1], c, args.kernel_number_type):
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

            e.dense_shapes = [int(np.round(uniform.rvs(args.dense_shapes[0],
                                                            args.dense_shapes[1] - args.dense_shapes[0])))]
            for j in range(dense_layer_numb-1):
                c = int(np.round(uniform.rvs(args.dense_shapess[0], args.dense_shapes[1] - args.dense_shapes[0])))
                while check_num_tie(e.dense_shapes[j-1], c, args.dense_shape_type):
                    c = int(np.round(uniform.rvs(args.dense_shapes[0], args.dense_shapes[1] - args.dense_shapes[0])))
                    e.dense_shapes.append(c)

            ################################################################################### Learning params
            e.shuffle = np.random.choice([True, False])
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
    experiments = grid_search(args)
elif args.search_strategy == "random":
    experiments = random_search(args)

############################################ creazione dei file per lo scheduler

i=0
for e in experiments:
    sript_path = os.path.join(root_dir, "scripts", str(i).zfill(5) + "_fall.pbs")  #--------------------------- dove vanno gli script?
    copyfile(os.path.join(root_dir, "template_script.txt"), sript_path)  #-------------------------------------- da settare virtual env su template
    command = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn python "
    command += " --exp-index " + str(e.id) + \
               " --score-path " + str(args.scorePath) + \
               " --trainset-list " + str(args.trainNameLists) + \
               " --case " + str(args.case) + \
               " --test-list-names " + str(args.testNamesLists) + \
               " --dev-list-name " + str(args.devNamesLists) + \
               " --input-type " + str(args.input_type) + \
               " --cnn-input-shape " + str(args.cnn_input_shape) + \
               " --conv-layers-numb " + str(e.conv_layer_num) + \
               " --kernels-number " + str(e.kernel_number) + \
               " --pool-type " + str(e.pool_type)

    for k in range(len(e.kernel_shape)):
        command += " --kernel-shape " + str(e.kernel_shape[k]) + \
                   " --strides " + str(e.strides[k]) + \
                   " --max-pool-shape " + str(e.m_pool[k])

    command += " --cnn-init " + str(e.cnn_init) + \
               " --cnn-conv-activation " + str(e.cnn_conv_activation) + \
               " --cnn-dense-activation " + str(e.cnn_dense_activation) + \
               " --border-mode " + str(e.border_mode) + \
               " --w-reg " + str(args.w_reg) + \
               " --b-reg " + str(args.b_reg) + \
               " --act-reg " + str(args.a_reg) + \
               " --w-constr " + str(args.w_constr) + \
               " --b-constr " + str(args.b_constr) + \
               " --dense-layers-numb " + str(e.dense_layer_numb) + \
               " --dense-shape " + str(e.dense_shapes) + \
               " --epoch " + str(args.epoch) + \
               " --batch-size " + str(e.batch_size) + \
               " --optimizer " + str(e.optimizer) + \
               " --loss " + str(e.loss) + \
               " --no-shuffle " + str(e.shuffle) + \
               " --no-bias " + str(e.bias)

    if args.log:
        command += " --logging "
    if args.fit_net:
        command += " --fit-net "

    with open(sript_path, "a") as f:
        f.write(command)

    i+=1

with open(os.path.join(root_dir, "scripts", "experiments.json"), "w") as outfile:
    json.dump(experiments, outfile)