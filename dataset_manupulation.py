#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:36:52 2017

@author: buckler
"""
import os
import numpy as np
from keras import backend as K


def load_A3FALL(spectrogramsPath):
    '''
    Carica tutto il dataset (spettri) in una lista di elementi [filename , matrix ]
    '''
    print("Loading A3FALL dataset");
    a3fall=list();
    for root, dirnames, filenames in os.walk(spectrogramsPath):
        i=0;
        for file in filenames:
            matrix=np.load(os.path.join(root,file));
            data=[file,matrix];
            a3fall.append(data)
            i+=1;
    return a3fall

def awgn_padding_set( set_to_pad, loc=0.0, scale=1.0):
    # find matrix with biggest second axis
    dim_pad=np.amax([len(k[1][2]) for k in set_to_pad]);
    awgn_padded_set = []
    for e in set_to_pad:
        row, col = e[1].shape;
        # crete an rowXcol matrix with awgn samples
        awgn_matrix = np.random.normal(loc, scale, size=(row,dim_pad-col));
        awgn_padded_set.append([e[0],np.hstack((e[1],awgn_matrix))]);
    return awgn_padded_set 

def reshape_set(set_to_reshape, channels=1):

    n_sample=len(set_to_reshape);
    row, col = set_to_reshape[0][1].shape;
    label = []
    shaped_matrix = np.empty((n_sample,channels,row,col));
    for i in range(len(set_to_reshape)):
        label.append(set_to_reshape[i][0]);
        shaped_matrix[i][0]=set_to_reshape[i][1]
    return shaped_matrix,  label

def split_A3FALL_simple(data,train_tag=None):
    '''
    Splitta il dataset in train e test set: train tutti i background, mentre test tutto il resto
    (da amplicare in modo che consenta lo split per la validation)
    '''
    if train_tag==None:
        train_tag=['classic_','rock_','ha_']
#        if test_tag=None:
#            test_tag=[]
    
    data_train=[d for d in data if any(word in d[0] for word in train_tag)] #controlla se uno dei tag è presente nnel nome del file e lo assegna al trainset
    data_test=[d for d in data if d not in data_train]#tutto cioò che non è train diventa test
    
    return data_train, data_test

def split_A3FALL_from_lists(data, listpath, namelist):
    '''
    Richede in inglesso la cartella dove risiedono i file di testo che elencano i vari segnali che farano parte di un voluto set di dati.
    Inltre in namelist vanno specificati i nomi dei file di testo da usare.
    Ritorna una lista contentete le liste dei dataset di shape: (len(namelist),data.shape)
    '''
    sets=list();
    for name in namelist:
        sets.append(select_list(os.path.join(listpath,name),data));
    return sets
    
def select_list(filename,dataset):
    '''
    Dato in ingesso un file di testo, resituisce una array contenete i dati corrispondenti elencati nel file
    '''
    subset=list()
    with open(filename) as f:
        content = f.readlines();
        content = [x.strip().replace('.wav','.npy') for x in content] #remove the '\n' at the end of string
        subset = [s for s in dataset if any(name in s[0] for name in content)] #select all the data presetn in the list
    return subset        
    
def normalize_data(data,mean=None,std=None):
    '''
    normalizza media e varianza del dataset passato
    se data=None viene normalizzato tutto il dataset A3FALL
    se mean e variance = None essi vengono calcolati in place sui data
    '''

    if bool(mean) ^ bool(std):#xor operator
        raise("Error!!! Provide both mean and variance")
    elif mean==None and std==None: #compute mean and variance of the passed data
        data_conc = concatenate_matrix(data);
        mean=np.mean(data_conc)
        std=np.std(data_conc)   
                                        
    data_std= [[d[0],((d[1]-mean)/std)] for d in data]#normalizza i dati: togle mean e divide per std
    
    return data_std, mean , std;
    
def concatenate_matrix(data):
    '''
    concatena gli spettri in un unica matrice: vule una lista e restituisce un array
    '''
    data_=data.copy()
    data_.pop(0)
    matrix=data[0][1]
    for d in data_:
        np.append(matrix,d[1], axis=1)
    return matrix
    