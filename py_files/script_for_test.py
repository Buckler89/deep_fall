#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: buckler
"""




#config variable


import numpy
foldDict = {'fold1': 0, 'fold2': 0, 'fold3': 0, 'fold4': 0}
dictsckeleton = {'AucDevs': foldDict, 'f1Devs': foldDict, 'CmDevs': foldDict, 'AucTest': foldDict, 'CmTest': foldDict, 'f1Test': foldDict, 'cmTot': 0, 'f1Final': 0}

dictsckeleton['AucDevs']['fold1']

dictsckeleton = {'AucDevsFold1': 0, 'AucDevsFold2': 0, 'AucDevsFold3': 0, 'AucDevsFold4': 0,
                 'f1DevsFold1': 0, 'f1DevsFold2': 0, 'f1DevsFold3': 0, 'f1DevsFold4': 0,
                 'CmDevsFold1': 0, 'CmDevsFold2': 0, 'CmDevsFold3': 0, 'CmDevsFold4': 0,
                 'AucTestFold1': 0, 'AucTestFold2': 0, 'AucTestFold3': 0, 'AucTestFold4': 0,
                 'CmTestFold1': 0, 'CmTestFold2': 0, 'CmTestFold3': 0, 'CmTestFold4': 0,
                 'f1TestFold1': 0, 'f1TestFold2': 0, 'f1TestFold3': 0, 'f1TestFold4': 0,
                 'cmFinal': 0,
                 'f1Final': 0}
