#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: buckler
"""


import random
import numpy
from matplotlib import pyplot
from autoencoder import ROCCurve, compute_optimal_th

x = [random.gauss(3,1) for _ in range(22)]
y = [random.gauss(0,2) for _ in range(22)]

bins = numpy.linspace(-20, 20, 1000)

pyplot.hist(x, bins, alpha=0.5, label='x')
pyplot.hist(y, bins, alpha=0.5, label='y')
pyplot.legend(loc='upper right')
pyplot.show()
data=[]
labels=[]
for d in x:
    data.append(d)
    labels.append(1)
for d in y:
    data.append(d)
    labels.append(0)

fpr, tpr, roc_auc, thresholds = ROCCurve(labels, data, pos_label=1, makeplot='yes', opt_th_plot='yes')
optimal_th, indx = compute_optimal_th(fpr, tpr, thresholds, method='std')
ROCCurve(labels, data, indx, pos_label=1, makeplot='yes', opt_th_plot='yes')

#ROCCurve(labels, data, None, pos_label=1, makeplot='yes', opt_th_plot='yes')
