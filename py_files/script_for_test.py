#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: buckler
"""


import autoencoder
import numpy as np
import dataset_manupulation as dm
import utility as u
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score

#config variable


pat='/media/buckler/DataSSD/Phd/fall_detection/framework/autoencoder_fall_detection/logs/case6/process_1.log'

u.logcleaner(pat)