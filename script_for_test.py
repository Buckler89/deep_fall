#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:00:29 2017

@author: buckler
"""


import autoencoder
import numpy as np
import dataset_manupulation as dm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score

#config variable


tp=0;
fn=0;
tn=0;
fp=0;
i=0;
y_pred=np.zeros(len(euclidean_distances))
for d in euclidean_distances:
    if d > optimal_th:
        y_pred[i]=1;
        if numeric_label[i]==1:
            tp+=1;
        else:
            fp+=1;
    else:
        y_pred[i]=0;
        if numeric_label[i]==1:
            fn+=1;
        else:
            tn+=1;
    i+=1;
#        tpr=tp/(tp+fn)
#        tnr=tn/(tn+fp)
#        fpr=fp/(fp+tn)
#        fnr=fn/(fn+tp)
print("confusion matrix:");  
print("\t Fall \t NoFall")
print("Fall \t"+str(tp)+"\t"+str(fn))
print("NoFall \t"+str(fp)+"\t"+str(tn))
print("F1measure: "+str(f1_score(numeric_label,y_pred,pos_label=0)))
print(classification_report(numeric_label,y_pred,target_names=['NoFall','Fall']))


print("\t Fall \t NoFall")
print("Fall \t"+str(my_cm[0,0])+"\t"+str(my_cm[0,1]))
print("NoFall \t"+str(my_cm[1,0])+"\t"+str(my_cm[1,1]))