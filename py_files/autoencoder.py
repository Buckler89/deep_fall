#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: buckler
"""

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from keras.layers import Input, Dense, Flatten, Reshape, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.models import Model,load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, f1_score
import matplotlib

from scipy.spatial.distance import euclidean
#import matplotlib.image as img


class autoencoder_fall_detection():

    def __init__(self, kernel_shape, number_of_kernel, fit=True):
        print("__init__")
        self._fit_net = fit; #se è False carica il modello e i pesi dal disco.

        self._ks = kernel_shape
        self._nk = number_of_kernel
        self._config=0;
        self._weight=0;
        self._autoencoder=0


    def define_cnn_arch(self, params):
        print("define_arch")
         # ---------------------------------------------------     - Encoding
        d=params.input_shape[0]
        h=params.input_shape[1]
        w=params.input_shape[2]

        input_img=Input(shape=params.input_shape)
        x=input_img

        for i in range(len(params.kernel_number)):

            x = Convolution2D(params.kernel_number[i],
                              params.kernel_shape[i][0],
                              params.kernel_shape[i][1],
                              init=params.init,
                              activation=params.conv_activation,
                              border_mode=params.border_mode,
                              subsample=params.strides,
                              W_regularizer=params.w_reg,
                              b_regularizer=params.b_reg,
                              activity_regularizer=params.a_reg,
                              W_constraint=params.w_constr,
                              b_constraint=params.b_constr,
                              bias=params.bias)(x)

            if params.border_mode=='same':
                ph=params.kernel_shape[i][0]-1
                pw=params.kernel_shape[i][1]-1
            else:
                ph=pw=0
            h=(int(h-params.kernel_shape[i][0]+ph)/params.strides[0])+1
            w=(int(w-params.kernel_shape[i][1]+pw)/params.strides[1])+1
            d=params.kernel_number[i]

            if not params.m_pool_only_end:
                x = MaxPooling2D(params.m_pool[i], border_mode='same')(x)
                # if border=='valid' h=int(h/params.params.m_pool[i][0])
                h=int(h/params.params.m_pool[i][0])+1
                w=int(w/params.params.m_pool[i][1])+1

        if params.m_pool_only_end:
             x = MaxPooling2D(params.m_pool[0], border_mode='same')(x)
             # if border=='valid' h=int(h/params.params.m_pool[i][0])
             h=int(h/params.params.m_pool[i][0])+1
             w=int(w/params.params.m_pool[i][1])+1

        x = Flatten()(x)

        x=Dense(d*h*w,
                init=params.init,
                activation=params.dense_activation,
                W_regularizer=params.w_reg,
                b_regularizer=params.b_reg,
                activity_regularizer=params.a_reg,
                W_constraint=params.w_constr,
                b_constraint=params.b_constr,
                bias=params.bias)(x)

        for i in range(len(params.dense_layers_inputs)):
            x=Dense(params.dense_layers_inputs[i],
                    init=params.init,
                    activation=params.dense_activation,
                    W_regularizer=params.w_reg,
                    b_regularizer=params.b_reg,
                    activity_regularizer=params.a_reg,
                    W_constraint=params.w_constr,
                    b_constraint=params.b_constr,
                    bias=params.bias)(x)

        # ---------------------------------------------------------- Decoding

        for i in range(len(params.dense_layers_inputs)-2,-1,-1): # backwards indices last excluded
            x=Dense(params.dense_layers_inputs[i],
                    init=params.init,
                    activation=params.dense_activation,
                    W_regularizer=params.w_reg,
                    b_regularizer=params.b_reg,
                    activity_regularizer=params.a_reg,
                    W_constraint=params.w_constr,
                    b_constraint=params.b_constr,
                    bias=params.bias)(x)

        x=Dense(d*h*w,
                init=params.init,
                activation=params.dense_activation,
                W_regularizer=params.w_reg,
                b_regularizer=params.b_reg,
                activity_regularizer=params.a_reg,
                W_constraint=params.w_constr,
                b_constraint=params.b_constr,
                bias=params.bias)(x)

        x = Reshape((d,h,w))(x)


-----------------------------------------------------------------------------------------------------------
        for i in range(len(params.kernel_number)-1,-1,-1):

            x = Convolution2D(params.kernel_number[i],
                              params.kernel_shape[i][0],
                              params.kernel_shape[i][1],
                              init=params.init,
                              activation=params.conv_activation,
                              border_mode=params.border_mode,
                              subsample=params.strides,
                              W_regularizer=params.w_reg,
                              b_regularizer=params.b_reg,
                              activity_regularizer=params.a_reg,
                              W_constraint=params.w_constr,
                              b_constraint=params.b_constr,
                              bias=params.bias)(x)

            if params.border_mode=='same':
                ph=params.kernel_shape[i][0]-1
                pw=params.kernel_shape[i][1]-1
            else:
                ph=pw=0
            h=(int(h-params.kernel_shape[i][0]+ph)/params.strides[0])+1
            w=(int(w-params.kernel_shape[i][1]+pw)/params.strides[1])+1
            d=params.kernel_number[i]

            if not params.m_pool_only_end:
                x = UpSampling2D(params.m_pool[i])(x)
                h=h*params.params.m_pool[i][0]
                w=w*params.params.m_pool[i][1]

        if params.m_pool_only_end:
             x = UpSampling2D(params.m_pool[i])(x)
             h=h*params.params.m_pool[i][0]
             w=w*params.params.m_pool[i][1]

        dh=h-params.input_shape[1]
        dw=w-params.input_shape[2]
        
        if dh>0:
            




calcolo differenze con l'input e croppo e zeropaddo



        
        x = ZeroPadding2D(padding=(0,0,0,1))(x);
        x = Cropping2D(cropping=((1, 2), (0, 0)))(x)

        decoded = Convolution2D(1, self._ks[0], self._ks[1], activation='tanh', border_mode='same')(x) 
        
#        layer1 = Model(input_img, decoded);
#        layer1.summary();
        self._autoencoder = Model(input_img, decoded)
                
        return self._autoencoder





















    def model_compile(self, model=None, optimizer='adadelta', loss='mse'):
        '''
        compila il modello con i parametri passati: se non viene passato compila il modello istanziato dalla classe
        '''
        print("model_compile")

        if model==None:
            self._autoencoder.compile(optimizer='adadelta', loss='mse');
        else:
            model.compile(optimizer='adadelta', loss='mse');
        
    def model_fit(self,x_train, y_train, x_test=None, y_test=None, nb_epoch=50, batch_size=128, shuffle=True ):
        print("model_fit")
        
        if not self._fit_net:
            #if i want to load from disk the model
            autoencoder=load_model('my_model.h5')
            autoencoder.load_weights('my_model_weights.h5')
            self._autoencoder=autoencoder;
        else:
            if x_test != None and y_test != None:
                self._autoencoder.fit(x_train, x_train,
                                nb_epoch=nb_epoch,
                                batch_size=batch_size,
                                shuffle=True,
                                validation_data=(x_test, x_test))
            else:
                self._autoencoder.fit(x_train, x_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True)
            #save the model an weights on disk
            self._autoencoder.save('my_model.h5')
            self._autoencoder.save_weights('my_model_weights.h5')
            #save the model and wetight on varibles
    #        self._config = self._autoencoder.get_config();
    #        self._weight = self._autoencoder.get_weights()
            
        self._fit_net=False;     
        return self._autoencoder
    
    def save_model(model):
            '''
            salva il modello e i pesi. 
            TODO gestire nomi dei file in maniera intelligente in base ai parametri e case, in 
            modo tale che siano riconoscibili alla fine
            '''
            model.save('my_model.h5')
            model.save_weights('my_model_weights.h5')       
    
    def reconstruct_spectrogram(self,x_test):
        '''
        decodifica i vettori in ingresso.
        '''
        print("reconstruct_spectrogram")

        decoded_imgs=self._autoencoder.predict(x_test)
            
#to load from variable
#autoencoder = Model.from_config(net_config)
#autoencoder.set_weights(net_weight)

#        n = 41
#        plt.figure(figsize=(20*4, 4*4))
#        for i in range(1,n):
#            # display original
#            ax = plt.subplot(2, n, i)
#            plt.imshow(x_test[i].reshape(28, 28))
#            plt.gray()
#            ax.get_xaxis().set_visible(False)
#            ax.get_yaxis().set_visible(False)
#        
#            # display reconstruction
#            ax = plt.subplot(2, n, i + n)
#            plt.imshow(decoded_imgs[i].reshape(28, 28))
#            plt.gray()
#            ax.get_xaxis().set_visible(False)
#            ax.get_yaxis().set_visible(False)
#        plt.show()
        return decoded_imgs

    def reconstruct_handwritedigit_mnist(self,x_test):   # @Diego -> da cancellare?
        '''
        vuole in ingresso un vettore con shape (1,1,28,28), la configurazione del modello e i pesi 
        '''
        print("reconstruct_handwritedigit_mnist")

        decoded_imgs=self._autoencoder.predict(x_test)       
        
        plt.figure()
        # display original
        ax = plt.subplot(2, 1, 1)
        plt.imshow(x_test.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        ax = plt.subplot(2, 1, 2)
        plt.imshow(decoded_imgs.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show() 
        
 
    def compute_distances(self,x_test,decoded_images):
        '''
        calcola le distanze euclide tra 2 vettori di immagini con shape (n_img,1,row,col)
        ritorna un vettore con le distanze con shape (n_img,1)
        '''
        print("compute_distance")

        #e_d2d = np.zeros(x_test.shape)
        e_d = np.zeros(x_test.shape[0])

            
        for i in range(decoded_images.shape[0]):
            #e_d2d[i,0,:,:] = euclidean_distances(decoded_images[i,0,:,:],x_test[i,0,:,:])
            #e_d[i] = euclidean_distances(decoded_images[i,0,:,:],x_test[i,0,:,:]).sum();
            e_d[i] = euclidean(decoded_images[i,0,:,:].flatten(),x_test[i,0,:,:].flatten())
                
        return e_d;
    
    def labelize_data(self,y):
        '''
        labellzza numericamente i nomi dei file
        assegna 1 se è una caduta del manichino, 0 altrimenti
        
        '''
        print("labelize_data")

        i=0
        true_numeric_labels=list();
        for d in y:
            if 'rndy' in d:
                true_numeric_labels.append(1);
            else:
                true_numeric_labels.append(0);
            i+=1;
                             
        return true_numeric_labels
    
    def compute_score(self, original_image, decoded_images, labels):
        print("compute_score")

        true_numeric_labels=self.labelize_data(labels);
        euclidean_distances=self.compute_distances(original_image,decoded_images);
        
                                                  
        fpr, tpr, roc_auc, thresholds = self.ROCCurve(true_numeric_labels, euclidean_distances, pos_label=1, makeplot='no', opt_th_plot='no')
        if max(fpr)!=1 or max(tpr)!=1 or min(fpr)!=0 or min(tpr)!=0: #in teoria questi mi e max dovrebbero essere sempre 1 e 0 rispettivamente
            print("max min tpr fpr error");
        optimal_th, indx = self.compute_optimal_th(fpr,tpr,thresholds,method = 'std');
        self.ROCCurve(true_numeric_labels, euclidean_distances, indx, pos_label=1, makeplot='yes', opt_th_plot='yes')
        
                                            
             
        #compute tpr fpr fnr tnr metrics                
#        npoint=5000
#        minth=min(euclidean_distances);
#        maxth=max(euclidean_distances);
#        step=(maxth-minth)/npoint;
#        ths=np.arange(minth,maxth,step);
#        tp=np.zeros(len(ths));
#        fn=np.zeros(len(ths));
#        tn=np.zeros(len(ths));
#        fp=np.zeros(len(ths));
#        
#        k=0;
#        for th in ths:
#            i=0;
#            for d in euclidean_distances:
#                if d > th:
#                    if true_numeric_labels[i]==1:
#                        tp[k]+=1;
#                    else:
#                        fp[k]+=1;
#                else:
#                    if true_numeric_labels[i]==1:
#                        fn[k]+=1;
#                    else:
#                        tn[k]+=1;                
#                i+=1;    
#            k+=1;
#        tpr=tp/(tp+fn)
#        tnr=tn/(tn+fp)
#        fpr=fp/(fp+tn)
#        fnr=fn/(fn+tp)
        #---------------------------DET----------------------

        #self.DETCurve(fpr,fnr)
       
        #---------------------------myROC----------------------
        
#        plt.figure()
#        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#        plt.plot([0, 1], [0, 1], 'k--')
#        plt.xlim([0.0, 1.0])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.title('Receiver operating characteristic')
#        plt.legend(loc="lower right")
#        plt.show()
        
        #--------------------------CONFUSION MATRIX---------------------
        tp=0;
        fn=0;
        tn=0;
        fp=0;
        i=0;
        y_pred=np.zeros(len(euclidean_distances))
        for d in euclidean_distances:
            if d > optimal_th:
                y_pred[i]=1;
                if true_numeric_labels[i]==1:
                    tp+=1;
                else:
                    fp+=1;
            else:
                y_pred[i]=0;
                if true_numeric_labels[i]==1:
                    fn+=1;
                else:
                    tn+=1;
            i+=1;
#        tpr=tp/(tp+fn)
#        tnr=tn/(tn+fp)
#        fpr=fp/(fp+tn)
#        fnr=fn/(fn+tp)
        print("confusion matrix:");
        #sk_cm=confusion_matrix(true_numeric_labels,y_pred);
        my_cm=np.array([[tp,fn],[fp,tn]]);
        print("\t Fall \t NoFall")
        print("Fall \t"+str(tp)+"\t"+str(fn))
        print("NoFall \t"+str(fp)+"\t"+str(tn))
        print("F1measure: "+str(f1_score(true_numeric_labels,y_pred,pos_label=1)))
        print(classification_report(true_numeric_labels,y_pred,target_names=['NoFall','Fall']))
        
        
        return roc_auc, optimal_th, my_cm, true_numeric_labels, y_pred;
    
    def compute_optimal_th(self,fpr, tpr, thresholds, method='std'):
        '''
        http://medind.nic.in/ibv/t11/i4/ibvt11i4p277.pdf
        ci sono molti metodi per trovare l ottima th:
            1-'std' minumum of distances from point (0,1)
                min(d^2), d^2=[(0-fpr)^2+(1-tpr)^2]
            2-'xxx' definire delle funzioni costo TODO
        '''
        if method=='std':
            indx = ((0-fpr)**2+(1-tpr)**2).argmin();
            optimal_th = thresholds[indx];
            return optimal_th, indx;
        
    def ROCCurve(self,y_true, y_score, indx=None, pos_label=1, makeplot='yes',opt_th_plot='no'):
        print("roc curve:");                         
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label);
        roc_auc = auc(fpr, tpr)
        
        if makeplot=='yes':
        # Plot of a ROC curve for a specific class
        
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            if opt_th_plot=='yes' and indx!=None:
                plt.plot(fpr[indx],tpr[indx], 'ro');
            plt.show()
            
        return fpr,tpr,roc_auc,thresholds
        
    def DETCurve(self,fpr,fnr):
        """
        Given false positive and false negative rates, produce a DET Curve.
        The false positive rate is assumed to be increasing while the false
        negative rate is assumed to be decreasing.
        """
        print("DETCurve")

        #axis_min = min(fps[0],fns[-1])
        fig,ax = plt.subplots(figsize=(10, 10), dpi=600)
        plt.plot(fpr,fnr)
        plt.yscale('log')
        plt.xscale('log')
        ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks(ticks_to_use)
        ax.set_yticks(ticks_to_use)
        plt.axis([0.001,50,0.001,50])
        
        
    def print_score(self, cm, y_pred, y_true):
        '''
        print the final results for the all fold test 
        '''
        print("FINAL REPORT")
        print("\t Fall \t NoFall")
        print("Fall \t"+str(cm[0,0])+"\t"+str(cm[0,1]))
        print("NoFall \t"+str(cm[1,0])+"\t"+str(cm[1,1]))
        
        print("F1measure: "+str(f1_score(y_true,y_pred,pos_label=1)))
        print(classification_report(y_true,y_pred,target_names=['NoFall','Fall']));