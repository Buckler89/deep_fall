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
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib

from scipy.spatial.distance import euclidean
#import matplotlib.image as img


class autoencoder_fall_detection():
    
    def __init__(self, kernel_shape=[3,3], number_of_kernel=[16,8,8]):
        print("__init__")
        self._debug_load_directly=0;#se è a 1 carica il modello e i pesi dal disco.

        self._ks = kernel_shape
        self._nk = number_of_kernel
        self._config=0;
        self._weight=0;
        self._autoencoder=0
        

    
    def define_arch(self):
        print("define_arch")

        input_img = Input(shape=(1, 129, 197))


        x = Convolution2D(self._nk[0], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(self._nk[1], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(self._nk[2], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        # at this point the representation is (8, 4, 4) i.e. 128-dimensional
        
        x = Flatten()(x)
        x = Dense(3400,activation='tanh')(x)
        encoded = Dense(64,activation='tanh')(x)
        #-------------------------------------
        x = Dense(3400,activation='tanh')(encoded)
        x = Reshape((8, 17, 25))(x)
        
        x = Convolution2D(self._nk[2], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(self._nk[1], self._ks[0], self._ks[1], activation='tanh', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(self._nk[0], self._ks[0], self._ks[1], activation='tanh')(x)
        x = UpSampling2D((2, 2))(x)
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
        
    def model_fit(self,x_train, y_train, x_test=None, y_test=None, nb_epoch=50, batch_size=128, shuffle=True, ):
        print("model_fit")
        
        if self._debug_load_directly:
            #if i want to load from disk the model
            autoencoder=load_model('my_model.h5')
            autoencoder.load_weights('my_model_weights.h5')
            self._autoencoder=autoencoder;
        else:
            if x_test != None and y_test != None:
                self._autoencoder.fit(x_train, x_train,
                                nb_epoch=50,
                                batch_size=128,
                                shuffle=True,
                                validation_data=(x_test, x_test))
            else:
                self._autoencoder.fit(x_train, x_train,
                        nb_epoch=50,
                        batch_size=128,
                        shuffle=True)
            #save the model an weights on disk
            self._autoencoder.save('my_model.h5')
            self._autoencoder.save_weights('my_model_weights.h5')
            #save the model and wetight on varibles
    #        self._config = self._autoencoder.get_config();
    #        self._weight = self._autoencoder.get_weights()
            
        self._debug_load_directly=1;     
        return self._autoencoder
        
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

    def reconstruct_handwritedigit_mnist(self,x_test):
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
        
 
    def compute_distance(self,x_test,decoded_images):
        '''
        calcola le distanze euclide tra 2 vettori di immagini con shape (n_img,1,row,col)
        ritorna un vettore con le distanze con shape .....
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
        
        '''
        print("labelize_data")

        i=0
        numeric_label=list();
        for d in y:
            if 'rndy' in d:
                numeric_label.append(1);
            else:
                numeric_label.append(0);
            i+=1;
                             
        return numeric_label
    
    def compute_score(self, original_image, decoded_images, labels):
        print("compute_score")

        numeric_label=self.labelize_data(labels);
        e_d=self.compute_distance(original_image,decoded_images);
             
           
        print("roc curve:");                         
        fpr, tpr, thresholds = roc_curve(numeric_label, e_d, pos_label=1);
        roc_auc = auc(fpr, tpr)
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
        plt.show()
        
        #---------------------------DET----------------------
        npoint=5000
        minth=min(e_d);
        maxth=max(e_d);
        step=(maxth-minth)/npoint;
        ths=np.arange(minth,maxth,step);
        tp=np.zeros(len(ths));
        fn=np.zeros(len(ths));
        tn=np.zeros(len(ths));
        fp=np.zeros(len(ths));
        
        k=0;
        for th in ths:
            i=0;
            for d in e_d:
                if d > th:
                    if numeric_label[i]==1:
                        tp[k]+=1;
                    else:
                        fp[k]+=1;
                else:
                    if numeric_label[i]==1:
                        fn[k]+=1;
                    else:
                        tn[k]+=1;                
                i+=1;    
            k+=1;
        tpr=tp/(tp+fn)
        tnr=tn/(tn+fp)
        fpr=fp/(fp+tn)
        fnr=fn/(fn+tp)
        self.DETCurve(fpr,fnr)
        
        
        #su quale theshold?
#        print("confusion matrix:");  
#             
#        confusion_matrix(numeric_label,)  
        #fmeasure(numeric_label,e_d) ??? 
                
        return roc_auc 
        
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