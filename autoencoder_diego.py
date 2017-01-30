#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: buckler
"""

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
          
import sys
from keras.layers import Input, Dense, Flatten, Reshape, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model,load_model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import euclidean
#import matplotlib.image as img
import random

__load_directly__=0;#se è a 1 carica il modello e i pesi dal disco.

class autoencoder_fall_detection():
    
    def __init__(self, kernel_shape, number_of_kernel):
        self._ks = kernel_shape
        self._nk = number_of_kernel
        self._config=0;
        self._weight=0;
        self._autoencoder=0
        
    ############LOAD DATA
#    def load_dataset_img(self,spectrogramsPath,listsPath):
#        '''
#        Carico il dataset di immagini generiche e lo splitto semplicemente in train e test set. Solo per fare i primi test. Qui non considero 
#        la validation phase
#        '''
#        print("Loading A3FFALL dataset");
#        n_channels=1#questo andrebbe modificato se gli spetteri hanno pure i delta e deltadelta
#
#        #leggo un immagine a caso dentro il dataset per sapere le dimensioni e inizializzare il vettore
#        #che conterrà tutte le immagini. Tutte le immagini devo essere della stessa dimensione.
#        example_image=img.imread(os.path.join(spectrogramsPath,random.choice(os.listdir(spectrogramsPath))))
#        #inizialixe matrix 4D shape (n_nample,n_channel,row,col)
#        n_sample=(len([name for name in os.listdir(spectrogramsPath) if os.path.isfile(os.path.join(spectrogramsPath,name))]))
#        allData=np.zeros((n_sample,n_channels,example_image.shape[1],example_image.shape[0]))#controllare se righe colonne sono al posto giusto!!
#
#        for root, dirnames, filenames in os.walk(spectrogramsPath):
#            i=0;
#            for file in filenames:
#                allData[i,0,:,:]=img.imread(os.path.join(root,file))[:,:,0];
#                i+=1;
#        self.allData=allData;
        
    def load_zp_dataset(self,zp_spectrogramsPath,listsPath):#TODO gestione liste liste !!!
        '''
        Carico il dataset (spettrogrammi) e lo splitto semplicemente in train e test set. Solo per fare i primi test. Qui ancora non considero 
        la validation phase:listpath è inutilizzato
        '''
        print("Loading A3FALL dataset");
        n_channels=1#questo andrebbe modificato se gli spetteri hanno pure i delta e deltadelta

        #leggo un immagine a caso dentro il dataset per sapere le dimensioni e inizializzare il vettore
        #che conterrà tutte le immagini. Tutte le immagini devo essere della stessa dimensione.
        example_image=np.load(os.path.join(zp_spectrogramsPath,random.choice(os.listdir(zp_spectrogramsPath))))
        #inizialixe matrix 4D shape (n_nample,n_channel,row,col)
        n_sample=(len([name for name in os.listdir(zp_spectrogramsPath) if os.path.join(zp_spectrogramsPath,name).endswith('.npy')])) #leggo solo i file
        allData=np.zeros((n_sample,n_channels,example_image.shape[0],example_image.shape[1]))#controllare se righe colonne sono al posto giusto!!

        for root, dirnames, filenames in os.walk(zp_spectrogramsPath):
            i=0;
            for file in filenames:
                allData[i,0,:,:]=np.load(os.path.join(root,file));
                i+=1;
        self.allData=allData;
    
    def load_A3FALL(self,spectrogramsPath):
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
        self.a3fall=a3fall
        return a3fall
        
    def split_A3FALL_simple(self,train_tag=None):
        '''
        splitta il dataset in train e test set: train tutti i background, mentre test tutto il resto
        (da amplicare in modo che consenta lo split per la validation)
        '''
        if train_tag==None:
            train_tag=['classic_','rock_','ha_']
#        if test_tag=None:
#            test_tag=[]
        
        self.a3fall_train=[d for d in self.a3fall if any(word in d[0] for word in train_tag)] #controlla se uno dei tag è presente nnel nome del file e lo assegna al trainset
        self.a3fall_test=[d for d in self.a3fall if d not in self.a3fall_train]#tutto cioò che non è train diventa test
        
        return self.a3fall_train, self.a3fall_test
        
    def normalize_data(self,data=None,mean=None,std=None):
        '''
        normalizza media e varianza del dataset passato
        se data=None viene normalizzato tutto il dataset A3FALL
        se mean e variance = None essi vengono calcolati in place sui data
        '''
        if data==None:
            data=self.a3fall

        if bool(mean) ^ bool(std):#xor operator
            raise("Error!!! Provide both mean and variance")
        elif mean==None and std==None: #compute mean and variance of the passed data
            data_conc = self.concatenate_matrix(data);
            mean=np.mean(data_conc)
            std=np.std(data_conc)   
                                            
        data_std= [[d[0],((d[1]-mean)/std)] for d in data]#normalizza i dati: togle mean e divide per std
        
        return data_std, mean , std;
        
    def concatenate_matrix(self,data):
        '''
        concatena gli spettri in un unica matrice: vule una lista e restituisce un array
        '''
        data_=data.copy()
        data_.pop(0)
        matrix=data[0][1]
        for d in data_:
            np.append(matrix,d[1], axis=1)
        return matrix
        
    def standardize_data(self,data_std): #controllare uso memoria: sto tenendo troppi narray
        self.data_std=data_std;
        self.mean=np.mean(data_std, axis=(0,2,3))#faccio la media senza cosiderare i channel ( ovvero asse 1)
        self.data_std -= self.mean
        self.std=np.std(data_std, axis=(0,2,3))
        self.data_std /= (self.std + K.epsilon())
        
        return self.data_std, self.mean , self.std 
        
    def pre_process_data_mnist(self,):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
        x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
        
        #remove some data 4 from trainset
        X_train=[x for x, y in zip(x_train, y_train) if (y < 9)]
        X_train=np.array(X_train)
        return  (X_train, y_train, x_test, y_test)
    
    ############END LOAD DATA
    
    def network_architecture_autoencoder(self):
    
        input_img = Input(shape=(1, 28, 28))
        
#        x = Convolution2D(int(self._nk[0]), int(self._ks[0,0]), int(self._ks[0,1]), activation='tanh', border_mode='same')(input_img)
#        x = MaxPooling2D((2, 2), border_mode='same')(x)
#        x = Convolution2D(self._nk[1], self._ks[1,0], self._ks[1,1], activation='tanh', border_mode='same')(x)
#        x = MaxPooling2D((2, 2), border_mode='same')(x)
#        x = Convolution2D(self._nk[2], self._ks[1,0], self._ks[2,1], activation='tanh', border_mode='same')(x)
#        encoded = MaxPooling2D((2, 2), border_mode='same')(x)
#        
#        # at this point the representation is (8, 4, 4) i.e. 128-dimensional
#        
#        x = Convolution2D(self._nk[2], self._ks[2,0], self._ks[2,1], activation='tanh', border_mode='same')(encoded)
#        x = UpSampling2D((2, 2))(x)
#        x = Convolution2D(self._nk[1], self._ks[1,0], self._ks[1,1], activation='tanh', border_mode='same')(x)
#        x = UpSampling2D((2, 2))(x)
#        x = Convolution2D(self._nk[0], self._ks[0,0], self._ks[0,1], activation='tanh')(x)
#        x = UpSampling2D((2, 2))(x)
#        decoded = Convolution2D(1, self._ks[0,0], self._ks[0,1], activation='sigmoid', border_mode='same')(x)

        x = Convolution2D(16, 3, 3, activation='tanh', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='tanh', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        # at this point the representation is (8, 4, 4) i.e. 128-dimensional
        
        x = Flatten()(x)
        x = Dense(128,activation='tanh')(x)
        encoded = Dense(64,activation='tanh')(x)
        #-------------------------------------
        x = Dense(128,activation='tanh')(encoded)
        x = Reshape((8, 4, 4))(x)
        
        x = Convolution2D(8, 3, 3, activation='tanh', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, 3, 3, activation='tanh', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3, 3, activation='tanh')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)  
#        layer1 = Model(input_img, decoded);
#        layer1.summary();
        self._autoencoder = Model(input_img, decoded)
        self._autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder_model = self._autoencoder
        
        return autoencoder_model
    
    def network_architecture_autoencoder_fit(self,X_train, y_train, x_test, y_test):

        self._autoencoder.fit(X_train, X_train,
                        nb_epoch=50,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        
        #save the model an weights on disk
        self._autoencoder.save('my_model.h5')
        self._autoencoder.save_weights('my_model_weights.h5')
        #save the model and wetight on varibles
#        self._config = self._autoencoder.get_config();
#        self._weight = self._autoencoder.get_weights()
        fitted_autoencoder = self._autoencoder
        return fitted_autoencoder
        
    def reconstruct_images(self,x_test):

        if __load_directly__:
            #if i want to load from disk the model
            autoencoder=load_model('my_model.h5')
            autoencoder.load_weights('my_model_weights.h5')
            decoded_imgs = autoencoder.predict(x_test)

        else:
            #norma operation
            decoded_imgs=self._autoencoder.predict(x_test)
            
#to load from variable
#autoencoder = Model.from_config(net_config)
#autoencoder.set_weights(net_weight)

        n = 41
        plt.figure(figsize=(20*4, 4*4))
        for i in range(1,n):
            # display original
            ax = plt.subplot(2, n, i)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            # display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        return decoded_imgs

    def reconstruct_image(self,x_test):
        '''
        vuole in ingresso un vettore con shape (1,1,28,28), la configurazione del modello e i pesi 
        '''
        #how to load model from hard disk
        #autoencoder = load_model('my_model.h5')
        #autoencoder.load_weights('my_model_weights.h5')
        #how to load model from python variables
#        autoencoder = Model.from_config(net_config)
#        autoencoder.set_weights(net_weight)
        
        #decoded_imgs = autoencoder.predict(x_test)
        if __load_directly__:
            #if i want to load from disk the model
            autoencoder=load_model('my_model.h5')
            autoencoder.load_weights('my_model_weights.h5')
            decoded_imgs = autoencoder.predict(x_test)

        else:
            #norma operation
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
        
        
    def data_spectrogram():
        print("start calculate spectrogram e save it to disk")
        
    def compute_distance(self,x_test,decoded_images):
        '''
        calcola le distanze euclide tra 2 vettori di immagini con shape (n_img,1,row,col)
        ritorna un vettore con le distanze con shape .....
        '''
        #e_d2d = np.zeros(x_test.shape)
        e_d = np.zeros(x_test.shape[0])

            
        for i in range(decoded_images.shape[0]):
            #e_d2d[i,0,:,:] = euclidean_distances(decoded_images[i,0,:,:],x_test[i,0,:,:])
            #e_d[i] = euclidean_distances(decoded_images[i,0,:,:],x_test[i,0,:,:]).sum();
            e_d[i] = euclidean(decoded_images[i,0,:,:].flatten(),x_test[i,0,:,:].flatten())
                
        return e_d;

    def compute_score(self,x_test,decoded_images,y_test):
        
        e_d=self.compute_distance(x_test,decoded_images);
        print("roc curve:");                         
        fpr, tpr, thresholds = roc_curve(y_test, e_d, pos_label=9);
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
#        matrix_e_d_target=np.zeros((2,x_test.shape[0]));
#        matrix_e_d_target[0,:]=e_d;
#        matrix_e_d_target[1,:]=y_test;
#        thrshold=np.arange(0,max(e_d),0.1);
#        for t in thrashold:
#            e_d[t]<t
#        for i in range(matrix_e_d_target.shape[1]):
#            if matrix_e_d_target[1,i]!=0:
                
        return 
        
