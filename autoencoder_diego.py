#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:43:32 2017

@author: buckler
"""

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
          
          
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


class autoencoder_fall_detection():
    
    def __init__(self, kernel_shape, number_of_kernel):
        self.ks = kernel_shape
        self.nk = number_of_kernel
    ############LOAD DATA

    def pre_process_data_mnist(self,):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
        x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
        
        #remove some data 4 from trainset
        X_train=[x for x, y in zip(x_train, y_train) if (y < 1)]
        X_train=np.array(X_train)
        return  (X_train, y_train, x_test, y_test)
    
    ############END LOAD DATA
    
    def network_architecture_autoencoder(self,X_train, y_train, x_test, y_test):
    
        input_img = Input(shape=(1, 28, 28))
        
#        x = Convolution2D(int(self.nk[0]), int(self.ks[0,0]), int(self.ks[0,1]), activation='relu', border_mode='same')(input_img)
#        x = MaxPooling2D((2, 2), border_mode='same')(x)
#        x = Convolution2D(self.nk[1], self.ks[1,0], self.ks[1,1], activation='relu', border_mode='same')(x)
#        x = MaxPooling2D((2, 2), border_mode='same')(x)
#        x = Convolution2D(self.nk[2], self.ks[1,0], self.ks[2,1], activation='relu', border_mode='same')(x)
#        encoded = MaxPooling2D((2, 2), border_mode='same')(x)
#        
#        # at this point the representation is (8, 4, 4) i.e. 128-dimensional
#        
#        x = Convolution2D(self.nk[2], self.ks[2,0], self.ks[2,1], activation='relu', border_mode='same')(encoded)
#        x = UpSampling2D((2, 2))(x)
#        x = Convolution2D(self.nk[1], self.ks[1,0], self.ks[1,1], activation='relu', border_mode='same')(x)
#        x = UpSampling2D((2, 2))(x)
#        x = Convolution2D(self.nk[0], self.ks[0,0], self.ks[0,1], activation='relu')(x)
#        x = UpSampling2D((2, 2))(x)
#        decoded = Convolution2D(1, self.ks[0,0], self.ks[0,1], activation='sigmoid', border_mode='same')(x)

        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        encoded = Dense(32,activation='tanh')(x)
        # at this point the representation is (8, 4, 4) i.e. 128-dimensional
        x = Dense(32,activation='tanh')(encoded)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(16, 3, 3, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)  

     
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        
        autoencoder.fit(X_train, X_train,
                        nb_epoch=50,
                        batch_size=128,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        
        #save the model an weights
        autoencoder.save('my_model.h5')
        autoencoder.save_weights('my_model_weights.h5')
        config = autoencoder.get_config();
        weight=autoencoder.get_weights()
    
        return config, weight
        
    def reconstruct_image(self,x_test, net_config, net_weight):
    
        #autoencoder = load_model('my_model.h5')
        #autoencoder.load_weights('my_model_weights.h5')
        autoencoder = Model.from_config(net_config)
        autoencoder.set_weights(net_weight)
        
        decoded_imgs = autoencoder.predict(x_test)
        
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