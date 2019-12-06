#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:35:32 2019

@author: wilsonforero
"""

from keras.layers import Dropout, Conv2D, BatchNormalization, MaxPooling2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD

def create_classifier_model(n_channels, n_samples, n_classes = 2, compile_model = True):
    """
    Parámetros:
    ____
    - n_channels: Número de canales del set EEG
    - n_samples: Número de muestras que tiene cada epoch
    - n_classes: Cantidad de categorías que tiene el modelo (por defecto 2)
    - compile_model: Entrega el modelo compilado con optimizador SGD con lr = 0.05
    Retorna:
    ____
    - model: Modelo con las capas definidas y compilado si así se marcó
    - str: Tipo de modelo generado, en este caso retorna 'classifier'
    """
    model = Sequential()
    
    # CONV POOL BLOCK 1
    # Input size: 44 (canales) x 522 (muestras) x 1 (features)
    # Convolución: 25 filtros, cada uno de 1x10
    # Output size: (44-1+1)x(522-10+1)x25 (filtros) = 44x513x25
    # Maxpooling: 25 filtros cada uno de 1x3 (ojo, es MaxPooling2D)
    # output size: 1x171x25
    
    # msotaquira: acá se arranca con Conv2D, pero para lograr la convolución 1D se usan kernels de 1x10 y 44x1 respectivamente.
    
    input_shape = (n_channels,n_samples,1)
    model.add(Conv2D(filters=25, kernel_size=(1,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    #model.add(Dropout(0.5))
    #model.add(Conv2D(filters=25, kernel_size=(n_channels,1), activation='linear', strides=1,  data_format='channels_last'))
    model.add(Conv2D(filters=50, kernel_size=(n_channels,1), activation='linear', strides=1,  data_format='channels_last'))
    model.add(BatchNormalization(momentum=0.5))
    #model.add(Conv2D(filters=25, kernel_size=(n_channels,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(1,3)))
    
    
    
    # CONV POOL BLOCK 1
    # Input size: 1x171x25 (la salida de la capa anterior)
    # Reshape: se debe reajustar el tamaño anterior para que quede de 171x25 y así poder aplicar Conv1D
    # Convolución: 50 filtros, cada uno de 10x25 (se usa sólo 10 en "kernel_size", pues la otra dimensión ya es 25)
    # Output size: 1x162x50
    # Maxpooling: 50 filtros cada uno de 1x3 (ojo, es MaxPooling1D)
    # output size: 1x54x50
    
    #model.add(Reshape((current_output_shape[2],current_output_shape[3])))
    model.add(Dropout(0.25))
    #model.add(Conv1D(filters=50, kernel_size=(10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    #model.add(Conv2D(filters=25, kernel_size=(1,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    #model.add(Conv2D(filters=12, kernel_size=(1,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    model.add(Conv2D(filters=50, kernel_size=(1,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    
    model.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
    model.add(Activation('elu'))
    #model.add(MaxPooling1D(pool_size=3))
    model.add(MaxPooling2D(pool_size=(1,3)))
    
    # CONV POOL BLOCK 3
    # Input size: 1x54x50 (la salida de la capa anterior)
    # Convolución: 100 filtros, cada uno de 10x50 (se usa sólo 10 en "kernel_size", pues la otra dimensión ya es 50)
    # Output size: 1x162x50
    # Maxpooling: 100 filtros cada uno de 1x3 (ojo, es MaxPooling1D)
    # output size: 1x15x100
    model.add(Dropout(0.25))
    #model.add(Conv1D(filters=100, kernel_size=(10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    #model.add(Conv2D(filters=50, kernel_size=(1,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    #model.add(Conv2D(filters=25, kernel_size=(1,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    model.add(Conv2D(filters=100, kernel_size=(1,10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    model.add(BatchNormalization(momentum=0.5, epsilon=1e-5))
    model.add(Activation('elu'))
    #model.add(MaxPooling1D(pool_size=3))
    model.add(MaxPooling2D(pool_size=(1,3)))
    
    # CONV POOL BLOCK 4
    # Input size: 1x15x100 (la salida de la capa anterior)
    # Convolución: 200 filtros, cada uno de 10x100 (se usa sólo 10 en "kernel_size", pues la otra dimensión ya es 100)
    # Output size: 1x6x200
    # Maxpooling: 100 filtros cada uno de 1x3 (ojo, es MaxPooling1D)
    # output size: 1x2x200
    model.add(Dropout(0.25))
    #model.add(Conv1D(filters=100, kernel_size=(2), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    #model.add(Conv2D(filters=100, kernel_size=(1,2), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    #model.add(Conv2D(filters=50, kernel_size=(1,2), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    model.add(Conv2D(filters=200, kernel_size=(1,2), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation('elu'))
    #model.add(MaxPooling1D(pool_size=3))
    model.add(MaxPooling2D(pool_size=1))
    
    
    
    # CLASSIFICATION LAYER
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    if compile_model:
        optimizer = SGD(lr=0.05)
        model.compile(optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
    return model, 'classifier'

def create_autoencoder_256(n_channels, compile_model=True):
    """
    Este método crea un modelo autoencoder pero únicamente funciona con sets que tengan
    256 muestras, por eso no tiene estos argumentos
    Parámetros:
    ____
    - compile_model: Entrega el modelo compilado con optimizador SGD con lr = 0.05
    Retorna:
    ____
    - model: Modelo con las capas definidas y compilado si así se marcó
    - str: Tipo de modelo generado, en este caso retorna 'autoencoder'
    """
    from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
    from keras.models import Model
    
    n_samples=256
    input_shape = (n_channels,n_samples,1)
    print(input_shape)
    input_layer = Input(input_shape)
    # Capa 1
    x = Conv2D(filters=25, kernel_size=(1,10), activation='linear', strides=1, data_format='channels_last') (input_layer)
    x = x = Conv2D(filters=50, kernel_size=(n_channels,1), activation='linear', strides=1,  data_format='channels_last') (x)
    x = BatchNormalization(momentum=0.5) (x)
    x = Activation('elu') (x)
    x = MaxPooling2D(pool_size=(1,3)) (x)
    
    x = Dropout(0.25)(x)
    # Capa 2
    x = Conv2D(filters=50, kernel_size=(1,10), activation='linear', strides=1, data_format='channels_last')(x)
    x = BatchNormalization(momentum=0.5, epsilon=1e-5) (x)
    x = Activation('elu') (x)
    x = MaxPooling2D(pool_size=(1,3)) (x)
    
    x = Dropout(0.25) (x)
    
    # Capa 3
    x = Conv2D(filters=100, kernel_size=(1,10), activation='linear', strides=1, data_format='channels_last') (x)
    x = BatchNormalization(momentum=0.5, epsilon=1e-5) (x)
    x = Activation('elu') (x)
    encoded = MaxPooling2D(pool_size=(1,3)) (x)
    

    x = Conv2DTranspose (filters=100, kernel_size=(1,20), activation='linear', strides=1, data_format='channels_last') (encoded)
    x = Activation('elu') (x)

    x = UpSampling2D ((1, 3)) (x)
    x = Conv2DTranspose (filters=50, kernel_size=(1,10), activation='linear', strides=1, data_format='channels_last') (x)
    x = Activation('elu') (x)

    x = UpSampling2D ((1, 3)) (x)
    x = Conv2DTranspose (filters=25, kernel_size=(1,14), activation='linear', strides=1, data_format='channels_last') (x)
    x = Activation('elu') (x)

    x = UpSampling2D ((1,1)) (x)
    decoded = Conv2DTranspose (filters=1, kernel_size=(n_channels,1), activation='linear', strides=1,  data_format='channels_last') (x)
    x = Activation('elu') (x)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder, 'autoencoder'
    

model, _ = create_autoencoder_256()
model.summary()
