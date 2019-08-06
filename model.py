#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wilsonforero 
"""

from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD

n_files = 2192
n_channels = 32
n_samples = 384
model = Sequential()


## Construcción del modelo 

model = Sequential()

# entrada = (#samples, #channels) (esto acorde a la documentación) 
# considerando que los epochs son la entrada
# es decir cada uno tendrá esas dimensiones
input_shape = (n_samples, n_channels)
# Capa #1
model.add(Conv1D(filters=25, kernel_size=10, activation='linear', strides=1,  input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3))

# Capa #2
model.add(Conv1D(filters=50, kernel_size=10, activation='linear',strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3))

# Capa #3
model.add(Conv1D(filters=100, kernel_size=10, activation='linear',strides=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=3))

#Capa #4
model.add(Flatten())
model.add(Dense(6, activation='softmax'))

optimizer = SGD(lr=0.025)
model.compile(optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, batch_size=200, epochs=50, verbose=1)









