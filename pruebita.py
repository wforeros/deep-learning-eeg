#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wilsonforero
"""
#%%
import utilities
import os.path as op
import random
import numpy as np
#%%

# =============================================================================
# Carga de los datos guardados en el set_package especificado
# =============================================================================
set_package = utilities.SETS_PACKAGES[0]

# En este array irán diccionarios cuya llave es el label y su key es la data
train_array = []
# for para recorrar todas las carpetas de sets 'Set1-500ms', 'Set2-100ms', 'Set3-0ms'
#for set_package in utilities.SETS_PACKAGES:
    # for para recorrer todas las sub categorías: 'AlE', 'AlR', 'NE', 'NR'...
for sub_category in utilities.SUB_CATEGORIES:
    # Toma la categoría que es 'Emoticon' o 'Real'
    category = utilities.get_category_from_subcategory(sub_category)
    # Genera la ruta correspondiente a ese paquete de sets, categoría
    # subcategoría y carpeta Train
    path = op.join(
            utilities.PROCESSED_AND_CLASSIFIED_FOLDER, 
            set_package,
            category, 
            sub_category,
            'Train'
        )
    # Toma todos los archivos con mne que terminan en .fif
    files = utilities.get_files_with_mne(directory=path, extension='.fif')
    # Itera todos los archivos
    for file in files:
        # Toma toda la info de esos archivos (normalizados)
        data = file.get_data()
        # Recorre los epochs de ese archivo
        for epoch in range(data.shape[0]):
            # Clasifica su label (numérico) 
            label = utilities.subcategory_to_label(sub_category)
            # Se almacena en la lista al label como key y la data como value
            train_array.append({label: data[epoch, :, :]})
            


# En este array irán diccionarios cuya llave es el label y su key es la data
test_array = []

# for para recorrar todas las carpetas de sets 'Set1-500ms', 'Set2-100ms', 'Set3-0ms'
#for set_package in utilities.SETS_PACKAGES:
    # for para recorrer todas las sub categorías: 'AlE', 'AlR', 'NE', 'NR'...
for sub_category in utilities.SUB_CATEGORIES:
    # Toma la categoría que es 'Emoticon' o 'Real'
    category = utilities.get_category_from_subcategory(sub_category)
    # Genera la ruta correspondiente a ese paquete de sets, categoría
    # subcategoría y carpeta Train
    path = op.join(
            utilities.PROCESSED_AND_CLASSIFIED_FOLDER, 
            set_package,
            category, 
            sub_category, 
            'Test'
        )
    # Toma todos los archivos con mne que terminan en .fif
    files = utilities.get_files_with_mne(directory=path, extension='.fif')
    # Itera todos los archivos
    for file in files:
        # Toma toda la info de esos archivos (normalizados)
        data = file.get_data()
        # Recorre los epochs de ese archivo
        for epoch in range(data.shape[0]):
            # Clasifica su label (numérico) 
            label = utilities.subcategory_to_label(sub_category)
            # Se almacena en la lista al label como key y la data como value
            test_array.append({label: data[epoch, :, :]})
                

data = None
category = None
epoch = None
files = None
label = None
path = None
sub_category = None

random.shuffle(test_array)
random.shuffle(train_array)


n_files_train = len(train_array)
n_files_test = len(test_array)
# Se toma un diccionario
dictionary = train_array[0]
# Se toma la data de ese diccionario ya que cualquier set debe tener mismo número 
# de muestras y canales, el values es para acceder a la data que de dimension (32, nSamples)
# el next y el iter es para tomar solo uno de esos arrays ya que values contiene varios
sample_set = next(iter(dictionary.values()))
# Se toma el número de canales que tiene el set
n_channels = sample_set.shape[0]
# Se toma el número de muestras que tiene el set
n_samples = sample_set.shape[1]
# Se crea un arreglo de numpy con esas dimensiones
# Se cambia el orden porque la idea es que primero estén las muestras y luego
# canales
#%%

# =============================================================================
# Creacion de sets de entrenamiento y validación
# =============================================================================


# Creación de los arreglos numpy que serán entrada a la red para prueba
X_test = np.zeros((n_files_test, n_channels, n_samples))
Y_test = np.zeros((n_files_test, 6))


#for test_counter in range(len(test_array)):
#    dictionary = test_array[test_counter]
#    # key es el label y value la data
#    for key, value in dictionary.items():
#        one_hot = utilities.label_to_onehot(key)
#        Y_test[test_counter,:] = one_hot
#        # Se hace el transpose porque originalmente viene como (canales, muestras)
#        # pero la red espera que sea (muestras, canales) y transpose hace eso
#        X_test[test_counter, :, :] = value


for test_counter, dictionary in enumerate(test_array):
    # key es el label y value la data
    for key, value in dictionary.items():
        one_hot = utilities.label_to_onehot(key)
        Y_test[test_counter,:] = one_hot
        # Se hace el transpose porque originalmente viene como (canales, muestras)
        # pero la red espera que sea (muestras, canales) y transpose hace eso
        X_test[test_counter, :, :] = value
        
test_array = None

# Creación de los arreglos numpy que serán entrada a la red para entrenamiento
X_train = np.zeros((n_files_train, n_channels, n_samples))
Y_train = np.zeros((n_files_train, 6))
for counter, dictionary in enumerate(train_array):
    # key es el label y value la data
    for key, value in dictionary.items():
        one_hot = utilities.label_to_onehot(key)
        Y_train[counter,:] = one_hot
        # Se hace el transpose porque originalmente viene como (canales, muestras)
        # pero la red espera que sea (muestras, canales) y transpose hace eso
        X_train[counter, :, :] = value
    
train_array = None
dictionary = None
k = None
v = None

 

#dictionary = [{0: 'a'}, {1:'b'}, {2:'c'}, {3:'d'}]
#prueba = dictionary[0].popitem()[0]
#random.shuffle(dictionary)
#for key in keys_values:
#    print (key)


#%%

# =============================================================================
# Reshape de los sets de entrenamiento y prueba para el modelo corregido
# =============================================================================

files_shape, channels_shape, samples_shape  = X_train.shape
X_train = X_train.reshape(files_shape, channels_shape, samples_shape, 1)

files_shape, channels_shape, samples_shape  = X_test.shape
X_test = X_test.reshape(files_shape, channels_shape, samples_shape, 1)

#%%

# =============================================================================
# Modelo corregido
# =============================================================================

from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling1D, MaxPooling2D, Activation, Flatten, Dense, Reshape
from keras.models import Sequential
from keras.optimizers import SGD

#n_files = 2192
#n_channels = 44
#n_samples = 522
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
model.add(Conv2D(filters=25, kernel_size=(n_channels,1), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(1,3)))

# CONV POOL BLOCK 1
# Input size: 1x171x25 (la salida de la capa anterior)
# Reshape: se debe reajustar el tamaño anterior para que quede de 171x25 y así poder aplicar Conv1D
# Convolución: 50 filtros, cada uno de 10x25 (se usa sólo 10 en "kernel_size", pues la otra dimensión ya es 25)
# Output size: 1x162x50
# Maxpooling: 50 filtros cada uno de 1x3 (ojo, es MaxPooling1D)
# output size: 1x54x50

current_output_shape = model.output_shape
model.add(Reshape((current_output_shape[2],current_output_shape[3])))
model.add(Conv1D(filters=50, kernel_size=(10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=3))

# CONV POOL BLOCK 3
# Input size: 1x54x50 (la salida de la capa anterior)
# Convolución: 100 filtros, cada uno de 10x50 (se usa sólo 10 en "kernel_size", pues la otra dimensión ya es 50)
# Output size: 1x162x50
# Maxpooling: 100 filtros cada uno de 1x3 (ojo, es MaxPooling1D)
# output size: 1x15x100
model.add(Conv1D(filters=100, kernel_size=(10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=3))

# CONV POOL BLOCK 4
# Input size: 1x15x100 (la salida de la capa anterior)
# Convolución: 200 filtros, cada uno de 10x100 (se usa sólo 10 en "kernel_size", pues la otra dimensión ya es 100)
# Output size: 1x6x200
# Maxpooling: 100 filtros cada uno de 1x3 (ojo, es MaxPooling1D)
# output size: 1x2x200
model.add(Conv1D(filters=200, kernel_size=(5), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
model.add(Activation('elu'))
#model.add(MaxPooling1D(pool_size=3))

# CLASSIFICATION LAYER
model.add(Flatten())
model.add(Dense(6, activation='softmax'))



optimizer = SGD(lr=0.025)
model.compile(optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])

model.fit(x=X_train, y=Y_train, batch_size=200, epochs=50, verbose=1)


#%%

# =============================================================================
# Evaluar modelo
# =============================================================================
model.evaluate(x=X_test, y=Y_test, batch_size=200, verbose=1)

#%%

# =============================================================================
# Guardar modelo
# =============================================================================
# serialize model to JSON

name_to_save = 'model_{package}'.format(package=set_package)
model_json = model.to_json()
json_name = '{name}.json'.format(name=name_to_save)
with open(json_name, "w") as json_file:
    json_file.write(model_json)
    
# Serializar en h5
    
h5_name = '{name}.h5'.format(name=name_to_save)
model.save_weights(h5_name)
print("Modelo guardado con el nombre:", name_to_save)


#%%

# =============================================================================
# Carga de un modelo guardado
# =============================================================================

model_name = 'model_{package}'.format(package=set_package)

from keras.models import model_from_json
json_name = '{name}.json'.format(name=model_name)
json_file = open(json_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
h5_name = "{name}.h5".format(name=model_name)
loaded_model.load_weights(h5_name)
print('Modelo con nombre {name} cargado'.format(name=model_name))

loaded_model.compile(optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
loaded_model.evaluate(x=X_train, y=Y_train, batch_size=200, verbose=1)




#%%

# =============================================================================
# Este es el que no sirvió
# =============================================================================
#from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, Dense
#from keras.models import Sequential
#from keras.optimizers import SGD
### Construcción del modelo 
#
#model = Sequential()
#
## entrada = (#samples, #channels) (esto acorde a la documentación) 
## considerando que los epochs son la entrada
## es decir cada uno tendrá esas dimensiones
#input_shape = (n_samples, n_channels)
## Capa #1
#model.add(Conv1D(filters=25, kernel_size=10, activation='linear', strides=1,  input_shape=input_shape))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=3))
#
## Capa #2
#model.add(Conv1D(filters=50, kernel_size=10, activation='linear',strides=1))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=3))
#
## Capa #3
#model.add(Conv1D(filters=100, kernel_size=10, activation='linear',strides=1))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling1D(pool_size=3))
#
##Capa #4
#model.add(Flatten())
#model.add(Dense(6, activation='softmax'))
#
#optimizer = SGD(lr=0.025)
#model.compile(optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
#
##for i in range(len(X_train)):
##    setsito = X_train[i]
#model.fit(x=X_train, y=Y_train, batch_size=200, epochs=50, verbose=1)
