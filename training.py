#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wilsonforero
"""

# 
#%%
import sys
sys.path.insert(1, '/Users/wilsonforero/Documents/Universidad/Proyecto/python')
import deep_eeg_core
import os.path as op
import random
import numpy as np
#%%

# =============================================================================
# Carga de los datos guardados en el set_package especificado
# =============================================================================
set_package = deep_eeg_core.SETS_PACKAGES[1]
#%%

# En este array irán diccionarios cuya llave es el label y su key es la data
train_array = []
# for para recorrar todas las carpetas de sets 'Set1-500ms', 'Set2-100ms', 'Set3-0ms'
#for set_package in deep_eeg_core.SETS_PACKAGES:
    # for para recorrer todas las sub categorías: 'AlE', 'AlR', 'NE', 'NR'...

file_counter = 0
for sub_category in deep_eeg_core.SUB_CATEGORIES:
    # Toma la categoría que es 'Emoticon' o 'Real'
    category = deep_eeg_core.get_category_from_subcategory(sub_category)
    # Genera la ruta correspondiente a ese paquete de sets, categoría
    # subcategoría y carpeta Train
    path = op.join(
            deep_eeg_core.PROCESSED_AND_CLASSIFIED_FOLDER, 
            set_package,
            category, 
            sub_category,
            'Train'
        )
    # Toma todos los archivos con mne que terminan en .fif
    files = deep_eeg_core.get_files_with_mne(directory=path, extension='.fif')
    # Itera todos los archivos
    for file in files:
        # Toma toda la info de esos archivos (normalizados)
        data = file.get_data()
        file_counter+=1
        # Recorre los epochs de ese archivo
        for epoch in range(data.shape[0]):
            # Clasifica su label (numérico) 
#            label = deep_eeg_core.subcategory_to_label(sub_category)
            
            if 'E' in sub_category:
                label = 0
            else:
                label = 1
            # Se almacena en la lista al label como key y la data como value
            train_array.append({label: data[epoch, :, :]})
            
#n_clases = len(deep_eeg_core.SUB_CATEGORIES)
n_clases = 2

# En este array irán diccionarios cuya llave es el label y su key es la data
test_array = []

# for para recorrar todas las carpetas de sets 'Set1-500ms', 'Set2-100ms', 'Set3-0ms'
#for set_package in deep_eeg_core.SETS_PACKAGES:
file_counter_test = 0
# for para recorrer todas las sub categorías: 'AlE', 'AlR', 'NE', 'NR'...
for sub_category in deep_eeg_core.SUB_CATEGORIES:
    # Toma la categoría que es 'Emoticon' o 'Real'
    category = deep_eeg_core.get_category_from_subcategory(sub_category)
    # Genera la ruta correspondiente a ese paquete de sets, categoría
    # subcategoría y carpeta Train
    path = op.join(
            deep_eeg_core.PROCESSED_AND_CLASSIFIED_FOLDER, 
            set_package,
            category, 
            sub_category, 
            'Test'
        )
    # Toma todos los archivos con mne que terminan en .fif
    files = deep_eeg_core.get_files_with_mne(directory=path, extension='.fif')
    # Itera todos los archivos
    for file in files:
        file_counter_test +=1
        # Toma toda la info de esos archivos (normalizados)
        data = file.get_data()
        # Recorre los epochs de ese archivo
        for epoch in range(data.shape[0]):
            # Clasifica su label (numérico) 
#            label = deep_eeg_core.subcategory_to_label(sub_category)
            if 'E' in sub_category:
                label = 0
            else:
                label = 1
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
emoticon_counter = 0
#irr_counter = 0
real_counter = 0
#ale_counter = 0
#ire_counter = 0
#ne_counter = 0
for counter, dictionary in enumerate(train_array):
    # key es el label y value la data
    for key, value in dictionary.items():
        if key == 0:
            emoticon_counter += 1
        elif key == 1:
            real_counter += 1

train_limit = min([emoticon_counter, real_counter])


emoticon_counter_test = 0
real_counter_test = 0
for counter, dictionary in enumerate(test_array):
    # key es el label y value la data
    for key, value in dictionary.items():
        if key == 0:
            emoticon_counter_test += 1
        elif key == 1:
            real_counter_test += 1

test_limit = min([emoticon_counter_test, real_counter_test])
#lower = 
#%%

# =============================================================================
# Creacion de sets de entrenamiento y validación
# =============================================================================


# Creación de los arreglos numpy que serán entrada a la red para prueba
X_test = np.zeros((test_limit * n_clases, n_channels, n_samples))
Y_test = np.zeros((test_limit * n_clases, n_clases))

emoti_counter = 0
real_counter = 0
test_counter = 0


for dictionary in test_array:
    # key es el label y value la data
    for key, value in dictionary.items():
        append_value = False
        if key == 0:
            emoti_counter += 1
            if emoti_counter < test_limit:
                append_value = True
        elif key == 1:
            real_counter += 1
            if real_counter < test_limit:
                append_value = True
        if append_value:
            one_hot = np.zeros(2)
            one_hot[key] = 1
            Y_test[test_counter,:] = one_hot
            X_test[test_counter, :, :] = value
            test_counter += 1
        
test_array = None

alr_counter = 0
irr_counter = 0
nr_counter = 0
ale_counter = 0
ire_counter = 0
ne_counter = 0

emoti_counter = 0
real_counter = 0

# Creación de los arreglos numpy que serán entrada a la red para entrenamiento
X_train = np.zeros((train_limit * n_clases, n_channels, n_samples))
Y_train = np.zeros((train_limit * n_clases, n_clases))
counter = 0
# test_limit = 1057
for dictionary in train_array:
    # key es el label y value la data
    for key, value in dictionary.items():
        append_value = False
        if key == 0:
            emoti_counter += 1
            if emoti_counter < train_limit:
                append_value = True
        elif key == 1:
            real_counter += 1
            if real_counter < train_limit:
                append_value = True
        if append_value:
            one_hot = np.zeros(2)
            one_hot[key] = 1
            Y_train[counter,:] = one_hot
            X_train[counter, :, :] = value
            counter += 1
            
#%%
            
emoti_counter_test = 0
real_counter_test = 0
for counter, y in enumerate(Y_test):
    if y[0] == 0:
        real_counter_test += 1
    else:
        emoti_counter_test += 1
        
emoti_counter_train = 0
real_counter_train = 0
for counter, y in enumerate(Y_train):
    if y[0] == 0:
        real_counter_train += 1
    else:
        emoti_counter_train += 1
#%% 
        
train_array = None
dictionary = None
k = None
v = None

 


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

from keras.layers import Dropout, Conv1D, Conv2D, BatchNormalization, MaxPooling1D, MaxPooling2D, Activation, Flatten, Dense, Reshape
from keras.models import Sequential
from keras.optimizers import SGD, Adam

#%%

import models

# Creación del modelo
model, model_type = models.create_classifier_model(n_channels=n_channels, n_samples=n_samples)
#model, model_type = models.create_autoencoder_256(n_channels=n_channels, n_samples=n_samples)

# En caso que el modelo sea autoencoder tendrá que tratar de generar
# a la salida la misma entrada, este modelo NO fue muy probado
# y las únicas pruebas realizadas fueron con el set de 256 muestras, 
# con los demás NO sirve
if model_type == 'autoencoder':
    Y_train = X_train
    Y_test = X_test
#%%

model.fit(
        x=X_train, y=Y_train, 
        validation_data=(X_test,Y_test), 
        batch_size=200, epochs=1000, 
        verbose=1, shuffle=True
    )

#%%
ceros = 0
ones = 0
for counter, y in enumerate(Y_test):
    if y[0] == 0:
        ceros += 1
    else:
        ones += 1
#%%

# =============================================================================
# Evaluar modelo
# =============================================================================
result = model.evaluate(x=X_test, y=X_test, batch_size=100, verbose=1)
print('Resultado al evaluar', result)

#%%
from matplotlib import pyplot as plt
pred = model.predict(X_test)

plt.plot(X_test[24][2])
plt.show()
plt.plot(pred[24][2])

#%%

# =============================================================================
# Guardar modelo
# =============================================================================
# serialize model to JSON

#name_to_save = 'model_{package}-0.1Hza40HzFiltrosDoble'.format(package=set_package)
name_to_save = 'autoencoder_primer_prueba_con_elu'
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

#model_name = 'model_{package}-0.1Hza40HzMismaCantidad'.format(package=set_package)
#model_name = 'model_Set3-0ms-0.1Hza40HzFiltrosCuarta<-Dropout0.25-momentum0.5-Relu'
#model_name = "autoencoder_primer_prueba_con_elu"
model_name = 'model_Set2-100ms-0.1Hza40HzMismaCantidad'
from keras.optimizers import SGD, Adam
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

optimizer = SGD(lr=0.05)
loaded_model.compile(optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
#loaded_model.fit(x=X_train, y=Y_train, batch_size=200, verbose=1)

#%%
loaded_model.fit(x=X_train, y=Y_train, validation_data=(X_test,Y_test), batch_size=200, epochs=1000, verbose=1, shuffle=True)


