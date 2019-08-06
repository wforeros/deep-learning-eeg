#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 02:22:22 2019

@author: wilsonforero
"""


import mne
import numpy as np
import utilities
import os.path as op

ans = mne.read_epochs_eeglab(op.join(utilities.CLASSIFIED_SETS_FOLDER,'Set1-500ms/Emoticon/AlE/90010_AlE.set'))

# (epochs, canales, muestras)
original_data = ans.get_data()

# intento de cálculo del promedio


# Toma cada epoch y suma todas las muestras que tuvo ese epoch en los diferentes
# canales, quedando al final un arreglo de 16x256
summation_per_epoch = np.sum(original_data, axis=1)

# Toma cada canal y suma las muestras que tuvo en cada epoch dejando al final
# un arreglo de 32x256
summation_per_channel = np.sum(original_data, axis=0)



# Se divide cada uno por su cantidad de datos  que para ambos casos es 256
# entonces toma cada uno de los 256 valores de cada epoch y lo divide en 256
#prom_per_epoch = summation_per_epoch / (summation_per_epoch.shape[1])
# entonces toma cada uno de los 256 valores de cada canal y lo divide en 256
#prom_per_channel = summation_per_channel / (summation_per_channel.shape[1])

# Por cada promedio se tomaron 32 datos para ser sumados pero no es entre
# 256 por que es cierto que el proceso se hizo 256 veces por cada epoch
# es decir 256 promedios por epoch, pero cada promedio llevó una suma de 32 valores
prom_per_epoch = summation_per_epoch / (32.)
# Por cada promedio se tomaron 16 datos para ser sumados pero no es entre
# 256 por que es cierto que el proceso se hizo 256 veces por cada canal
# es decir 256 promedios por canal, pero cada promedio llevó una suma de 16 valores
prom_per_channel = summation_per_channel / (16.)

# Ahora se calculará la diferencia de cada valor acorde a su promedio
# primero se hará con los canales

# Las diferencias entre el valor en un canal, epoch y muestra puntual con respecto
# al promedio de ese canal, por tanto se vuelve a tener un arreglo de las dimensiones
# del original
diffs_channel = np.zeros((original_data.shape[0], original_data.shape[1], original_data.shape[2]))
# un for de 0 a 31
for channel in range(prom_per_channel.shape[0]):
    # se iteral sobre los epochs originales
    for epoch in range(original_data.shape[0]):
        # Se itera sobre los 256 valores, sería lo mismo que poner
        # original_data.shape[2]
        for sample in range(prom_per_channel.shape[1]):
            # Entonces según el orden de los for el arreglo se llenará así:
            # toma un canal, de ese canal toma un epoch y de ese epoch una muestra
            # para luego restarle el promedio que se calculó en ese canal para esa muestra
            diff = original_data[epoch, channel, sample] - prom_per_channel[channel, sample]      
            
            # De una vez se eleva al cuadrado la diferencia
            diffs_channel[epoch, channel, sample] = pow(diff, 2.0);

# Las diferencias entre el valor en un canal, epoch y muestra puntual con respecto
# al promedio de ese epoch, por tanto se vuelve a tener un arreglo de las dimensiones
# del original
diffs_epochs = np.zeros((original_data.shape[0], original_data.shape[1], original_data.shape[2]))
# un for de 0 a 15
for epoch in range(prom_per_epoch.shape[0]):
    # se iteral sobre los canales originales
    for channel in range(original_data.shape[1]):
        # Se itera sobre los 256 valores, sería lo mismo que poner
        # original_data.shape[2]
        for sample in range(prom_per_epoch.shape[1]):
            # Entonces según el orden de los for el arreglo se llenará así:
            # toma un canal, de ese canal toma un epoch y de ese epoch una muestra
            # para luego restarle el promedio que se calculó en ese canal para esa muestra
            diff = original_data[epoch, channel, sample] - prom_per_epoch[epoch, sample]      
            
            # De una vez se eleva al cuadrado la diferencia
            diffs_epochs[epoch, channel, sample] = pow(diff, 2.0);
            
# Etapa de suma de las diferencias 
# se volverá a arreglos 2d
summation_diffs_channels = np.sum(diffs_channel, axis=0)

summation_diffs_epochs = np.sum(diffs_epochs, axis=1)


# Finalmente vuelve a dividirse entre el total de datos - 1
summation_diffs_channels /= 15.

# Cada epoch tenía 
summation_diffs_epochs /= 31.

standard_deviation_channels = np.sqrt(summation_diffs_channels)
standard_deviation_epochs = np.sqrt(summation_diffs_epochs)



# =============================================================================
# Con numpy
# =============================================================================
summation_per_epoch = np.sum(original_data, axis=1)
summation_per_channel = np.sum(original_data, axis=0)



prom_per_channel = summation_per_channel / (16.)
prom_per_epoch = summation_per_epoch / (32.)

prom_per_channel = np.median(original_data, axis=0)
prom_per_epoch = np.median(original_data, axis=1)

standard_deviation_channels_np = np.std(original_data, axis = 0)
standard_deviation_epochs_np = np.std(original_data, axis = 1)

normalized_channels = np.copy(original_data)
assert(prom_per_channel.shape[0] == standard_deviation_channels_np.shape[0])
for channel in range(original_data.shape[1]):
    for epoch in range(original_data.shape[0]):
        for sample in range(original_data.shape[2]):
            # Se toma el valor original
            channel_value = original_data[epoch, channel, sample]
            # Se toma el valor del promedio para ese canal en esa muestra
            prom_value = prom_per_channel[channel, sample]
            # Se toma la desviación para ese canal en esa muestra
            std_deviation = standard_deviation_channels_np[channel, sample]
            # Se opera acorde a la fórmula
            normalized_data = (channel_value - prom_value) / std_deviation
            # Se almacena el valor normalizado
            normalized_channels[epoch, channel, sample] = normalized_data
        
#normalized_channels = prom_per_channel/standard_deviation_channels_np
#normalized_epochs = prom_per_epoch/standard_deviation_epochs_np
            
ans._data = normalized_channels
ans.save('90100-epo.fif', verbose=1)