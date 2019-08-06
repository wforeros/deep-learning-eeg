#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:42:53 2019

@author: wilsonforero
"""

import mne
import numpy as np
import utilities
import os.path as op
import matplotlib.pyplot as plt

ans = mne.read_epochs_eeglab(op.join(utilities.CLASSIFIED_SETS_FOLDER,'Set1-500ms/Emoticon/AlE/90010_AlE.set'))

# (epochs, canales, muestras)
original_data = ans.get_data()


# =============================================================================
# Con numpy
# =============================================================================


        
#normalized_channels = prom_per_channel/standard_deviation_channels_np
#normalized_epochs = prom_per_epoch/standard_deviation_epochs_np
            

ans._data = utilities.normalize_array(original_data)
ans.save('90100-epo.fif', verbose=1)

#%%

# =============================================================================
# En esta sección se realiza la normalización de datos
# =============================================================================
sets_package='Set3-0ms'


for sub_cat in utilities.SUB_CATEGORIES:
    category = 'Emoticon' if 'E' in sub_cat else 'Real'
    directory = op.join(utilities.CLASSIFIED_SETS_FOLDER, sets_package, category, sub_cat)
    mne_files = utilities.get_files_with_mne(directory)
    output_folder = op.join(
                utilities.MAIN_SETS_FOLDER, 'procesado_python', sets_package, category, sub_cat
            )
    utilities.create_directory(output_folder)
    for file in mne_files:
        original_data = file.get_data()
        normalized_data = utilities.normalize_array(original_data)
        file_name = utilities.get_mne_filename(file)
        normalized_data = utilities.normalize_array(original_data)
        file._data = normalized_data
        file_path = op.join(output_folder, '{}{}'.format(file_name, '-epo.fif'))
        file.save(file_path, verbose=1)
        





#%%
# =============================================================================
# Comparación del resultado de la gráfica
# =============================================================================
epoch = 0
channel_data=1
fif_data = ans.get_data()

set_channel_data = original_data[epoch, channel_data, :]
fif_channel_data = fif_data[epoch, channel_data, :]

plt.plot(set_channel_data)
plt.title(('Archivo ' + str(90100) + '.set'))
plt.show()

plt.plot(fif_channel_data)
plt.title('Archivo ' + str(90100) + '-epo.fif')
plt.show()
