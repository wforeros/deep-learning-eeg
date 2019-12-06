#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wilsonforero
"""
__author__ = 'Wilson Forero'

import core
import os.path as op
import os

#sets_package='Set3-0ms'
sets_package = core.SETS_PACKAGES[2]
#%%
# =============================================================================
# En esta sección se realiza la normalización de datos
# =============================================================================



for sub_cat in core.SUB_CATEGORIES:
    category = 'Emoticon' if 'E' in sub_cat else 'Real'
    directory_origin = op.join(core.CLASSIFIED_SETS_FOLDER, sets_package, category, sub_cat)
    mne_files = core.get_files_with_mne(directory_origin)
    output_folder = op.join(
                core.PROCESSED_WITH_PYTHON_FOLDER, sets_package, category, sub_cat
            )
    core.create_directory(output_folder)
    i = 0
    for file in mne_files:
        # Los canales Fp1 y Fp2 fueron usados para remover EOG por Gratton
        file.drop_channels(['Fp1', 'Fp2'])
        original_data = file.get_data()
        normalized_data = core.normalize_array(original_data, mode='epochs')
#        print('Ya normalizado', normalized_data.shape, 'iter', i)
        file_name = core.get_mne_filename(file)
        normalized_data = core.normalize_array(original_data)
        file._data = normalized_data
        file_path = op.join(output_folder, '{}{}'.format(file_name, '-epo.fif'))
        file.save(file_path, verbose=1)
        
#%%
# =============================================================================
# Split data
# =============================================================================
        
test_size = 20.#%


for sub_cat in core.SUB_CATEGORIES:
    category = 'Emoticon' if 'E' in sub_cat else 'Real'
    origin_folder = op.join(core.PROCESSED_WITH_PYTHON_FOLDER, sets_package, category, sub_cat)
    all_files = os.listdir(path=origin_folder)
    output_folder = op.join(
                core.PROCESSED_AND_CLASSIFIED_FOLDER, sets_package, category, sub_cat
            )
    core.create_directory(output_folder)
    core.split_sets(origin_folder, output_folder)
