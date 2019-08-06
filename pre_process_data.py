#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wilsonforero
"""
__author__ = 'Wilson Forero'

import utilities
import os.path as op
import os

#%%
# =============================================================================
# En esta sección se realiza la normalización de datos
# =============================================================================
#sets_package='Set3-0ms'
sets_package = utilities.SETS_PACKAGES[1]


for sub_cat in utilities.SUB_CATEGORIES:
    category = 'Emoticon' if 'E' in sub_cat else 'Real'
    directory_origin = op.join(utilities.CLASSIFIED_SETS_FOLDER, sets_package, category, sub_cat)
    mne_files = utilities.get_files_with_mne(directory_origin)
    output_folder = op.join(
                utilities.PROCESSED_WITH_PYTHON_FOLDER, sets_package, category, sub_cat
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
# Split data
# =============================================================================
        
test_size = 20.#%
sets_package='Set2-100ms'


for sub_cat in utilities.SUB_CATEGORIES:
    category = 'Emoticon' if 'E' in sub_cat else 'Real'
    origin_folder = op.join(utilities.PROCESSED_WITH_PYTHON_FOLDER, sets_package, category, sub_cat)
    all_files = os.listdir(path=origin_folder)
    output_folder = op.join(
                utilities.PROCESSED_AND_CLASSIFIED_FOLDER, sets_package, category, sub_cat
            )
    utilities.create_directory(output_folder)
    utilities.split_sets(origin_folder, output_folder)
