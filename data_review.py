
import deep_eeg_core
import os.path as op

"""
En este archivo se extraen los sets que:
______

    - Tienen menos de 16 epochs, estos se separarán en el diccionario "files_wless_epochs"
    - TIenen menos de las muestras que deberían tener a una frecuencia de sampleo
    de 256Hz, estos se separarán en el diccionario "files_wless_samples"
"""

# =============================================================================
# En esta sección están los valores que se deben manipular
# =============================================================================
# Tiempo que se tomó antes de mostrar el estímulo (en ms)
prev_time = 100

# Se genera el nombre del grupo de sets, por ejemplo Set1-500ms
sets_package='Set2-'+str(prev_time)+'ms'


#%%
# =============================================================================
# En esta sección es para hacerlo manual
# =============================================================================

sub_category = 'AlE'
#sub_category = 'IrE'
#sub_category = 'NE'

#sub_category = 'IrR'
#sub_category = 'AlR'
#sub_category = 'IrR'

category = 'Emoticon' if 'E' in sub_category else 'Real'

# Generación de la ruta a los sets clasificados (último paso del Matlab)
path = op.join(deep_eeg_core.CLASSIFIED_SETS_FOLDER, sets_package, category, sub_category)
    
mne_files = deep_eeg_core.get_files_with_mne(path)


files_wless_epochs, files_wless_samples, _ = deep_eeg_core.check_files(mne_files, prev_time)

#%%


# =============================================================================
# Sección para que sea más rápido
# Este código saca un diccionario con todos los archivos de un 'sets_package'
# =============================================================================

# En este diccionario estarán todos los archivos separados por sub categoría que no 
# cumplen con el número de epochs y/o muestras
all_files = {}
for sub_cat in deep_eeg_core.SUB_CATEGORIES:
    category = 'Emoticon' if 'E' in sub_cat else 'Real'
    path = op.join(deep_eeg_core.CLASSIFIED_SETS_FOLDER, sets_package, category, sub_cat)
    mne_files = deep_eeg_core.get_files_with_mne(path)
    files_wless_epochs, files_wless_samples, ideal_samples_amount = deep_eeg_core.check_files(mne_files, prev_time)
    
    # Creción del diccionario que contendrá el 100% de la información
    consolidated_data = {
                'Menos epochs': files_wless_epochs,
                'Menos muestras (ideal: '+str(ideal_samples_amount)+')': files_wless_samples
            }
    
    all_files.update({ 
                sub_cat: consolidated_data
            })

#%% 

# =============================================================================
# Mostrar resultados del paso anterior
# =============================================================================
list(consolidated_data.items())





