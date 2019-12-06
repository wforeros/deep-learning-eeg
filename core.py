#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wilsonforero
"""
import os.path as op
import os
import mne
import math
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile


# =============================================================================
# Sección de constantes
# =============================================================================


# Esta carpeta contiene TODOS los sets 
MAIN_SETS_FOLDER = '/Users/wilsonforero/Documents/Universidad/Proyecto/python/sets'
#MAIN_SETS_FOLDER = 'sets'

PACKAGE_1 = 'Set1-500ms'
PACKAGE_2 = 'Set2-100ms'
PACKAGE_3 = 'Set3-0ms'

RAW_SETS_FOLDER = op.join(MAIN_SETS_FOLDER, 'raw')

# Nombre de la carpeta qur contiene los sets clasificados, es decir el
# último paso de Matlab
CLASSIFIED_SETS_FOLDER = op.join(MAIN_SETS_FOLDER, 'Clasificado')

PROCESSED_WITH_PYTHON_FOLDER = op.join(MAIN_SETS_FOLDER, 'procesado_mne_python')

PROCESSED_WITH_PYTHON_FOLDER = op.join(MAIN_SETS_FOLDER, 'procesado_python')

PROCESSED_AND_CLASSIFIED_FOLDER = op.join(MAIN_SETS_FOLDER, 'procesado_clasificado_python')

TEST_SETS = op.join(MAIN_SETS_FOLDER, 'tests')

SETS_PACKAGES = [PACKAGE_1, PACKAGE_2, PACKAGE_3]
SUB_CATEGORIES = ['AlE', 'IrE', 'NE', 'IrR', 'NR', 'AlR']

REAL_SUB_CATEGORIES = ['AlR', 'NR', 'IrR']

LABELS = {'AlR': 0, 'NR': 1, 'IrR': 2, 'AlE': 3, 'NE': 4, 'IrE': 5}


# =============================================================================
# Sección de funciones
# =============================================================================

def label_to_onehot(label):
    assert (label < 6)
    one_hot = np.zeros(6)
    one_hot[label] = 1
    return one_hot
    
    
    
def subcategory_to_label(sub_category):
    """
    Parámetros:
    _____
    - sub_category (str): Debe ser una de las siguientes categorías: AlE, IrE, NE, IrR, NR o AlR
    Retorna:
    _____
    - label (int): Retorna la etiqueta correspondiente, si no encuentra la categoría retorna None
    """
    
    return LABELS.get(sub_category)

def get_category_from_subcategory(sub_category):
    """
    Parámetros:
    _____
    - sub_category (str): Debe ser una de las siguientes categorías: AlE, IrE, NE, IrR, NR o AlR
    Retorna:
    _____
    - category (int): Retorna la categoría, por defecto retorna 'Real'
    """
    category = 'Emoticon' if 'E' in sub_category else 'Real'
    return category

def get_mne_filename(mne_file, remove_extension=True):
    """
    Parámetros:
    _____
        mne_file: Archivo de mne al que se quiere extraer el nombre 
    Retorna:
    _____
        file_name (str): Nombre del archivo. Si remove_extension es falso entonces saldrá un solo string de la forma nombre.extension
    Ejemplo:
    _____
        >>> mne_file = mne.read_epochs_eeglab(90010_AlE.set')
        >>> file_name = utilities.get_mne_filename(mne_file)
    """
    
    
    # Retorna todo el directorio entonces se separa
    file_name = mne_file.filename.split('/')
    # Se toma el último valor que es el nombre
    file_name = file_name[len(file_name)-1]
    if remove_extension:
        file_name = file_name.split('.')[0]
    return file_name


            
def get_files_with_mne(directory, extension='.set'):
    """
    Esta función retorna los archivos abiertos con mne con la extensión especificada, 
    esta retorna un arreglo con los archivos abiertos.
    
    Args:
    ____
    - directory (str): Ruta que va al directorio en el cual se encuentran los archivos
    - extension (str): Extensión de los archivos que se desea leer (por defecto es .set)
    
    Returns:
    ____
    - array: Arreglo que contiene todos los sarchivos abiertos con mne
        
    Examples:
    ____
        Se requiere acceder a los sets de datos clasificados del set1
        para el caso de emoticones alegres, es decir la ruta: 
        /Users/wilsonforero/Documents/Universidad/Proyecto/python/sets/Set1-500ms/Emoticon/AlE
        Entonces se usa esta función
        
        >>> sets_package='Set1-500ms'
        >>> sub_category='AlE'
        >>> path = op.join(CLASSIFIED_SETS_FOLDER, sets_package, 'Emoticon', sub_category)
        >>> files=get_files_with_mne(path) # Retorna todos los archivos con esa extensión
    """
    sets = []
    for file in os.listdir(path=directory):
        if file.endswith(extension):
            file_path = op.join(directory, file)
            set_file = mne.read_epochs(file_path) if extension == '.fif' else mne.read_epochs_eeglab(file_path)
            if set_file is  None:
                print('\n\n\n====================================\n' +
                          'Fallo al cargar un archivo .set con el nombre:', file +
                          '\n====================================\n\n\n')
            else:
                sets.append(set_file)
    print('Un total de', len(sets), ' archivos fueron abiertos con mne')
    return sets#op.join(MAIN_SETS_FOLDER, path, file_name)
    
    
def check_files(mne_files, prev_time, epochs=16):
    """
    **Esta función comprueba que los sets tengan la misma cantidad de epochs y samples**
    ____
    Args:
    ____
        - mne_sets (array): Arreglo de con los sets ya abiertos con mne, para esto usar la función `~utilities.get_files_with_mne`
        - prev_time (int): El tiempo (en ms) que se tomó previo a la aparición del estímulo, por ejemplo: 500ms
        - epochs (int): La cantidad de epochs que debería tener cada archivo
    Returns:
    ____
        - files_wless_epochs: Diccionario que contiene los nombres de los archivos con menos epochs que la referencia
        - files_wless_samples: Diccionario que contiene los nombres de los archivos con menos muestras que las que debería tener (con fs=256Hz)
        - ideal_samples_amount (int): Valor de las muestras que deberían tener los archivos (acorde al prev_time)
    Examples:
    ____
        >>> import utilities
        >>> path = 'Users/usuario/una/ruta/cualquiera'
        >>> mne_files = utilities.get_files_with_mne(path)
        >>> files_wless_epochs, files_wless_samples = utilities.check_files(mne_files)
    """
    # Detección de qué archivos tienen menos de 16 epochs
    files_wless_epochs = {}
    for file in mne_files:
        epochs_amount = file.events.shape[0]
        if not (epochs_amount == epochs):
            # Retorna todo el directorio entonces se separa
            file_name = get_mne_filename(file, remove_extension=False)
            # En files quedan los archivos que tienen menos de 16 epochs
            files_wless_epochs.update({file_name: epochs_amount})
            
    #Se calcula el número de muestras que debería tener el archivo 
    #sabiendo que dura un segundo pero varía en el tiempo previo al estímulo
    #y se redondea al número superior
    ideal_samples_amount = math.ceil((prev_time/1000.)*256. + 256.)
    files_wless_samples = {}
    
    # Detección de qué archivos tienen menos muestras que las que deberían tener
    for file in mne_files:
        samples = file.times.shape[0]
        if not (samples == ideal_samples_amount):
            # Retorna todo el directorio entonces se separa
            file_name = get_mne_filename(file, remove_extension=False)
            # En files quedan los archivos que tienen menos de 16 epochs
            files_wless_samples.update({file_name: samples})
    return files_wless_epochs, files_wless_samples, ideal_samples_amount

 

def normalize_array(original_data, mode='epochs'):
    """
    Parámetros:
    _____
        - original_data: Arreglo 3D (epochs, canales, muestras) de los datos originales del archivo. Para cargar los archivos usar la función `~utilities.get_files_with_mne`
        - mode: Modo de normalización si desea normalizar los canales usar `channels`, para el caso de epochs cualquier otro valor
    
    Retorna:
    _____
        - arreglo: Un arreglo con los datos normalizados acorde al método propuesto por Cecotti y Gräser (2011)
    
    """
    epochs, channels, samples = original_data.shape
    normalized = np.zeros((epochs, channels, samples))
    if mode == 'epochs':
#        prom_per_channel = np.median(original_data, axis=0)
#        print(prom_per_channel.shape, 'TEST' )
#        standard_deviation_channels_np = np.std(original_data, axis = 0)
#        normalized_channels = np.copy(original_data)
        epochs, channels, samples = original_data.shape
        normalized = np.zeros((epochs, channels, samples))
        print('Tamaño', normalized.shape)
#        assert(prom_per_channel.shape[0] == standard_deviation_channels_np.shape[0])
        # El -1 es porque en ese canal solo se encuentran 0, por tanto esos valores se dejan así
        for channel in range(original_data.shape[1]):
            for epoch in range(original_data.shape[0]):
                # Se extraen las 256 muestras de ese epoch
                epoch_data = original_data[epoch, channel, :]
                # Promedio de las muestras de ese epoch
                prom_epoch = np.mean(epoch_data)
                # Desviacion de las muestras de ese epoch
                standard_deviation_epoch = np.std(epoch_data)
                for sample in range(original_data.shape[2]):
                    # Se toma el valor original
                    original_value = epoch_data[sample]
                    # Se toma el valor del promedio para ese canal en esa muestra
#                    prom_value = prom_per_channel[channel, sample]
                    # Se toma la desviación para ese canal en esa muestra
#                    std_deviation = standard_deviation_channels_np[channel, sample]
                    # Se opera acorde a la fórmula
#                    normalized_data = (original_value - prom_value) / std_deviation
                    # Se opera valor a valor
                    normalized_data = (original_value - prom_epoch) / standard_deviation_epoch
                    # Se almacena el valor normalizado
                    normalized[epoch, channel, sample] = normalized_data
        return normalized
    elif mode == 'test':
        max_value = 0
        abs_min_value = 0
        for channel in range(channels):
            ch_data = original_data[:, channel, :]
            max_value = np.amax(ch_data)
            abs_min_value = np.abs(np.amin(ch_data))
            value_to_norm = max_value if max_value > abs_min_value else abs_min_value
            for epoch in range(epochs):
                for sample in range(samples): 
                    normalized[epoch, channel, sample] = sample / value_to_norm
    else:
        epochs, channels, samples = original_data.shape
        normalized = np.zeros((epochs, channels, samples))
        print('Tamaño', normalized.shape)
#        assert(prom_per_channel.shape[0] == standard_deviation_channels_np.shape[0])
        # El -1 es porque en ese canal solo se encuentran 0, por tanto esos valores se dejan así
        for epoch in range(original_data.shape[0]):
            for channel in range(original_data.shape[1]):
                # Se extraen las 256 muestras de ese epoch
                ch_data = original_data[epoch, channel, :]
                # Promedio de las muestras de ese epoch
                prom_ch = np.mean(ch_data)
                # Desviacion de las muestras de ese epoch
                standard_deviation_ch = np.std(ch_data)
                for sample in range(original_data.shape[2]):
                    # Se toma el valor original
                    original_value = ch_data[sample]
                    # Se toma el valor del promedio para ese canal en esa muestra
#                    prom_value = prom_per_channel[channel, sample]
                    # Se toma la desviación para ese canal en esa muestra
#                    std_deviation = standard_deviation_channels_np[channel, sample]
                    # Se opera acorde a la fórmula
#                    normalized_data = (original_value - prom_value) / std_deviation
                    # Se opera valor a valor
                    normalized_data = (original_value - prom_ch) / standard_deviation_ch
                    # Se almacena el valor normalizado
                    normalized[epoch, channel, sample] = normalized_data
        return normalized

def create_directory(directory, print_msg=True):
    """
    Parámetros:
    _____
        - directory (str): Nombre del directorio que se quiere crear
    Ejemplo:
    _____
        >>> import utilities
        >>> directory = 'prueba/crear/directorio'
        >>> utilities.create_directory(directory)
        El directorio: "prueba/crear/directorio" ha sido creado
    """
    assert (isinstance(directory, str))
    msg = ''
    try:
        os.path
        os.makedirs(directory)    
        msg = 'El directorio: "' + directory +  '" ha sido creado'
    except FileExistsError:
        msg = 'El directorio: "' + directory +  '" ya existe'
    
    if print_msg:
        print(msg)

def split_sets(origin_folder, output_folder, train_folder_name='Train', test_folder_name='Test', test_size=20.):
    """
    Parámetros:
    _____
    - origin_folder (str): Carpeta en la que se encuentran los archivos que se van a utilizar 
    - output_folder (str): Carpeta en la que se guardarán los archivos separados
    - train_folder_name (str): Nombre de la carpeta de entrenamiento
    - test_folder_name (str): Nombre de la carpeta de pruebas
    - test_size (float): Porcentaje de archivos que serán usados para test
    Anotaciones:
    _____
    - Esta función NO crea el directorio de salida, este debe existir al momento de ejecutar esta función
    - Esta función sólo crea los directorios de prueba y entrenamiento dentro de la ruta 'output_folder'
    """
    print(output_folder)
    all_files = os.listdir(origin_folder)
    test_percentage = (float(test_size) / 100.)
    train, test = train_test_split(all_files,
                                   test_size=test_percentage,
                                   shuffle=True)
    # X_train, X_test, Y_train, Y_test = train_test_split(ans,ans, test_size=0.15, random_state=5)
    train_directory = op.join(output_folder, train_folder_name)
    test_directory = op.join(output_folder, test_folder_name)
    create_directory(train_directory)
    create_directory(test_directory)
    for file in train:
        file_path = op.join(origin_folder, file)
        destination_path = op.join(train_directory, file)
        copyfile(file_path, destination_path)

    for file in test:
        file_path = op.join(origin_folder, file)
        destination_path = op.join(test_directory, file)
        copyfile(file_path, destination_path)
    print('Proceso finalizado')
    
def create_model2():
    
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
    model.add(Conv2D(filters=25, kernel_size=(44,1), activation='linear', strides=1,  data_format='channels_last'))
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
    model.add(Conv1D(filters=200, kernel_size=(10), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=3))
    
    # CLASSIFICATION LAYER
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
    
    
    optimizer = SGD(lr=0.025)
    model.compile(optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
    
    #model.fit(x=X_train, y=Y_train, batch_size=200, epochs=50, verbose=1)
