# Deep Learning aplicado a señales EEG
En este repositorio se encuentran los diferentes archivos empleados para la implementación de un modelo de deep learning encargado de clasificar señales EEG en dos grandes categorías
- Emoticón 
- Cara Real

Esto teniendo una señal EEG (resultado de usar como estímulo uno de estos tipos) a la entrada del modelo pre-procesada.


------------


### Scripts del repositorio:
- **data_review.py**: Se encarga de hacer la revisión del data set, revisa cada archivo y en caso de que no cuente con las muestras que debería tener ese archivo será incluído en un diccionario, este es mostrado al final del script

- **deep_eeg_core.py**: En este script se encuentra el núcleo de este proyecto, es utilizado en todos los procesos, se recomienda revisar cada una de las funciones pues cuentan co su respectivo docstring definiendo que son cada uno de los parámetros, funcionamiento y qué retorna

> IMPORTANTE: Este script contiene variables correspondientes al nombre asignado a cada carpeta de sets, rutas que dependerán de cada computador, etiquetas entre otras... Éstas son usadas a lo largo del proyecto, por tanto es necesario actualizarlas de forma manual (en el código) para su correcto funcionamiento


- **load_models.py**: Script no finalizado pero incluído en el repositorio (no se usa en ninguno de los scripts)

- **models.py**: Tiene funciones para crear los modelos clasificador y autoencoder256 (sólo puede ser entrenado con datos de 256 muestras), este último aún en desarrollo

- **pre_process_data.py**: En este script se realizan las siguientes tareas: Elimar canales FP1 y FP2 (usados para EOG), normalización de los datos, almacenamiento en archivos terminados en*"-epo.fif"* y división en carpetas de **entrenamiento** y **test**, con porcentaje por defecto de 80% y 20% respectivamente

> La ecuación empleada para normalizar fue:

$$\frac{canal  -  \mu}{\sigma}$$

- **training.py**: Lectura de datos .fif, asignación de categoría para cada epoch, separación en arreglos de numpy X contiene epochs y Y contiene labels one hot, creación del modelo con la ayuda del script models.py, entrenamiento del modelo y almacenamiento del mismo


------------


### Dependencias:

- [NumPy](https://numpy.org/ "NumPy")
- [mne](https://mne.tools/stable/install/index.html "mne")
- [Scikit Learn](https://scikit-learn.org/stable/install.html "Scikit Learn")
- [Keras](https://keras.io/#installation "Keras")
- [TensorFlow](https://www.tensorflow.org/install/pip "TensorFlow")


------------

### Recomendaciones:

Se recomienda para trabajos a futuro desarrollar el modelo auto-encoder, haciendo una “ruptura” del modelo antes de la capa de aplanado “Flatten”, esto con el fin de incluir las capas inversas que se tienen hasta ese punto e inicialmente lograr recrear la señal original al modelo que no será realmente igual, pero si conservará la estructura y no presentará grandes variaciones entre los diferentes puntos, consiguiendo así una reconstrucción con menor ruido, logrando una estructura definida y probada del modelo CAE (convolutional auto-encoder). Es posible usar un modelo CAE como herramienta de visualización de las zonas en las que se presenta mayor actividad (activación) en el cerebro, todo esto acorde a la señal EEG presentada a la entrada del modelo.
