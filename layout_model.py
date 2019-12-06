# Implementación del modelo de la Figura 1 del artículo Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human brain mapping, 38(11), 5391-5420.

from keras.layers import Conv1D, Conv2D, BatchNormalization, MaxPooling1D, MaxPooling2D, Activation, Flatten, Dense, Reshape
from keras.models import Sequential
from keras.optimizers import SGD

n_files = 2192
n_channels = 44
n_samples = 522
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
model.add(Conv2D(filters=25, kernel_size=(44,1), activation='linear', strides=1, input_shape=input_shape, data_format='channels_last'))
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






