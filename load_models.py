#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:02:14 2019

@author: wilsonforero
"""


# =============================================================================
# Carga de un modelo guardado
# Pero no fue terminado realmente, ir a training.py
# =============================================================================


from keras.models import model_from_json
model_name='model_1'
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
loaded_model.evaluate(x=X_test, y=Y_test, batch_size=200, verbose=1)
