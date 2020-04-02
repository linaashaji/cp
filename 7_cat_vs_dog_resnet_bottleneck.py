#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 00:02:27 2020

@author: lachaji
"""

import numpy as np
import os
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D

work_dir = 'dogs_vs_cats_dataset/data'

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = os.path.join(work_dir, 'train')
validation_data_dir = os.path.join(work_dir, 'test')
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 16
input_shape = (img_height, img_width, 3)
epoch_steps = nb_train_samples // batch_size
test_steps = nb_validation_samples // batch_size


base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
base_model.trainable = False


global_average_layer = GlobalMaxPooling2D()(base_model.output)
prediction_layer = Dense(1, activation='sigmoid')(global_average_layer)

model = Model(inputs=base_model.input, outputs=prediction_layer)


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=epoch_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=test_steps)


################### FINE-TUNING ################
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
    
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.fit_generator(train_generator,  
                    epochs=5, 
                    validation_data=validation_generator)