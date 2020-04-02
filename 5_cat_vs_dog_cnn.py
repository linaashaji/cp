import numpy as np
import os
import tensorflow as tf

work_dir = 'dogs_vs_cats_dataset/data'

image_height, image_width = 150, 150
train_dir = os.path.join(work_dir, 'train')
test_dir = os.path.join(work_dir, 'test')
no_classes = 2
no_validation = 800
epochs = 2
batch_size = 200
no_train = 2000
no_test = 800
input_shape = (image_height, image_width, 3)
epoch_steps = no_train // batch_size
test_steps = no_test // batch_size


def simple_cnn(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

simple_cnn_model = simple_cnn(input_shape)

generator_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)


# Download the train dataset and test dataset,
# extract them into 2 different folders named as “train” and “test”.
# The train folder should contain ‘n’ folders each containing images of respective classes.
# For example, In the Dog vs Cats data set, the train folder should have 2 folders, 
# namely “Dog” and “Cats” containing respective images inside them.

train_images = generator_train.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height))


# batch_size: Set this to some number that divides your total number of images in your test set exactly.
# Why this only for test_generator?
# Actually, you should set the “batch_size” in both train and valid generators to some number 
# that divides your total number of images in your train set and valid respectively, 
# but this doesn’t matter before because even if batch_size doesn’t match the number 
# of samples in the train or valid sets and some images gets missed out every time 
# we yield the images from generator, it would be sampled the very next epoch you train.
# But for the test set, you should sample the images exactly once, no less or no more. 
# If Confusing, just set it to 1(but maybe a little bit slower).

test_images = generator_test.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    target_size=(image_width, image_height))

simple_cnn_model.fit_generator(
    train_images,
    steps_per_epoch=epoch_steps,
    epochs=epochs,
    validation_data=test_images,
    validation_steps=test_steps)

