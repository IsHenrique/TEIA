from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image 
from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.models import Model

img_width, img_height = 128, 96

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 606
nb_validation_samples = 114
epochs = 10
batch_size = 20
classes = 2

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True )

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
input_tensor = Input(shape=(128, 96, 3))
pre_trained_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(128,96,3))

for layer in pre_trained_model.layers[:]:
    layer.trainable = True

x = pre_trained_model.output
x = Flatten()(x)
#x = Dense(32, activation='relu')(x)
predictions = Dense(classes, activation='softmax')(x)

model = Model(inputs=pre_trained_model.input, outputs=predictions)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples//batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=nb_validation_samples//batch_size)



