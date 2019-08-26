from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image 

img_width, img_height = 128, 96

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 606
nb_validation_samples = 114
epochs = 100
batch_size = 20

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True )

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples//batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=nb_validation_samples//batch_size)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save_weights('first_try.binary')

img_pred1 = image.load_img('data/validation/edible/Lactariusrufus4.jpg', target_size = (128, 96))
img_pred2 = image.load_img('data/validation/edible/Boletusbadius3.jpg', target_size = (128, 96))
img_pred3 = image.load_img('data/validation/edible/Kuehneromycesmutabilis2.jpg', target_size = (128, 96))
img_pred4 = image.load_img('data/validation/edible/Lepistanuda4.jpg', target_size = (128, 96))
img_pred5 = image.load_img('data/validation/edible/Morchellaelata1.jpg', target_size = (128, 96))
img_pred6 = image.load_img('data/validation/inedible/Amanitaceciliae2.jpg', target_size = (128, 96))
img_pred7 = image.load_img('data/validation/inedible/Entolomavernum2.jpg', target_size = (128, 96))
img_pred8 = image.load_img('data/validation/inedible/Tricholomasejunctum1.jpg', target_size = (128, 96))
img_pred9 = image.load_img('data/validation/inedible/Cortinariustraganus3.jpg', target_size = (128, 96))
img_pred10 = image.load_img('data/validation/inedible/Tricholomopsisdecora2.jpg', target_size = (128, 96))
img_pred1 = image.img_to_array(img_pred1)
img_pred2 = image.img_to_array(img_pred2)
img_pred3 = image.img_to_array(img_pred3)
img_pred4 = image.img_to_array(img_pred4)
img_pred5 = image.img_to_array(img_pred5)
img_pred6 = image.img_to_array(img_pred6)
img_pred7 = image.img_to_array(img_pred7)
img_pred8 = image.img_to_array(img_pred8)
img_pred9 = image.img_to_array(img_pred9)
img_pred10 = image.img_to_array(img_pred10)

img_pred1 = np.expand_dims(img_pred1, axis = 0)
img_pred2 = np.expand_dims(img_pred2, axis = 0)
img_pred3 = np.expand_dims(img_pred3, axis = 0)
img_pred4 = np.expand_dims(img_pred4, axis = 0)
img_pred5 = np.expand_dims(img_pred5, axis = 0)
img_pred6 = np.expand_dims(img_pred6, axis = 0)
img_pred7 = np.expand_dims(img_pred7, axis = 0)
img_pred8 = np.expand_dims(img_pred8, axis = 0)
img_pred9 = np.expand_dims(img_pred9, axis = 0)
img_pred10 = np.expand_dims(img_pred10, axis = 0)

rslt1 = model.predict(img_pred1)
rslt2 = model.predict(img_pred2)
rslt3 = model.predict(img_pred3)
rslt4 = model.predict(img_pred4)
rslt5 = model.predict(img_pred5)
rslt6 = model.predict(img_pred6)
rslt7 = model.predict(img_pred7)
rslt8 = model.predict(img_pred8)
rslt9 = model.predict(img_pred9)
rslt10 = model.predict(img_pred10)

print(rslt1)
print(rslt2)
print(rslt3)
print(rslt4)
print(rslt5)
print(rslt6)
print(rslt7)
print(rslt8)
print(rslt9)
print(rslt10)
