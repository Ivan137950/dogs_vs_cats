#from cgi import test
import os, shutil


# original_dataset_dir = "/Users/limon/for_saves/photos_of_d&c/train"
base_dir = "Users/limon/OneDrive/Рабочий стол/new"

#os.mkdir(base_dir)

train_dir=os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir=os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir=os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

train_cats_dir=os.path.join(train_dir, 'cats')
#os.mkdir(train_cats_dir)
train_dogs_dir=os.path.join(train_dir, 'dogs')
#os.mkdir(train_dogs_dir)

validation_cats_dir=os.path.join(validation_dir, 'cats')
#os.mkdir(validation_cats_dir)
validation_dogs_dir=os.path.join(validation_dir, 'dogs')
#os.mkdir(validation_dogs_dir)

test_cats_dir=os.path.join(test_dir, 'cats')
#os.mkdir(test_cats_dir)
test_dogs_dir=os.path.join(test_dir, 'dogs')
#os.mkdir(test_dogs_dir)


# fnames = [f'cat.{i}.jpg' for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]
# for fname in fnames:
#      src = os.path.join(original_dataset_dir, fname)
#      dst = os.path.join(test_cats_dir, fname)
#      shutil.copyfile(src, dst)

# fnames = [f'dog.{i}.jpg' for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = [f'dog.{i}.jpg' for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)


from keras import layers
from keras import models

#1st model //       acc == 74%

#model=models.Sequential()
# model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Conv2D(64,(3,3), activation='relu'))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Conv2D(128,(3,3), activation='relu'))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Conv2D(128,(3,3), activation='relu'))
# model.add(layers.MaxPool2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# #print( model.summary())

from keras import optimizers
#model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

# keras.preprocessing.image     ///  module for working with images in keras

from keras.preprocessing.image import ImageDataGenerator

# train_datagen=ImageDataGenerator(rescale=1./255)
# test_datagen=ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#      # This is the target directory
#         train_dir,
#         # All images will be resized to 150x150
#         target_size=(150, 150),
#         batch_size=20,
#         # Since we use binary_crossentropy loss, we need binary labels
#         class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')
 

# history = model.fit(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)
# model.save('cats_and_dogs_small_1.keras')




# data augmentation   // EXAMPLE

# datagen = ImageDataGenerator(
#       rotation_range=40, # angle of rotarion
#       width_shift_range=0.2, # transport picture in horizontal
#       height_shift_range=0.2, # transport picture in vertical
#       shear_range=0.2,  # shearing transforms
#       zoom_range=0.2,
#       horizontal_flip=True,   #random flipping
#       fill_mode='nearest') # strategy of filling pixels

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
 input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))   # !!!!!!!!!!!!!!
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])

train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
 
test_data=ImageDataGenerator(1./255)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')
validation_generator=train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

history=model.fit(
    train_generator,
    steps_per_epoch=62,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_2.keras')

#  # plots:

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()