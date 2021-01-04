from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
import os
import time
from numba import jit


os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

rootPath = './datasets/cat-and-dog/'

imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.1
)

trainGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'training_set'),
    target_size=(64, 64),
    subset='training'
)

validationGen = imageGenerator.flow_from_directory(
    os.path.join(rootPath, 'training_set'),
    target_size=(64, 64),
    subset='validation'
)


model = Sequential()
model.add(ResNet50(include_top=True, weights=None, input_shape=(64, 64, 3), classes=2))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc'],
)


epochs = 1000


def train1():
    model.fit_generator(
        trainGen,
        epochs=epochs,
        steps_per_epoch=1,
        validation_data=validationGen,
        validation_steps=1,
    )

@jit(forceobj=True)
def train2():
    model.fit_generator(
        trainGen,
        epochs=epochs,
        steps_per_epoch=1,
        validation_data=validationGen,
        validation_steps=1,
    )

start = time.time()
train1()
print("===============================================================")
print("General Learning time : " + str(time.time() - start) + " " + "seconds")
print("===============================================================")

train2()
print("===============================================================")
print("Numba Learning time : " + str(time.time() - start) + " " + "seconds")
print("===============================================================")
