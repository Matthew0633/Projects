from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# [load_data]
TRAIN_DIR = 'videos/images/1/train/'
TEST_DIR = 'videos/images/1/train/'
LR = 0.001
BATCH_SIZE = 30

IM_WIDTH = 500  # generated img size
IM_HEIGHT = 275

train_datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
                    TRAIN_DIR,
                    target_size = (IM_WIDTH, IM_HEIGHT),
                    batch_size = BATCH_SIZE,
                    class_mode = 'categorical',
                    shuffle = True)

test_generator = test_datagen.flow_from_directory(
                    TEST_DIR,
                    target_size=(IM_WIDTH, IM_HEIGHT),
                    batch_size=BATCH_SIZE,
                    class_mode='categorical',
                    shuffle=False)