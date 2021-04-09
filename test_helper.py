from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from IPython.display import display
from IPython.core.display import Image
from keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from playsound import playsound

def load_prep_img(file):
    image = load_img(file)
    display(image)

    IM_WIDTH = 500
    IM_HEIGHT = 275

    image_resize = load_img(file, target_size=(IM_WIDTH, IM_HEIGHT))
    image_arr_dm3 = img_to_array(image_resize)
    image_arr_dm4 = image_arr_dm3.reshape((1, IM_WIDTH, IM_HEIGHT, 3))
    return image_arr_dm4

if __name__ == __main__:
    pass