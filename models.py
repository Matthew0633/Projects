import os
import numpy as np
from IPython.display import display

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


class Baby_CNN(tf.keras.Model):
    """ subclass of tf.keras.Model

        Argument:
        config --- dict

    """
    def __init__(self, config: dict):
        super(Baby_CNN, self).__init__()

        # hyperparameters
        self.w, self.h = config['general']['img_w'], config['general']['img_h']
        self.kernel_size = config['model']['kernel_size']
        self.pool_size = config['model']['pool_size']
        self.n1_filter, self.n2_filter, self.n3_filter = config['model']['n_filters']
        self.n_dense_hidden = config['model']['n_dense_hidden']
        self.dropout_conv = config['model']['dropout_conv']
        self.dropout_dense = config['model']['dropout_dense']
        self.n_block = config['model']['n_block']
        self.class_num = len(config)

        # layers
        self.input_layer = tf.keras.layers.Input((self.w, self.h, 3))
        self.cnninput = tf.keras.layers.Conv2D(filters=self.n1_filter, kernel_size=5, activation=tf.nn.relu,
                                            padding='SAME')
        self.cnn = tf.keras.layers.Conv2D(filters=self.n2_filter, kernel_size=5, activation=tf.nn.relu,
                                          padding='SAME')
        self.maxpool = tf.keras.layers.MaxPool2D(padding='SAME')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.n_dense_hidden, activation=tf.nn.relu)
        self.dropout_conv = tf.keras.layers.Dropout(self.dropout_conv)
        self.dropout_dense = tf.keras.layers.Dropout(self.dropout_dense)
        self.outputs = tf.keras.layers.Dense(self.class_num, activation='softmax')

    def call(self, x):
        x = self.cnninput(x)
        for _ in range(self.n_block):
            x = self.cnn(x)
            x = self.maxpool(x)
            x = self.dropout_conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout_dense(x)
        x = self.outputs(x)

        return x

    def build_summary(self):
        built_model = tf.keras.Model(inputs=[self.input_layer], outputs=[model.call(self.input_layer)])
        print(built_model.summary())


class Pretrained_baby_CNN(Baby_CNN):
    def __init__(self, config: dict, pretrained_model: str = 'vgg19'):
        super(Pretrained_baby_CNN, self).__init__()

        self.model_names = ['resnet50', 'vgg19', 'inceptionresnet19', 'densenet', 'nasnet', 'mohbilenetv2']

        self.dropout_dense = 0.1
        self.dropout_dense = tf.keras.layers.Dropout(self.dropout_dense)
        self.resnet50 = ResNet50(weights='imagenet', include_top=False,
                                                        input_shape=(self.w, self.h, 3))
        self.vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(self.w, self.h, 3),
                             pooling=None, classes=2)
        self.inceptionresnet19 = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(self.w, self.h, 3),
                                                     pooling=None, classes=2)
        self.densenet = DenseNet201(include_top=False, weights='imagenet', input_shape=(self.w, self.h, 3),
                                    pooling=None, classes=2)
        self.nasnet = NASNetLarge(include_top=False, weights='imagenet', input_shape=(self.w, self.h, 3),
                                  pooling=None, classes=2)
        self.mohbilenetv2 = MobileNetV2(alpha=1.0, include_top=False, weights='imagenet', input_shape=(self.w, self.h, 3),
                                        pooling=None, classes=2)

        self.model_list = [self.resnet50, self.vgg19, self.inceptionresnet19, self.densenet, self.nasnet, self.mohbilenetv2]
        self.pretrained_layer = self.models[self.model_list.index(pretrained_model)]

    def call(self, x):
        x = self.pretrained_layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout_dense(x)
        x = self.output(x)
        return x

if __name__ == '__main__':
    config = {'general': {'img_w': 500, 'img_h': 275,
                          'train_dir': 'videos/images/1/train/',
                          'test_dir': 'videos/images/1/train/',
                          'train_csv_dir': './csvdata',
                          'test_csv_dir': './csvdata'
                          },
              'model': {'n_block': 5,
                        'kernel_size': (3, 3),
                        'pool_size': (2, 2),
                        'n_filters': [32, 64, 128],
                        'n_dense_hidden': 1024,
                        'dropout_conv': 0.3,
                        'dropout_dense': 0.3},
              'train': {'learning_rate': 0.001, 'batch_size': 256}
              }

    model = Baby_CNN(config)
    model = Pretrained_baby_CNN(config, pretrained_model='vgg19')

    model.build_summary()
