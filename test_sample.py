import os
from glob import glob

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img

from test_helper import load_prep_img


def test_sample(config):
    """test sample images with pretrained model

    requirement: locate sample files at "test_samples" directory

    """
    labels = config['general']['labels']

    # load test_data
    img_paths = glob('test_samples/*.bmp')

    # load model
    model = load_model('pretrained_model.h5')
    #model = Pretrained_baby_CNN(config) # pretrained_model
    #model = model.load_weights()

    # load test images and predict
    for path in tqdm(img_paths):
        img_arr4 = load_prep_img(path)
        yhat = model.predict(img_arr4)
        idx = np.argmax(yhat[0])

        plt.imshow(load_img(path))
        plt.title(os.path.basename(path))
        print('%s (%.2f%%)' % (labels[idx], yhat[0][idx] * 100))

if __name__ == '__main__':

    config = {'general': {'labels': ['difficult..','safe','danger'],
                          'img_w': 500,
                          'img_h': 275,
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

    test_sample(config)