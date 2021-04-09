from glob import glob
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from test_helper import load_prep_img
from models import pretrained_baby_CNN

config = {
'IM_WIDTH': 500,
'IM_HEIGHT': 275,
'class_num' : 2,
'n_block' : 5,
'kernel_size' : (3, 3),
'pool_size' : (2, 2),
'n_filters' : [32, 64, 128],
'n_dense_hidden' : 1024,
'dropout_conv' : 0.3,
'dropout_dense' : 0.3
}

# load test_data
img_paths = glob('test_samples/*.bmp')

# load model
model = load_model('pretrained_model.h5')
#model = pretrained_baby_CNN(config) # pretrained_model
#model = model.load_weights()

# load test images and predict
for path in tqdm(img_paths):
    img_arr4 = load_prep_img(path)
    yhat = model.predict(img_arr4)
    idx = np.argmax(yhat[0])
    print('%s (%.2f%%)' % (custom_labels[idx], yhat[0][idx] * 100))

if __name__ == '__main__':
    pass