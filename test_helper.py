from glob import glob

from keras.preprocessing.image import load_img, img_to_array
from IPython.display import display

def load_prep_img(file):
    image = load_img(file)
    display(image)

    IM_WIDTH = 500
    IM_HEIGHT = 275

    image_resize = load_img(file, target_size=(IM_WIDTH, IM_HEIGHT))
    image_arr_dm3 = img_to_array(image_resize)
    image_arr_dm4 = image_arr_dm3.reshape((1, IM_WIDTH, IM_HEIGHT, 3))
    return image_arr_dm4

if __name__ == '__main__':
    sample_path = glob('test_samples/*.bmp')[0]
    image_arr_dm4 = load_prep_img(sample_path)
    print(image_arr_dm4.shape)