# import
import argparse
from IPython.display import display
from glob import glob
from tqdm import tqdm

from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img
from playsound import playsound

from test_helper import load_prep_img
from utils  import get_config

def main(config):

    test_dir = config['test_dir']
    voice_dir = config['voice_dir']
    model_dir = config['model_dir']

    categories = ['safe', 'danger']

    # load images
    img_paths = glob(test_dir + '/*.bmp') # need update to cv2 video frames

    for path in tqdm(img_paths):
        image = load_img(path)
        display(image)
        image_arr_dm4 = load_prep_img(path)

    # Showing if the baby's in safe or danger
    if 'eat' in path[:-3]:
        model = models.load_model(model_dir + '/Baby_eat.h5')  # load pretrained model
        categories = ['difficult..', 'danger', 'safe']
    elif 'door' in path[:-3]:
        model = models.load_model(model_dir + '/Baby_door.h5')
        categories = ['difficult..', 'danger', 'safe']
    elif 'fall' in path[:-3]:
        model = models.load_model(model_dir + '/Baby_fall.h5')
        categories = ['difficult..', 'danger', 'safe']
    elif 'kitchen' in path[:-3]:
        model = models.load_model(model_dir + '/Baby_kitchen.h5')
        categories = ['difficult..', 'safe', 'danger']
    else:
        print('사진에 적합한 모델이 없습니다.')

    predict = model.predict_classes(image_arr_dm4)
    answer = categories[int(predict)]
    print(answer)

    # Playing Mom's voice in each situation, Mom and baby hears it immediately
    if ('eat' in path[:-3]) & (answer == 'danger'):
        return playsound(voice_dir + '/EatWarning.mp3')
    elif ('door' in path[:-3]) & (answer == 'danger'):
        return playsound(voice_dir + '/DoorWarning.mp3')
    elif ('fall' in path[:-3]) & (answer == 'danger'):
        return playsound(voice_dir + '/FallWarning.mp3')
    elif ('kitchen' in path[:-3]) & (answer == 'danger'):
        return playsound(voice_dir + '/FallWarning.mp3')


if __name__ == '__main__':
    # argparse -> config
    parser = argparse.ArgumentParser(description='Test parser')
    parser.add_argument('-testdir', '--test_dir', help='test_dir')
    parser.add_argument('-v', '--voice_dir', help='voice_dir')
    parser.add_argument('-v', '--model_dir', help='model_dir')

    # generate config
    config = get_config(parser.args, is_train = False)

    # run test
    main(config)