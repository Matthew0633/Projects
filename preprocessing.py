import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def preprocess_data(config, is_test = False, from_dir = True):
    """ load and augment img data

        Arguments:
            config --- arg parsed config
            is_test --- train or test
            from_dir --- datagen.flow_from_directory or datagen.flow_from_dataframe

        Returns:
            - train_generator, val_generator (is_test=False)
            - test_generator (is_test=True)
    """


    IM_WIDTH = config['train']['img_w']  # generated img size
    IM_HEIGHT = config['train']['img_h']

    target_size = (IM_WIDTH, IM_HEIGHT)

    BS = config['train']['batch_size']

    train_dir = config['general']['train_dir']  # videos/images/1/train/'
    test_dir = config['general']['test_dir']  # 'videos/images/1/train/'

    train_csv_dir = config['general']['train_csv_dir']
    test_csv_dir = config['general']['test_csv_dir']


    traindf = pd.read_csv(train_csv_dir)
    testdf = pd.read_csv(test_csv_dir)

    if not is_test:
        datagen = ImageDataGenerator(rescale=1. / 255.,
                                     shear_range=.2,
                                     zoom_range=.2,
                                     horizontal_flip=True,
                                     validation_split=.1)

        if from_dir:
            train_generator = datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=BS,
                subset='traning',
                class_mode='categorical',
                shuffle=True)

            valid_generator = datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=BS,
                subset='validation',
                class_mode='categorical',
                shuffle=False)
            return train_generator, valid_generator

        else:
            train_generator = datagen.flow_from_dataframe(dataframe=traindf,
                                                          directory=train_dir,  # "./train/"
                                                          x_col="id",
                                                          y_col="label",
                                                          subset="training",
                                                          batch_size=BS,
                                                          seed=42,
                                                          shuffle=True,
                                                          class_mode="categorical",
                                                          target_size=target_size)

            valid_generator = datagen.flow_from_dataframe(dataframe=traindf,
                                                          directory=train_dir,  # "./train/"
                                                          x_col="id",
                                                          y_col="label",
                                                          subset="validation",
                                                          batch_size=BS,
                                                          seed=42,
                                                          shuffle=True,
                                                          class_mode="categorical",
                                                          target_size=target_size)
            return train_generator, valid_generator

    else:
        datagen = ImageDataGenerator(rescale=1. / 255.,
                                     shear_range=.2,
                                     zoom_range=.2,
                                     horizontal_flip=True)
        if from_dir:
            test_generator = datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=BS,
                class_mode='categorical',
                shuffle=False)
            return test_generator

        else:
            test_generator = datagen.flow_from_dataframe(dataframe = testdf,
                                                         directory= test_dir, # "./test/"
                                                         x_col="id",
                                                         y_col=None,
                                                         batch_size=BS,
                                                         seed=42,
                                                         shuffle=False,
                                                         class_mode=None,
                                                         target_size=target_size)
            return test_generator

if __name__ == '__main__':
    config = {'general': {'labels': ['difficult..','safe','danger'],
                          'img_w': 500, 'img_h': 275,
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

    train_generator, valid_generator = preprocess_data(config, is_test=False, from_dir=True)
    test_generator = preprocess_data(config, is_test=True, from_dir=True)
