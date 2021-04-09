import tensorflow as tf

from models import pretrained_baby_CNN
from train_helper import viz_history


def main(config):
      CASE = 'door' # door, eat, fall, kitchen

      TRAIN_DIR = 'videos/images/1/train/'
      TEST_DIR = 'videos/images/1/train/'

      LR = config['train']['learning_rate']
      BATCH_SIZE = config['train']['batch_size']

      model = pretrained_baby_CNN(config)

      print(model.summary())

      model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(lr = LR), metrics = ['accuracy'])

      print(model.summary())

      history = model.fit_generator(
            train_generator,
            steps_per_epoch = (train_generator.samples / train_generator.batch_size) ,
            epochs = 10,
            validation_data = test_generator,
            validation_steps = test_generator.samples / test_generator.batch_size,
            verbose = 1)

      # show train history
      viz_history(history)

      if not os.path.isdir('./pretrained_model'):
          os.mkdir('./pretrained_model')

      # save model
      tf.keras.models.save_model('pretrained_model/Baby_{}.h5'.format(CASE))

if __name__ == '__main__':
      # argparse -> config
      

      # temp_config
      config = {'general': {'IM_WIDTH': 500, 'IM_HEIGHT': 275},
                'model': {'class_num': 2,
                          'n_block': 5,
                          'kernel_size': (3, 3),
                          'pool_size': (2, 2),
                          'n_filters': [32, 64, 128],
                          'n_dense_hidden': 1024,
                          'dropout_conv': 0.3,
                          'dropout_dense': 0.3},
                'train': {'learning_rate': 0.001, 'batch_size': 256}
                }
      
      # run train
      main(config)



