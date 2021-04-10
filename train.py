import os
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from models import Baby_CNN, Pretrained_baby_CNN
from train_helper import viz_history
from preprocessing import preprocess_data

from utils  import read_json, get_config

def main(config):
    CASE = 'door' # door, eat, fall, kitchen

    LR = config['train']['learning_rate']
    BS = config['train']['batch_size']

    train_generator, val_generator = preprocess_data(config, is_test = False, from_dir = True)

    #model = Baby_CNN(config)
    model = Pretrained_baby_CNN(config)

    print(model.build_summary())

    model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr = LR), metrics = ['accuracy'])

    # callbacks
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    LRonPlateau = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                                verbose=1, mode='max', min_lr=0.00001)
    checkpoint = ModelCheckpoint('best_model1.h5', monitor='val_acc', verbose=1,
                               save_best_only=True, save_weights_only=True)

    callbacks = [earlystopping, LRonPlateau, checkpoint]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = (train_generator.samples / train_generator.batch_size) ,
        epochs = 10,
        validation_data = val_generator,
        validation_steps = val_generator.samples / BS,
        verbose = 1, callbacks=callbacks)

    # show train history
    viz_history(history)

    if not os.path.isdir('./pretrained_model'):
      os.mkdir('./pretrained_model')

    # save model
    tf.keras.models.save_model('pretrained_model/Baby_{}.h5'.format(CASE))

if __name__ == '__main__':
    # argparse -> config
    parser = argparse.ArgumentParser(description='Train parser')
    parser.add_argument('-c', '--config_file', help='config_file')
    parser.add_argument('-l', '--labels', help='labels', action='append')
    parser.add_argument('-w', '--img_w', help='img_w')
    parser.add_argument('-h', '--img_h', help='img_h')
    parser.add_argument('-traindir', '--train_dir', help='train_dir')
    parser.add_argument('-traincsv', '--train_csv_dir', help='train_csv_dir')
    parser.add_argument('-b', '--n_block', help='n_block')
    parser.add_argument('-k', '--kernel_size', help='kernel_size')
    parser.add_argument('-p', '--pool_size', help='pool_size')
    parser.add_argument('-f', '--n_filters', help='n_filters', action = 'append')
    parser.add_argument('-d', '--n_dense_hidden', help='n_dense_hidden')
    parser.add_argument('-dr_c', '--dropout_conv', help='dropout_conv')
    parser.add_argument('-dr_d', '--dropout_dense', help='dropout_dense')
    parser.add_argument('-lr', '--learning_rate', help='learning_rate')
    parser.add_argument('-bs', '--batch_size', help='batch_size')
    args = parser.parse_args()

    # generate config
    if args.config_file:
        config = read_json(args)
    else:
        config = get_config(args, is_train = True)

    # run train
    main(config)



