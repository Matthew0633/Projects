# import
import json

# read config from json file

def save_config():
    # temp_config
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

    with open('config.json', 'w') as outfile:
        json.dump(config, outfile)

    print('config saved!')
    outfile.close()

def read_json(args):
    """read config from json file

    :param args: parser arguments
    :return: config dict
    """
    json_path = args.config_file
    config = json.load(json_path)
    return config

# generate config
def get_config(args, is_train = True):
    """ get config dict from argparser

    :param args: parser arguments
    :param is_train: train or test
    :return: config
    """
    config={}

    if is_train:
        config['general']['labels'] = args.labels
        config['general']['img_w'] = args.img_w
        config['general']['img_h'] = args.img_h
        config['general']['train_dir'] = args.train_dir
        config['general']['train_csv_dir'] = args.train_csv_dir

        config['model']['n_block'] = args.n_block
        config['model']['kernel_size'] = args.kernel_size
        config['model']['pool_size'] = args.pool_size
        config['model']['n_filters'] = args.n_filters
        config['model']['n_dense_hidden'] = args.n_dense_hidden
        config['model']['dropout_conv'] = args.dropout_conv
        config['model']['dropout_dense'] = args.dropout_dense

        config['model']['learning_rate'] = args.learning_rate
        config['model']['batch_size'] = args.batch_size

    else:
        config['testdir'] = args.testdir
        config['voice_dir'] = args.v
        config['model_dir'] = args.model_dir

    return config

if __name__ == '__main__':
    save_config()