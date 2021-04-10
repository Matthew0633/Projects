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
        config['general']['test_dir'] = args.test_dir
        config['general']['train_csv_dir'] = args.train_csv_dir
        config['general']['test_csv_dir'] = args.test_csv_dir

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