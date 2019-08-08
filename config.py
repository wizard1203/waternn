from pprint import pprint
from pprint import pformat
import os
import logging

class Config:
    # data
    data_dir = ' '
    out_pred_dir = '/home/zhtang/water/txt/'
    out = 'predict'

    nets = ['waternetsmallfl', 'watercnndsnetf_in4_out58', 'waterdsnetf_in4_out58', 'waterdsnetf_self_define']

    # pretrained
    pretrained = None

    # architecture of network
    customize = True
    arch = 'waternet'
    growth_rate = 128
    num_init_features = 1536
    num_classes = 34

    train_num_workers = 8
    test_num_workers = 8

    # optimizers
    optim = 'SGD'
    use_adam = False

    # param for optimizer
    # lr = {}
    # weight_decay = {}
    # lr_decay = {}

    # lr['waterdsnetf_self_define'] = 0.6
    # weight_decay['waterdsnetf_self_define'] = 0.00005
    # lr_decay['waterdsnetf_self_define'] = 0.33

    # lr['waternetsmallfl'] = 0.01
    # weight_decay['waternetsmallfl'] = 0.00005
    # lr_decay['waternetsmallfl'] = 0.33

    # lr['watercnndsnetf_in4_out58'] = 0.1
    # weight_decay['watercnndsnetf_in4_out58'] = 0.00005
    # lr_decay['watercnndsnetf_in4_out58'] = 0.33

    # lr['waterdsnetf_in4_out58'] = 0.2
    # weight_decay['waterdsnetf_in4_out58'] = 0.00005
    # lr_decay['waterdsnetf_in4_out58'] = 0.33

    activation = 'relu'

    lr = 0.6
    weight_decay = 0.00005
    lr_decay = 0.33  #

    # record i-th log
    kind = '0'

    # set gpu :
    # gpu = True

    # visualization
    env = 'water-nn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'water'

    # training
    epoch = 120

    # if eval
    evaluate = False

    test_num = 10000
    # model
    load_path = None
    save_path = '~/water/modelparams'

    # len(labels_dict) == 34
    labels_dict_34 = (1036, 1066, 1064, 1032, 1004, 1038, 1040, 1030, 1014,
    1008, 1002, 1006, 1078, 1010, 1046, 1052,
    1056, 1080, 1060, 1018, 1020, 1016, 1022, 1026, 1042,
    1024, 1028, 1062, 1044, 1058, 1048, 1050, 1034, 1012
    )

    # len(labels_dict) == 58
    labels_dict_58 = (379, 385, 390, 391, 392, 406, 414, 415, 416, 417, 418, 419, 420, 422,
    425, 434, 435, 436, 438, 439, 440, 441, 443, 444, 445, 446, 447, 448, 449,
    450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 464, 465, 466, 468, 512,
    513, 514, 515, 517, 518, 519, 520, 557, 558, 559, 560, 561, 562
    )
    labels_dict = ()

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
        if opt.customize:
            logging_name = 'log' + '_self_' + opt.arch + '_'+ opt.optim + opt.kind + '.txt' 
        else:
            logging_name = 'log' + '_default_' + opt.arch  + '_' + opt.optim + opt.kind + '.txt'
        if not os.path.exists('log'):
            os.mkdir('log')

        if opt.arch == 'waterdsnetf':
            self.labels_dict = self.labels_dict_34
        elif opt.arch == 'waterdsnetf_in4_out58':
            self.labels_dict = self.labels_dict_58
        elif opt.arch == 'waterdsnetf_self_define':
            self.labels_dict = self.labels_dict_34

        logging_path = os.path.join('log', logging_name) 
    
        logging.basicConfig(level=logging.DEBUG,
                        filename=logging_path,
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')
        logging.info('Logging for {}'.format(opt.arch))
        logging.info('======user config========')
        logging.info(pformat(self._state_dict()))
        logging.info('==========end============')
        # logging.info('optim : [{}], batch_size = {}, lr = {}, weight_decay= {}, momentum = {}'.format( \
        #                 args.optim, args.batch_size,
        #                 args.lr, args.weight_decay, args.momentum) )


    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}
    

opt = Config()
