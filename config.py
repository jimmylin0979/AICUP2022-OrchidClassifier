class DefualtConfig(object):

    ###################################################################
    # Model
    model_name = 'STN_ViT'
    pretrained_model = 'google/vit-base-patch16-224-in21k'
    # pretrained_model = 'google/vit-base-patch32-384'
    
    # model_name = 'ConvNeXt'
    # # pretrained_model = 'facebook/convnext-base-224'
    # pretrained_model = 'facebook/convnext-base-384'

    # model_name = 'Swin_ViT'
    # # pretrained_model = 'microsoft/swin-base-patch4-window7-224'
    # pretrained_model = 'microsoft/swin-base-patch4-window12-384'

    # model_name = 'CVT'
    # pretrained_model = 'microsoft/cvt-w24-384-22k'
    
    resize = 384

    ###################################################################

    model_path = 'model.pth'
    ema_path = 'ema.pth'
    load_model = False
    num_classes = 219

    do_MixUp = True

    do_cutMix = True
    beta = 1.0
    
    mix_prob = 0.2

    ###################################################################
    # Training
    start_epoch = 0
    num_epochs = 105
    earlyStop_interval = 600

    do_semi = False
    semi_start_epoch = 40

    batch_size = 8
    lr = 5e-5
    lr_warmup_epoch = 5
    cosine_tmax = 101

    ###################################################################
    # GPU Settings
    # use_gpu = True
    use_gpu_index = 0

    ###################################################################
    # DataLoader
    num_workers = 6

    # Dataset
    trainset_path = './data/dataset/train'
    unlabeledset_path = './data/dataset/unlabeled'
    testset_path = './data/dataset/test'

    train_valid_split = 0.2  # ratio of valid set

    ###################################################################

    def __init__(self) -> None:
        '''
        '''
        pass

    def parse(self, kwargs):
        '''
        '''
        print('User config : ')
        pass
