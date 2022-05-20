
class DefualtConfig(object):

    ###################################################################
    # Model
    model_name = 'ViT'
    model_path = './model.pth'
    ema_path = './ema.pth'
    load_model = False
    num_classes = 219

    # Model : ViT
    pretrained_model = 'google/vit-base-patch16-224-in21k'

    ###################################################################
    # Training
    start_epoch = 0
    num_epochs = 100
    earlyStop_interval = 25

    batch_size = 16
    lr = 0.0002
    lr_warmup_epoch = 0

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