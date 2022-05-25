#
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR

#
from torchvision import transforms

#
from torch_ema import ExponentialMovingAverage

# 
import numpy as np

#
from sklearn.model_selection import train_test_split

#
from ptflops import get_model_complexity_info

#
from tqdm import tqdm

#
import argparse

#
import os
import shutil

#
import models
from data.dataset import OrchidDataSet
from config import DefualtConfig
from utils import get_confidence_score
from utils import mixup_data, mixup_criterion
from utils import CutMixCollator, CutMixCriterion
from utils.self_supervised import get_pseudo_labels
from optim.scheduler import GradualWarmupScheduler

###################################################################################

config = DefualtConfig()
device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available() else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')

def main(logdir):

    # Step 1 : prepare logging writer
    writer = SummaryWriter(log_dir=logdir)

    # Step 2 : 
    print(config.model_name)
    model = getattr(models, config.model_name)(config)
    if config.load_model:
        model.load_state_dict(torch.load(config.model_path))
    model.to(device)

    # 
    # Metrics : FLOPs, Params
    resize = (224, 224) if config.model_name != 'ConvNeXt' else (384, 384)
    macs, params = get_model_complexity_info(model, (3, resize[0], resize[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('pthflops : ')
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # Step 3 : DataSets}
    # Data Augumentation
    transform_set = [
        transforms.RandomResizedCrop((resize[0])),
        # transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # transforms.RandAugment()
        # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
    ]

    transform_set = transforms.Compose([

        # # Reorder transform randomly
        transforms.RandomOrder(transform_set),

        # Resize the image into a fixed shape
        transforms.Resize(resize),

        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # 
        transforms.RandomErasing()
    ])
    ds = OrchidDataSet(config.trainset_path, transform_set=transform_set)
    ds_unlabeled = None
    if config.do_semi:
        ds_unlabeled = OrchidDataSet(config.unlabeledset_path, transform_set=transform_set)

    # Step 3
    # Deal with imbalance dataset
    #   For the classification task, we use cross-entropy as the measurement of performance.
    #   Since the wafer dataset is serverly imbalance, we add class weight to make it classifier better
    class_weights = [1 - (ds.targets.count(c))/len(ds) for c in range(config.num_classes)]
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # criterion = LabelSmoothingCrossEntropy()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-8)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    if config.load_model:
        ema = ema.load_state_dict(torch.load(config.ema_path))

    # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler_steplr = CosineAnnealingLR(optimizer, T_max=20)
    # scheduler_steplr = ExponentialLR(optimizer, gamma=0.9)
    # if config.lr_warmup_epoch > 0:
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=config.lr_warmup_epoch, after_scheduler=scheduler_steplr)

    # Step 4
    # train_loader, valid_loader = get_loader(ds)
    ds_train, ds_valid = get_train_valid_ds(ds)

    # Step 5
    history = {'train_acc' : [], 'train_loss' : [], 'valid_acc' : [], 'valid_loss' : []}
    best_epoch, best_loss = 0, 1e100
    nonImprove_epochs = 0

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    #
    assert not(config.do_cutMix and config.do_MixUp), "Only support one of the mix-based augmentation"

    for epoch in range(config.start_epoch, config.start_epoch + config.num_epochs):

        print('=' * 150)

        # 
        # if config.lr_warmup_epoch > 0:
        scheduler_warmup.step(epoch + 1)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        print(f'Epoch {epoch}, LR = {optimizer.param_groups[0]["lr"]}')

        # 
        collator = torch.utils.data.dataloader.default_collate
        if config.do_cutMix:
            collator = CutMixCollator(config.cutMix_alpha)
        
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, collate_fn=collator, num_workers=config.num_workers, pin_memory=True)
        # if epoch == 35:
        #     torch.save(model.state_dict(), f'{config.model_path[:-4]}_normal.pth')
        #     torch.save(ema.state_dict(), f'{config.ema_path[:-4]}_normal.pth')
            
        if epoch >= 35 and config.do_semi:
            # Obtain pseudo-labels for unlabeled data using trained model.
            print(f"[ Train | Start pseudo labeling]")
            pseudo_set = get_pseudo_labels(model, ds_unlabeled)

            if pseudo_set != None:
                # Construct a new dataset and a data loader for training.
                # This is used in semi-supervised learning only.
                concat_dataset = ConcatDataset([ds_train, pseudo_set])
                train_loader = DataLoader(concat_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator, num_workers=config.num_workers, pin_memory=True)
                
        valid_loader = DataLoader(ds_valid, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

        # 
        train_criterion = criterion
        if config.do_cutMix:
            train_criterion = CutMixCriterion(reduction='mean')

        train_acc, train_loss = train(model, train_loader, train_criterion, optimizer, ema)
        print(f"[ Train | {epoch + 1:03d}/{config.num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        
        # 
        valid_acc, valid_loss = valid(model, valid_loader, criterion, None)
        print(f"[ Valid | {epoch + 1:03d}/{config.num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        
        # 
        valid_acc_ema, valid_loss_ema = valid(model, valid_loader, criterion, ema)
        print(f"[ Valid | {epoch + 1:03d}/{config.num_epochs:03d} ] loss = {valid_loss_ema:.5f}, acc = {valid_acc_ema:.5f} (EMA)")
        

        # Append the training statstics into history
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        # Tensorboard Visualization
        writer.add_scalar("Train/train_acc", train_acc, epoch)
        writer.add_scalar("Valid/valid_acc", valid_acc, epoch)
        writer.add_scalar("Valid/valid_acc_ema", valid_acc_ema, epoch)
        writer.add_scalar("Train/train_loss", train_loss, epoch)
        writer.add_scalar("Valid/valid_loss", valid_loss, epoch)
        writer.add_scalar("Valid/valid_loss_ema", valid_loss_ema, epoch)

        # EarlyStop
        # if the model improves, save a checkpoint at this epoch
        if valid_loss_ema < best_loss:
            best_loss = valid_loss_ema
            best_epoch = epoch
            torch.save(model.state_dict(), f'{logdir}/{config.model_path}')
            torch.save(ema.state_dict(), f'{logdir}/{config.ema_path}')
            get_confidence_score(model, loader=valid_loader, use_gpu_index=config.use_gpu_index, batch_size=config.batch_size, outpu_file_path=f'{logdir}/prediction-Confidence-best.csv')
            print(f'Saving model with loss {valid_loss_ema:.4f}'.format(valid_loss_ema))
            nonImprove_epochs = 0
        else:
            nonImprove_epochs += 1

        # Stop training if your model stops improving for "config['early_stop']" epochs.    
        if nonImprove_epochs >= config.earlyStop_interval:
            break
    
    torch.save(model.state_dict(), f'{logdir}/last_{config.model_path}')
    torch.save(ema.state_dict(), f'{logdir}/last_{config.ema_path}')
    print(f'Best epoch: {best_epoch} with loss {best_loss}')

    writer.flush()
    writer.close()

    # Step 6 : Explanation & Visualization
    get_confidence_score(model, loader=valid_loader, use_gpu_index=config.use_gpu_index, batch_size=config.batch_size, outpu_file_path=f'{logdir}/last-prediction-Confidence.csv')

###################################################################################

def get_train_valid_ds(ds):

    # Split the train/test with each class should appear on both train/test dataset
    valid_split = config.train_valid_split

    indices = list(range(len(ds)))  # indices of the dataset
    train_indices, valid_indices = train_test_split(indices, test_size=valid_split, stratify=ds.targets)
    
    # Creating sub dataset from valid indices
    # Do not shuffle valid dataset, let the image in order
    valid_indices.sort()
    ds_valid = torch.utils.data.Subset(ds, valid_indices)

    ds_train = torch.utils.data.Subset(ds, train_indices)

    return ds_train, ds_valid

def get_loader(ds):

    # Split the train/test with each class should appear on both train/test dataset
    valid_split = config.train_valid_split

    indices = list(range(len(ds)))  # indices of the dataset
    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_split, stratify=ds.targets)
    
    # Creating sub dataset from valid indices
    # Do not shuffle valid dataset, let the image in order
    valid_indices.sort()
    ds_valid = torch.utils.data.Subset(ds, valid_indices)

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

    # Construct data loaders.
    train_loader = DataLoader(
        ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(
        ds_valid, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, valid_loader


def train(model, train_loader, criterion, optimizer, ema):
    
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        # imgs = (batch_size, 3, 224, 224)
        # labels = (batch_size)
        imgs, labels = batch
        imgs = imgs.to(device)

        if config.do_cutMix:
            if isinstance(labels, (tuple, list)):
                targets1, targets2, lam = labels
                labels = (targets1.to(device), targets2.to(device), lam)
            else:
                labels = labels.to(device)

        if config.do_MixUp:
            labels = labels.to(device)

        if config.do_MixUp:
            imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha=0.2, use_cuda=torch.cuda.is_available())
            imgs, targets_a, targets_b = map(Variable, (imgs, targets_a, targets_b))

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        # loss = criterion(logits, labels.to(device))

        loss = None
        if config.do_MixUp:
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        elif config.do_cutMix:
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # # STN : Allow transformes to do like translation, cropping, isotropic scaling but rotation
        # #           , with a intention to let STN learns where to focus on instead of how to transform the image.
        # # Below is what matrix should look like : 
        # #    [ x_ratio, 0 ] [offset_X]
        # #    [ 0, y_ratio ] [offset_y]
        if config.model_name == "STN_ViT":
            model.fc_loc[-1].weight.grad[1].zero_()
            model.fc_loc[-1].weight.grad[3].zero_()

        # Update the parameters with computed gradients.
        optimizer.step()
        ema.update()

        # # Clip the gradient norms for stable training.
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Compute the accuracy for current batch.
        acc = torch.tensor([0])
        if not config.do_cutMix:        
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    return acc.item(), loss.item()

def valid(model, valid_loader, criterion, ema=None):
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    if ema is not None:

        with ema.average_parameters():

            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = model(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    
    else:
        
        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    return acc.item(), loss.item()

###################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AICUP - Orchid Classifier')

    # parser.add_argument('--lr', default=2e-5, type=float,
    #                     help='Base learning rate')
    # parser.add_argument('--bs', default=32, type=int, help='Batch size')
    # parser.add_argument('--e', default=50, type=int, help='Numbers of epoch')
    # parser.add_argument('--v', default=50, type=int, help='Experiment version')
    # parser.add_argument('--device', default=-1, type=int,
    #                     help='GPU index, -1 for cpu')
    parser.add_argument('--logdir', default='model', type=str, required=True, 
                                help='The folder to store the training stats of current model')

    args = parser.parse_args()

    # 
    assert not os.path.isdir(os.path.join(os.getcwd(), args.logdir)), "Already has a folder with the same name"
    os.mkdir(args.logdir)

    shutil.copy('./config.py', f'{args.logdir}/config.py')

    main(args.logdir)


###################################################################################