#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ExponentialLR

#
from torchvision import transforms

#
from torch_ema import ExponentialMovingAverage

#
import numpy as np
import pandas as pd

#
from tqdm import tqdm

#
import os
import shutil
import argparse

#
import models
from data.dataset import OrchidDataSet
from utils import get_confidence_score
from config import DefualtConfig

###################################################################################

# config = DefualtConfig()
# device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available() else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')
config = None
device = torch.device('cpu')

BATCH_SIZE = 16

def get_fileName(ds):

    fileNames = []

    for i in range(len(ds.imgs)):
        fileNames.append(ds.imgs[i][0])

    return fileNames


def test(output_file_path='predictions.csv'):
    '''
    @ Params:

    '''

    # Mapping
    ds = OrchidDataSet(config.trainset_path, transform_set=None)
    idx_to_class = {}
    for k in ds.class_to_idx:
        idx_to_class[ds.class_to_idx[k]] = k

    # Step 1 : Model Define & Load
    model = getattr(models, config.model_name)(config)  
    device = torch.device(f'cuda:{config.use_gpu_index}' if torch.cuda.is_available() else'cpu') if config.use_gpu_index != -1 else torch.device('cpu')
    model = model.to(device)
    if torch.cuda.is_available() is True:
        model = model.cuda()
    model.load_state_dict(torch.load(f'./saved/{args.model}/{config.model_path}', map_location=device))

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    ema.load_state_dict(torch.load(f'./saved/{args.model}/{config.ema_path}', map_location=device))

    # Step 2 : DataSet & DataLoader
    resize = resize = (config.resize, config.resize)
    transform_set = transforms.Compose([

        # Resize the image into a fixed shape
        transforms.Resize(resize),

        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds_test = OrchidDataSet(config.testset_path, transform_set=transform_set)
    # test_loader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=config.num_workers)

    # Step 3 : Make prediction via trained model
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    # Initialize a list to store the predictions.
    predictions = []

    with ema.average_parameters():

        # Iterate the validation set by batches.
        for batch in tqdm(test_loader):

            # A batch consists of image data and corresponding labels.
            imgs, _ = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            predictions += logits.argmax(dim=-1)

    # imgs_file_names = os.listdir(config.testset_path)
    imgs_file_names = get_fileName(ds_test)

    # Step 4 : Save predictions into the file.
    with open(output_file_path, "w") as f:

        # The first row must be "Id, Category"
        f.write("filename,category\n")

        # For the rest of the rows, each image id corresponds to a predicted class.
        for i, pred in  enumerate(predictions):
            ans = idx_to_class[pred.item()]
            imgs_file_name = imgs_file_names[i].split("\\")
            # f.write(f"{imgs_file_names[i][-16:]},{ans}\n")
            f.write(f"{imgs_file_name[-1]},{ans}\n")

    # Step 5 : Rearrange
    # Rearrange the predictions into a dataframe.
    df_predict = pd.read_csv(output_file_path)
    df_sample = pd.read_csv('./submission_template.csv')
    df_arrange = pd.merge(df_sample, df_predict, how='inner', on=['filename'])
    df_arrange = df_arrange.drop('category_sample', axis=1)
    df_arrange.to_csv(f'rearrange_{output_file_path}', index=False)

    # 
    # # Step 6 : Explanation & Visualization
    # # get_confidence_score(model, loader=test_loader, use_gpu_index=config.use_gpu_index, batch_size=config.batch_size, outpu_file_path=f'{output_file_path[:-3]}_Confidence.csv')
    # get_confidence_score(model, loader=test_loader, use_gpu_index=config.use_gpu_index, batch_size=BATCH_SIZE, outpu_file_path=f'{output_file_path[:-3]}_Confidence.csv')


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
    # parser.add_argument('--logdir', default='model', type=str, required=True, 
    #                             help='The folder to store the training stats of current model')

    parser.add_argument('--model', default='model_swin', type=str, required=True,
                                help='The name of the model')

    # parser.add_argument('--output', default=f'predictions.csv', type=str,
    #                             help='The file to store the predictions')

    args = parser.parse_args()

    # First, select the chosen model's config file, and replace it with the current one in the main folder
    # Remove the old config file
    if os.path.exists('config.py'):
        os.remove('config.py')
    
    # Then copy the new config file from the model folder, and rename it to config.py
    shutil.copyfile(f'./saved/{args.model}/config.py', 'config.py')

    # 
    config = DefualtConfig()

    # Second
    test(output_file_path=f'prediction_{args.model}.csv')


###################################################################################