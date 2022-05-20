#
import torch
import torch.nn as nn
import torch.nn.functional as F

#
import numpy as np

# Hugging face API
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification

from models import BaseModule
from config import DefualtConfig


class ConvNeXt(nn.Module):

    def __init__(self, config: DefualtConfig):

        super(ConvNeXt, self).__init__()

        self.config = config
        self.num_labels = config.num_classes

        ###############################################
        # ViT
        pretrained_model = 'facebook/convnext-xlarge-384-22k-1k'
        self.feature_extractor = ConvNextFeatureExtractor.from_pretrained(pretrained_model)
        self.model = ConvNextForImageClassification.from_pretrained(pretrained_model)
        
        # Classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, self.num_labels)

    # def feature_extract(self, imgs):
    #     '''
    #     '''
    #     # Change input array into list with each batch being one element

    #     # convert it to numpy array first
    #     device = torch.device('cpu')
    #     if imgs.device != torch.device('cpu'):
    #         device = torch.device(f'cuda:{self.config.use_gpu_index}')

    #     imgs = imgs.cpu().numpy()
    #     imgs = np.split(np.squeeze(np.array(imgs)), imgs.shape[0])

    #     # Remove unecessary dimension
    #     for index, array in enumerate(imgs):
    #         imgs[index] = np.squeeze(array)

    #     # Apply feature extractor, stack back into 1 tensor and then convert to tensor
    #     # imgs = (batch_size, 3, 224, 224)
    #     imgs = torch.tensor(np.stack(self.feature_extractor(imgs)['pixel_values'], axis=0))
    #     imgs = imgs.to(device)

    #     return imgs

    def forward(self, x, labels=None):
        '''
        Model forward function
        '''

        # Feature extraction
        x = self.feature_extractor(x, return_tensors="pt")

        # Swin-ViT
        x = self.model(pixel_values=x)
        # x = self.dropout(x.last_hidden_state[:, 0])
        x = torch.mean(x.last_hidden_state[:, ], 1)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
