#
import torch

#
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Inherit from torchvision.datasets.ImageFolder


class OrchidDataSet(ImageFolder):

    def __init__(self, root, transform_set):

        # # Data Augumentation
        # self.transform_set = [
        #     transforms.ColorJitter(brightness=0.5, contrast=0.5),
        #     # transforms.GaussianBlur(kernel_size=3),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     # transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=45),
        # ]

        # self.transform = transforms.Compose([
        #     # transforms.Grayscale(num_output_channels=3),

        #     # Reorder transform randomly
        #     transforms.RandomOrder(transform_set),

        #     # Resize the image into a fixed shape
        #     transforms.Resize((224, 224)),

        #     # ToTensor() should be the last one of the transforms.
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        #
        super(OrchidDataSet, self).__init__(
            root=root, transform=transform_set)

    # help to get images for visualizing
    def getbatch(self, indices):
        '''
            @ Params : 
                1. indices (python.list)
            @ Returns : 
                1. images (torch.tensor with shape (1, ))
                2. labels (torch.tensor with shape (1, ))
        '''
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            # transform_ToTensor =  transforms.Compose([
            #                         transforms.Resize((224, 224)),
            #                         transforms.ToTensor()])
            # image = transform_ToTensor(image)

            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)
