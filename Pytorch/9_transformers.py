'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self,transforms=None):
        xy=np.loadtxt('./data/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.n_samples=xy.shape[0]
        self.x=xy[:,1:]
        self.y=xy[:,[0]]
        self.transforms=transforms
    
    def __getitem__(self, index):
        sample= self.x[index],self.y[index]
        if(self.transforms):
            sample=self.transforms(sample)
        return sample
        
    
    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self,sample):
        x,y=sample
        return torch.from_numpy(x),torch.from_numpy(y)

class MulTransform:
    def __init__(self,factore):
        self.factore=factore
        
    def __call__(self,sample):
        x,y=sample
        return x*self.factore,y

    
#creating dataset
print('Without Transform')
dataset=WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset=WineDataset(transforms=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed=torchvision.transforms.Compose([ToTensor(),MulTransform(2)])
dataset=WineDataset(transforms=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)








