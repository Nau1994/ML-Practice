import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        xy=np.loadtxt('./data/wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.n_samples=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,1:])
        self.y_data=torch.from_numpy(xy[:,[0]])
    
    def __getitem__(self, index):
        return self.x_data,self.y_data
    
    def __len__(self):
        return self.n_samples
    
#creating dataset
datasets=WineDataset()
# first_data=datasets[0]
# features,labels=first_data
# print(features,labels)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!

train_loader=DataLoader(dataset=datasets,
                        batch_size=4,
                        shuffle=True)

dataiter=iter(train_loader)
data=next(dataiter)
features,labels=data
# print(features,labels)

# Dummy Training loop
num_epochs = 2
total_samples = len(datasets)
n_iterations = math.ceil(total_samples/4)
for epoch in range(2):
    for i,(inputs,labels) in enumerate(train_loader):
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')


# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)

# look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)
