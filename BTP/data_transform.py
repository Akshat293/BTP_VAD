import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# 1) load dataset

class WineDataset(Dataset):
    
        def __init__(self,transform=None):
            # data loading
            xy = np.loadtxt('/Users/akshatsaxena/Desktop/BTP/data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
            self.n_samples = xy.shape[0]
            print(self.n_samples)
    
            # here the first column is the class label, the rest are the features
            self.x_data = xy[:, 1:] # rows all, cols 1 to end
            self.y_data = xy[:, [0]] # rows all, cols 0

            self.transform = transform
            
    
        # dataset[0]
        def __getitem__(self, index):
            sample= self.x_data[index], self.y_data[index]
            if self.transform:
                sample=self.transform(sample)
            return sample
    
        # len(dataset)
        def __len__(self):
            return self.n_samples


class ToTensor: # convert numpy to tensor
    def __call__(self,sample):
        inputs,labels=sample
        return torch.from_numpy(inputs),torch.from_numpy(labels)

class MulTransform:
    def __init__(self,factor):
        self.factor=factor
    def __call__(self,sample): # sample is a tuple of (inputs,labels)
        inputs,labels=sample
        inputs*=self.factor
        return inputs,labels

dataset = WineDataset(transform=ToTensor())

first_data=dataset[0] # unpacking
features,labels=first_data # unpacking
print(type(features),type(labels))


"""
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the Dataset object. All you need to do is write a callable

class and pass it as an argument to the transform parameter of the Dataset
object. You can write as many callable classes as you want and use them

On Images
---------
1) torchvision.transforms
2) torch.nn.functional

CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip
RandomRotation, Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Custom Transform
----------------
You can write your own custom transforms by writing your own callable class
and implementing the __call__ method. You can use __init__ method to pass
any required parameters.

"""