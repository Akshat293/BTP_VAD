# Data loading and preprocessing

# Path: data_loader.py

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Create dataset
class WineDataset(Dataset):
    
        def __init__(self):
            # data loading
            xy = np.loadtxt('/Users/akshatsaxena/Desktop/BTP/data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
            self.n_samples = xy.shape[0]
            print(self.n_samples)
    
            # here the first column is the class label, the rest are the features
            self.x_data = torch.from_numpy(xy[:, 1:]) # rows all, cols 1 to end
            self.y_data = torch.from_numpy(xy[:, [0]]) # rows all, cols 0
            
    
        # dataset[0]
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
    
        # len(dataset)
        def __len__(self):
            return self.n_samples


dataset = WineDataset()
dataloader=DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=0)

# Data iteration

dataiter = iter(dataloader)
data = next(dataiter)

features, labels = data
print(features, labels)

# Training loop

num_epochs = 2

total_samples = len(dataset)

n_iterations = math.ceil(total_samples/4)

print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader): # enumerate gives us the index of the current batch
        # forward, backward, update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

        

