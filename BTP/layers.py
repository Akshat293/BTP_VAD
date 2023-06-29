import torch
import torch.nn as nn



class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,1)


    def forward(self,x):
        out=torch.relu(self.linear1(x))
        out=torch.sigmoid(self.linear2(out))
        return out


