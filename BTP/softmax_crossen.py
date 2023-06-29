# Softmax and cross-entropy loss function


import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x=np.array([2.0,1.0,0.1])
outputs=softmax(x)
print('softmax numpy:',outputs)

x=torch.tensor([2.0,1.0,0.1])
outputs=torch.softmax(x,dim=0)
print('softmax torch:',outputs)

# Cross entropy loss ?
# Cross entropy loss is a loss function used in classification problems.
# It is a measure of the difference between two probability distributions.

def cross_entropy(actual,predicted):
    loss=-np.sum(actual*np.log(predicted))
    return loss # loss is always a scalar

Y=np.array([1,0,0])

# y_pred_good
Y_pred_good=np.array([0.7,0.2,0.1])
l1=cross_entropy(Y,Y_pred_good)
print('loss1 numpy:',l1)

# y_pred_bad
Y_pred_bad=np.array([0.1,0.3,0.6])
l2=cross_entropy(Y,Y_pred_bad)
print('loss2 numpy:',l2)

# Cross entropy loss with torch
loss=nn.CrossEntropyLoss()
Y=torch.tensor([0])

# nsamples x nclasses = 1x3\
Y_pred_good=torch.tensor([[2.0,1.0,0.1]])
Y_pred_bad=torch.tensor([[0.5,2.0,0.3]])

l1=loss(Y_pred_good,Y)
l2=loss(Y_pred_bad,Y)

print('loss1 torch:',l1.item())
print('loss2 torch:',l2.item())

# Types of activation functions
# 1) Sigmoid:- Sigmoid function is used to map the input value between 0 and 1.
# 2) Tanh:- Tanh function is used to map the input value between -1 and 1.
# 3) ReLU:- ReLU function is used to map the input value between 0 and infinity.
# 4) Leaky ReLU:- Leaky ReLU function is used to map the input value between 0 and infinity.
# 5) Softmax:_ Softmax function is used to map the input value between 0 and 1.

