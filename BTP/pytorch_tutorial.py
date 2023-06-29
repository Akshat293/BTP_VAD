import torch
import numpy as np
x=np.ones(5)
print(x.dtype)
# Create a tensor with 3 elements
b=torch.from_numpy(x)
print(b)

# a tensor using torch.one 
a=torch.ones(5,requires_grad=True)
print(a)


