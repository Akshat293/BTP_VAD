import torch
import numpy as np

x=torch.randn(3,requires_grad=True)
print(x)
# y=x*2
# print(y)
# y=y.mean()
# y.backward()
# print(x.grad)


# If we dont want the required grat to stop[ to stop the gradient from flowing back to x]
# we can use the following
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():

# x=x.requires_grad_(False)
# print(x)

# x=x.detach()
# print(x)

# with torch.no_grad():
#     y=x*2
#     x=x+0
#     print(x)


# Dummy Model training

# weights=torch.ones(4,requires_grad=True)
# for epoch in range(3):
#     model_output=(weights*weights*4).sum()   # y=w^2*4
#     model_output.backward()
#     print(weights.grad)
#     weights.grad.zero_()


# Pytorch build in optimizer
# weights=torch.ones(4,requires_grad=True)
# optimizer=torch.optim.SGD([weights],lr=0.01)
# for epoch in range(3):
#     model_output=(weights*weights*4).sum()   # y=w^2*4
#     model_output.backward()
#     optimizer.step()
#     print(weights.grad)
#     optimizer.zero_grad()


