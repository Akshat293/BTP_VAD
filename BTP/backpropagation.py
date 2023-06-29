import torch as tp
import numpy as np

# Backpropagation
x=tp.tensor(1.0)
y=tp.tensor(2.0)

w=tp.tensor(1.0,requires_grad=True)

# forward pass and compute the loss
y_hat=x*w
loss=(y_hat-y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)

