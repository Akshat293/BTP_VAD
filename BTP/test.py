import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import feedforward


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.load('model.ckpt')
# # Test the model
img = Image.open('test.png')
img=img.resize((28,28))
img = np.array(img)
img = torch.from_numpy(img).float()
img = img.to(device)

# Forward pass
outputs = model(img)
_,predictions = torch.max(outputs,1)
print(predictions.tolist())
# Forward pass
