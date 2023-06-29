# Voice Activity Detection using Convolutional Neural Network

# Importing the libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import librosa.display
import IPython.display as ipd
import pandas as pd
import glob
import random
import time
import pickle
import sys
import argparse
import math
import scipy.io.wavfile as wav
import scipy.signal as signal
import scipy.fftpack as fftpack
import scipy.stats as stats
import scipy.io as sio
import scipy
import sklearn
import sklearn.metrics
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.tree

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
num_classes = 2
batch_size = 100
learning_rate = 0.001

# 


