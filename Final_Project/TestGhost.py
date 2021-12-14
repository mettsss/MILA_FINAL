# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from ghostnet.ghost_net import ghost_net
# from keras.datasets import mnist
import pandas as pd
import numpy as np
import torch

from torch import nn,optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import SimpleITK as sitk

import os

# 检验GPU是否可用
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def TestGhost(test_data):
    '''

    Parameters
    ----------
    test_data : torch.Tensor
        The shape of the input data should be (batch_size, 1, 256, 256)

    Returns
    predict_y: torch.Tensor
    The prediction made by the ghostnet 
    '''
    model = torch.load('GhostNet.pkl') #load_model make sure that this file is in the same file of the model
    model = model.to(device)   #transfer the model to the target
    predicted_digit = model(test_data.to(device))
    _, predicted = torch.max(predicted_digit, 1)
    predicted = predicted.cpu() #finally transfer it to cpu tensor
    return predicted

