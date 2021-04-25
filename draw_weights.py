#Helper for draw weights

import configargparse
import os
import random
import math

import torch

import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

import helper
import data.base as data
import module.srvp as srvp
from metrics.ssim import ssim_loss
from metrics.lpips.loss import PerceptualLoss
from metrics.fvd.score import fvd as fvd_score

from PIL import Image

import cv2



#------------------------------------------------------------------------------
#Draw network
def print_model(model):
    i = 0
    for param_tensor in model.state_dict():
        tensor = model.state_dict()[param_tensor]
        L = len(list(tensor.size()))
        num = tensor.numel()
        print(i, "\t", num, "\t", param_tensor, "\t", tensor.size(), "\t", len(list(tensor.size())))
        i = i + 1
        
        
    img = np.ones((300,300,1),np.uint8)*255
    cv2.imwrite('output_weights/network.png',img)
    

    #exit from program  
    print("exiting by calling os._exit(0)...");
    os._exit(0)
        
    
    
#------------------------------------------------------------------------------

