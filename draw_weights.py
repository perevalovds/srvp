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
#Print network structure
def print_model(model):
    i = 0
    for param_tensor in model.state_dict():
        tensor = model.state_dict()[param_tensor]
        L = len(list(tensor.size()))
        num = tensor.numel()
        print(i, "\t", num, "\t", param_tensor, "\t", tensor.size(), "\t", len(list(tensor.size())))
        i = i + 1
        
        
#------------------------------------------------------------------------------
def mapf(x, a, b, A, B):
  return (x - a) * (B - A) / (b - a) + A
  
  
def clamp(v, minv, maxv): 
    return max(minv, min(v, maxv))
    
#------------------------------------------------------------------------------
#Draw network
IMAGE = []
w = 1200
h = 0
scly = 0.08
indy = 10
tensor_y = []


#------------------------------------------------------------------------------
def put_pixel_float(x, y, v_float, minv, maxv, zoom):
    global IMAGE
    
    col = clamp(math.floor(mapf(v_float, minv, maxv, 0, 255)), 0, 255)                    
    gx = x * zoom
    gy = y * zoom
    for a in range(zoom):
        for b in range(zoom):
            IMAGE[gy + b, gx + a, 0] = col
            IMAGE[gy + b, gx + a, 1] = col
            IMAGE[gy + b, gx + a, 2] = col 

#------------------------------------------------------------------------------
def draw_model(model):

    global IMAGE, w, h, scly, indy, tensor_y

    #compute size of image
    h = 0

    i = 0
    prev_1d = False
    for param_tensor in model.state_dict():
        tensor = model.state_dict()[param_tensor]
        num = tensor.numel()
        Size = list(tensor.size())
        Len = len(Size)
        print(i, "\t", num, "\t", param_tensor, "\t", Size, "\t", Len)    
        i = i + 1
        
        if Len > 1:
            h += indy            
            tensor_y.append(h)  #store Y for this element
            
            h += math.floor(Size[1] * scly)
        
            prev_1d = False
        if Len == 1:
            tensor_y.append(h)  #store Y for this element

            h = h + 1
            #if not prev_1d:
            #    h += indy
            prev_1d = True
                
    h += indy
    #print("tensor_y", tensor_y)

    #create image
    zoom = 2
    print("   gen image", w, h, " zoom", zoom)
    
    IMAGE = np.zeros((h*zoom,w*zoom,3),np.uint8)
    
    
    i = 0
    for param_tensor in model.state_dict():
        tensor = model.state_dict()[param_tensor]        
        Size = list(tensor.size())
        Len = len(Size)
                        
        minv = torch.min(tensor)
        maxv = torch.max(tensor)
                
        if Len > 1:
            Y = tensor_y[i]
            i = i + 1

            hh = Size[1]
            h1 = math.floor(hh * scly)
            w1 = Size[0]
            x0 = (w-w1)//2
            
            for y in range(h1):
                for x in range(w1):
                    yy = y * hh // h1
                    v = 0
                    if (Len == 2):
                        v = tensor[x,yy]
                    if (Len == 4):
                        v = tensor[x,yy,0,0] #Just ignoring other components...
                    #print(v)   #it seems normally v = [-1..1]
                    put_pixel_float(x + x0, y + Y, v, minv, maxv, zoom)
        if Len == 1:
            Y = tensor_y[i]
            i = i + 1

            w1 = Size[0]
            x0 = (w-w1)//2
            y = 0
            for x in range(w1):
                v = tensor[x]
                put_pixel_float(x + x0, y + Y, v, minv, maxv, zoom)

    
    #save
    cv2.imwrite('output_weights/network.png',IMAGE)
    
    #exit from program  
    print("exiting by calling os._exit(0)...");
    os._exit(0)
        
    
    
#------------------------------------------------------------------------------

