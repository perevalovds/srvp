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
zoom = 2

tensor_y = []
tensor_w = []
tensor_h = []



#------------------------------------------------------------------------------
def put_pixel_rgb(x, y, red, green, blue):
    global IMAGE, zoom
    
    gx = x * zoom
    gy = y * zoom
    for b in range(zoom):
        for a in range(zoom):
            IMAGE[gy + b, gx + a, 0] = red
            IMAGE[gy + b, gx + a, 1] = green
            IMAGE[gy + b, gx + a, 2] = blue

#------------------------------------------------------------------------------
def put_pixel_float(x, y, v_float, minv, maxv):
    global IMAGE, zoom
    
    col = clamp(math.floor(mapf(v_float, minv, maxv, 0, 255)), 0, 255)                    
    put_pixel_rgb(x, y, col, col, col)

#------------------------------------------------------------------------------
def save_model_image(iteration):   
    image_file = 'output_weights/network_' + f'{iteration:05d}' + ".png"                    
    cv2.imwrite(image_file,IMAGE)
    
#------------------------------------------------------------------------------
def draw_model(model):

    global IMAGE, w, h, scly, indy, zoom, tensor_y

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
        
        if Len == 0:
            tensor_y.append(0) 
            tensor_w.append(0)
            tensor_h.append(0)
        
        if Len > 1:
            h += indy            
            tensor_y.append(h)  #store Y for this element
            h1 = math.floor(Size[1] * scly)
            tensor_w.append(Size[0])

            tensor_h.append(h1)            
            h += h1            
        
            prev_1d = False
        if Len == 1:
            tensor_y.append(h)  #store Y for this element
            tensor_w.append(Size[0])
            tensor_h.append(1)
            
            h = h + 1
            #if not prev_1d:
            #    h += indy
            prev_1d = True
                
    h += indy
    #print("tensor_y", tensor_y)

    #create image
    print("   gen image", w, h, " zoom", zoom)
    
    IMAGE = np.zeros((h*zoom,w*zoom,3),np.uint8)
    
    
    i = 0
    for param_tensor in model.state_dict():
        tensor = model.state_dict()[param_tensor]        
        Size = list(tensor.size())
        Len = len(Size)
                        
        minv = torch.min(tensor)
        maxv = torch.max(tensor)
               
        Y = tensor_y[i]
        w1 = tensor_w[i]
        h1 = tensor_h[i]
        #print(i, Y, w1, h1)

        i = i + 1
            
        if Len > 1:

            hh = Size[1]
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
                    put_pixel_float(x + x0, y + Y, v, minv, maxv)
        if Len == 1:
            x0 = (w-w1)//2
            y = 0
            for x in range(w1):
                v = tensor[x]
                put_pixel_float(x + x0, y + Y, v, minv, maxv)

    
    #exit from program  
    #print("exiting by calling os._exit(0)...");
    #os._exit(0)
        
    
#------------------------------------------------------------------------------
#draw on model red pixels in damaged parts

def draw_red_points(id, x, yy, hh):
    global IMAGE, w, h, scly, indy, tensor_y
    
    Y = tensor_y[id]
    w1 = tensor_w[id]
    h1 = tensor_h[id]
    #print(i, Y, w1, h1)

    x0 = (w-w1)//2
    y = yy * h1 // hh
        
    put_pixel_rgb(x + x0, y + Y, 0, 0, 255)
        
    
#------------------------------------------------------------------------------
