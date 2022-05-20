#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:53:47 2022

@author: spi-2017-12
"""


import _init_paths
import os
import time
from copy import deepcopy
from typing import List, Tuple
import pickle
import torch
from torch.autograd import Variable
from torch import nn
import math
import numpy as np
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 




class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False).double().to(device)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        
        mkpts0 = input[0]
        mkpts1 = input[1]
        valid_scores = input[2]
        src_i = mkpts0
        src_corr_i = torch.matmul(mkpts1, valid_scores.transpose(1, 0).contiguous()).to(device)
        src_centered_i = mkpts0 - mkpts0.mean(dim=1, keepdim=True)   
        src_corr_centered_i = src_corr_i - src_corr_i.mean(dim=1, keepdim=True)
        H_i = torch.matmul(src_centered_i.to(device), src_corr_centered_i.transpose(1, 0)
                           .contiguous().to(device)).to(device)
        u, s, v = torch.svd(H_i)
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
        r_det = torch.det(r)
        if r_det < 0:
            u, s, v = torch.svd(H_i)
            v = torch.matmul(v.to(device), self.reflect).to(device)
            r = torch.matmul(v, u.transpose(1, 0).contiguous().to(device)).to(device)
        r=torch.eye(3,dtype=float,device=device)
        tb=torch.matmul(-r, src_i.mean(dim=1, keepdim=True).to(device)) 
        + src_corr_i.mean(dim=1, keepdim=True).to(device)

        return r, tb


if __name__ == '__main__':
    # to check : nbatchs,emebd dim, nb pts
    src_embedding = Variable(torch.rand(4,512,256))
    src=Variable(torch.rand(4,256,3))
    # print(src.size())
    scores=Variable(torch.rand(4,256,256))
    matches = Variable(torch.rand(4,256))
    svd=SVDHead()
    t=svd(src,src,scores,matches,matches)
    # print(t[0][1])
    # print(t[1][1])
