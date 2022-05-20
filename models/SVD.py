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
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 




class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        scores = input[2]
        
        batch_size = src.size(0)

       # d_k = src_embedding.size(1)
       # scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3).to(device)


    # def forward(self, *input):
    #     src_embedding = input[0]
    #     tgt_embedding = input[1]
    #     src = input[2]
    #     tgt = input[3]
    #     batch_size = src.size(0)

    #     d_k = src_embedding.size(1)
    #     scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
    #     scores = torch.softmax(scores, dim=2)
    #     src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

    #     src_centered = src - src.mean(dim=2, keepdim=True)

    #     src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

    #     H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

    #     U, S, V = [], [], []
    #     R = []

    #     for i in range(src.size(0)):
    #         u, s, v = torch.svd(H[i])
    #         r = torch.matmul(v, u.transpose(1, 0).contiguous())
    #         r_det = torch.det(r)
    #         if r_det < 0:
    #             u, s, v = torch.svd(H[i])
    #             v = torch.matmul(v, self.reflect)
    #             r = torch.matmul(v, u.transpose(1, 0).contiguous())
    #             # r = r * self.reflect
    #         R.append(r)

    #         U.append(u)
    #         S.append(s)
    #         V.append(v)

    #     U = torch.stack(U, dim=0)
    #     V = torch.stack(V, dim=0)
    #     S = torch.stack(S, dim=0)
    #     R = torch.stack(R, dim=0)

    #     t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
    #     return R, t.view(batch_size, 3)
if __name__ == '__main__':
    # to check : nbatchs,emebd dim, nb pts
    src_embedding = Variable(torch.rand(4,512,256))
    src=Variable(torch.rand(4,3,256))
    print(src.size())
    scores=Variable(torch.rand(4,256,256))
    
    svd=SVDHead()
    t=svd(src,src,scores)
    print(t[0][1])
    print(t[1][1])
