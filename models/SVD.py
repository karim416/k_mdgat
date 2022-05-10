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
        skip=True
        
        src = input[0]
        tgt = input[1]
        scores = input[2]
        
        indices0 = input[3] 
        indices1 = input[4] 
        
        batch_size = src.size(0)

       # d_k = src_embedding.size(1)
       # scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)

        scores = torch.softmax(scores, dim=2)
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        
        # src_centered = src - src.mean(dim=2, keepdim=True)
        # src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
        # H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        
        U, S, V = [], [], []
        R = []
        t = []
        for i in range(src.size(0)) :
            
            # On applique la SVD uniquement au points ayant des correspondances !
            kpts0=input[0].permute(0,2,1)[i].cpu().numpy()
            kpts1=input[1].permute(0,2,1)[i].cpu().numpy()
            matches=indices0[i].cpu().numpy()
            matches1=indices1[i].cpu().numpy()
            
            matches[matches==256]=-1
            matches1[matches1==256]=-1
            valid = matches > -1
            valid_scores=scores[i,valid]
            valid_scores=valid_scores[:,matches[valid]].to(device)
            
#            print('Matched points :', valid_scores.size()[1])
            
            
            # On a finalement les points source et target matchés :
            mkpts0 = torch.tensor(kpts0[valid],dtype=torch.double).permute(1,0).to(device)
            mkpts1 = torch.tensor(kpts1[matches[valid]],dtype=torch.double).permute(1,0).to(device)
            
        
 
            #src_corr_i = torch.matmul(tgt[i], scores[i].transpose(1, 0).contiguous())
            #src_centered_i = src[i] - src[i].mean(dim=1, keepdim=True)
            if valid_scores.size()[1] > 3 :
                src_i = mkpts0
                src_corr_i = torch.matmul(mkpts1, valid_scores.transpose(1, 0).contiguous()).to(device)
                src_centered_i = mkpts0 - mkpts0.mean(dim=1, keepdim=True)   
            else:
                # si on n'a pas assez de correspondances , on calcule la svd sur tous les points ?
                kpts0=src[i]
                kpts1=tgt[i]
                src_i = kpts0
                src_corr_i = torch.matmul(kpts1, scores[i].transpose(1, 0).contiguous()).to(device)
                src_centered_i = kpts0.to(device) - kpts0.mean(dim=1, keepdim=True).to(device)    
                
            src_corr_centered_i = src_corr_i - src_corr_i.mean(dim=1, keepdim=True)
            H_i = torch.matmul(src_centered_i, src_corr_centered_i.transpose(1, 0)
                               .contiguous().to(device)).to(device)
            u, s, v = torch.svd(H_i)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H_i)
                v = torch.matmul(v.to(device), self.reflect).to(device)
                r = torch.matmul(v, u.transpose(1, 0).contiguous().to(device)).to(device)
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)
            tb=torch.matmul(-r, src_i.mean(dim=1, keepdim=True).to(device)) 
            + src_corr_i.mean(dim=1, keepdim=True).to(device)
            t.append(tb)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)
        t = torch.stack(t, dim=0)

        return R, t.view(batch_size, 3).to(device)


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

