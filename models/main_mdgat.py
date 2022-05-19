#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:33:59 2022

@author: spi-2017-12
"""
from copy import deepcopy
import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import inspect
import sys
import os
from pointnet_util import PointNetSetKptsMsg, PointNetSetAbstraction
from DGCNN import DGCNN , DGCNN_leaky

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import _init_paths
from load_data import SparseDataset
import torch.multiprocessing
import time
from utils.utils_test import (calculate_error, plot_match,max_distance)
from models.superglue import SuperGlue
from trans_mdgat import MDGAT
from scipy.spatial.distance import cdist



class main_mdgat(nn.Module):
    
    default_config = {
        'descriptor_dim': 128,
        'embed_dim': 128,
        'keypoint_encoder': [32, 64, 128],
        'descritor_encoder': [64, 128],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'points_transform' : False

    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.part1 = MDGAT(self.config)
        self.part2 = MDGAT(self.config)
        self.part3 = MDGAT(self.config)
                
        if torch.cuda.is_available():
            device=torch.device('cuda:{}'.format(opt.local_rank[0]))
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     net = torch.nn.DataParallel(net, device_ids=opt.local_rank)
            # else:
            self.part1 = torch.nn.DataParallel(self.part1)
            self.part2 = torch.nn.DataParallel(self.part2)
            self.part3 = torch.nn.DataParallel(self.part3)


        
    def forward(self, data,epoch):
        
        pred = self.part1(data,epoch)
        # On applique la 1 ère Transformation
        data['keypoints0'] = pred['keypoints0']
        
        pred = self.part2(data,epoch)
        # On applique une 2ème Transformation
        data['keypoints0'] = pred['keypoints0']
        #
        pred = self.part3(data,epoch)
        return pred


    def load_model (self) :
        path_checkpoint = parentdir+'/part1.pth'          
        checkpoint = torch.load(path_checkpoint, map_location='cpu')#{'cuda:2':'cuda:0'})  
        self.part1.load_state_dict(checkpoint['net']) 
        ##
        path_checkpoint = parentdir+'/part2.pth'          
        checkpoint = torch.load(path_checkpoint, map_location='cpu')#{'cuda:2':'cuda:0'})  
        self.part2.load_state_dict(checkpoint['net']) 
        ##
        path_checkpoint = parentdir+'/part3.pth'          
        checkpoint = torch.load(path_checkpoint, map_location='cpu')#{'cuda:2':'cuda:0'})  
        self.part3.load_state_dict(checkpoint['net']) 
            


        print('Model loaded ')

parser = argparse.ArgumentParser(
    description='Point cloud matching and pose evaluation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--visualize', type=bool, default=False,
    help='Visualize the matches')

parser.add_argument(
    '--vis_line_width', type=float, default=0.2,
    help='the width of the match line open3d visualization')

parser.add_argument(
    '--calculate_pose', type=bool, default=False,
    help='Registrate the point cloud using the matched point pairs and calculate the pose')

parser.add_argument(
    '--learning_rate', type=int, default=0.0001,
    help='Learning rate')
    
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')

parser.add_argument(
    '--train_path', type=str, default=parentdir+'/KITTI/',
    help='Path to the directory of training scans.')

parser.add_argument(
    '--model_out_path', type=str, default=parentdir+'/models/checkpoint',
    help='Path to the directory of output model')

parser.add_argument(
    '--memory_is_enough', type=bool, default=False, 
    help='If true load all the scans')


parser.add_argument(
    '--local_rank', type=int, default=[0,1,2,3], 
    help='Gpu rank')

parser.add_argument(
    '--txt_path', type=str, default=parentdir+'/KITTI/preprocess-random-full',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--keypoints_path', type=str, default=parentdir+'/KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--resume_model', type=str, default=parentdir+'/pre-trained/trans_best_model.pth',
    help='Number of skip frames for training')



parser.add_argument(
    '--resume', type=bool, default=False, # True False
    help='Resuming from existing model')


parser.add_argument(
    '--loss_method', type=str, default='gap_loss', 
    help='triplet_loss superglue gap_loss')

parser.add_argument(
    '--net', type=str, default='mdgat', 
    help='mdgat; superglue')

parser.add_argument(
    '--mutual_check', type=bool, default=False,
    help='perform')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    help='Mdgat structure. None means connect all the nodes.')

parser.add_argument(
    '--l', type=int, default=9, 
    help='Layers number in GNN')

parser.add_argument(
    '--descriptor', type=str, default='FPFH', 
    help='FPFH pointnet FPFH_only msg')

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='USIP')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=False, 
    help='make kepoints number')

parser.add_argument(
    '--max_keypoints', type=int, default=256,
    help='Maximum number of keypoints'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--threshold', type=float, default=0.5, 
    help='Ground truth distance threshold')

parser.add_argument(
    '--triplet_loss_gamma', type=float, default=0.5,
    help='Threshold for triplet loss and gap loss')

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')

parser.add_argument(
    '--train_step', type=int, default=3,  
    help='Training step when using pointnet: 1,2,3')

parser.add_argument(
    '--descriptor_dim',  type=int, default=128, 
    help=' features dim ')

parser.add_argument(
    '--embed_dim',  type=int, default=128, 
    help='DGCNN output dim ')

parser.add_argument(
    '--points_transform', type=bool, default=False,  # True False
    help='If applies [R,t] to source set ')

parser.add_argument(
    '--test_seq', nargs="+", type=int, default=[4], 
    help='sequences for test ')

if __name__ == '__main__':
    
    import pickle
    from load_data import SparseDataset
    from torch.autograd import Variable
    opt = parser.parse_args()


    test_set = SparseDataset(opt, opt.test_seq)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=opt.batch_size, num_workers=1, drop_last=True, pin_memory = True)
    print('==================\nData imported')



    config = {
            'net': {
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
                'lr': opt.learning_rate,
                'loss_method': opt.loss_method,
                'k': opt.k,
                'descriptor': opt.descriptor,
                'mutual_check': opt.mutual_check,
                'triplet_loss_gamma': opt.triplet_loss_gamma,
                'train_step':opt.train_step,
                'L':opt.l,
                'points_transform' : opt.points_transform,
                'descriptor_dim' : opt.descriptor_dim,
                'embed_dim' : opt.embed_dim,
            }
        }
    
    # initialisation du modèle

    net = main_mdgat(config.get('net', {}))
    
    if torch.cuda.is_available():
        device=torch.device('cuda:{}'.format(opt.local_rank[0]))
    else:
        device = torch.device("cpu")
    net.double().to(device)
    net.load_model()
    # if torch.cuda.is_available():
    #     device=torch.device('cuda:{}'.format(opt.local_rank[0]))
    #     # if torch.cuda.device_count() > 1:
    #     #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     #     net = torch.nn.DataParallel(net, device_ids=opt.local_rank)
    #     # else:
    #     net = torch.nn.DataParallel(net)
    # else:
    #     device = torch.device("cpu")
    #     print("### CUDA not available ###")
        
    # net.double().to(device)


        
    # edited_data={}
    
    # for batch, pred in enumerate(test_loader):
    #     net.double().eval()                
    #     if batch > 0 : break # Pour s'arreter à un seul batch
    #     for k in pred:
    #         if k!='idx0' and k!='idx1' and k!='sequence':
    #             if type(pred[k]) == torch.Tensor:
    #                 pred[k] = Variable(pred[k].to(device))
    #             else:
    #                 pred[k] = Variable(torch.stack(pred[k]).to(device))
    #     # On applique Superglue
    #     data = net(pred,200)
    #     pred = {**pred, **data}	
    #     edited_data = {**edited_data,**pred}
        

