#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 00:27:30 2022

@author: spi-2017-12
"""
import argparse
import numpy as np
import torch
import os
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.neighbors import KDTree
import time
import inspect
import sys
import _init_paths
from trans_mdgat import  MDGAT

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 



torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


def load_kitti_gt_txt(txt_root, seq):
    '''
    :param txt_root:
    :param seq
    :return: [{anc_idx: *, pos_idx: *, seq: *}]                
     '''
    dataset = []
    with open(os.path.join(parentdir+'/KITTI/preprocess-random-full/', '%02d'%seq, 'groundtruths.txt'), 'r') as f:
        lines_list = f.readlines()
        for i, line_str in enumerate(lines_list):
            if i == 0:
                # skip the header line
                continue
            line_splitted = line_str.split()
            anc_idx = int(line_splitted[0])
            pos_idx = int(line_splitted[1])

            data = {'seq': seq, 'anc_idx': anc_idx, 'pos_idx': pos_idx}
            dataset.append(data)
    # dataset.pop(0)
    return dataset

def make_dataset_kitti_distance(txt_path, mode):
        if mode == 'train':
            seq_list = [0,1,2,3,4,5,6,7,9]
        elif mode == 'val':
            seq_list = [8]
        elif mode == 'test':
            seq_list = [10]
            
        elif type(mode[0])==int:
            seq_list=mode
            
        else:
            raise Exception('Invalid mode.')

        dataset = []
        for seq in seq_list:
            dataset += (load_kitti_gt_txt(txt_path, seq))
           
        return dataset, seq_list

class SparseDataset(Dataset):
    """Sparse correspondences dataset.  
    Reads images from files and creates pairs. It generates keypoints, 
    descriptors and ground truth matches which will be used in training."""

    def __init__(self, opt, mode):

        self.train_path = opt.train_path
        self.keypoints = opt.keypoints
        self.keypoints_path = opt.keypoints_path
        self.descriptor = opt.descriptor
        self.nfeatures = opt.max_keypoints
        self.threshold = opt.threshold
        self.ensure_kpts_num = opt.ensure_kpts_num
        self.mutual_check = opt.mutual_check
        self.memory_is_enough = opt.memory_is_enough
        self.txt_path = opt.txt_path
        self.dataset, self.seq_list = make_dataset_kitti_distance(self.txt_path, mode)

        self.calib={}
        self.pose={}
        self.pc = {}
                
        config = {
                 'net': {
                     'sinkhorn_iterations': 100,
                     'match_threshold': 0.2,
                     'lr': 1e-4,
                     'loss_method': 'gap_loss',
                     'k': opt.k,
                     'descriptor': 'DGCNN',
                     'mutual_check': False,
                     'triplet_loss_gamma': 0.5,
                     'train_step':3,
                     'L': 9,
                     'points_transform' : False,
                     'descriptor_dim' : 128,
                     'embed_dim' : 128,
                     'use_normals' : False,
                     'train_part' : 1
                 }
             }

        if torch.cuda.is_available():
            device=torch.device('cuda:{}'.format(opt.local_rank[0]))
        else:
            device = torch.device("cpu")
            print("### CUDA not available ###")
        path_checkpoint = parentdir+'/part1.pth'          
        checkpoint = torch.load(path_checkpoint,map_location=device)#{'cuda:2':'cuda:0'})   
        self.net=MDGAT(config.get('net', {}))
        self.net = torch.nn.DataParallel(self.net)
        self.net.load_state_dict(checkpoint['net']) 
        print('Resume from ', path_checkpoint)
        self.net.to(device)
        

            
        for seq in self.seq_list:
            sequence = '%02d'%seq
            calibpath = os.path.join(parentdir+'/KITTI/', 'calib/sequences', sequence, 'calib.txt')
            posepath = os.path.join(parentdir+'/KITTI/', 'poses', '%02d.txt'%seq)
            with open(calibpath, 'r') as f:
                for line in f.readlines():
                    _, value = line.split(':', 1)
                    try:
                        calib = np.array([float(x) for x in value.split()])
                    except ValueError:
                        pass
                    calib = np.reshape(calib, (3, 4))    
                    self.calib[sequence] = np.vstack([calib, [0, 0, 0, 1]])
            
            poses = []
            with open(posepath, 'r') as f:
                for line in f.readlines():
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
                self.pose[sequence] = poses

            '''If memory is enough, load all the data'''
            if self.memory_is_enough:
                pcs = []
                folder = os.path.join(self.keypoints_path, sequence)
                folder = os.listdir(folder)   
                folder.sort(key=lambda x:int(x[:-4]))
                for idx in range(len(folder)):
                    file = os.path.join(self.keypoints_path, sequence, folder[idx])
                    if os.path.isfile(file):
                        pc = np.fromfile(file, dtype=np.float32)
                        pcs.append(pc)
                    else:
                        pcs.append([0])
                self.pc[sequence] = pcs


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
             

        # d=[9,236,390,259,1048,171,395,296]
        # d=[259,296]
        # idx=d[idx]
        index_in_seq = self.dataset[idx]['anc_idx']
        index_in_seq2 = self.dataset[idx]['pos_idx']
        seq = self.dataset[idx]['seq']
        # trans = self.dataset[idx]['trans']
        # rot = self.dataset[idx]['rot']

        # relative_pos = self.dataset[idx]['anc_idx']

        if self.memory_is_enough:
            sequence = sequence = '%02d'%seq
            pc_np1 = self.pc[sequence][index_in_seq]

            pc_np1 = pc_np1.reshape((-1, 37))
            kp1 = pc_np1[:, :3]
            score1 = pc_np1[:, 3]
            descs1 = pc_np1[:, 4:]
            pose1 = self.pose[sequence][index_in_seq] 

            pc_np2 = self.pc[sequence][index_in_seq2]
            pc_np2 = pc_np2.reshape((-1, 37))
            kp2 = pc_np2[:, :3]
            score2 = pc_np2[:, 3]
            descs2 = pc_np2[:, 4:]
            pose2 = self.pose[sequence][index_in_seq2]

            T_cam0_velo = self.calib[sequence]
            # q = np.asarray([rot[3], rot[0], rot[1], rot[2]])
            # t = np.asarray(trans)
            # relative_pose = RigidTransform(q, t)
        else:
            sequence = '%02d'%seq
            pc_np_file1 = os.path.join(parentdir+'/KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise', sequence, '%06d.bin' % (index_in_seq))
            pc_np1 = np.fromfile(pc_np_file1, dtype=np.float32)

            pc_np_file2 = os.path.join(parentdir+'/KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise', sequence, '%06d.bin' % (index_in_seq2))
            pc_np2 = np.fromfile(pc_np_file2, dtype=np.float32)
            
            pc_np1 = pc_np1.reshape((-1, 37))
            kp1 = pc_np1[:, :3]

            pc_np2 = pc_np2.reshape((-1, 37))
            kp2 = pc_np2[:, :3]
                
            score1 = pc_np1[:, 3]
            descs1 = pc_np1[:, 4:]
            # pose1 = dataset.poses[index_in_seq]
            pose1 = self.pose[sequence][index_in_seq]
            # pc1 = dataset.get_velo(index_in_seq)

            score2 = pc_np2[:, 3]
            descs2 = pc_np2[:, 4:]
            # pose2 = dataset.poses[index_in_seq2]
            pose2 = self.pose[sequence][index_in_seq2]

            T_cam0_velo = self.calib[sequence]

        if self.descriptor == 'pointnet' or self.descriptor == 'pointnetmsg':
            pc_file1 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq)
            pc_file2 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq2)
            pc1 = np.fromfile(pc_file1, dtype=np.float32)
            pc2 = np.fromfile(pc_file2, dtype=np.float32)
            pc1 = pc1.reshape((-1, 8))
            pc2 = pc2.reshape((-1, 8))
            pc1, pc2 = torch.tensor(pc1, dtype=torch.double), torch.tensor(pc2, dtype=torch.double)
   
        if self.ensure_kpts_num:
            # kp1_num = min(self.nfeatures, len(kp1))
            # kp2_num = min(self.nfeatures, len(kp2))
            valid1 = score1>10
            valid2 = score2>10
            kp1=kp1[valid1]
            kp2=kp2[valid2]
            score1=score1[valid1]
            score2=score2[valid2]
            descs1=descs1[valid1]
            descs2=descs2[valid2]
            kp1_num = self.nfeatures
            kp2_num = self.nfeatures
            if kp1_num < len(kp1):
                kp1 = kp1[:kp1_num]
                score1 = score1[:kp1_num]
                descs1 = descs1[:kp1_num]
            else:
                while kp1_num > len(kp1):
                    kp1 = np.vstack((kp1[:(kp1_num-len(kp1))], kp1))
                    score1 = np.hstack((score1[:(kp1_num-len(score1))], score1))
                    descs1 = np.vstack((descs1[:(kp1_num-len(descs1))], descs1))
            
            if kp2_num < len(kp2):
                kp2 = kp2[:kp2_num]
                score2 = score2[:kp2_num]
                descs2 = descs2[:kp2_num]
            else:
                while kp2_num > len(kp2):
                    kp2 = np.vstack((kp2[:(kp2_num-len(kp2))], kp2))
                    score2 = np.hstack((score2[:(kp2_num-len(score2))], score2))
                    descs2 = np.vstack((descs2[:(kp2_num-len(descs2))], descs2))
        else:
            kp1_num = len(kp1)
            kp2_num = len(kp2)
        kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1]) 
        kp2_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp2])

        vis_registered_pointcloud = False
        if vis_registered_pointcloud:
            pc_file1 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq)
            pc_file2 = os.path.join(self.train_path, 'kitti_randomsample_16384_n8', sequence, '%06d.bin' % index_in_seq2)
            pc1 = np.fromfile(pc_file1, dtype=np.float32)
            pc2 = np.fromfile(pc_file2, dtype=np.float32)
            pc1 = pc1.reshape((-1, 8))
            pc2 = pc2.reshape((-1, 8))

            kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc1]) 
            kp2_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in pc2])

        scores1_np = np.array(score1) 
        scores2_np = np.array(score2)

        kp1_np = torch.tensor(kp1_np, dtype=torch.double)
        pose1 = torch.tensor(pose1, dtype=torch.double)
        kp2_np = torch.tensor(kp2_np, dtype=torch.double)
        pose2 = torch.tensor(pose2, dtype=torch.double)
        T_cam0_velo = torch.tensor(T_cam0_velo, dtype=torch.double)
        T_gt = torch.einsum('ab,bc,cd,de->ae', torch.inverse(T_cam0_velo), torch.inverse(pose1), pose2, T_cam0_velo) # T_gt: transpose kp2 to kp1

        '''transform pose from cam0 to LiDAR'''
        kp1w_np = torch.einsum('ki,ij,jm->mk', pose1, T_cam0_velo, kp1_np.T)
        kp2w_np = torch.einsum('ki,ij,jm->mk', pose2, T_cam0_velo, kp2_np.T)
        
        kp1w_np = kp1w_np[:, :3]
        kp2w_np = kp2w_np[:, :3]

        vis_registered_keypoints = False
        if vis_registered_keypoints:
            point_cloud_o3d = o3d.geometry.PointCloud()
            point_cloud_o3d.points = o3d.utility.Vector3dVector(kp1w_np.numpy())
            point_cloud_o3d.paint_uniform_color([0, 1, 0])
            point_cloud_o3d2 = o3d.geometry.PointCloud()
            point_cloud_o3d2.points = o3d.utility.Vector3dVector(kp2w_np.numpy())
            point_cloud_o3d2.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([point_cloud_o3d, point_cloud_o3d2])

        dists = cdist(kp1w_np, kp2w_np)

        '''Find ground true keypoint matching'''
        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)
        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < self.threshold]

        '''For calculating repeatibility'''
        rep = len(min1f)

        '''
        If you got high-quality keypoints, you can set the 
        mutual_check to True, otherwise, it is better to 
        set to False
        '''
        match1, match2 = -1 * np.ones((len(kp1)), dtype=np.int16), -1 * np.ones((len(kp2)), dtype=np.int16)
        if self.mutual_check:
            xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
            matches = np.intersect1d(min1f, xx)

            match1[min1[matches]] = matches
            match2[matches] = min1[matches]
        else:
            match1[min1v < self.threshold] = min1f

            min2v = np.min(dists, axis=0)
            min2f = min1[min2v < self.threshold]
            match2[min2v < self.threshold] = min2f

        kp1_np = kp1_np[:, :3]
        kp2_np = kp2_np[:, :3]

        norm1, norm2 = np.linalg.norm(descs1, axis=1), np.linalg.norm(descs2, axis=1)
        norm1, norm2 = norm1.reshape(kp1_num, 1), norm2.reshape(kp2_num, 1)
        descs1, descs2  = np.multiply(descs1, 1/norm1), np.multiply(descs2, 1/norm2)

        descs1, descs2 = torch.tensor(descs1, dtype=torch.double), torch.tensor(descs2, dtype=torch.double)
        scores1_np, scores2_np = torch.tensor(scores1_np, dtype=torch.double), torch.tensor(scores2_np, dtype=torch.double)

        pred = {
            # 'skip': False,
            'keypoints0': torch.unsqueeze(kp1_np,0).double(),
            'keypoints1': torch.unsqueeze(kp2_np,0).double(),
            'gt_matches0': torch.unsqueeze(torch.tensor(match1),0).double(),
            'gt_matches1': torch.unsqueeze(torch.tensor(match2),0).double(),
            'T_gt': torch.unsqueeze(T_gt,0).double(),
        } 
        with torch.no_grad():
            self.net.eval().double()
            for k in pred:
                if k!='idx0' and k!='idx1' and k!='sequence':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].to('cpu').detach()).double()
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).to('cpu').detach()).double()
            data = self.net(pred,200)
            
            
        kp1_npp = torch.squeeze(data['keypoints0'],0).double()
        kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in kp1_npp.to('cpu').detach()]) 
        kp1_np = torch.tensor(kp1_np, dtype=torch.double)
        kp1_np = kp1_np[:, :3]
        
        
        return{
            # 'skip': False,
            'keypoints0': kp1_np,
            'keypoints1': kp2_np,
            'descriptors0': descs1,
            'descriptors1': descs2,
            'scores0': scores1_np,
            'scores1': scores2_np,
            'gt_matches0': match1,
            'gt_matches1': match2,
            'sequence': sequence,
            'idx0': index_in_seq,
            # 'idx1': index_in_seq2,
            # 'pose1': pose1,
            # 'pose2': pose2,
            # 'T_cam0_velo': T_cam0_velo,
            'T_gt': T_gt,
            # 'cloud0': pc1,
            # 'cloud1': pc2,
            # 'all_matches': list(all_matches),
            # 'file_name': file_name
            'rep': rep
        } 







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
    '--local_rank', type=int, default=0, 
    help='Gpu rank.')

parser.add_argument(
    '--txt_path', type=str, default=parentdir+'/KITTI/preprocess-random-full',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--keypoints_path', type=str, default=parentdir+'/KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--resume_model', type=str, default=parentdir+'/pre-trained/model.pth',
    help='Number of skip frames for training')

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

    opt = parser.parse_args()
    test_set = SparseDataset(opt, opt.test_seq)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=opt.batch_size, num_workers=1, drop_last=True, pin_memory = True)

    for i, pred in enumerate(test_loader):
        if i>2:break
        print(i)
