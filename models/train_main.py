#encoding: utf-8
import pickle
from pathlib import Path
import argparse
import torch
from torch.autograd import Variable
import torch.multiprocessing
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
from superglue import SuperGlue
from main_mdgat import main_mdgat
from trans_mdgat import MDGAT

import inspect
import sys
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(
    description='Point cloud matching training ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations')

parser.add_argument(
    '--learning_rate', type=float, default=0.0001,  #0.0001
    help='Learning rate')

parser.add_argument(
    '--epoch', type=int, default=1,
    help='Number of epoches')

parser.add_argument(
    '--memory_is_enough', type=bool, default=False, 
    help='If memory is enough, load all the data')
        
parser.add_argument(
    '--batch_size', type=int, default=1, #12
    help='Batch size')

parser.add_argument(
    '--local_rank', type=int, default=[0,1], 
    help='Gpu rank')

parser.add_argument(
    '--resume', type=bool, default=False, # True False
    help='Resuming from existing model')

parser.add_argument(
    '--net', type=str, default='mdgat', 
    help='Choose net structure : mdgat superglue')

parser.add_argument(
    '--loss_method', type=str, default='gap_loss',
    help='Choose loss function : superglue triplet_loss gap_loss')

parser.add_argument(
    '--mutual_check', type=bool, default=False,  # True False
    help='If perform mutual check')

parser.add_argument(
    '--k', type=int, default=[128, None, 128, None, 64, None, 64, None], 
    help='Mdgat structure. None means connect all the nodes.')

parser.add_argument(
    '--l', type=int, default=9, 
    help='Layers number of GNN')

parser.add_argument(
    '--descriptor', type=str, default='FPFH', 
    help='Choose keypoint descriptor : FPFH pointnet pointnetmsg FPFH_gloabal FPFH_only')

parser.add_argument(
    '--keypoints', type=str, default='USIP', 
    help='Choose keypoints : sharp USIP lessharp')

parser.add_argument(
    '--ensure_kpts_num', type=bool, default=False, 
    help='')

parser.add_argument(
    '--max_keypoints', type=int, default=512,  #1024
    help='Maximum number of keypoints'
            ' (\'-1\' keeps all keypoints)')

parser.add_argument(
    '--dataset', type=str, default='kitti',  
    help='Used dataset')

parser.add_argument(
    '--resume_model', type=str, default='./your_model.pth',
    help='Path to the resumed model')

parser.add_argument(
    '--train_path', type=str, default=parentdir+'/KITTI', 
    help='Path to the directory of training scans.')

parser.add_argument(
    '--keypoints_path', type=str, default=parentdir+'/KITTI/keypoints/tsf_256_FPFH_16384-512-k1k16-2d-nonoise',
    help='Path to the directory of kepoints.')

parser.add_argument(
    '--txt_path', type=str, default=parentdir+'/KITTI/preprocess-random-full', 
    help='Path to the directory of pairs.')

parser.add_argument(
    '--model_out_path', type=str, default=parentdir+'/KITTI/checkpoint',
    help='Path to the directory of output model')

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
    '--train_step', type=int, default=3,  
    help='Training step when using pointnet: 1,2,3')


parser.add_argument(
    '--train_seq', nargs="+", type=int, default=[4], 
    help='sequences for train ')

parser.add_argument(
    '--eval_seq',nargs="+",  type=int, default=[9], 
    help='sequences for evaluation ')
    
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
    '--use_normals', type=bool, default=False,  # True False
    help='use normals to compute scores')

parser.add_argument(
    '--train_part',  type=int, default=1, 
    help='part 1 2 3')
        
if __name__ == '__main__':
    opt = parser.parse_args()
    
    from load_data import SparseDataset
    
    if opt.net == 'raw':
        opt.k = None
        opt.l = 9
    if opt.mutual_check:
        model_name = '{}-k{}-batch{}-{}-{}-{}' .format(opt.net, opt.k, opt.batch_size, opt.loss_method, opt.descriptor, opt.keypoints)
    else:
        model_name = 'nomutualcheck-{}-k{}-batch{}-{}-{}-{}' .format(opt.net, opt.k, opt.batch_size, opt.loss_method, opt.descriptor, opt.keypoints)
    

    if torch.cuda.is_available():
        device=torch.device('cuda:{}'.format(opt.local_rank[0]))
    else:
        device = torch.device("cpu")
   # model_out_path = '{}/{}/{}{}-k{}-{}-{}' .format(opt.model_out_path, opt.dataset, opt.net, opt.l, opt.k, opt.loss_method, opt.descriptor)
    # if opt.descriptor == 'pointnet' or opt.descriptor == 'pointnetmsg':
    #     model_out_path = '{}/train_step{}' .format(model_out_path, opt.train_step)
    # model_out_path = '{}/{}' .format(model_out_path, model_name)
    model_out_path = opt.model_out_path
    model_out_path = Path(model_out_path)
    model_out_path.mkdir(exist_ok=True, parents=True)

    print("Train",opt.net,"net with \nStructure k:",opt.k,"\nDescriptor: ",opt.descriptor,"\nLoss: ",opt.loss_method,"\nin Dataset: ",opt.dataset,
    "\n====================",
    "\nmodel_out_path: ", model_out_path)
   
    if opt.resume:        
        path_checkpoint = parentdir+'/'+opt.resume_model  
        checkpoint = torch.load(path_checkpoint) 
        lr = checkpoint['lr_schedule']  # lr = opt.learning_rate # lr = checkpoint['lr_schedule']
        start_epoch = checkpoint['epoch'] + 1 
        loss = checkpoint['loss']
        best_loss = loss
    else:
        start_epoch = 1
        best_loss = 1e6
        lr=opt.learning_rate
    
    config = {
            'net': {
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
                'lr': lr,
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
                'use_normals' : opt.use_normals,
                'train_part' : opt.train_part
            }
        }
    
    if opt.net == 'superglue':
        net = SuperGlue(config.get('net', {}))
    else:
        # initialisation du modèle
        path_checkpoint = parentdir+'/part1.pth'          
        checkpoint = torch.load(path_checkpoint,map_location=device)#{'cuda:2':'cuda:0'})  
    
        MG1=MDGAT(config.get('net', {}))

        if torch.cuda.is_available():
        #    device=torch.device('cuda:{}'.format(opt.local_rank[0]))
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                MG1 = torch.nn.DataParallel(MG1, device_ids=opt.local_rank)
            else:
                MG1 = torch.nn.DataParallel(MG1)
        else:
            device = torch.device("cpu")
            print("### CUDA not available ###")
            
        MG1.load_state_dict(checkpoint['net']) 
        net = main_mdgat(config.get('net', {}),MG1)
        
    if opt.resume:
        net.load_state_dict(checkpoint['net']) 
        optimizer = torch.optim.Adam(net.part2.parameters(), lr=config.get('net', {}).get('lr'))
        print('Resume from:', opt.resume_model, 'at epoch', start_epoch, ',loss', loss, ',lr', lr,'.\nSo far best loss',best_loss,
        "\n====================")
    else:
        optimizer = torch.optim.Adam(net.part2.parameters(), lr=lr)
        print('====================\nStart new training')



        
    for param in net.part1.parameters():
            param.requires_grad = False
              
 #   net.module.load_model()
    net.part1.eval()
    net.part2.train()
    
    if torch.cuda.is_available():
        device=torch.device('cuda:{}'.format(opt.local_rank[0]))
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     net = torch.nn.DataParallel(net, device_ids=opt.local_rank)
        # else:
        #     net = torch.nn.DataParallel(net)
    else:
        device = torch.device("cpu")
        print("### CUDA not available ###")
    net.double().to(device)
    

    train_set = SparseDataset(opt,opt.train_seq)
    val_set = SparseDataset(opt,opt.eval_seq)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, batch_size=opt.batch_size, num_workers=1, drop_last=True, pin_memory = True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, num_workers=1, drop_last=True, pin_memory = True)
    print('==================\nData imported')

    mean_loss = []

    for epoch in range(start_epoch, opt.epoch+1):

        net.part1.eval()
        net.part2.train()
        epoch_loss = 0
        epoch_gap_loss = 0
        epoch_t_loss = 0
        current_loss = 0
        net.to(device)
        train_loader = tqdm(train_loader) 
        begin = time.time()
        for i, pred in enumerate(train_loader):
            for k in pred:
                if k!='idx0' and k!='idx1' and k!='sequence':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].to(device))
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).to(device))

            
            data = net(pred,200)
            
            for k, v in pred.items(): 
                pred[k] = v[0]
            pred = {**pred, **data}

            if 'skip_train' in pred: # no keypoint
                continue

            optimizer.zero_grad()


            # Gap loss
            Loss = (pred['loss']) 
            Loss = torch.mean(Loss)
            
            # Transformation loss
            ''' On rétropropage la moyenne ou uniquement la dernière loss ...'''

            T_Loss = pred['t_loss']
            T_Loss = torch.mean(T_Loss) 
            

            # sum
            if opt.train_part == 1 : 
                if epoch > -1 : # 100 :
                    a = 1e1
                else :
                    a = 1
            else :
                a = 1e1       
                
            tot_loss= T_Loss  + a * Loss
            tot_loss.backward()
            optimizer.step()    
            
            epoch_gap_loss += a * Loss.item()
            epoch_loss += tot_loss.item()
            epoch_t_loss += T_Loss.item()
        
            # print('\n  part1')
            # for name, param in net.module.part1.gnn.named_parameters():
            #     print(name, param.grad)

          #  lr_schedule.step()
            
            del pred, data, i
            
        print('\nepoch = ',epoch,' -------- loss = ', epoch_loss/len(train_loader)
              , ' T loss = ' , epoch_t_loss/len(train_loader)  , ' Gap loss = ', epoch_gap_loss /len(train_loader) )

