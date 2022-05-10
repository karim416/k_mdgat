#encoding: utf-8
from pathlib import Path
import argparse
import torch
from torch.autograd import Variable
import torch.multiprocessing
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter
from superglue import SuperGlue
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
    '--learning_rate', type=int, default=0.0001,  #0.0001
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
    '--local_rank', type=int, default=[0,1,2,3], 
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
    '--descriptor_dim',  type=int, default=256, 
    help=' features dim ')
    
parser.add_argument(
    '--embed_dim',  type=int, default=256, 
    help='DGCNN output dim ')
        
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
        best_loss = 1
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
                'L':opt.l
            }
        }
    
    if opt.net == 'superglue':
        net = SuperGlue(config.get('net', {}))
    else:
        net = MDGAT(config.get('net', {}))

    if torch.cuda.is_available():
        device=torch.device('cuda:{}'.format(opt.local_rank[0]))
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     net = torch.nn.DataParallel(net, device_ids=opt.local_rank)
        # else:
        net = torch.nn.DataParallel(net)
    else:
        device = torch.device("cpu")
        print("### CUDA not available ###")
    net.to(device)

    if opt.resume:
        net.load_state_dict(checkpoint['net']) 
        optimizer = torch.optim.Adam(net.parameters(), lr=config.get('net', {}).get('lr'))
        print('Resume from:', opt.resume_model, 'at epoch', start_epoch, ',loss', loss, ',lr', lr,'.\nSo far best loss',best_loss,
        "\n====================")
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        print('====================\nStart new training')


    train_set = SparseDataset(opt, 'train')
    val_set = SparseDataset(opt, 'val')
    
    val_loader = torch.utils.data.DataLoader(dataset=val_set, shuffle=False, batch_size=opt.batch_size, num_workers=1, drop_last=True, pin_memory = True)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batch_size, num_workers=1, drop_last=True, pin_memory = True)
    print('==================\nData imported')
    mean_loss = []
    for epoch in range(start_epoch, opt.epoch+1):
        epoch_loss = 0
        current_loss = 0
        net.double().train() 
        train_loader = tqdm(train_loader) 
        begin = time.time()
        for i, pred in enumerate(train_loader):
            for k in pred:
                if k!='idx0' and k!='idx1' and k!='sequence':
                    if type(pred[k]) == torch.Tensor:
                        pred[k] = Variable(pred[k].to(device))
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).to(device))
            
            data = net(pred)

            for k, v in pred.items(): 
                pred[k] = v[0]
            pred = {**pred, **data}

            if 'skip_train' in pred: # no keypoint
                continue

            optimizer.zero_grad()

            #print(pred['loss'].size())

            # Gap loss
            Loss = (pred['loss'])
            Loss = torch.mean(Loss)
            # Transformation loss
            T_Loss = (pred['t_loss'])
            T_Loss = torch.mean(T_Loss)
#            print(Loss)
#            print(T_Loss)
            # sum
            tot_loss= 0.01 * T_Loss + Loss
            
            epoch_loss += tot_loss.item()
            tot_loss.backward()
            optimizer.step()
            # lr_schedule.step()

            del Loss, pred, data, i

        # validation

        begin = time.time()
        with torch.no_grad():
            if epoch >= 0 and epoch%1==0:
                mean_val_loss = []
                for i, pred in enumerate(val_loader):
                    ### eval ###
                    net.eval()                
                    for k in pred:
                        # if k != 'file_name' and k!='cloud0' and k!='cloud1':
                        if k!='idx0' and k!='idx1' and k!='sequence':
                            if type(pred[k]) == torch.Tensor:
                                pred[k] = Variable(pred[k].cuda().detach())
                            else:
                                pred[k] = Variable(torch.stack(pred[k]).cuda().detach())
                            # print(type(pred[k]))   #pytorch.tensor
                    
                    data = net(pred) 
                    pred = {**pred, **data}

                    Loss = pred['loss']
                    # Transformation loss
                    T_Loss = (pred['t_loss'])
                    
                    # sum
                    tot_loss= 0.01 * T_Loss + Loss
                    
                    mean_val_loss.append(tot_loss) 
                    
         
            timeconsume = time.time() - begin
            mean_val_loss = torch.mean(torch.stack(mean_val_loss)).item()
            epoch_loss /= len(train_loader)

            print('Validation loss: {:.4f}, epoch_loss: {:.4f},  best val loss: {:.4f}' .format(mean_val_loss, epoch_loss, best_loss))
            checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': optimizer.state_dict()['param_groups'][0]['lr'],
                    'loss': mean_val_loss
                }
            if epoch == opt.epoch : 
                print('Last epoch model')
                best_loss = mean_val_loss
                model_out_fullpath = "{}/last_model_epoch_{}(val_loss{}).pth".format(model_out_path, epoch, best_loss)
                torch.save(checkpoint, model_out_fullpath)
                print('time consume: {:.1f}s, last loss: {:.4f}, Checkpoint saved to {}' .format(timeconsume, best_loss, model_out_fullpath))                
            if (mean_val_loss <= best_loss + 1e-5): 
                best_loss = mean_val_loss
                model_out_fullpath = "{}/best_model_epoch_{}(val_loss{}).pth".format(model_out_path, epoch, best_loss)
                torch.save(checkpoint, model_out_fullpath)
                print('time consume: {:.1f}s, So far best loss: {:.4f}, Checkpoint saved to {}' .format(timeconsume, best_loss, model_out_fullpath))
            elif epoch%50 == 0:
                model_out_fullpath = "{}/model_epoch_{}.pth".format(model_out_path, epoch)
                torch.save(checkpoint, model_out_fullpath)
                print("Epoch [{}/{}] done. Epoch Loss {:.4f}. Checkpoint saved to {}"
                    .format(epoch, opt.epoch, epoch_loss, model_out_fullpath))

    #     #     # ================================================================== #
    #     #     #                        Tensorboard Logging                         #
    #     #     # ================================================================== #
    #     #     logger.add_scalar('Train/val_loss',mean_val_loss,epoch)
    #     #     logger.add_scalar('Train/epoch_loss',epoch_loss,epoch)
    #     #     print("log file saved to {}\n"
    #     #         .format(log_path))
