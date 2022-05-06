from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def get_graph_feature(x, k=20):
    # x = x.squeeze()
    k = min(k,x.size()[2])
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    global device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx.to(device) + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512,k=10):
        super(DGCNN, self).__init__()
        self.k = k # Num of nearest neighbors to use
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x,self.k)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x,x4,x3



class DGCNN_leaky(nn.Module):
    def __init__(self, emb_dims=512,k=10):
        super(DGCNN_leaky, self).__init__()
        self.k = k # Num of nearest neighbors to use
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x,self.k)
        x = F.leaky_relu_(self.bn1(self.conv1(x)),negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.leaky_relu_(self.bn2(self.conv2(x)),negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.leaky_relu_(self.bn3(self.conv3(x)),negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.leaky_relu_(self.bn4(self.conv4(x)),negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.leaky_relu_(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x,x4,x3
    
    
    


if __name__ == '__main__':
    #  (taille batch, xyz , nb pts)
    sim_data = Variable(torch.rand(2,3,3))
    print(sim_data.size())
    dgcnn=DGCNN_leaky(1088)
    out,_,_=dgcnn(sim_data)
    print(out.size())
