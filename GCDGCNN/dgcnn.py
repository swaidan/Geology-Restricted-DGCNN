"""
This is a modified version of DGCNN
The backbone code is taken from the original author below:
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, depth_lim=3, idx=None):
    height = 32
    batch_size = x.size(0)
    num_points = x.size(2)
    width = int(num_points / height)
    k = int(np.ceil(k * width))
    x = x.view(batch_size, -1, num_points)
    if idx is None and depth_lim < 32:
        new_x = x.reshape(batch_size, x.shape[1], height, width)
        for i in range(height):
            low_lim = max(0, i - depth_lim)
            high_lim = min(height, i + depth_lim + 1)
            range_lim = high_lim - low_lim
            x_limited = new_x[:, :, low_lim:high_lim, :].reshape(batch_size, x.shape[1], -1)
            if i == 0:
                idx = knn(x_limited, k=k)[:, :width, :] + (width * i)
            elif i == height - 1:
                new_idx = knn(x_limited, k=k)[:, -width:, :] - ((range_lim - 1)*width) + (width * i)
                idx = torch.concat((idx, new_idx), dim=1)
            else:
                if low_lim == 0:
                    new_idx = knn(x_limited, k=k)[:, i*width:(i+1)*width, :] - (i*width) + (width * i)
                    idx = torch.concat((idx, new_idx), dim=1)
                elif high_lim == height:
                    ii = height - i
                    new_idx = knn(x_limited, k=k)[:, -ii*width:(-ii+1)*width, :] - ((range_lim - ii)*width)  + (width * i)
                    idx = torch.concat((idx, new_idx), dim=1)
                else:
                    new_idx = knn(x_limited, k=k)[:, depth_lim*width:(depth_lim+1)*width, :] - (depth_lim*width) + (width * i)
                    idx = torch.concat((idx, new_idx), dim=1)
    else:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class ModDGCNN(nn.Module):
    def __init__(self, k=40, depth_lim=3, convs=[256, 128, 128, 64, 64], out_dim=64*6):
        super(ModDGCNN, self).__init__()
        self.k = k
        self.depth_lim = depth_lim
        self.layers = nn.ModuleList()

        for i in range(len(convs) - 1):
            self.layers.append(
                nn.Sequential(nn.Conv2d(convs[i]*2, convs[i+1], kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(convs[i+1]),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   )
            )


        self.lastconv = nn.Sequential(nn.Conv1d(np.sum(convs[1:]), out_dim, kernel_size=1, bias=False))
        

    def forward(self, x):
        for i, conv in enumerate(self.layers):
            if i == 0:
                x0 = get_graph_feature(x, k=self.k, depth_lim=self.depth_lim)
                x0 = conv(x0)
                x0 = x0.max(dim=-1, keepdim=False)[0]
                xnew = x0
                x_con = x0
            else:
                xnew = get_graph_feature(xnew, k=self.k, depth_lim=self.depth_lim)
                xnew = conv(xnew)
                xnew = xnew.max(dim=-1, keepdim=False)[0]
                x_con = torch.cat((x0, xnew), dim=1)
                x0 = x_con
    
        x = self.lastconv(x_con)

        return x