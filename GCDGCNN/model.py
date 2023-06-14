import torch
import torch.nn as nn
from pretrainedmodels import resnet34
from GCDGCNN.dgcnn import ModDGCNN


class GCDGCNN(nn.Module):
    def __init__(self, out_channels=6, depth_lim=2, k=0.5):
        super(GCDGCNN, self).__init__()

        resnet_layers = resnet34()
        self.layer1, self.layer2, self.layer3, = resnet_layers.layer1, resnet_layers.layer2, resnet_layers.layer3

        self.layer0 = torch.nn.Sequential(torch.nn.Conv2d(2, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        torch.nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(16, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True)
        )

        self.dgcnn = ModDGCNN(k=k, depth_lim=depth_lim, convs=[256*1, 128*1, 128*1, 64*1, 64*1], out_dim=out_channels*64)

        self.shuffle = nn.PixelShuffle(8)

    def forward(self, x):
        x_size = x.size()

        depth = torch.arange(0, 1, step=1/x.shape[2]).reshape(x.shape[2], 1)
        depth = depth * torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        x = torch.concat((x, depth.to(x.device)), dim=1)

        gx = self.layer0(x)
        B, C, H, W = gx.size()
        gx = self.layer1(gx)
        gx = self.layer2(gx)
        gx = self.layer3(gx)
        B, C, H, W = gx.size()
        gx = self.dgcnn(gx.reshape(B, C, -1))

        gx = gx.reshape(B, -1, H, W)

        gx = self.shuffle(gx)

        gx = gx[:, :, :x_size[2], :x_size[3]]

        return gx