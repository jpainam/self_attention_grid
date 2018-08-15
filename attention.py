import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.misc import  imsave
import numpy as np
import random

class AttentionModule(nn.Module):
    """ Online attention Layer"""
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inter1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.inter2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)


    def forward(self, x):
        b, c, h, w = x.size()
        #print(x.size())
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inter1(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.softmax(x)
        x = torch.reshape(x, (b, c, h, w))
        #out = input_img + x
        #out = x.detach().cpu().numpy()
        #print(out[0].shape)
        #img = np.transpose(out[0], (1, 2, 0))
        #imsave("./attention/%d_img.png" % (random.randint(0, 100)), img)
        return x


if __name__ == "__main__":
    # net = ft_net(751)
    net = AttentionModule()
    # print(net)
    input = Variable(torch.FloatTensor(8, 3, 64, 128))
    output = net(input)
    print('net output size:')
    print(output.size())
