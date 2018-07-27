# -*- coding: utf-8 -*-

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.serialization import load_lua
from torchvision.utils import save_image

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self, nlabels):
        """
        Constructor
        """
        super().__init__()
        self.nlabels = nlabels
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_attention = nn.Conv2d(512, 1, 1)
        nn.init.kaiming_normal_(self.conv_attention.weight)
        self.conv_proc_detail = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_proc_detail.weight)
        self.fc6 = nn.Linear(512 * 7 * 7 + 512, 4096)
        nn.init.kaiming_normal_(self.fc6.weight)
        self.fc7 = nn.Linear(4096, 4096)
        nn.init.kaiming_normal_(self.fc7.weight)
        self.fc8 = nn.Linear(4096, self.nlabels)
        nn.init.kaiming_normal_(self.fc8.weight)

    '''def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch weights

        Args:
            path: path for the luatorch weights
        """
        model = load_lua(path, unknown_classes=True)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if hasattr(layer, "weight"):
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = layer.weight.view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = layer.bias.view_as(self_layer.bias)[...]
                # else:
                #     self_layer = getattr(self, "fc%d" % (block))
                #     block += 1
                #     self_layer.weight.data[...] = layer.weight.view_as(self_layer.weight)[...]
                #     self_layer.bias.data[...] = layer.bias.view_as(self_layer.bias)[...]
    '''
    def get_vgg_parameters(self):
        """ Function to obtain the vgg pretrained parameters. Useful for freezing.

        Returns: pre-trained parameters

        """
        parameters = []
        for block in self.block_size:
            for num in range(block):
                layer = getattr(self, "conv_%d_$d"(block + 1, num + 1))
                parameters += list(layer.parameters())
        return parameters

    def get_att_parameters(self):
        """ Function to obtain the attention parameters.

        Returns: attention params.

        """
        parameters = []
        parameters += list(self.conv_attention.parameters())
        parameters += list(self.conv_proc_detail.parameters())
        parameters += list(self.fc6.parameters())
        parameters += list(self.fc7.parameters())
        parameters += list(self.fc8.parameters())
        return parameters

    def attend(self, x):
        """ Computes the attention mask on the input images

        Args:
            x: input images

        Returns: attention mask

        """
        b, c, h, w = x.size()
        self.input_size = (h, w)
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        self.pool4 = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(self.pool4))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        self.pool5 = F.max_pool2d(x, 2, 2)
        b, c, h, w = self.pool5.size()
        att = self.conv_attention(self.pool5).view(b, h * w)
        return F.softmax(att, -1).view(b, 1, h, w)

    def crop(self, x, multiple=7):
        """ Adjusts the high resolution image feature size to be multiple of 7

        Args:
            x: input features
            multiple: multiple to adjust to

        Returns: cropped feature map

        """
        b, c, h, w = x.size()
        h_ = h % multiple
        w_ = w % multiple
        return x[:, :, 0:(h - h_), 0:(w - w_)]

    def reprocess(self, x):
        """ Reprocesses high resolution images

        Args:
            x: input image

        """
        b, c, h, w = x.size()
        self.input_size = (h, w)
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        self.pool4 = F.max_pool2d(x, 2, 2)

    def classify(self, x_high, attention):
        """ Final classifier. Applies attention and outputs class logits.

        Args:
            x_high: high res image
            attention: attention mask

        Returns:

        """
        x_high = self.conv_proc_detail(self.crop(x_high))
        b, c, h, w = x_high.size()
        x_high = x_high.view(b, c, 7, h // 7, 7, w // 7)
        attended = x_high * attention.view(b, 1, 7, 1, 7, 1)
        attended = F.normalize(attended.view(b, c, -1).sum(-1), 2, -1)
        x_low = F.normalize(self.pool5.view(b, 512 * 7 * 7), 2, -1)
        concat = torch.cat([attended, x_low], -1)
        fc6 = F.relu(self.fc6(concat), True)
        fc6 = F.dropout(fc6, 0.5, inplace=True)
        fc7 = F.relu(self.fc7(fc6), True)
        fc7 = F.dropout(fc7, 0.5, inplace=True)
        return self.fc8(fc7)

    def forward(self, x1):
        """ Model forward function

        Args:
            x1: low res image
            x2: [high res] image

        Returns:

        """
        b1, c1, h1, w1 = x1.size()
        #b2, c2, h2, w2 = x2.size()
        att = self.attend(x1)
        #print(att.size())
        ##att = att.view(att.size(0), 3, 64, 128))
        #save_image(denorm(att.data), './attention/vggnetd.png')
        #exit(0)
        #if (h1, w1) != (h2, w2):
        self.reprocess(x1)
        return self.classify(self.pool4, att)


if __name__ == "__main__":
    import numpy as np

    #im = cv2.imread('images/ak.png') - np.array([129.1863, 104.7624, 93.5940]).reshape((1, 1, 3))
    #im2 = cv2.resize(im, (448, 448))
    #im = im.transpose((1, 2, 0)).reshape((1, 3, 224, 224))
    #im2 = im2.transpose((1, 2, 0)).reshape((1, 3, 448, 448))
    #im = torch.Tensor(im).cuda()
    #im2 = torch.Tensor(im2).cuda()
    #model = VGG_16(nlabels=751).cuda()
    from torchvision import models
    model = models.vgg16(pretrained=True).cuda()
    print(model)
    exit(0)
    #print(model(im, im).max())
    #print(model(im, im2).max())
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)