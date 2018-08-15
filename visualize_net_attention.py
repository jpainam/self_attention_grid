from __future__ import print_function, division

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
import argparse
from attention_module import AttentionModule
from basic_layers import ResidualBlock

parser = argparse.ArgumentParser(description='Visualizing the attention map')
parser.add_argument('--model_path', default='ran', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/paul/datasets/market1501/pytorch', type=str, help='training dir path')
parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
opt = parser.parse_args()

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.set_device(int(os.environ["CUDA_VISIBLE_DEVICES"]))

num_class = 751
# Image Preprocessing 
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor()])

test_dataset = datasets.ImageFolder(os.path.join(opt.data_dir, 'query'), transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batchsize, shuffle=True,
                                          num_workers=2)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class ResidualAttentionModel(nn.Module):
    def __init__(self, num_class):
        super(ResidualAttentionModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule(256, 256, (56, 56), (28, 28), (14, 14))
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule(512, 512, (28, 28), (14, 14), (7, 7))
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule(1024, 1024, (14, 14), (7, 7), (4, 4))
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.num_class = num_class
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out1 = self.attention_module1(out)
        out = self.residual_block2(out1)
        out2 = self.attention_module2(out)
        out = self.residual_block3(out2)
        # print(out.data)
        out3 = self.attention_module3(out)
        out = self.residual_block4(out3)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out1, out2, out3 = [g.pow(2).mean(1) for g in (out1, out2, out3)]
        return out, x, out1, out2, out3


model = ResidualAttentionModel(num_class=num_class).cuda()
save_model = os.path.join("./model", opt.model_path)
model.load_state_dict(torch.load(save_model))
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import imsave


def visualize_test_set(model, test_loader, how_many=100):
    num = 0
    if not os.path.exists('./output'):
        os.makedirs('./output')
    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs, image, feature1, feature2, feature3 = model(images)
        # save image
        print('process batch %d' % num)
        image = image.view(image.size(0), 3, 224, 224)
        feature1 = feature1[:, 0:3, :, :]  # .view(feature1.size(0), 3, 28, 28)
        feature2 = feature2[:, 0:3, :, :]  # .view(feature2.size(0), 3, 28, 28)
        feature3 = feature3[:, 0:3, :, :]  # .view(feature3.size(0), 3, 28, 28)
        idx = 0
        for feature in (feature1, feature2, feature3):
            #feature = denorm(feature.data)
            img_array = feature.detach().cpu().numpy()
            for i in range(opt.batchsize):
                img = img_array[i]
                img = np.moveaxis(img, 0, -1)
                img = Image.fromarray(img.astype('uint8'), 'RGB')
                img = img.resize((64, 128), Image.ANTIALIAS)
                imsave('./output/feature%d-%d.png' % (idx + 1, num + 1), img)
            idx = idx + 1

        save_image(denorm(image.data), './output/image-%d.png' % (num + 1))

        '''
        save_image(denorm(feature1.data), './output/feature1-%d.png' % (num + 1))
        save_image(denorm(feature2.data), './output/feature2-%d.png' % (num + 1))
        save_image(denorm(feature3.data), './output/feature3-%d.png' % (num + 1))

        save_image(denorm(feature1.data), './output/feature1-%d.eps' % (num + 1))
        save_image(denorm(feature2.data), './output/feature2-%d.eps' % (num + 1))
        save_image(denorm(feature3.data), './output/feature3-%d.eps' % (num + 1))'''
        num += 1
        print("Success {}".format(num))
        if num == how_many:
            break


if __name__ == "__main__":
    visualize_test_set(model=model, test_loader=test_loader, how_many=100)
