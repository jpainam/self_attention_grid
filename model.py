import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ResNetAttentionModel(nn.Module):

    def __init__(self, num_class):
        super(ResNetAttentionModel, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, num_class)

        # conv attention for layer 1 (256x56x56)
        self.conv_att1 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # conv attention for layer 2 (512x28x28)
        #self.conv_att2 = nn.Conv2d(256, 1, 1)
        # conv attention for layer 3 (512x28x28)
        self.conv_att2 = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        # conv attention for layer 3 (1024x14x14)
        self.conv_att3 = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def attend(self, attention, x):
        pool = F.max_pool2d(x, 2, 2)
        b, c, h, w = pool.size()
        att = attention(pool).view(b, 1 * h * w)
        att = F.upsample(att.view(b, 1, h, w), scale_factor=2, mode='bilinear', align_corners=True)
        b, c, h, w = att.size()
        att = F.softmax(att.view(b, 1 * h * w), -1)
        return att.view(b, 1, h, w)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        att1 = self.attend(self.conv_att1, x)
        x = x * att1

        x = self.model.layer2(x)
        x = self.model.layer3(x)
        att3 = self.attend(self.conv_att3, x)
        x = x * att3
        x = self.model.layer4(x)
        b, c, h, w = x.size()
        x = self.model.avgpool(x)
        x = F.normalize(x.view(b, c, -1).sum(-1), 2, -1)
        # classifier
        x = self.classifier(x)
        return x

import cv2
if __name__ == "__main__":
    #im = cv2.imread('img.jpg') - np.array([129.1863, 104.7624, 93.5940]).reshape((1, 1, 3))
    #im = cv2.resize(im, (224, 224))
    #im = im.transpose((1, 2, 0)).reshape((1, 3, 224, 224))
    #im2 = im2.transpose((1, 2, 0)).reshape((1, 3, 448, 448))
    #im = torch.Tensor(im).cuda()
    #im2 = torch.Tensor(im2).cuda()

    x = torch.randn(3, 3, 128, 64)
    from torchvision import transforms
    transform = transforms.Compose([
       transforms.ToPILImage(),
        transforms.Resize(size=24),
        transforms.ToTensor()
    ])
    x = [transform(x_) for x_ in x]
    model = ResNetAttentionModel(751).cuda()

    pred = model(x)
    img = x.detach().cpu().numpy()
    img = img[0]
    img = img[:3, :, :]
    import numpy as np

    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    from scipy.misc import imsave, imresize

    img = imresize(img, (128, 64))
    imsave("./images/att4.png", img)
    exit(0)
