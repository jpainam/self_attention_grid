import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms


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


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
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


# Define the ResNet50-based Model#
class ResNetAttention(nn.Module):

    def __init__(self, num_class):
        super(ResNetAttention, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, num_class)
        self.conv_attention = nn.Sequential(
            nn.Conv2d(2048, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        #nn.init.kaiming_normal_(self.conv_attention.weight)
        self.conv_attention.apply(weights_init_kaiming)
        #self.conv_proc_detail = nn.Sequential(
        #    nn.Conv2d(2048, 2048, 3, stride=1, padding=1),
        #    nn.BatchNorm2d(2048),
        #    nn.ReLU(inplace=True)
        #)
        #nn.init.kaiming_normal_(self.conv_proc_detail.weight)
        #self.conv_proc_detail.apply(weights_init_kaiming)


    def reprocess(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        self.pool4 = F.max_pool2d(x, 2, 2)


    def attend(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        self.pool = F.max_pool2d(x, 2, 2)
        b, c, h, w = self.pool.size()
        b, c, h, w = x.size()
        att = self.conv_attention(x)
        #att = F.upsample(att.view(b, 1, h, w), scale_factor=2, mode='bilinear', align_corners=True)
        att_mask = F.softmax(att.view(b, 1 * h * w), -1).view(b, 1, h, w)
        return att_mask

    def crop(self, x, h_multiple=4, w_multiple=2):
        """ Adjusts the high resolution image feature size to be multiple of 7

        Args:
            x: input features
            multiple: multiple to adjust to

        Returns: cropped feature map

        """
        b, c, h, w = x.size()
        h_ = h % h_multiple
        w_ = w % w_multiple
        return x[:, :, 0:(h - h_), 0:(w - w_)]

    def forward(self, x):
        b1, c1, h1, w1 = x.size()
        x2 = F.upsample(x, scale_factor=2)
        b2, c2, h2, w2 = x2.size()
        att = self.attend(x)
        if (h1, w1) != (h2, w2):
            self.reprocess(x2)

        x_high = self.pool4
        b, c, h, w = x_high.size()
        b_att, c_att, h_att, w_att = att.size()
        #x_high = F.max_pool2d(x_high, 2, 2)
        #x_high = x_high[:, :, 0:(h - h_att), 0:(w - w_att)]
        #print(x_high.size())
        #print(att.size())
        #exit(0)
        # x_high = self.conv_proc_detail(x_high)
        #att = F.upsample(att, scale_factor=2, mode='bilinear', align_corners=True)
        attended = x_high * att
        #attended = F.normalize(attended.view(b, c, -1).sum(-1), 2, -1)
        x = self.model.avgpool(attended)
        x = F.normalize(x.view(b, c, -1).sum(-1), 2, -1)
        #x = torch.squeeze(x)
        x = self.classifier(x)
        return x


'''
x = att.view(b, 1, h, w)
        x = x.detach().cpu()
        #x = x.detach().cpu().numpy()
        import torch
        #x = torch.randn(32, 1, 3, 3)#
        #print(type(x))
        #print(x.size())
        #from torchvision import transforms
        #transform = transforms.Compose([
        #   transforms.ToPILImage(),
        #    transforms.Resize(size=24),
        #    transforms.ToTensor()
        #])
        #x = [transform(x_) for x_ in x]
        #save_image(x, './images/att.png')
print(x.size())

        img = x.detach().cpu().numpy()
        img = img[0]
        img = img[0]
        import numpy as np
        img = np.transpose(img, (1, 0))
        print(img.shape)
        from scipy.misc import imsave
        imsave("./images/1dimconv1.png", img)
        exit(0)
        save_image(x, "images/conv1.png")
'''


# Define the DenseNet121-based Model
class DenseNetModel(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
