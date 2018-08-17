# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from residual_attention_network import ResidualAttentionModel

######################################################################
# Options
# python test.py --model_path default/model_39.pth
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_path', default='model_best', type=str, help='Model path')
parser.add_argument('--test_dir', default='/home/paul/datasets/market1501/pytorch', type=str, help='./test_data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query')

opt = parser.parse_args()

test_dir = opt.test_dir
#torch.cuda.set_device(int(os.environ["CUDA_VISIBLE_DEVICES"]))
num_class = 751
data_transforms = transforms.Compose([
    transforms.Resize((160, 64), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                  ['gallery', 'query', 'multi-query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in
               ['gallery', 'query', 'multi-query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

#---- Single GPU training --
def load_network(network):
    save_path = os.path.join('./model', opt.model_path)
    network.load_state_dict(torch.load(save_path))
    return network

#-----multi-gpu training---------
def load_network1(network):
    save_path = os.path.join('./model', opt.model_path)
    state_dict = torch.load(save_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:] # remove `module.`
        new_state_dict[namekey] = v
    # load params
    network.load_state_dict(new_state_dict)
    return network

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
mquery_path = image_datasets['multi-query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)
mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
from resnet_attention import ResNetAttention
from model import ResNetAttentionModel
model_structure = ResNetAttention(num_class)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer

model.model.fc = nn.Sequential()
model.classifier = nn.Sequential()


# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model, dataloaders['gallery'])
query_feature = extract_feature(model, dataloaders['query'])
if opt.multi:
    mquery_feature = extract_feature(model, dataloaders['multi-query'])

# Save to Matlab for check
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('./pytorch_result.mat', result)
if opt.multi:
    result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
    scipy.io.savemat('./multi_query.mat', result)