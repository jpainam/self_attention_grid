# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from torchvision import datasets, transforms
import matplotlib

matplotlib.use('agg')
import time
import os
from model import ResNetModel, DenseNetModel
from random_erasing import RandomErasing
import json

######################################################################
# Options
# python train.py --use_dense --train_all
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_path', default='market1501', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/paul/datasets/market1501/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0.8, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet')
opt = parser.parse_args()

data_dir = opt.data_dir
model_path = opt.model_path
if opt.use_dense:
    model_path = os.path.join(model_path, 'dense')

transform_train_list = [
    transforms.Resize((288, 144), interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=16)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# inputs, classes = next(iter(dataloaders['train']))

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                # print("Current Loss {}".format(loss.item()))
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]
                # last_model_wts = model.state_dict()
                save_network(model, epoch)

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                best_model_wts = model.state_dict()

            y_loss[phase].append(valid_epoch_loss)
            y_err[phase].append(1.0 - valid_epoch_acc)

        print('Epoch [{}/{}] train loss: {:.4f} acc: {:.4f} '
              'valid loss: {:.4f} acc: {:.4f}'.format(epoch, num_epochs - 1, train_epoch_loss,
                                                      train_epoch_acc, valid_epoch_loss, valid_epoch_acc))

        print()
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best')
    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []


######################################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'model_%s.pth' % epoch_label
    save_path = os.path.join('./model', model_path, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(int(os.environ["CUDA_VISIBLE_DEVICES"]))


if opt.use_dense:
    model = DenseNetModel(len(class_names))
else:
    model = ResNetModel(len(class_names))

if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer = optim.SGD([
    {'params': base_params, 'lr': 0.01},
    {'params': model.model.fc.parameters(), 'lr': 0.1},
    {'params': model.classifier.parameters(), 'lr': 0.1}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dir_name = os.path.join('./model', model_path)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

# save opts
with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

if __name__ == "__main__":
    start_time = time.time()
    print(model)
    exit(0)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                        num_epochs=150)
    print('Training time: {:10f} minutes'.format((time.time() - start_time) / 60))
