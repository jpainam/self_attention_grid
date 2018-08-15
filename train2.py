# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
# from torchvision.utils import save_image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os
import copy
from random_erasing import RandomErasing
from reid_attention import VGG_16
from image_folder_loader import ImageFolderLoader

plt.ion()

######################################################################
# Options
# python train2.py --use_dense --train_all
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_path', default='market1501', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/paul/datasets/market1501/pytorch', type=str, help='training dir path')
parser.add_argument('--batchsize', default=24, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--resume', action='store_true', help='Resume training')
opt = parser.parse_args()

data_dir = opt.data_dir
model_path = opt.model_path
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using GPU")
else:
    print("Not using GPU")
    exit(0)
# torch.cuda.set_device(0)
TRAIN = 'train'
VAL = 'val'

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms_1 = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally.
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}
data_transforms_2 = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally.
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((448, 448), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}

if opt.erasing_p > 0:
    data_transforms_1[TRAIN] = data_transforms_1[TRAIN] + [
        RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    data_transforms_2[TRAIN] = data_transforms_1[TRAIN] + [
        RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

image_datasets = {
    x: ImageFolderLoader(
        os.path.join(data_dir, x),
        transform_1=data_transforms_1[x],
        transform_2=data_transforms_2[x]
    )
    for x in [TRAIN, VAL]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=opt.batchsize,
        shuffle=True, num_workers=16
    )
    for x in [TRAIN, VAL]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL]}

for x in [TRAIN, VAL]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

print("Classes: ")
class_names = image_datasets[TRAIN].classes


# print(image_datasets[TRAIN].classes)#


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)
        scheduler.step()
        for i, data in enumerate(dataloaders[TRAIN]):
            # Use half training dataset
            # if i >= train_batches / 2:
            #    break
            input1, input2, labels, _ = data
            if use_gpu:
                input1, input2, labels = Variable(input1.cuda()), \
                                         Variable(input2.cuda()), Variable(labels.cuda())
            else:
                input1, input2, labels = Variable(input1), Variable(input2), Variable(labels)

            optimizer.zero_grad()
            outputs = vgg(input1, input2)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            print(loss.item())
            if i % 100 == 0:
                print("\rTraining batch {}/{}-{}".format(i, train_batches, loss.item()))
            acc_train += torch.sum(preds == labels.data)
            del input1, input2, labels, outputs, preds
            torch.cuda.empty_cache()

        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train / dataset_sizes[TRAIN]
        avg_acc = acc_train / dataset_sizes[TRAIN]

        vgg.train(False)
        vgg.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloaders[VAL]):
                input1, input2, labels, _ = data
                if use_gpu:
                    input1, input2, labels = input1.cuda(), input2.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = vgg(input1, input2)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss_val += loss.item()
                acc_val += torch.sum(preds == labels.data)
                if i % 100 == 0:
                    print("\rValidation batch {}/{}-{}".format(i, val_batches, loss.item()))
                del input1, input2, labels, outputs, preds
                torch.cuda.empty_cache()

        avg_loss_val = loss_val / dataset_sizes[VAL]
        avg_acc_val = acc_val / dataset_sizes[VAL]

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
        torch.save(vgg16.state_dict(), './model/model%s.pth' % epoch)

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    plt.savefig("visualize.png")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


def visualize_model(vgg, num_images=6):
    was_training = vgg.training

    # Set model for evaluation
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in enumerate(dataloaders[VAL]):
        input1, input2, labels = data
        size = input1.size()[0]

        if use_gpu:
            input1, input2, labels = Variable(input1.cuda(), volatile=True), \
                                     Variable(input2.cuda(), volatile=True), \
                                     Variable(labels.cuda(), volatile=True)
        else:
            input1, input2, labels = Variable(input1, volatile=True), \
                                     Variable(input2, volatile=True), Variable(labels, volatile=True)

        outputs = vgg(input1, input2)
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(input1.size()[0])]

        print("Ground truth:")
        show_databatch(input1.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(input1.data.cpu(), predicted_labels)

        del input1, input2, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()

        images_so_far += size
        if images_so_far >= num_images:
            break

    vgg.train(mode=was_training)  # Revert model back to original training state


vgg16 = VGG_16(len(class_names))
if use_gpu:
    vgg16.cuda()  # .cuda() will move everything to the GPU side

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    vgg16 = nn.DataParallel(vgg16)

# vgg16.load_state_dict(torch.load("./model/model.pth"))
# print(vgg16.classifier[6].out_features)  # 1000

# Freeze training for all layers
# for param in vgg16.features.parameters():
#    param.require_grad = False

# Newly created modules have require_grad=True by default
# num_features = vgg16.classifier[6].in_features
# features = list(vgg16.classifier.children())[:-1]  # Remove last layer
# features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with 4 outputs
# vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
# print(vgg16)

resume_training = opt.resume

if resume_training:
    print("Loading pretrained model..")
    vgg16.load_state_dict(torch.load('./model/model.pth'))
    print("Loaded!")

if __name__ == "__main__":
    # Get a batch of training data
    input1, input2, classes, _ = next(iter(dataloaders[TRAIN]))
    print(input1.size())
    print(input2.size())
    # show_databatch(input1, classes)
    # show_databatch(input2, classes)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)
    torch.save(vgg16.state_dict(), './model/model.pth')
    # eval_model(vgg16, criterion)
    # visualize_model(vgg16, num_images=4)
